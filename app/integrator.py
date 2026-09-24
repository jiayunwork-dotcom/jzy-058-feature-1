"""自适应时间推进内核：Dormand–Prince RK45（带局部误差控制）。

为什么不用定步长显式欧拉：
  本问题的时间尺度在稀释率贴近临界值时会拉得极长（慢特征值趋于 0），
  而初值离稳态很远时局部导数又可能很陡。固定步长要么在缓变段浪费
  大量步数，要么在陡变段放大截断误差，甚至把浓度推成物理上不可能的
  负数。这里改用嵌入式 5(4) 阶 Runge–Kutta（Dormand–Prince，FSAL）：
  每一步同时给出五阶解与四阶解，二者之差作为局部误差估计，按误差
  自动放大/缩小步长，误差超差的步直接作废重推。

物理兜底（与误差控制双保险）：
  * 推进中若出现 NaN/Inf、或右端函数在试探阶段报错（例如中间态把
    基质探成负值），该步作废、缩小步长重试；
  * 五阶解只允许出现与容差同量级的微小负越界，就地夹回 0；明显
    的负值同样作废重推——整条轨迹绝不输出负浓度；
  * 步长缩到数值下限仍无法通过误差/物理检查，抛出
    :class:`IntegrationError`，绝不带着坏结果继续；
  * 内部步数设硬上限，防止任何工况把计算量拖失控。

积分器只认右端函数 ``f(t, y) -> dy/dt``，对活性污泥动力学一无所知；
动力学关系统一由 :mod:`app.dynamics` 借助 :mod:`app.kinetics` 装配。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt
from typing import Callable, Sequence

# 二维状态（基质 S、污泥 X），向量以 (s, x) 元组表示
Vector = tuple[float, float]
RHS = Callable[[float, Vector], Vector]

# ---------- Dormand–Prince 5(4) Butcher 系数（导入时一次性算成 float） ----------

C2, C3, C4, C5, C6 = 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1.0

A21 = 1 / 5
A31, A32 = 3 / 40, 9 / 40
A41, A42, A43 = 44 / 45, -56 / 15, 32 / 9
A51, A52, A53, A54 = 19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729
A61, A62, A63, A64, A65 = (
    9017 / 3168,
    -355 / 33,
    46732 / 5247,
    49 / 176,
    -5103 / 18656,
)
# 第五阶权重即最后一行（FSAL：第七次求值等于下一步的第一次）
B1, B3, B4, B5, B6 = 35 / 384, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84
# 误差权重 = 五阶权重 − 四阶权重（第二、第七级四阶权重为 0/1）
E1, E3, E4, E5, E6, E7 = (
    71 / 57600,
    -71 / 16695,
    71 / 1920,
    -17253 / 339200,
    22 / 525,
    -1 / 40,
)

SAFETY = 0.9          # 步长更新安全系数
MIN_FACTOR = 0.2      # 接受步的单步收缩下限
MAX_FACTOR = 5.0      # 单步放大上限
REJECT_FACTOR = 0.5   # 物理/误差拒绝时的固定收缩比

# 内部（计算）步数硬上限：accepted + rejected 累计，防计算量失控
DEFAULT_MAX_STEPS = 200_000

# 相对/绝对精度默认值：工艺量级下既能平滑收敛，又不浪费步数；
# 留出一个量级余量，使极端初值下的陡瞬态不必把步长压到失控
DEFAULT_RTOL = 1e-6
DEFAULT_ATOL = 1e-9


class IntegrationError(RuntimeError):
    """积分器无法在步数/步长下限内得到满足误差与物理约束的推进。"""


@dataclass
class IntegrationStats:
    """一次积分的计算量观测（便于排查/测试，不参与 HTTP 输出）。"""

    accepted_steps: int = 0
    rejected_steps: int = 0
    rhs_evaluations: int = 0


@dataclass(frozen=True)
class IntegrationOutput:
    """与请求输出时刻逐点对齐的积分结果。"""

    times: list[float] = field(default_factory=list)
    states: list[Vector] = field(default_factory=list)
    stats: IntegrationStats = field(default_factory=IntegrationStats)


def _rms_norm(values: Sequence[float], scales: Sequence[float]) -> float:
    """按分量误差标度归一化的均方根范数。"""
    total = 0.0
    for value, scale in zip(values, scales):
        ratio = value / scale
        total += ratio * ratio
    return sqrt(total / len(values))


def _is_finite_vector(vec: Vector) -> bool:
    return all(value == value and abs(value) < 1e300 for value in vec)


def _initial_step_size(
    rhs: RHS,
    t0: float,
    y0: Vector,
    f0: Vector,
    span: float,
    rtol: float,
    atol: float,
    stats: IntegrationStats,
) -> float:
    """Hairner/Wanner 的首步步长启发式：用一阶导数估一个可靠的起步尺度。"""
    sc = tuple(atol + abs(value) * rtol for value in y0)
    d0 = _rms_norm(y0, sc)
    d1 = _rms_norm(f0, sc)
    if d0 < 1e-5 or d1 < 1e-5:
        h0 = 1e-6
    else:
        h0 = 0.01 * d0 / d1
    h0 = min(h0, span) if span > 0 else h0

    y1 = tuple(y_i + h0 * f_i for y_i, f_i in zip(y0, f0))
    try:
        f1 = rhs(t0 + h0, y1)
        stats.rhs_evaluations += 1
    except (ValueError, ArithmeticError):
        return max(h0 * 1e-3, 1e-12)
    d2 = _rms_norm(tuple(a - b for a, b in zip(f1, f0)), sc) / h0

    if max(d1, d2) <= 1e-15:
        h1 = max(1e-6, h0 * 1e-3)
    else:
        h1 = (0.01 / max(d1, d2)) ** 0.2
    return min(100.0 * h0, h1, span) if span > 0 else min(100.0 * h0, h1)


def integrate(
    rhs: RHS,
    t0: float,
    y0: Vector,
    output_times: Sequence[float],
    *,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
    max_steps: int = DEFAULT_MAX_STEPS,
) -> IntegrationOutput:
    """沿时间轴积分一阶 ODE 组，在 ``output_times`` 逐点取样。

    ``output_times`` 必须从 ``t0`` 起严格递增。内部步长由误差控制
    自动决定；每个输出时刻通过“收窄当前步”精确落点，不依赖跨大步长
    插值，因此输出疏密不影响末端状态的计算口径——这是“取点更密不
    改变末端稳态”这条自洽性的数值保障。
    """
    if not output_times or output_times[0] != t0:
        raise ValueError("output_times 必须以 t0 为首个输出时刻")

    span = output_times[-1] - t0
    if span < 0:
        raise ValueError("输出时刻必须单调不减")
    # 步长数值下限：绝对地板与相对全程的地板取大
    h_floor = max(1e-12, span * 1e-12)

    stats = IntegrationStats()
    t = t0
    y = (float(y0[0]), float(y0[1]))

    k1 = rhs(t, y)
    stats.rhs_evaluations += 1
    h = max(
        _initial_step_size(rhs, t, y, k1, span, rtol, atol, stats),
        h_floor,
    )

    times: list[float] = [t0]
    states: list[Vector] = [y]

    def attempt(step: float):
        """推进一步，返回 (接受?, 新状态, FSAL斜率, 误差范数, 建议步长)。

        任何中间态把右端函数逼到报错（如负基质进 Monod）、产生非有限
        值或明显负浓度，都判定本步不可接受。
        """
        try:
            k2 = rhs(
                t + C2 * step,
                (y[0] + step * A21 * k1[0], y[1] + step * A21 * k1[1]),
            )
            k3 = rhs(
                t + C3 * step,
                (
                    y[0] + step * (A31 * k1[0] + A32 * k2[0]),
                    y[1] + step * (A31 * k1[1] + A32 * k2[1]),
                ),
            )
            k4 = rhs(
                t + C4 * step,
                (
                    y[0] + step * (A41 * k1[0] + A42 * k2[0] + A43 * k3[0]),
                    y[1] + step * (A41 * k1[1] + A42 * k2[1] + A43 * k3[1]),
                ),
            )
            k5 = rhs(
                t + C5 * step,
                (
                    y[0]
                    + step
                    * (A51 * k1[0] + A52 * k2[0] + A53 * k3[0] + A54 * k4[0]),
                    y[1]
                    + step
                    * (A51 * k1[1] + A52 * k2[1] + A53 * k3[1] + A54 * k4[1]),
                ),
            )
            k6 = rhs(
                t + C6 * step,
                (
                    y[0]
                    + step
                    * (
                        A61 * k1[0]
                        + A62 * k2[0]
                        + A63 * k3[0]
                        + A64 * k4[0]
                        + A65 * k5[0]
                    ),
                    y[1]
                    + step
                    * (
                        A61 * k1[1]
                        + A62 * k2[1]
                        + A63 * k3[1]
                        + A64 * k4[1]
                        + A65 * k5[1]
                    ),
                ),
            )
        except (ValueError, ArithmeticError):
            return False, y, k1, 0.0, step * REJECT_FACTOR

        y_new = (
            y[0]
            + step
            * (B1 * k1[0] + B3 * k3[0] + B4 * k4[0] + B5 * k5[0] + B6 * k6[0]),
            y[1]
            + step
            * (B1 * k1[1] + B3 * k3[1] + B4 * k4[1] + B5 * k5[1] + B6 * k6[1]),
        )
        if not _is_finite_vector(y_new):
            return False, y, k1, 0.0, step * REJECT_FACTOR

        # 物理兜底：明显的负越界说明截断误差已破坏物理意义，作废重推；
        # 只把与状态量级相称的微小抖动夹回 0，整条轨迹恒不为负。
        clamped: list[float] = []
        for old, new in zip(y, y_new):
            neg_limit = 1e-10 + 1e-8 * max(1.0, abs(old))
            if new < -neg_limit:
                return False, y, k1, 0.0, step * REJECT_FACTOR
            clamped.append(0.0 if new < 0.0 else new)
        y_safe = (clamped[0], clamped[1])

        try:
            k7 = rhs(t + step, y_safe)
        except (ValueError, ArithmeticError):
            return False, y, k1, 0.0, step * REJECT_FACTOR
        if not _is_finite_vector(k7):
            return False, y, k1, 0.0, step * REJECT_FACTOR

        stats.rhs_evaluations += 6  # k2..k7
        sc = tuple(
            atol + rtol * max(abs(old), abs(new))
            for old, new in zip(y, y_safe)
        )
        err_vec = (
            step
            * (
                E1 * k1[0] + E3 * k3[0] + E4 * k4[0]
                + E5 * k5[0] + E6 * k6[0] + E7 * k7[0]
            ),
            step
            * (
                E1 * k1[1] + E3 * k3[1] + E4 * k4[1]
                + E5 * k5[1] + E6 * k6[1] + E7 * k7[1]
            ),
        )
        error = _rms_norm(err_vec, sc)

        if error > 1.0:
            factor = max(0.1, SAFETY * error ** (-1 / 5))
            return False, y, k1, error, step * factor
        if error == 0.0:
            factor = MAX_FACTOR
        else:
            factor = min(MAX_FACTOR, max(MIN_FACTOR, SAFETY * error ** (-1 / 5)))
        return True, y_safe, k7, error, step * factor

    def step_until(target: float) -> None:
        nonlocal t, y, k1, h
        while t < target - 1e-14 * max(1.0, span):
            remaining = target - t
            shortened = h > remaining
            step = remaining if shortened else h
            accepted, y_candidate, k_next, _error, suggested = attempt(step)
            if stats.accepted_steps + stats.rejected_steps >= max_steps:
                raise IntegrationError(
                    f"内部步数超过上限 {max_steps}，请检查工况或拆短仿真时长"
                )
            if accepted:
                t += step
                y = y_candidate
                k1 = k_next  # FSAL：本步末斜率即下一首斜率
                stats.accepted_steps += 1
                # 控制器的建议是按“这一步实际长度”给的：为对齐输出点而
                # 临时收窄的步只把建议作为下限，下一段恢复自然尺度 h，
                # 避免一次收窄把后续所有步永久压小
                if shortened:
                    h = max(h, suggested, h_floor)
                else:
                    h = max(suggested, h_floor)
            else:
                stats.rejected_steps += 1
                # 被拒的一定是按当前自然尺度（或其与剩余区间的较小值）
                # 推出的步，直接采纳控制器的收窄建议
                h = max(suggested, h_floor)
                if step <= h_floor and h <= h_floor:
                    raise IntegrationError(
                        "步长已缩至数值下限仍无法满足误差/物理约束"
                    )

    for target in output_times[1:]:
        if target < t:
            raise ValueError("输出时刻必须单调不减")
        step_until(target)
        times.append(target)
        states.append(y)

    return IntegrationOutput(times=times, states=states, stats=stats)
