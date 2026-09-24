"""CSTR 动态仿真：沿时间轴积分基质/污泥耦合 ODE。

模型（与稳态核算同一套 Monod 动力学，无衰减项）::

    dX/dt = ( μ(S) − D ) · X          污泥：增殖 − 出流稀释
    dS/dt = D · (S0 − S) − μ(S)·X/Y   基质：进水补充 − 出流带走 − 微生物消耗
    μ(S)  = mu_max · S / (Ks + S)     见 :mod:`app.kinetics`

与稳态求解器（:mod:`app.solver`）的自洽关系：非冲刷工况下仿真跑得
足够久，末端状态收敛到同一组参数的稳态代数解；冲刷工况下 X 单调
趋零、S 回升至 S0。两条计算路径落到同一个点。

积分器：Dormand–Prince 嵌入式 RK45 对 + PI 步长控制（误差自适应），
不采用定步长显式推进——后者在稀释率贴近临界或初值远离稳态时会
放大误差、算出负浓度。每个被接受的步再做一次可行性钳制：浮点
残差造成的微小负浓度抬回 0，并按全物料衡算配对其搭档变量。
注意 S + X/Y 并非守恒量——对它求导得 dM/dt = D·(S0 − M)，即
M(t) = S0 + (M(0) − S0)·e^(−D·t) 有解析轨迹，钳制以该解析值为
基准，既不引入负浓度，也不凭空增减总物料。

数值口径：时间推进是数值近似，内部一律用 float（双精度）；
:mod:`app.solver` 的稳态判定仍用 Decimal 精确比较，二者分工不变。
所有函数均为纯函数，不持有任何跨调用可变状态，并发仿真互不干扰。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from app.kinetics import monod_growth_rate
from app.solver import (
    MAX_SCAN_POINTS,
    ParameterValidationError,
    ProcessParameters,
    _as_decimal,
    solve_steady_state,
)

# ---- 上限保护：输出点数与内部积分步数都有硬顶，防止计算量失控 ----
MAX_TRAJECTORY_POINTS = MAX_SCAN_POINTS  # 与扫描同一量级：10 000
MAX_DURATION = Decimal("1000000")        # 仿真时长上限（与参数同时间单位）
MAX_INTERNAL_STEPS = 2_000_000           # 积分器内部步数保险丝

# ---- 误差控制容差：足以让末端与稳态代数解的偏差远小于展示精度 ----
RTOL = 1e-9
ATOL = 1e-11

# PI 步长控制增益（Gustafsson 常用整定），比纯积分控制更平稳
_PI_BETA1 = 0.7
_PI_BETA2 = -0.4

# 步长变化限幅：宁可多走几步，也不让步长暴涨暴跳引起振荡
_STEP_GROWTH_MAX = 6.0
_STEP_SHRINK_MIN = 0.2
_SAFETY = 0.9

# 可行性钳制阈值：只兜浮点残差量级的负值；更大的负值说明积分器
# 本身出问题，直接报错而不是悄悄掩盖
_CLAMP_TOLERANCE = 1e-9

# Dormand–Prince RK45（DOPRI5）系数表
_RK_A = (
    (),
    (1 / 5,),
    (3 / 40, 9 / 40),
    (44 / 45, -56 / 15, 32 / 9),
    (19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729),
    (9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656),
    (35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84),
)
_RK_B5 = (35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0.0)
_RK_B4 = (
    5179 / 57600, 0.0, 7571 / 16695, 393 / 640,
    -92097 / 339200, 187 / 2100, 1 / 40,
)


class SimulationValidationError(ParameterValidationError):
    """动态仿真输入非法。继承参数校验错误，沿用同一套结构化错误信封。"""


@dataclass(frozen=True)
class InitialState:
    """反应器初始状态：基质浓度 s、污泥浓度 x，均须非负。"""

    s: float
    x: float


@dataclass(frozen=True)
class TrajectoryPoint:
    """轨迹上的一个采样点。"""

    t: float
    s: float
    x: float


@dataclass(frozen=True)
class SimulationResult:
    """一次动态仿真的完整结果。"""

    points: tuple[TrajectoryPoint, ...]
    is_washout: bool          # 该工况按稳态判定是否处于冲刷区
    converged: bool           # 末端导数是否已落到数值零（达到稳态）
    internal_steps: int       # 积分器实际接受的步数（审计用）


def _as_non_negative_decimal(value: Any, field: str) -> Decimal:
    """转 Decimal 并要求非负；类型/有限性检查与求解器同一口径。"""
    dec = _as_decimal(value, field)
    if dec < 0:
        raise SimulationValidationError({field: "不能为负"})
    return dec


def build_initial_state(*, s: Any, x: Any) -> InitialState:
    """校验并构造初始状态。负浓度（含 −0.0）一律拒绝。"""
    converted: dict[str, Decimal] = {}
    errors: dict[str, str] = {}
    for field, raw in (("S_init", s), ("X_init", x)):
        try:
            converted[field] = _as_non_negative_decimal(raw, field)
        except ParameterValidationError as exc:
            errors.update(exc.errors)
    if errors:
        raise SimulationValidationError(errors)
    return InitialState(s=float(converted["S_init"]), x=float(converted["X_init"]))


def build_time_grid(*, duration: Any, num_points: Any) -> tuple[float, int]:
    """校验仿真时长与输出点数，返回 (duration_float, num_points)。

    时长须为正且不超过上限；点数须为 [2, MAX_TRAJECTORY_POINTS] 内
    的整数（布尔、小数、字符串一律拒绝）。
    """
    errors: dict[str, str] = {}

    try:
        dur = _as_decimal(duration, "duration")
        if dur <= 0:
            errors["duration"] = "仿真时长必须为正数"
        elif dur > MAX_DURATION:
            errors["duration"] = f"仿真时长超过上限 {MAX_DURATION}"
    except ParameterValidationError as exc:
        errors.update(exc.errors)

    n: int | None = None
    if isinstance(num_points, bool) or not isinstance(num_points, int):
        errors["num_points"] = "输出点数必须是整数"
    else:
        n = num_points
        if n < 2:
            errors["num_points"] = "输出点数至少为 2（起点与终点）"
        elif n > MAX_TRAJECTORY_POINTS:
            errors["num_points"] = (
                f"输出点数超过上限 {MAX_TRAJECTORY_POINTS}，请降低采样密度"
            )

    if errors:
        raise SimulationValidationError(errors)
    assert n is not None
    return float(dur), n


@dataclass(frozen=True)
class _FloatParams:
    """积分循环内使用的参数口径（进入循环前一次性转换）。

    ``mu_max``/``ks`` 保持 Decimal 以直接复用 Monod 内核；
    其余为 float，避免循环里反复转换。
    """

    s0: float
    d: float
    mu_max: Decimal
    ks: Decimal
    y: float

    @classmethod
    def from_process(cls, params: ProcessParameters) -> "_FloatParams":
        return cls(
            s0=float(params.s0),
            d=float(params.dilution),
            mu_max=params.mu_max,
            ks=params.ks,
            y=float(params.y),
        )


def _rhs(s: float, x: float, p: _FloatParams) -> tuple[float, float]:
    """ODE 右端：返回 (dS/dt, dX/dt)。动力学复用 Monod 内核。"""
    mu = float(
        monod_growth_rate(Decimal(str(max(s, 0.0))), p.mu_max, p.ks)
    )
    return p.d * (p.s0 - s) - mu * x / p.y, (mu - p.d) * x


def _rk45_step(
    s: float, x: float, h: float, p: _FloatParams
) -> tuple[float, float, float, float]:
    """单步 Dormand–Prince 推进，返回 5 阶解与 4 阶解各一份。"""
    k_s = [0.0] * 7
    k_x = [0.0] * 7
    k_s[0], k_x[0] = _rhs(s, x, p)
    for stage in range(1, 7):
        a = _RK_A[stage]
        s_stage = s + h * sum(a[j] * k_s[j] for j in range(stage))
        x_stage = x + h * sum(a[j] * k_x[j] for j in range(stage))
        k_s[stage], k_x[stage] = _rhs(s_stage, x_stage, p)
    s5 = s + h * sum(b * k for b, k in zip(_RK_B5, k_s))
    x5 = x + h * sum(b * k for b, k in zip(_RK_B5, k_x))
    s4 = s + h * sum(b * k for b, k in zip(_RK_B4, k_s))
    x4 = x + h * sum(b * k for b, k in zip(_RK_B4, k_x))
    return s5, x5, s4, x4


def _clamp_feasible(
    s: float, x: float, m_ref: float, y: float
) -> tuple[float, float]:
    """把浮点残差造成的微小负浓度抬回可行域。

    ``m_ref`` 是全物料衡算量 M = S + X/Y 在当前时刻的解析值
    （M 服从 dM/dt = D·(S0−M)，可精确求出）。若某一分量轻微
    越界（|负值| ≤ 钳制阈值），将其归零并按 M 配对其搭档，
    既不引入负浓度，也不凭空增减总物料。真出现大幅负值说明
    积分器失稳，报错而非掩盖。
    """
    if s >= 0.0 and x >= 0.0:
        return s, x
    if s < 0.0 and -s <= _CLAMP_TOLERANCE * max(1.0, m_ref):
        # 基质轻微越界：归零，污泥按物料衡算补齐
        return 0.0, max(0.0, m_ref * y)
    if x < 0.0 and -x <= _CLAMP_TOLERANCE * max(1.0, m_ref * y):
        # 污泥轻微越界：归零，基质按物料衡算补齐
        return max(0.0, m_ref), 0.0
    raise ArithmeticError(
        f"积分器产生显著负浓度 (S={s!r}, X={x!r})，请检查工况与步长控制"
    )


def _hermite(
    t: float,
    t0: float, s0: float, x0: float, ds0: float, dx0: float,
    t1: float, s1: float, x1: float, ds1: float, dx1: float,
) -> tuple[float, float]:
    """三次 Hermite 插值：用段两端的状态与导数插出中间时刻。"""
    h = t1 - t0
    if h <= 0.0:
        return s1, x1
    u = (t - t0) / h
    u2 = u * u
    u3 = u2 * u
    h00 = 2 * u3 - 3 * u2 + 1
    h10 = u3 - 2 * u2 + u
    h01 = -2 * u3 + 3 * u2
    h11 = u3 - u2
    s = h00 * s0 + h10 * h * ds0 + h01 * s1 + h11 * h * ds1
    x = h00 * x0 + h10 * h * dx0 + h01 * x1 + h11 * h * dx1
    return s, x


def simulate(
    params: ProcessParameters,
    initial: InitialState,
    duration: float,
    num_points: int,
) -> SimulationResult:
    """沿 [0, duration] 积分 ODE，返回 num_points 个等距采样点。

    参数假定已过 :func:`app.solver.build_parameters` 校验；时长与
    点数假定已过 :func:`build_time_grid` 校验。积分自适应推进，
    采样只在被接受的解上做 Hermite 插值，不额外引入格式误差。
    """
    p = _FloatParams.from_process(params)
    # 冲刷判定与稳态求解器保持同一口径（含 S* ≥ S0 的物料衡算情形）
    is_washout = solve_steady_state(params).is_washout

    t_end = float(duration)
    dt_sample = t_end / (num_points - 1)

    # 全物料衡算量 M = S + X/Y 服从 dM/dt = D·(S0−M)，解析轨迹已知，
    # 可行性钳制以它为基准
    m0 = initial.s + initial.x / p.y

    def m_reference(t_now: float) -> float:
        return p.s0 + (m0 - p.s0) * math.exp(-p.d * t_now)

    # 初始步长：由初始导数量级估计，再限幅到采样间隔内
    ds_init, dx_init = _rhs(initial.s, initial.x, p)
    scale = max(abs(initial.s), abs(initial.x) / p.y, p.s0, 1.0)
    deriv_scale = max(abs(ds_init), abs(dx_init) / p.y, 1e-30)
    h = min(0.01 * scale / deriv_scale, dt_sample, t_end)
    h = max(h, t_end * 1e-12)

    points: list[TrajectoryPoint] = [TrajectoryPoint(0.0, initial.s, initial.x)]
    t = 0.0
    s, x = initial.s, initial.x
    # 当前 Hermite 插值段的端点状态与导数
    seg_t0, seg_s0, seg_x0, seg_ds0, seg_dx0 = t, s, x, ds_init, dx_init
    seg_t1, seg_s1, seg_x1, seg_ds1, seg_dx1 = t, s, x, ds_init, dx_init

    err_old = 1.0
    steps = 0
    sample_idx = 1

    while t < t_end:
        if steps >= MAX_INTERNAL_STEPS:
            raise ArithmeticError(
                f"积分步数超过上限 {MAX_INTERNAL_STEPS}，工况可能过于刚性"
            )
        h = min(h, t_end - t)

        s5, x5, s4, x4 = _rk45_step(s, x, h, p)
        # 归一化误差：每个分量按 atol + rtol·|值| 缩放
        err_s = abs(s5 - s4) / (ATOL + RTOL * max(abs(s5), abs(s)))
        err_x = abs(x5 - x4) / (ATOL + RTOL * max(abs(x5), abs(x)))
        err = max(err_s, err_x)

        if err <= 1.0:
            # 步被接受：可行性钳制后推进
            s_new, x_new = _clamp_feasible(
                s5, x5, m_reference(t + h), p.y
            )
            t_new = t + h
            ds_new, dx_new = _rhs(s_new, x_new, p)

            # 本段成为新的插值段
            seg_t0, seg_s0, seg_x0 = t, s, x
            seg_ds0, seg_dx0 = seg_ds1, seg_dx1
            seg_t1, seg_s1, seg_x1 = t_new, s_new, x_new
            seg_ds1, seg_dx1 = ds_new, dx_new

            # 吐出落在本段内的所有采样点（三次 Hermite 插值）
            while sample_idx < num_points:
                t_sample = sample_idx * dt_sample
                if t_sample > t_new + 1e-15:
                    break
                s_s, x_s = _hermite(
                    t_sample,
                    seg_t0, seg_s0, seg_x0, seg_ds0, seg_dx0,
                    seg_t1, seg_s1, seg_x1, seg_ds1, seg_dx1,
                )
                s_s, x_s = _clamp_feasible(
                    s_s, x_s, m_reference(t_sample), p.y
                )
                points.append(
                    TrajectoryPoint(min(t_sample, t_end), s_s, x_s)
                )
                sample_idx += 1

            t, s, x = t_new, s_new, x_new
            steps += 1

            # PI 控制更新步长（err_old 为上一步误差；误差可能恰好
            # 为 0——例如线性段——先抬到下限再取负幂）
            err_safe = max(err, 1e-16)
            factor = (
                _SAFETY
                * err_safe ** (-_PI_BETA1 / 5.0)
                * err_old ** (-_PI_BETA2 / 5.0)
            )
            factor = min(_STEP_GROWTH_MAX, max(_STEP_SHRINK_MIN, factor))
            h *= factor
            err_old = err_safe
        else:
            # 步被拒绝：缩小步长重试
            factor = max(_STEP_SHRINK_MIN, _SAFETY * err ** (-1.0 / 5.0))
            h *= factor

    # 收敛判定：末端导数相对量级是否已落到数值零
    ds_end, dx_end = _rhs(s, x, p)
    end_scale = max(abs(s), abs(x) / p.y, p.s0, 1.0)
    converged = (
        abs(ds_end) <= 1e-7 * end_scale
        and abs(dx_end) <= 1e-7 * end_scale * p.y
    )

    return SimulationResult(
        points=tuple(points),
        is_washout=is_washout,
        converged=converged,
        internal_steps=steps,
    )
