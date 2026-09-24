"""活性污泥反应器动态仿真：从初值沿时间轴积分到（逼近）稳态。

动态方程（单级 CSTR，无衰减，与 :mod:`app.solver` 同一模型口径）::

    dX/dt = (μ(S) − D) · X
    dS/dt = D · (S0 − S) − μ(S) · X / Y

模块职责边界（不另起炉灶，动力学与稳态全部复用既有内核）：
  * Monod 比增长速率 μ 直接取自 :mod:`app.kinetics`；
  * 参数合法性由 :mod:`app.solver.build_parameters` 把关，初值、时长、
    输出粒度的补充校验同样抛 :class:`ParameterValidationError`，与
    稳态路径共用同一套结构化错误信封；
  * 末端收敛点用 :func:`app.solver.solve_steady_state` 一并解出带回，
    动态轨迹与稳态代数解必须落到同一个点，这条自洽性由测试钉死；
  * 本模块只负责装配右端函数、校验仿真设置、驱动
    :mod:`app.integrator` 做带误差控制的时间推进。

冲刷在动态上的表现与稳态判定一致：只要稳态求解器判定冲刷
（``D ≥ μmax``，或分式解 ``S ≥ S0`` 使正污泥平衡不存在），从
``S ≤ S0`` 出发时 μ(S) ≤ D 严格成立，污泥被单调冲刷至 0，基质
回升至 S0。
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_FLOOR
from typing import Any

from app.integrator import IntegrationStats, integrate
from app.kinetics import monod_growth_rate
from app.solver import (
    ParameterValidationError,
    ProcessParameters,
    SteadyStateSolution,
    parse_decimal,
    solve_steady_state,
)

# 输出点上限：调用方可控疏密，但绝不允许一次仿真刷出失控的点列
MAX_OUTPUT_POINTS = 5_000
# 调用方不给粒度时的默认输出点数（含 t=0 与 t=T 两端）
DEFAULT_OUTPUT_POINTS = 200
MIN_OUTPUT_POINTS = 2


@dataclass(frozen=True)
class InitialState:
    """反应器初始状态：初始基质浓度与初始污泥浓度。"""

    s: Decimal
    x: Decimal


@dataclass(frozen=True)
class SimulationSpec:
    """一次动态仿真的完整设置（参数已经过校验）。

    ``num_points`` 与 ``interval`` 互斥：前者指定等距点数，后者指定
    等距时间间隔（末端时刻 T 始终包含）。两者皆空时取默认点数。
    """

    parameters: ProcessParameters
    initial: InitialState
    duration: Decimal
    num_points: int | None = None
    interval: Decimal | None = None


@dataclass(frozen=True)
class TrajectoryPoint:
    """轨迹上一个带时刻的状态点。"""

    time: float
    s: float
    x: float
    mu: float


@dataclass(frozen=True)
class SimulationResult:
    """一次仿真的完整产物：轨迹 + 同参数稳态参照 + 计算量观测。"""

    points: list[TrajectoryPoint]
    steady_state: SteadyStateSolution
    duration: float
    stats: IntegrationStats


def build_initial_state(*, s: Any, x: Any) -> InitialState:
    """校验初始状态：两者都须为有限、非负数值（允许 S 或 X 恰好为 0）。"""
    errors: dict[str, str] = {}
    try:
        s_dec = parse_decimal(s, "S_init")
    except ParameterValidationError as exc:
        errors.update(exc.errors)
        s_dec = Decimal(0)
    try:
        x_dec = parse_decimal(x, "X_init")
    except ParameterValidationError as exc:
        errors.update(exc.errors)
        x_dec = Decimal(0)
    if "S_init" not in errors and s_dec < 0:
        errors["S_init"] = "初始基质浓度不能为负"
    if "X_init" not in errors and x_dec < 0:
        errors["X_init"] = "初始污泥浓度不能为负"
    if errors:
        raise ParameterValidationError(errors)
    return InitialState(s=s_dec, x=x_dec)


def build_simulation_spec(
    parameters: ProcessParameters,
    *,
    initial_state: InitialState,
    duration: Any,
    num_points: Any = None,
    interval: Any = None,
) -> SimulationSpec:
    """校验仿真时长与输出粒度，组装仿真设置。

    工艺参数本身沿用已校验的 :class:`ProcessParameters`，这里不重复
    把关。非法时长（非正/非数值）、点数越界、间隔非正、粒度导致点数
    超上限，一律走 :class:`ParameterValidationError`。
    """
    errors: dict[str, str] = {}

    try:
        duration_dec = parse_decimal(duration, "duration")
    except ParameterValidationError as exc:
        errors.update(exc.errors)
        duration_dec = Decimal(1)
    if "duration" not in errors and duration_dec <= 0:
        errors["duration"] = "仿真时长必须为正数"

    points: int | None = None
    interval_dec: Decimal | None = None

    if num_points is None and interval is None:
        # 调用方不给粒度：默认等距 200 点（含 0 与 T 两端）
        points = DEFAULT_OUTPUT_POINTS
    elif num_points is not None and interval is not None:
        # 显式 null 视作“未指定”，因此只有两者都给了非空值才算互斥
        errors["num_points"] = "num_points 与 interval 互斥，只能指定一种输出粒度"
    elif num_points is not None:
        if isinstance(num_points, bool) or not isinstance(num_points, int):
            errors["num_points"] = "输出点数必须是整数"
        elif num_points < MIN_OUTPUT_POINTS:
            errors["num_points"] = (
                f"输出点数至少为 {MIN_OUTPUT_POINTS}（需包含起点与终点）"
            )
        elif num_points > MAX_OUTPUT_POINTS:
            errors["num_points"] = f"输出点数不能超过上限 {MAX_OUTPUT_POINTS}"
        else:
            points = num_points
    elif interval is not None:
        try:
            interval_dec = parse_decimal(interval, "interval")
        except ParameterValidationError as exc:
            errors.update(exc.errors)
            interval_dec = Decimal(1)
        if "interval" not in errors and interval_dec <= 0:
            errors["interval"] = "输出时间间隔必须为正数"
        elif "interval" not in errors and duration_dec > 0:
            # 0, dt, 2dt, … 直到不超过 T 的最后一个整数倍点，T 始终补在末端
            full_steps = int(
                (duration_dec / interval_dec).to_integral_value(
                    rounding=ROUND_FLOOR
                )
            )
            total = full_steps + 1 + (1 if full_steps * interval_dec < duration_dec else 0)
            if total > MAX_OUTPUT_POINTS:
                errors["interval"] = (
                    f"该间隔下输出点数 {total} 超过上限 {MAX_OUTPUT_POINTS}，"
                    "请加大间隔或缩短仿真时长"
                )

    if errors:
        raise ParameterValidationError(errors)

    return SimulationSpec(
        parameters=parameters,
        initial=initial_state,
        duration=duration_dec,
        num_points=points,
        interval=interval_dec,
    )


def _output_grid(spec: SimulationSpec) -> list[float]:
    """生成从 0 到 T 的输出时刻序列（float），首尾必含。"""
    duration = float(spec.duration)
    if spec.num_points is not None:
        n = spec.num_points
        return [duration * i / (n - 1) for i in range(n)]

    assert spec.interval is not None
    interval = float(spec.interval)
    full_steps = int(
        (spec.duration / spec.interval).to_integral_value(rounding=ROUND_FLOOR)
    )
    times = [interval * i for i in range(full_steps + 1)]
    if times[-1] < duration:
        times.append(duration)
    return times


def simulate(spec: SimulationSpec) -> SimulationResult:
    """按仿真设置积分动态方程，返回逐时刻轨迹与同参数稳态参照。

    积分采用带局部误差控制与非负兜底的自适应 RK45
    （见 :mod:`app.integrator`），输出点只决定取样位置、不改变内部
    推进口径，因此取点更密不会改变末端收敛结果。
    """
    p = spec.parameters

    # Decimal 常量与 float 状态之间的唯一转换口：动力学严格复用
    # kinetics.monod_growth_rate，不在这里另写一份 Monod 公式。
    mu_max, ks = p.mu_max, p.ks
    d_f = float(p.dilution)
    s0_f = float(p.s0)
    y_f = float(p.y)

    def growth(s_f: float) -> float:
        return float(
            monod_growth_rate(Decimal(repr(s_f)), mu_max, ks)
        )

    def rhs(_t: float, state: tuple[float, float]) -> tuple[float, float]:
        s_f, x_f = state
        # 非负兜底之外再守一道：任何中间态基质为负都视为试探失败，
        # 交由积分器收窄步长，绝不把负 S 外推进 Monod。
        if s_f < 0.0:
            raise ValueError("中间态基质浓度为负")
        mu = growth(s_f)
        dx = (mu - d_f) * x_f
        ds = d_f * (s0_f - s_f) - mu * x_f / y_f
        return (ds, dx)

    times = _output_grid(spec)
    y0 = (float(spec.initial.s), float(spec.initial.x))
    output = integrate(rhs, 0.0, y0, times)

    points = [
        TrajectoryPoint(time=t, s=state[0], x=state[1], mu=growth(state[0]))
        for t, state in zip(output.times, output.states)
    ]

    # 稳态参照与冲刷判定完全交给既有稳态求解器，两条路径必须自洽
    steady = solve_steady_state(p)

    return SimulationResult(
        points=points,
        steady_state=steady,
        duration=times[-1],
        stats=_stats_snapshot(output.stats),
    )


def _stats_snapshot(stats: IntegrationStats) -> IntegrationStats:
    """积分统计按值留档，避免结果之间共享可变对象。"""
    return IntegrationStats(
        accepted_steps=stats.accepted_steps,
        rejected_steps=stats.rejected_steps,
        rhs_evaluations=stats.rhs_evaluations,
    )
