"""动态仿真内核测试：轨迹、与稳态解的自洽、冲刷、非负与粒度无关性。

这些用例直接钉死需求中的核心性质：
  1. 非冲刷工况仿真足够久，末端状态落在稳态求解器给出的点上；
  2. 冲刷工况（D ≥ μmax 及分式解 S ≥ S0）下污泥从任意初始投量
     单调趋零、基质回升至进水浓度；
  3. 整条轨迹 S、X 恒不为负；
  4. 输出点取密/取疏、按点数或按间隔取样，都不改变末端收敛点；
  5. 非法初值、时长、粒度被 ParameterValidationError 结构化拒绝。
"""

from __future__ import annotations

import pytest

from app.dynamics import (
    MAX_OUTPUT_POINTS,
    build_initial_state,
    build_simulation_spec,
    simulate,
)
from app.solver import (
    ParameterValidationError,
    build_parameters,
    solve_steady_state,
)


def params(s0=100, D=0.1, mu_max=0.5, Ks=10, Y=0.5):
    return build_parameters(s0=s0, dilution=D, mu_max=mu_max, ks=Ks, y=Y)


def run(p, si, xi, T, num_points=None, interval=None):
    spec = build_simulation_spec(
        p,
        initial_state=build_initial_state(s=si, x=xi),
        duration=T,
        num_points=num_points,
        interval=interval,
    )
    return simulate(spec)


# ---------- 正常收敛：动态末端 == 稳态代数解 ----------


def test_normal_regime_trajectory_converges_to_steady_state():
    # S* = 10·0.1/(0.5−0.1) = 2.5，X* = 0.5·(100−2.5) = 48.75
    p = params()
    result = run(p, si=0, xi=10, T=200, num_points=41)
    steady = solve_steady_state(p)
    final = result.points[-1]
    assert final.s == pytest.approx(float(steady.s), rel=1e-4)
    assert final.x == pytest.approx(float(steady.x), rel=1e-4)
    # 响应体里带回的稳态参照就是同一组参数下求解器给的点：两条路径同源
    assert result.steady_state.s == steady.s
    assert result.steady_state.x == steady.x


def test_different_initial_states_converge_to_same_point():
    # 从“近乎空池启动”和“高浓度扰动”出发，终点必须是同一个稳态
    p = params()
    steady = solve_steady_state(p)
    starts = [(0.0, 0.001), (100.0, 0.01), (50.0, 200.0), (0.0, 1000.0)]
    for si, xi in starts:
        result = run(p, si=si, xi=xi, T=400, num_points=21)
        final = result.points[-1]
        assert final.s == pytest.approx(float(steady.s), abs=2e-3)
        assert final.x == pytest.approx(float(steady.x), abs=2e-3)


def test_trajectory_can_overshoot_before_settling():
    # 从“基质满、污泥少”启动：S 先被快速消耗冲到稳态之下，再回升；
    # 动态路径必须能如实呈现这段过冲，而不是直接给终点
    p = params()
    result = run(p, si=100, xi=1, T=200, num_points=401)
    substrates = [pt.s for pt in result.points]
    assert min(substrates[10:]) < 2.5  # 中途确实下冲
    assert substrates[-1] == pytest.approx(2.5, abs=1e-3)


def test_trajectory_starts_at_initial_state_and_covers_duration():
    result = run(params(), si=12.0, xi=7.5, T=50, num_points=26)
    first, last = result.points[0], result.points[-1]
    assert first.time == 0.0
    assert first.s == 12.0 and first.x == 7.5
    assert last.time == pytest.approx(50.0)
    # 时间严格递增
    times = [pt.time for pt in result.points]
    assert times == sorted(times)
    assert len(set(times)) == len(times)


def test_zero_initial_biomass_stays_zero_and_substrate_fills():
    # X(0)=0：无泥可长（模型不接泥种），X 恒为 0，S 单调回升至 S0
    result = run(params(), si=50, xi=0, T=100, num_points=11)
    assert all(pt.x == 0.0 for pt in result.points)
    substrates = [pt.s for pt in result.points]
    assert substrates == sorted(substrates)
    assert substrates[-1] == pytest.approx(100, abs=1e-2)


# ---------- 冲刷：污泥单调趋零、基质回升 ----------


@pytest.mark.parametrize("x0", [0.001, 1.0, 50.0, 5000.0])
def test_dilution_washout_biomass_monotonic_to_zero_any_seed(x0):
    # D > μmax：任意初始污泥投量都被单调冲刷，S 回升到 S0
    p = params(D=0.6)
    assert solve_steady_state(p).is_washout
    result = run(p, si=100, xi=x0, T=120, num_points=121)
    xs = [pt.x for pt in result.points]
    assert all(x >= 0.0 for x in xs)
    assert all(xs[i + 1] <= xs[i] + 1e-12 for i in range(len(xs) - 1))
    assert xs[-1] < 1e-3 * max(x0, 1.0)
    assert result.points[-1].s == pytest.approx(100, abs=1e-2)


@pytest.mark.parametrize("x0", [0.01, 1.0, 50.0])
def test_critical_dilution_equal_mu_max_washes_out_slowly(x0):
    # D == μmax 临界点：慢冲刷，时间要给足；X 仍然单调不增
    p = params(D=0.5)
    assert solve_steady_state(p).is_washout
    result = run(p, si=100, xi=x0, T=400, num_points=81)
    xs = [pt.x for pt in result.points]
    assert all(xs[i + 1] <= xs[i] + 1e-12 for i in range(len(xs) - 1))
    assert xs[-1] < 1e-5
    assert result.points[-1].s == pytest.approx(100, abs=1e-3)


def test_balance_washout_substrate_fraction_above_influent():
    # S0=5、D=0.3：分式 S*=15 ≥ S0，正污泥平衡不存在 -> 冲刷
    p = params(s0=5, D=0.3)
    steady = solve_steady_state(p)
    assert steady.is_washout
    result = run(p, si=5, xi=10, T=150, num_points=31)
    xs = [pt.x for pt in result.points]
    assert all(xs[i + 1] <= xs[i] + 1e-12 for i in range(len(xs) - 1))
    assert xs[-1] < 1e-6
    assert result.points[-1].s == pytest.approx(5, abs=1e-3)


def test_washout_holds_with_substrate_below_influent_start():
    # 初始基质低于 S0 也不改变冲刷结论：X 单调衰减，S 最终回 S0
    p = params(D=0.6)
    result = run(p, si=10, xi=30, T=100, num_points=51)
    xs = [pt.x for pt in result.points]
    assert all(xs[i + 1] <= xs[i] + 1e-12 for i in range(len(xs) - 1))
    assert result.points[-1].s == pytest.approx(100, abs=1e-2)


# ---------- 临界稀释率附近（非冲刷侧）：慢但稳地收敛 ----------


def test_near_critical_below_mu_max_converges_given_enough_time():
    # D=0.45：S*=90、X*=5，慢特征值时间常数约 200，需长仿真
    p = params(D=0.45)
    steady = solve_steady_state(p)
    assert not steady.is_washout
    result = run(p, si=0, xi=1, T=2000, num_points=21)
    final = result.points[-1]
    assert final.s == pytest.approx(float(steady.s), rel=2e-3)
    assert final.x == pytest.approx(float(steady.x), rel=2e-3)


# ---------- 非负兜底 ----------


def test_concentrations_never_negative_across_far_starts():
    p = params()
    for si, xi in [(0.0, 0.0), (0.0, 1000.0), (1000.0, 1000.0),
                   (1e4, 1e4), (0.0001, 0.0001)]:
        result = run(p, si=si, xi=xi, T=200, num_points=101)
        assert all(pt.s >= 0.0 and pt.x >= 0.0 for pt in result.points), (si, xi)


def test_extreme_start_is_aborted_by_step_cap_not_negativity():
    # 极端陡瞬态超出内部步数预算时，积分器必须明确报错而不是吐出
    # 带负浓度或虚假收敛的轨迹（HTTP 层映射为 500）
    from app.integrator import IntegrationError

    p = params()
    with pytest.raises(IntegrationError):
        run(p, si=1e6, xi=1e6, T=200, num_points=21)


def test_non_negative_under_sweep_of_dilutions():
    for d in (0.05, 0.2, 0.35, 0.45, 0.49, 0.5, 0.51, 0.8):
        p = params(D=d)
        result = run(p, si=0, xi=80, T=300, num_points=61)
        assert all(pt.s >= 0.0 and pt.x >= 0.0 for pt in result.points), d


# ---------- 输出粒度不改变末端 ----------


def test_denser_output_does_not_change_terminal_state():
    p = params(D=0.2)
    coarse = run(p, si=0, xi=1, T=150, num_points=6)
    dense = run(p, si=0, xi=1, T=150, num_points=1001)
    steady = solve_steady_state(p)
    # 两者彼此一致（自适应全局误差量级 1e-6），且都落在稳态点上；
    # 误差控制按内部步推进，输出点只决定取样位置
    assert dense.points[-1].s == pytest.approx(coarse.points[-1].s, abs=1e-5)
    assert dense.points[-1].x == pytest.approx(coarse.points[-1].x, abs=1e-5)
    assert coarse.points[-1].s == pytest.approx(float(steady.s), abs=1e-3)
    assert coarse.points[-1].x == pytest.approx(float(steady.x), abs=1e-3)


def test_interval_grid_matches_point_grid_terminal_state():
    p = params()
    by_points = run(p, si=5, xi=20, T=120, num_points=13)
    by_interval = run(p, si=5, xi=20, T=120, interval=10)
    assert by_interval.points[-1].s == pytest.approx(
        by_points.points[-1].s, abs=1e-5
    )
    assert by_interval.points[-1].x == pytest.approx(
        by_points.points[-1].x, abs=1e-5
    )
    # interval 网格在非整除时刻把 T 补在末端
    times = [pt.time for pt in by_interval.points]
    assert times[0] == 0.0 and times[-1] == pytest.approx(120.0)


def test_interval_grid_appends_terminal_when_not_multiple():
    # T=105, dt=20 -> 0,20,40,60,80,100,105
    result = run(params(), si=10, xi=5, T=105, interval=20)
    times = [pt.time for pt in result.points]
    expected = [0, 20, 40, 60, 80, 100, 105]
    assert len(times) == len(expected)
    assert all(t == pytest.approx(e) for t, e in zip(times, expected))


# ---------- 非法输入：结构化拒绝 ----------


@pytest.mark.parametrize(
    "kwargs,field",
    [
        (dict(s=-1, x=1), "S_init"),
        (dict(s=1, x=-0.1), "X_init"),
        (dict(s="bad", x=1), "S_init"),
        (dict(s=True, x=1), "S_init"),
        (dict(s=float("nan"), x=1), "S_init"),
        (dict(s=1, x=float("inf")), "X_init"),
    ],
)
def test_invalid_initial_state_rejected(kwargs, field):
    with pytest.raises(ParameterValidationError) as exc:
        build_initial_state(**kwargs)
    assert field in exc.value.errors


@pytest.mark.parametrize(
    "kwargs,field",
    [
        (dict(duration=0), "duration"),
        (dict(duration=-10), "duration"),
        (dict(duration="soon"), "duration"),
        (dict(duration=10, num_points=1), "num_points"),
        (dict(duration=10, num_points=2.0), "num_points"),
        (dict(duration=10, num_points=MAX_OUTPUT_POINTS + 1), "num_points"),
        (dict(duration=10, interval=0), "interval"),
        (dict(duration=10, interval=-1), "interval"),
        (dict(duration=10, interval=0.001), "interval"),  # 点数超上限
        (dict(duration=10, num_points=5, interval=1), "num_points"),  # 互斥
    ],
)
def test_invalid_spec_rejected(kwargs, field):
    with pytest.raises(ParameterValidationError) as exc:
        build_simulation_spec(
            params(),
            initial_state=build_initial_state(s=1, x=1),
            **kwargs,
        )
    assert field in exc.value.errors


def test_default_grid_when_no_granularity():
    from app.dynamics import DEFAULT_OUTPUT_POINTS

    spec = build_simulation_spec(
        params(),
        initial_state=build_initial_state(s=1, x=1),
        duration=10,
    )
    result = simulate(spec)
    assert len(result.points) == DEFAULT_OUTPUT_POINTS
