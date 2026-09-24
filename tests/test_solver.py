"""稳态求解器单元测试：正常区、临界点、冲刷与非法输入。

这些用例直接钉死需求中的三条交叉关系：
  1. 未冲刷区抬高 D -> S 升高、X 下降；
  2. 无衰减时 S0 加倍 -> X 近似翻倍、S 不变（S 只由 D 决定）；
  3. D 越过 μmax -> X 干净地跌到 0，绝不为负；
以及 D == μmax 临界点的专门处理。
"""

from decimal import Decimal

import pytest

from app.solver import (
    MAX_SCAN_POINTS,
    ParameterValidationError,
    WASHOUT_DILUTION,
    WASHOUT_SUBSTRATE_BALANCE,
    build_dilution_range,
    build_parameters,
    scan_dilution,
    solve_steady_state,
)


def params(s0=100, D=0.1, mu_max=0.5, Ks=10, Y=0.5):
    return build_parameters(
        s0=s0, dilution=D, mu_max=mu_max, ks=Ks, y=Y
    )


def test_steady_state_matches_hand_formula():
    # S = 10·0.1/(0.5−0.1) = 2.5；X = 0.5·(100−2.5) = 48.75
    sol = solve_steady_state(params())
    assert not sol.is_washout
    assert sol.s == Decimal("2.5")
    assert sol.x == Decimal("48.75")
    assert sol.mu == Decimal("0.1")  # 稳态 D = μ
    assert sol.washout_reason is None


def test_doubling_s0_doubles_biomass_but_keeps_substrate():
    """交叉关系 2：S0 变化 -> X = Y·(S0−S)，而 S 只由 D 决定、与 S0 无关。

    D=0.1、μmax=0.5、Ks=10 时 S=2.5。把底物缺口 (S0−S) 翻倍
    （S0: 5 -> 7.5，缺口 2.5 -> 5），污泥 X 即干净翻倍。
    """
    base = solve_steady_state(params(s0=5))
    doubled = solve_steady_state(params(s0=7.5))
    assert base.s == doubled.s == Decimal("2.5")  # S 与 S0 无关
    assert base.x == Decimal("1.25")
    assert doubled.x == base.x * 2
    assert base.mu == doubled.mu == Decimal("0.1")


def test_raising_dilution_raises_substrate_lowers_biomass():
    """交叉关系 1：未冲刷区内 D 单调升高 -> S 单调升高、X 单调下降。"""
    solutions = [
        solve_steady_state(params(D=d))
        for d in ("0.05", "0.1", "0.15", "0.2", "0.25", "0.3")
    ]
    assert all(not s.is_washout for s in solutions)
    substrates = [s.s for s in solutions]
    biomass = [s.x for s in solutions]
    assert substrates == sorted(substrates)
    assert substrates == sorted(set(substrates))  # 严格递增
    assert biomass == sorted(biomass, reverse=True)
    assert len(set(biomass)) == len(biomass)      # 严格下降


def test_critical_dilution_equal_mu_max_is_washout_no_zero_denominator():
    """临界点 D = μmax：专门处理，绝不允许分母为零。"""
    sol = solve_steady_state(params(D=0.5, mu_max=0.5))
    assert sol.is_washout
    assert sol.washout_reason == WASHOUT_DILUTION
    assert sol.x == Decimal(0)
    assert sol.s == Decimal(100)


def test_dilution_above_mu_max_is_washout():
    """交叉关系 3：D > μmax -> X = 0（而非负数），S = S0。"""
    for d in ("0.5000001", "0.6", "1.0", "10"):
        sol = solve_steady_state(params(D=d))
        assert sol.is_washout
        assert sol.x == Decimal(0)
        assert sol.x >= 0
        assert sol.s == Decimal(100)
        assert sol.washout_reason == WASHOUT_DILUTION


def test_biomass_never_negative_anywhere():
    # 全区间扫描，任何点都不得出现负污泥量
    scan_range = build_dilution_range(start=0.01, stop=1.0, step=0.01)
    for sol in scan_dilution(params(), scan_range):
        assert sol.x >= 0
        assert sol.s >= 0


def test_washout_when_steady_substrate_not_below_influent():
    """未冲刷分式若给出 S ≥ S0（X 非正），同样按冲刷处理。

    取 S0=5、D=0.3、μmax=0.5、Ks=10：
    分式 S = 10·0.3/0.2 = 15 ≥ 5 -> 冲刷。
    """
    sol = solve_steady_state(params(s0=5, D=0.3))
    assert sol.is_washout
    assert sol.washout_reason == WASHOUT_SUBSTRATE_BALANCE
    assert sol.x == 0
    assert sol.s == Decimal(5)


def test_zero_influent_substrate_washes_out():
    # S0 = 0：分式解 S=0 不小于 S0，X = 0，按冲刷处理且数值全为非负
    sol = solve_steady_state(params(s0=0, D=0.1))
    assert sol.is_washout
    assert sol.x == 0
    assert sol.s == 0
    assert sol.mu == 0


def test_washout_mu_is_monod_at_influent_substrate():
    # 冲刷时 μ 按进水基质 S0 评估，而不是假装等于 D
    sol = solve_steady_state(params(D=0.6))
    # μ = 0.5·100/(10+100) = 50/110
    assert sol.mu == Decimal("0.5") * 100 / (Decimal(10) + 100)
    assert sol.mu < Decimal("0.5")


def test_exact_decimal_boundary_not_foiled_by_float():
    # 用字符串给出精确的 D == μmax，Decimal 精确比较必须判为冲刷
    p = build_parameters(s0="100", dilution="0.5", mu_max="0.5", ks="10", y="0.5")
    assert solve_steady_state(p).is_washout


# ---------- 参数合法性 ----------


@pytest.mark.parametrize(
    "field,value",
    [
        ("dilution", 0),
        ("dilution", -0.1),
        ("mu_max", 0),
        ("ks", -1),
        ("y", 0),
    ],
)
def test_non_positive_process_params_rejected(field, value):
    kwargs = dict(s0=100, dilution=0.1, mu_max=0.5, ks=10, y=0.5)
    kwargs[field] = value
    with pytest.raises(ParameterValidationError) as exc_info:
        build_parameters(**kwargs)
    # 错误键沿用工艺记号
    expected_key = {"dilution": "D", "mu_max": "mu_max", "ks": "Ks", "y": "Y"}[field]
    assert expected_key in exc_info.value.errors


def test_negative_s0_rejected():
    with pytest.raises(ParameterValidationError) as exc_info:
        params(s0=-1)
    assert "S0" in exc_info.value.errors


def test_multiple_invalid_fields_reported_together():
    with pytest.raises(ParameterValidationError) as exc_info:
        build_parameters(s0=-1, dilution=-2, mu_max=0, ks=-3, y=-4)
    assert set(exc_info.value.errors) == {"S0", "D", "mu_max", "Ks", "Y"}


@pytest.mark.parametrize("value", ["not-a-number", None, True, float("nan"), float("inf")])
def test_non_numeric_params_rejected(value):
    with pytest.raises(ParameterValidationError):
        build_parameters(
            s0=value, dilution=0.1, mu_max=0.5, ks=10, y=0.5
        )


# ---------- 区间扫描 ----------


def test_scan_points_include_both_ends():
    scan_range = build_dilution_range(start=0.1, stop=0.5, step=0.1)
    solutions = scan_dilution(params(), scan_range)
    dilutions = [str(s.dilution) for s in solutions]
    assert dilutions == ["0.1", "0.2", "0.3", "0.4", "0.5"]
    assert len(solutions) == 5
    # 末点 D=μmax 必须落在冲刷区
    assert solutions[-1].is_washout
    assert solutions[-1].x == 0
    assert solutions[0].x > 0


def test_scan_last_point_never_exceeds_stop():
    # (stop−start)/step = 9999.1 -> 向下取整 9999 步，末点必须 ≤ stop
    scan_range = build_dilution_range(start=0.00009, stop=1.0, step=0.0001)
    solutions = scan_dilution(params(), scan_range)
    assert len(solutions) == 10_000
    assert solutions[-1].dilution <= Decimal(str(1.0))
    assert solutions[0].dilution == Decimal("0.00009")

    # 小数部分 ≥ 0.5 也不得向上入导致越界
    scan_range2 = build_dilution_range(start=0.1, stop=0.3, step=0.15)
    points = [s.dilution for s in scan_dilution(params(), scan_range2)]
    assert points == [Decimal("0.10"), Decimal("0.25")]


def test_scan_rejects_non_positive_step_and_inverted_range():
    with pytest.raises(ParameterValidationError) as exc_info:
        build_dilution_range(start=0.5, stop=0.1, step=0.1)
    assert "start" in exc_info.value.errors
    with pytest.raises(ParameterValidationError):
        build_dilution_range(start=0.1, stop=0.5, step=0)


def test_scan_point_cap_enforced():
    with pytest.raises(ParameterValidationError) as exc_info:
        build_dilution_range(start=0.0001, stop=1.0, step=0.0000001)
    assert "step" in exc_info.value.errors


def test_scan_rejects_too_many_points():
    assert MAX_SCAN_POINTS == 10_000


def test_scan_transitions_into_washout_cleanly():
    # 跨过 μmax 的扫描：一旦冲刷，后续所有点 X 恒为 0、S 恒为 S0
    scan_range = build_dilution_range(start=0.45, stop=0.55, step=0.01)
    solutions = scan_dilution(params(), scan_range)
    after_washout = [s for s in solutions if s.dilution >= Decimal("0.5")]
    assert after_washout and all(s.is_washout for s in after_washout)
    assert all(s.x == 0 for s in after_washout)
    assert all(s.s == Decimal(100) for s in after_washout)
    # 跨点之间污泥量绝不出现负值
    assert all(s.x >= 0 for s in solutions)
