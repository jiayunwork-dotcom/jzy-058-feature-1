"""自适应 RK45 积分器自身性质测试（与活性污泥动力学无关的通用内核）。

钉死三件事：
  1. 已知解析解的线性问题（指数衰减、线性填充）积分到精度量级；
  2. 刚性度不同的问题靠自适应步长都能完成，输出点疏密不改变末端；
  3. 右端函数在负状态上报错时，积分器收窄步长绕开，输出恒非负；
  4. 非有限输出 / 超步数预算时明确抛 IntegrationError，不产坏数据。
"""

from __future__ import annotations

import pytest

from app.integrator import (
    DEFAULT_MAX_STEPS,
    IntegrationError,
    integrate,
)


def test_exponential_decay_with_analytic_solution():
    # y' = -k y, y(0)=1 -> y = e^{-kt}
    k = 0.7
    out = integrate(lambda t, y: (-k * y[0], 0.0), 0.0, (1.0, 0.0),
                    [i * 0.5 for i in range(21)])
    # 全局误差随步数累积到几个 rtol，属自适应积分的正常量级
    for t, state in zip(out.times, out.states):
        assert state[0] == pytest.approx(2.718281828459045 ** (-k * t), rel=5e-6)
    assert out.stats.rejected_steps >= 0


def test_linear_fill_with_analytic_solution():
    # y' = D(S0 - y), y(0)=0 -> y = S0(1 - e^{-Dt})
    d, s0 = 0.3, 100.0
    out = integrate(lambda t, y: (d * (s0 - y[0]), 0.0), 0.0, (0.0, 0.0),
                    [i * 2.0 for i in range(11)])
    for t, state in zip(out.times, out.states):
        assert state[0] == pytest.approx(
            s0 * (1.0 - 2.718281828459045 ** (-d * t)), rel=5e-6
        )


def test_output_density_does_not_change_endpoint():
    rhs = lambda t, y: (-0.5 * y[0], 0.0)  # noqa: E731
    coarse = integrate(rhs, 0.0, (3.0, 0.0), [0.0, 5.0, 10.0])
    dense = integrate(rhs, 0.0, (3.0, 0.0), [i * 0.05 for i in range(201)])
    # 末端差异远小于工艺结论所需精度：取点疏密不改变收敛点
    assert dense.states[-1][0] == pytest.approx(coarse.states[-1][0], abs=1e-7)


def test_negative_state_rejected_by_rhs_is_handled_by_shrinking():
    # 右端在基质为负时主动报错（模拟 Monod 对负 S 的拒绝）。取一条
    # 真值贴着 0、初斜率很陡的轨迹：粗糙的大步显式推进会把中间态探
    # 到负，积分器必须靠收窄步长绕开，整条轨迹保持非负。
    def rhs(t, y):
        s, x = y
        if s < 0.0:
            raise ValueError("负基质")
        # s' = -40(s - 0.01)：从 0 出发单调升到 0.01，起步时任何
        # 过度外推都可能越过 0；x' = -0.5x 同步衰减
        return (-40.0 * (s - 0.01), -0.5 * x)

    out = integrate(
        rhs, 0.0, (0.0, 5.0), [i * 1.5 for i in range(21)],
        rtol=1e-8, atol=1e-11,
    )
    assert all(s >= 0.0 and x >= 0.0 for s, x in out.states)
    assert out.states[-1][0] == pytest.approx(0.01, abs=1e-6)
    # t=30: x = 5·e^{-15} ≈ 1.5e-6
    assert out.states[-1][1] == pytest.approx(0.0, abs=1e-5)


def test_nonfinite_output_raises_integration_error_eventually():
    # y' = y^2 在有限时间爆破；超预算时必须报错而不是返回 Inf
    with pytest.raises(IntegrationError):
        integrate(
            lambda t, y: (y[0] * y[0], 0.0),
            0.0,
            (1.0, 0.0),
            [0.0, 5.0],
            max_steps=5_000,
        )


def test_max_steps_is_capped():
    assert DEFAULT_MAX_STEPS >= 10_000


def test_output_grid_must_start_at_t0():
    with pytest.raises(ValueError):
        integrate(lambda t, y: (0.0, 0.0), 1.0, (0.0, 0.0), [0.0, 1.0])
