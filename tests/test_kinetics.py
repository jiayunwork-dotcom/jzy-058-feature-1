"""动力学内核测试：Monod 比增长速率的基本形状。"""

from decimal import Decimal

import pytest

from app.kinetics import monod_growth_rate


def mu(s, mu_max=Decimal("0.5"), ks=Decimal("10")):
    return monod_growth_rate(Decimal(str(s)), mu_max, ks)


def test_monod_zero_substrate_is_zero():
    assert mu(0) == Decimal(0)


def test_monod_half_saturation_is_half_mu_max():
    # S = Ks 时 μ = μmax/2，这是最容易手工核对的点
    assert mu(10) == Decimal("0.25")


def test_monod_monotonically_increasing_and_bounded():
    values = [mu(s) for s in (1, 5, 10, 50, 100, 1000, 1_000_000)]
    assert values == sorted(values)
    for value in values:
        assert Decimal(0) < value < Decimal("0.5")
    # 高基质下渐近 μmax 但不越过
    assert mu(1_000_000) > Decimal("0.4999")


def test_monod_matches_formula():
    # μ = 0.5·30/(10+30) = 0.375
    assert mu(30) == Decimal("0.375")


def test_monod_rejects_negative_substrate():
    with pytest.raises(ValueError):
        monod_growth_rate(Decimal("-1"), Decimal("0.5"), Decimal("10"))
