"""动力学内核：Monod 比增长速率。

本模块只承载与具体求解方式无关的动力学关系，稳态求解逻辑见
``app.solver``，二者保持独立，便于单独复核与复用。
"""

from decimal import Decimal


def monod_growth_rate(
    substrate: Decimal,
    mu_max: Decimal,
    half_saturation: Decimal,
) -> Decimal:
    """按 Monod 动力学计算比增长速率 μ。

    μ = μmax · S / (Ks + S)

    参数假定已经过 :mod:`app.solver` 的合法性校验，调用方不应直接
    传入未经校验的值（例如非正的 Ks）。
    当 S = 0 时返回 0，不做负数基质的外推。
    """
    if substrate < 0:
        raise ValueError("基质浓度不能为负")
    if substrate == 0:
        return Decimal(0)
    return mu_max * substrate / (half_saturation + substrate)
