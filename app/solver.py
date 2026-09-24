"""CSTR 稳态求解：物料衡算与 Monod 动力学联立。

稳态关系（无衰减项）::

    D = μ = μmax · S / (Ks + S)
    S = Ks · D / (μmax − D)
    X = Y · (S0 − S)

物理边界——冲刷（washout）::

    D ≥ μmax           -> 菌种来不及繁殖被冲光
    S(未冲刷分式) ≥ S0 -> X 非正，同样说明工况已越过冲刷边界

进入冲刷状态后统一取 X = 0、S = S0、μ = μmax·S0/(Ks+S0)，
绝不把 D ≥ μmax 代回分式造成零分母或负污泥量。

所有数值以 :class:`decimal.Decimal` 精确表示，避免 D 恰好等于
μmax 时因浮点误差错判区段；仅在 HTTP 输出时转成 float。
"""

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_FLOOR
from math import isfinite
from typing import Any, Iterable

from app.kinetics import monod_growth_rate

# 冲刷原因取值（结构化回报，保持稳定的英文标识便于上游程序分支判断）
WASHOUT_DILUTION = "dilution_at_or_above_mu_max"
WASHOUT_SUBSTRATE_BALANCE = "steady_state_substrate_not_below_influent"

# 区间扫描允许的最大点数，防止步长过小导致点数失控
MAX_SCAN_POINTS = 10_000


class ParameterValidationError(ValueError):
    """工况参数非法。``errors`` 以字段名为键、人类可读原因为值。"""

    def __init__(self, errors: dict[str, str]):
        self.errors = dict(errors)
        super().__init__("; ".join(f"{k}: {v}" for k, v in self.errors.items()))


@dataclass(frozen=True)
class ProcessParameters:
    """一组 CSTR 活性污泥稳态工况参数。"""

    s0: Decimal          # 进水基质浓度
    dilution: Decimal    # 稀释率 D
    mu_max: Decimal      # 最大比增长速率
    ks: Decimal          # 半饱和常数
    y: Decimal           # 产率系数

    def with_dilution(self, dilution: Decimal) -> "ProcessParameters":
        """返回替换稀释率后的同一组参数（用于区间扫描）。"""
        return ProcessParameters(
            s0=self.s0,
            dilution=dilution,
            mu_max=self.mu_max,
            ks=self.ks,
            y=self.y,
        )


@dataclass(frozen=True)
class SteadyStateSolution:
    """单点稳态解。"""

    s: Decimal           # 稳态出水基质浓度
    x: Decimal           # 稳态污泥浓度
    mu: Decimal          # 该工况下的实际比增长速率
    is_washout: bool
    washout_reason: str | None
    dilution: Decimal    # 回显本次使用的 D，便于扫描结果直接核对
    mu_max: Decimal      # 回显 μmax，临界点判断可复核


def _as_decimal(value: Any, field: str) -> Decimal:
    """把入参转成 Decimal；拒绝布尔、非数值、NaN 与无穷。"""
    if isinstance(value, bool):
        raise ParameterValidationError({field: "必须是数值，不能是布尔值"})
    if isinstance(value, Decimal):
        dec = value
    elif isinstance(value, int):
        dec = Decimal(value)
    elif isinstance(value, float):
        if not isfinite(value):
            raise ParameterValidationError({field: "必须是有限数值，不能为 NaN 或无穷"})
        dec = Decimal(str(value))
    elif isinstance(value, str):
        try:
            dec = Decimal(value.strip())
        except (InvalidOperation, AttributeError):
            raise ParameterValidationError({field: f"无法解析为数值: {value!r}"})
    else:
        raise ParameterValidationError({field: f"不支持的数值类型: {type(value).__name__}"})
    if not dec.is_finite():
        raise ParameterValidationError({field: "必须是有限数值，不能为 NaN 或无穷"})
    return dec


def build_parameters(
    *,
    s0: Any,
    dilution: Any,
    mu_max: Any,
    ks: Any,
    y: Any,
) -> ProcessParameters:
    """校验并构造工况参数。

    非法情形（返回 :class:`ParameterValidationError`，错误按字段归集）：
      * D、μmax、Ks、Y 任一非正；
      * S0 为负；
      * 缺字段、类型不是数值、NaN、无穷。
    """
    raw = {
        "S0": s0,
        "D": dilution,
        "mu_max": mu_max,
        "Ks": ks,
        "Y": y,
    }
    converted: dict[str, Decimal] = {}
    errors: dict[str, str] = {}
    for field, raw_value in raw.items():
        try:
            converted[field] = _as_decimal(raw_value, field)
        except ParameterValidationError as exc:
            errors.update(exc.errors)
    if errors:
        raise ParameterValidationError(errors)

    if converted["S0"] < 0:
        errors["S0"] = "进水基质浓度不能为负"
    for field in ("D", "mu_max", "Ks", "Y"):
        if converted[field] <= 0:
            errors[field] = "必须为正数"
    if errors:
        raise ParameterValidationError(errors)

    return ProcessParameters(
        s0=converted["S0"],
        dilution=converted["D"],
        mu_max=converted["mu_max"],
        ks=converted["Ks"],
        y=converted["Y"],
    )


def solve_steady_state(params: ProcessParameters) -> SteadyStateSolution:
    """求单级 CSTR 的稳态出水基质与污泥浓度，并自行判定冲刷区段。

    判定顺序经过刻意安排：
      1. D ≥ μmax 直接按冲刷处理——这是分式 S = Ks·D/(μmax−D)
         的奇点，临界点 D = μmax 也在此兜住，分母绝不取零；
      2. 未冲刷分式解出的 S 若不小于 S0，物料衡算给出 X ≤ 0，
         物理上仍属冲刷，同样回退到 X = 0、S = S0；
      3. 其余为正常稳态，S 由 D 决定，X = Y·(S0 − S)。
    """
    # 1) 稀释率临界/越界：Exact Decimal 比较，临界点 D == μmax 也算冲刷
    if params.dilution >= params.mu_max:
        return SteadyStateSolution(
            s=params.s0,
            x=Decimal(0),
            mu=monod_growth_rate(params.s0, params.mu_max, params.ks),
            is_washout=True,
            washout_reason=WASHOUT_DILUTION,
            dilution=params.dilution,
            mu_max=params.mu_max,
        )

    # 2) 未冲刷区的分式解（此时分母严格为正）
    steady_substrate = (
        params.ks * params.dilution / (params.mu_max - params.dilution)
    )

    # 3) S ≥ S0 => X = Y(S0−S) ≤ 0，说明实际已落在冲刷边界之外
    if steady_substrate >= params.s0:
        return SteadyStateSolution(
            s=params.s0,
            x=Decimal(0),
            mu=monod_growth_rate(params.s0, params.mu_max, params.ks),
            is_washout=True,
            washout_reason=WASHOUT_SUBSTRATE_BALANCE,
            dilution=params.dilution,
            mu_max=params.mu_max,
        )

    biomass = params.y * (params.s0 - steady_substrate)
    return SteadyStateSolution(
        s=steady_substrate,
        x=biomass,
        mu=params.dilution,  # 稳态下净增长恰好被出流带走：D = μ
        is_washout=False,
        washout_reason=None,
        dilution=params.dilution,
        mu_max=params.mu_max,
    )


@dataclass(frozen=True)
class DilutionRange:
    """稀释率扫描区间 [start, stop]，步长 step，两端均为闭区间。"""

    start: Decimal
    stop: Decimal
    step: Decimal


def build_dilution_range(
    *,
    start: Any,
    stop: Any,
    step: Any,
) -> DilutionRange:
    """校验扫描区间。三者均须为正、有限；start ≤ stop；点数受上限保护。"""
    converted: dict[str, Decimal] = {}
    errors: dict[str, str] = {}
    for field, raw_value in (("start", start), ("stop", stop), ("step", step)):
        try:
            converted[field] = _as_decimal(raw_value, field)
        except ParameterValidationError as exc:
            errors.update(exc.errors)
    if errors:
        raise ParameterValidationError(errors)

    lo, hi, st = converted["start"], converted["stop"], converted["step"]
    if lo <= 0:
        errors["start"] = "起始稀释率必须为正数"
    if hi <= 0:
        errors["stop"] = "终止稀释率必须为正数"
    if st <= 0:
        errors["step"] = "步长必须为正数"
    if lo > hi:
        errors["start"] = "起始稀释率不能大于终止稀释率"
    if errors:
        raise ParameterValidationError(errors)

    scan_range = DilutionRange(start=lo, stop=hi, step=st)
    # n 为起点之后还能容纳的整步数（显式向下取整，绝不让末点越过 stop）；
    # 总点数 n+1（含两端）
    steps = ((hi - lo) / st).to_integral_value(rounding=ROUND_FLOOR)
    point_count = int(steps) + 1
    if point_count > MAX_SCAN_POINTS:
        raise ParameterValidationError(
            {
                "step": (
                    f"扫描点数 {point_count} 超过上限 {MAX_SCAN_POINTS}，"
                    "请加大步长或缩小区间"
                )
            }
        )
    return scan_range


def _scan_points(scan_range: DilutionRange) -> Iterable[Decimal]:
    """生成扫描点：D_i = start + i·step，且保证末点不越过 stop。"""
    steps = int(
        (
            (scan_range.stop - scan_range.start) / scan_range.step
        ).to_integral_value(rounding=ROUND_FLOOR)
    )
    for i in range(steps + 1):
        yield scan_range.start + scan_range.step * i


def scan_dilution(
    params: ProcessParameters,
    scan_range: DilutionRange,
) -> list[SteadyStateSolution]:
    """沿稀释率对同一组工艺参数做区间扫描，返回逐点稳态解。"""
    return [
        solve_steady_state(params.with_dilution(point))
        for point in _scan_points(scan_range)
    ]
