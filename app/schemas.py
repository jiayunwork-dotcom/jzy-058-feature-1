"""HTTP 接口的 Pydantic 输入/输出模型。

字段名刻意沿用工艺记号（S0、D、mu_max、Ks、Y），让上游设计程序
丢过来的 JSON 与工艺单口径一致。

校验分工：这里只挡“结构/类型”层面的错误（非数值、NaN、无穷、
布尔冒充数值）；正负号等工艺合法性统一在 :mod:`app.solver` 把关，
两处错误最终由同一套结构化错误信封回报。
"""

from __future__ import annotations

import math
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, field_validator

# 档案名允许的字符：字母数字、下划线、连字符、点；长度 1–128
NAME_PATTERN = r"^[A-Za-z0-9_.-]{1,128}$"


def _finite_number(value: object, field: str) -> float:
    """拒绝布尔、NaN、无穷；bool 是 int 子类必须先挡。"""
    if isinstance(value, bool):
        raise ValueError("必须是数值，不能是布尔值")
    if not isinstance(value, (int, float)):
        raise ValueError(f"{field} 必须是数值")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{field} 必须是有限数值，不能为 NaN 或无穷")
    return numeric


class ProcessParametersIn(BaseModel):
    """一组工艺参数输入（S0 允许为 0，其余正值由求解器校验）。"""

    model_config = ConfigDict(extra="forbid")

    S0: float
    D: float
    mu_max: float
    Ks: float
    Y: float

    @field_validator("S0", "D", "mu_max", "Ks", "Y", mode="before")
    @classmethod
    def _reject_non_finite(cls, value: object, info) -> float:
        return _finite_number(value, info.field_name)


class ScanRangeIn(BaseModel):
    """稀释率扫描区间输入。"""

    model_config = ConfigDict(extra="forbid")

    start: float
    stop: float
    step: float

    @field_validator("start", "stop", "step", mode="before")
    @classmethod
    def _reject_non_finite(cls, value: object, info) -> float:
        return _finite_number(value, info.field_name)


class ScanRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    parameters: ProcessParametersIn
    range: ScanRangeIn


class InitialStateIn(BaseModel):
    """初始状态输入：初始基质浓度 S 与初始污泥浓度 X（均须非负）。"""

    model_config = ConfigDict(extra="forbid")

    S: float
    X: float

    @field_validator("S", "X", mode="before")
    @classmethod
    def _reject_non_finite(cls, value: object, info) -> float:
        return _finite_number(value, info.field_name)


class SimulationSetupIn(BaseModel):
    """动态仿真设置：时长 + 输出粒度（点数与间隔互斥）。

    结构层只挡类型/有限性；正负号、互斥与点数上限等语义由
    :mod:`app.dynamics` 统一把关，与稳态路径共用结构化错误信封。
    """

    model_config = ConfigDict(extra="forbid")

    duration: float
    num_points: int | None = None
    interval: float | None = None

    @field_validator("duration", "interval", mode="before")
    @classmethod
    def _reject_non_finite(cls, value: object, info) -> float | None:
        if value is None:
            return None
        return _finite_number(value, info.field_name)

    @field_validator("num_points", mode="before")
    @classmethod
    def _reject_non_integer(cls, value: object) -> int | None:
        if value is None:
            return None
        if isinstance(value, bool):
            raise ValueError("必须是整数，不能是布尔值")
        if not isinstance(value, int):
            raise ValueError("必须是整数")
        return value


class SimulationRequest(SimulationSetupIn):
    """临时参数直接仿真：工艺参数 + 初始状态 + 仿真设置。"""

    model_config = ConfigDict(extra="forbid")

    parameters: ProcessParametersIn
    initial_state: InitialStateIn


class ProfileSimulationRequest(SimulationSetupIn):
    """凭档案仿真：参数从具名档取回，请求体只带初始状态与仿真设置。"""

    model_config = ConfigDict(extra="forbid")

    initial_state: InitialStateIn


class ProfileCreateIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    parameters: ProcessParametersIn

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        import re

        name = value.strip()
        if not name:
            raise ValueError("档案名不能为空")
        if not re.fullmatch(NAME_PATTERN, name):
            raise ValueError(
                "档案名只能包含字母、数字、下划线、连字符和点，长度 1–128"
            )
        return name


# ---------- 输出模型 ----------


def _f(value: Decimal | None) -> float | None:
    return None if value is None else float(value)


class SteadyStateOut(BaseModel):
    D: float
    S: float
    X: float
    mu: float
    mu_max: float
    is_washout: bool
    washout_reason: str | None = None


class ScanResponse(BaseModel):
    points: list[SteadyStateOut]
    count: int


class TrajectoryPointOut(BaseModel):
    """轨迹上一个带时刻的状态点。"""

    time: float
    S: float
    X: float
    mu: float


class SimulationResponse(BaseModel):
    """动态仿真结果：逐时刻轨迹 + 同参数稳态参照（供两条路径对拍）。"""

    duration: float
    count: int
    points: list[TrajectoryPointOut]
    final: TrajectoryPointOut
    steady_state: SteadyStateOut


class ProfileOut(BaseModel):
    name: str
    parameters: ProcessParametersIn
    created_at: str | None = None


class ProfileListResponse(BaseModel):
    profiles: list[ProfileOut]
    count: int


class ErrorDetail(BaseModel):
    field: str
    reason: str


class ErrorResponse(BaseModel):
    error: str
    code: str
    details: list[ErrorDetail] = []
