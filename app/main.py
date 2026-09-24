"""FastAPI 应用装配与 HTTP 接口。

职责边界：
  * :mod:`app.kinetics` —— Monod 动力学；
  * :mod:`app.solver`   —— 稳态求解、冲刷判定、参数合法性；
  * :mod:`app.dynamics` —— 动态仿真：ODE 时间推进积分；
  * :mod:`app.profiles` —— 具名参数档登记与取回；
  * :mod:`app.database` —— SQLite 持久化；
  * 本模块              —— 仅负责 HTTP 编解码、状态码与错误信封。

纯计算全部落在无状态的求解器内核上；档案操作各开独立连接与短事务，
并发请求之间不共享任何可变工况数据。
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.database import Database
from app.dynamics import (
    build_initial_state,
    build_time_grid,
    simulate,
)
from app.profiles import (
    DEMO_PROFILE_NAME,
    ProfileAlreadyExistsError,
    ProfileManager,
    StoredProfile,
)
from app.schemas import (
    ProcessParametersIn,
    ProfileCreateIn,
    ProfileListResponse,
    ProfileOut,
    ProfileSimulationRequest,
    ScanRangeIn,
    ScanRequest,
    ScanResponse,
    SimulationRequest,
    SimulationResponse,
    SteadyStateOut,
    TrajectoryPointOut,
)
from app.solver import (
    ParameterValidationError,
    build_dilution_range,
    build_parameters,
    scan_dilution,
    solve_steady_state,
)

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "sludge.db"


def _db_path() -> str:
    return os.environ.get("SLUDGE_DB_PATH", str(DEFAULT_DB_PATH))


def _parameters_from_input(payload: ProcessParametersIn):
    return build_parameters(
        s0=payload.S0,
        dilution=payload.D,
        mu_max=payload.mu_max,
        ks=payload.Ks,
        y=payload.Y,
    )


def _solution_output(solution) -> SteadyStateOut:
    return SteadyStateOut(
        D=float(solution.dilution),
        S=float(solution.s),
        X=float(solution.x),
        mu=float(solution.mu),
        mu_max=float(solution.mu_max),
        is_washout=solution.is_washout,
        washout_reason=solution.washout_reason,
    )


def _profile_output(profile: StoredProfile) -> ProfileOut:
    p = profile.parameters
    return ProfileOut(
        name=profile.name,
        parameters=ProcessParametersIn(
            S0=float(p.s0),
            D=float(p.dilution),
            mu_max=float(p.mu_max),
            Ks=float(p.ks),
            Y=float(p.y),
        ),
        created_at=profile.created_at,
    )


def _simulation_output(params, payload) -> SimulationResponse:
    """跑一遍动态仿真并组装响应；稳态解一并附上供核对终点。"""
    initial = build_initial_state(
        s=payload.initial_state.S_init, x=payload.initial_state.X_init
    )
    duration, num_points = build_time_grid(
        duration=payload.duration, num_points=payload.num_points
    )
    result = simulate(params, initial, duration, num_points)
    return SimulationResponse(
        points=[
            TrajectoryPointOut(t=pt.t, S=pt.s, X=pt.x)
            for pt in result.points
        ],
        count=len(result.points),
        is_washout=result.is_washout,
        converged=result.converged,
        steady_state=_solution_output(solve_steady_state(params)),
    )


def _validation_error_response(
    code: str, message: str, errors: dict[str, str], status_code: int = 422
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "error": message,
            "code": code,
            "details": [
                {"field": field, "reason": reason}
                for field, reason in errors.items()
            ],
        },
    )


def create_app(db_path: str | None = None) -> FastAPI:
    app = FastAPI(
        title="活性污泥 CSTR 求解服务",
        version="1.1.0",
        description=(
            "单级完全混合反应器（CSTR）Monod 动力学核算："
            "稳态求解、稀释率区间扫描、冲刷判定、具名参数档管理，"
            "以及从任意初值出发的时间推进动态仿真。"
        ),
    )
    app.state.db = Database(db_path or _db_path())
    app.state.manager = ProfileManager(app.state.db)
    # 拉起即可验证：内置可手工核对的有氧示范档
    app.state.manager.seed_demo()

    # ---------- 错误处理：统一结构化信封 ----------

    @app.exception_handler(ParameterValidationError)
    async def handle_parameter_error(_: Request, exc: ParameterValidationError):
        return _validation_error_response(
            code="invalid_parameters",
            message="工况参数不合法",
            errors=exc.errors,
        )

    @app.exception_handler(RequestValidationError)
    async def handle_request_validation_error(
        _: Request, exc: RequestValidationError
    ):
        errors: dict[str, str] = {}
        for err in exc.errors():
            loc = ".".join(str(p) for p in err["loc"] if p != "body") or "body"
            errors[loc] = err["msg"]
        return _validation_error_response(
            code="invalid_request",
            message="请求结构或字段类型不合法",
            errors=errors,
        )

    # ---------- 健康检查与示范档 ----------

    @app.get("/health", tags=["system"])
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/demo", tags=["profiles"], response_model=ProfileOut)
    async def get_demo_profile() -> ProfileOut:
        """返回内置示范档（参数与其登记名），供拉起后立刻手工核对。"""
        return _profile_output(app.state.manager.get(DEMO_PROFILE_NAME))

    # ---------- 单点稳态求解 ----------

    @app.post(
        "/api/solve",
        tags=["steady-state"],
        response_model=SteadyStateOut,
    )
    async def solve(payload: ProcessParametersIn) -> SteadyStateOut:
        params = _parameters_from_input(payload)
        return _solution_output(solve_steady_state(params))

    # ---------- 稀释率区间扫描 ----------

    @app.post(
        "/api/scan",
        tags=["steady-state"],
        response_model=ScanResponse,
    )
    async def scan(payload: ScanRequest) -> ScanResponse:
        params = _parameters_from_input(payload.parameters)
        scan_range = build_dilution_range(
            start=payload.range.start,
            stop=payload.range.stop,
            step=payload.range.step,
        )
        solutions = scan_dilution(params, scan_range)
        points = [_solution_output(sol) for sol in solutions]
        return ScanResponse(points=points, count=len(points))

    # ---------- 动态仿真（时间推进） ----------

    @app.post(
        "/api/simulate",
        tags=["dynamics"],
        response_model=SimulationResponse,
    )
    async def simulate_adhoc(payload: SimulationRequest) -> SimulationResponse:
        """对临时丢进来的一组参数，从给定初值沿时间积分到稳态。"""
        params = _parameters_from_input(payload.parameters)
        return _simulation_output(params, payload)

    # ---------- 具名参数档 ----------

    @app.post(
        "/api/profiles",
        tags=["profiles"],
        response_model=ProfileOut,
        status_code=201,
    )
    async def create_profile(payload: ProfileCreateIn) -> ProfileOut:
        params = _parameters_from_input(payload.parameters)
        profile = StoredProfile(name=payload.name, parameters=params)
        try:
            app.state.manager.create(profile)
        except ProfileAlreadyExistsError:
            return JSONResponse(
                status_code=409,
                content={
                    "error": f"工况档案已存在: {payload.name}",
                    "code": "profile_already_exists",
                    "details": [
                        {"field": "name", "reason": "同名档案已登记，"
                         "请更换名称或先删除旧档"}
                    ],
                },
            )
        return _profile_output(app.state.manager.get(payload.name))

    @app.get(
        "/api/profiles",
        tags=["profiles"],
        response_model=ProfileListResponse,
    )
    async def list_profiles() -> ProfileListResponse:
        profiles = app.state.manager.list_all()
        items = [_profile_output(p) for p in profiles]
        return ProfileListResponse(profiles=items, count=len(items))

    @app.get(
        "/api/profiles/{name}",
        tags=["profiles"],
        response_model=ProfileOut,
    )
    async def get_profile(name: str) -> ProfileOut:
        try:
            return _profile_output(app.state.manager.get(name))
        except KeyError:
            return JSONResponse(
                status_code=404,
                content={
                    "error": f"工况档案不存在: {name}",
                    "code": "profile_not_found",
                    "details": [],
                },
            )

    @app.delete("/api/profiles/{name}", tags=["profiles"], status_code=204)
    async def delete_profile(name: str) -> JSONResponse:
        if not app.state.manager.delete(name):
            return JSONResponse(
                status_code=404,
                content={
                    "error": f"工况档案不存在: {name}",
                    "code": "profile_not_found",
                    "details": [],
                },
            )
        return JSONResponse(status_code=204, content=None)

    @app.post(
        "/api/profiles/{name}/solve",
        tags=["profiles"],
        response_model=SteadyStateOut,
    )
    async def solve_saved_profile(name: str) -> SteadyStateOut:
        """凭名取回档案并复算其登记参数下的稳态解。"""
        try:
            params = app.state.manager.get_parameters(name)
        except KeyError:
            return JSONResponse(
                status_code=404,
                content={
                    "error": f"工况档案不存在: {name}",
                    "code": "profile_not_found",
                    "details": [],
                },
            )
        return _solution_output(solve_steady_state(params))

    @app.post(
        "/api/profiles/{name}/scan",
        tags=["profiles"],
        response_model=ScanResponse,
    )
    async def scan_saved_profile(
        name: str, payload: ScanRangeIn
    ) -> ScanResponse:
        """凭名取回档案，沿稀释率区间扫描复算。"""
        try:
            params = app.state.manager.get_parameters(name)
        except KeyError:
            return JSONResponse(
                status_code=404,
                content={
                    "error": f"工况档案不存在: {name}",
                    "code": "profile_not_found",
                    "details": [],
                },
            )
        scan_range = build_dilution_range(
            start=payload.start,
            stop=payload.stop,
            step=payload.step,
        )
        solutions = scan_dilution(params, scan_range)
        points = [_solution_output(sol) for sol in solutions]
        return ScanResponse(points=points, count=len(points))

    @app.post(
        "/api/profiles/{name}/simulate",
        tags=["profiles", "dynamics"],
        response_model=SimulationResponse,
    )
    async def simulate_saved_profile(
        name: str, payload: ProfileSimulationRequest
    ) -> SimulationResponse:
        """凭名取回档案参数，从给定初值沿时间积分到稳态。"""
        try:
            params = app.state.manager.get_parameters(name)
        except KeyError:
            return JSONResponse(
                status_code=404,
                content={
                    "error": f"工况档案不存在: {name}",
                    "code": "profile_not_found",
                    "details": [],
                },
            )
        return _simulation_output(params, payload)

    return app


app = create_app()
