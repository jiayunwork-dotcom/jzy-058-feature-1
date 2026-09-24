"""动态仿真 HTTP 接口端到端测试。

覆盖：正常收敛（末端对拍 /api/solve）、冲刷与临界稀释率、轨迹非负、
取点疏密不改变末端、非法初值/时长/粒度的结构化拒绝、具名档取回仿真、
积分失败 500 信封。
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.profiles import DEMO_PROFILE_NAME


@pytest.fixture()
def client(tmp_path) -> TestClient:
    app = create_app(db_path=str(tmp_path / "sim.db"))
    with TestClient(app) as test_client:
        yield test_client


PARAMS = {"S0": 100, "D": 0.1, "mu_max": 0.5, "Ks": 10, "Y": 0.5}


def simulate_body(**overrides):
    body = {
        "parameters": PARAMS,
        "initial_state": {"S": 0, "X": 10},
        "duration": 200,
        "num_points": 41,
    }
    body.update(overrides)
    return body


# ---------- 正常收敛与响应结构 ----------


def test_simulate_returns_trajectory_and_matches_steady_solver(client):
    resp = client.post("/api/simulate", json=simulate_body())
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert body["count"] == 41
    assert body["duration"] == pytest.approx(200.0)
    assert body["points"][0] == {"time": 0.0, "S": 0.0, "X": 10.0, "mu": 0.0}
    times = [p["time"] for p in body["points"]]
    assert times[0] == 0.0 and times[-1] == pytest.approx(200.0)

    final = body["final"]
    assert final == body["points"][-1]
    # 自洽性：动态末端与稳态求解器的点一致
    solved = client.post("/api/solve", json=PARAMS).json()
    assert final["S"] == pytest.approx(solved["S"], rel=1e-4)
    assert final["X"] == pytest.approx(solved["X"], rel=1e-4)
    # 同参数稳态参照随响应带回
    assert body["steady_state"] == solved

    # 整条轨迹非负
    assert all(p["S"] >= 0 and p["X"] >= 0 for p in body["points"])


def test_simulate_default_granularity_when_omitted(client):
    body = simulate_body()
    del body["num_points"]
    resp = client.post("/api/simulate", json=body)
    assert resp.status_code == 200
    assert resp.json()["count"] == 200


def test_simulate_interval_granularity(client):
    body = simulate_body(num_points=None, interval=25)
    resp = client.post("/api/simulate", json=body)
    assert resp.status_code == 200
    times = [p["time"] for p in resp.json()["points"]]
    assert times[0] == 0.0
    assert times[-1] == pytest.approx(200.0)
    assert times[-2] == pytest.approx(175.0)  # T 非整除时补末端


def test_simulate_trajectory_shows_overshoot(client):
    # 基质满、污泥少：S 先被吃下去再回稳，动态必须呈现中途过冲
    body = simulate_body(initial_state={"S": 100, "X": 1}, num_points=401)
    points = client.post("/api/simulate", json=body).json()["points"]
    assert min(p["S"] for p in points[10:]) < 2.5
    assert points[-1]["S"] == pytest.approx(2.5, abs=1e-3)


def test_output_density_does_not_change_terminal_state(client):
    coarse = client.post("/api/simulate", json=simulate_body(num_points=6)).json()
    dense = client.post(
        "/api/simulate", json=simulate_body(num_points=1001)
    ).json()
    assert dense["final"]["S"] == pytest.approx(coarse["final"]["S"], abs=1e-6)
    assert dense["final"]["X"] == pytest.approx(coarse["final"]["X"], abs=1e-6)


# ---------- 冲刷与临界 ----------


def test_simulate_washout_above_critical(client):
    body = simulate_body(
        parameters={**PARAMS, "D": 0.6},
        initial_state={"S": 100, "X": 50},
        duration=120,
        num_points=121,
    )
    result = client.post("/api/simulate", json=body).json()
    assert result["steady_state"]["is_washout"] is True
    xs = [p["X"] for p in result["points"]]
    assert all(xs[i + 1] <= xs[i] + 1e-12 for i in range(len(xs) - 1))
    assert xs[-1] < 0.05
    assert result["final"]["S"] == pytest.approx(100, abs=1e-1)
    assert all(p["S"] >= 0 and p["X"] >= 0 for p in result["points"])


def test_simulate_critical_dilution_equal_mu_max(client):
    body = simulate_body(
        parameters={**PARAMS, "D": 0.5},
        initial_state={"S": 100, "X": 20},
        duration=400,
        num_points=81,
    )
    result = client.post("/api/simulate", json=body).json()
    assert result["steady_state"]["is_washout"] is True
    assert result["steady_state"]["X"] == 0
    xs = [p["X"] for p in result["points"]]
    assert all(xs[i + 1] <= xs[i] + 1e-12 for i in range(len(xs) - 1))
    assert result["final"]["X"] < 1e-4
    assert result["final"]["S"] == pytest.approx(100, abs=1e-2)


def test_simulate_near_critical_below_mu_max_settles_slowly(client):
    # D=0.45：S*=90、X*=5，慢收敛，时长必须给足
    body = simulate_body(
        parameters={**PARAMS, "D": 0.45},
        initial_state={"S": 0, "X": 1},
        duration=2000,
        num_points=21,
    )
    result = client.post("/api/simulate", json=body).json()
    assert result["steady_state"]["is_washout"] is False
    solved = client.post("/api/solve", json={**PARAMS, "D": 0.45}).json()
    assert result["final"]["S"] == pytest.approx(solved["S"], rel=2e-3)
    assert result["final"]["X"] == pytest.approx(solved["X"], rel=2e-3)


# ---------- 非法输入：结构化拒绝 ----------


def _expect_422(resp):
    assert resp.status_code == 422
    envelope = resp.json()
    assert envelope["code"] == "invalid_parameters"
    return {d["field"] for d in envelope["details"]}


def test_simulate_rejects_negative_initial_state(client):
    fields = _expect_422(
        client.post(
            "/api/simulate", json=simulate_body(initial_state={"S": -1, "X": 5})
        )
    )
    assert "S_init" in fields
    fields = _expect_422(
        client.post(
            "/api/simulate", json=simulate_body(initial_state={"S": 1, "X": -5})
        )
    )
    assert "X_init" in fields


def test_simulate_rejects_non_positive_duration(client):
    for bad in (0, -10):
        fields = _expect_422(
            client.post("/api/simulate", json=simulate_body(duration=bad))
        )
        assert "duration" in fields


def test_simulate_rejects_bad_granularity(client):
    fields = _expect_422(
        client.post("/api/simulate", json=simulate_body(num_points=1))
    )
    assert "num_points" in fields
    fields = _expect_422(
        client.post(
            "/api/simulate", json=simulate_body(num_points=100_000)
        )
    )
    assert "num_points" in fields
    fields = _expect_422(
        client.post(
            "/api/simulate",
            json=simulate_body(num_points=None, interval=0),
        )
    )
    assert "interval" in fields
    # 间隔过细导致点数超上限
    fields = _expect_422(
        client.post(
            "/api/simulate",
            json=simulate_body(num_points=None, interval=0.001),
        )
    )
    assert "interval" in fields
    # 点数与间隔互斥
    fields = _expect_422(
        client.post(
            "/api/simulate",
            json=simulate_body(num_points=10, interval=5),
        )
    )
    assert "num_points" in fields


def test_simulate_rejects_non_numeric_and_missing(client):
    # 结构层错误：缺字段 / 非数值 -> invalid_request
    resp = client.post(
        "/api/simulate",
        json={"parameters": PARAMS, "initial_state": {"S": "abc", "X": 1},
              "duration": 10},
    )
    assert resp.status_code == 422
    assert resp.json()["code"] == "invalid_request"

    resp = client.post("/api/simulate", json={"duration": 10})
    assert resp.status_code == 422
    assert resp.json()["code"] == "invalid_request"

    # num_points 必须为整数
    resp = client.post(
        "/api/simulate", json=simulate_body(num_points=2.5)
    )
    assert resp.status_code == 422

    # 未知字段拒绝
    resp = client.post(
        "/api/simulate", json=simulate_body(decay=0.01)
    )
    assert resp.status_code == 422


def test_simulate_rejects_invalid_process_parameters(client):
    # 工艺参数非法仍走同一条稳态内核校验
    fields = _expect_422(
        client.post(
            "/api/simulate",
            json=simulate_body(parameters={**PARAMS, "D": -0.1}),
        )
    )
    assert "D" in fields


# ---------- 具名档取回后仿真 ----------


def test_simulate_saved_profile(client):
    created = client.post(
        "/api/profiles", json={"name": "run-a", "parameters": PARAMS}
    )
    assert created.status_code == 201

    resp = client.post(
        "/api/profiles/run-a/simulate",
        json={"initial_state": {"S": 0, "X": 10},
              "duration": 200, "num_points": 41},
    )
    assert resp.status_code == 200, resp.text
    result = resp.json()
    solved = client.post("/api/profiles/run-a/solve").json()
    assert result["final"]["S"] == pytest.approx(solved["S"], rel=1e-4)
    assert result["final"]["X"] == pytest.approx(solved["X"], rel=1e-4)
    assert result["steady_state"]["D"] == pytest.approx(0.1)


def test_simulate_demo_profile(client):
    resp = client.post(
        f"/api/profiles/{DEMO_PROFILE_NAME}/simulate",
        json={"initial_state": {"S": 0, "X": 10},
              "duration": 200, "num_points": 21},
    )
    assert resp.status_code == 200
    result = resp.json()
    assert result["final"]["S"] == pytest.approx(2.5, rel=1e-4)
    assert result["final"]["X"] == pytest.approx(48.75, rel=1e-4)


def test_simulate_unknown_profile_404(client):
    resp = client.post(
        "/api/profiles/nope/simulate",
        json={"initial_state": {"S": 0, "X": 1}, "duration": 10},
    )
    assert resp.status_code == 404
    assert resp.json()["code"] == "profile_not_found"


def test_simulate_saved_profile_validates_setup(client):
    client.post(
        "/api/profiles", json={"name": "run-b", "parameters": PARAMS}
    )
    resp = client.post(
        "/api/profiles/run-b/simulate",
        json={"initial_state": {"S": 0, "X": 1}, "duration": -1},
    )
    assert resp.status_code == 422
    assert resp.json()["code"] == "invalid_parameters"


# ---------- 积分器失控时的服务端信封 ----------


def test_simulate_extreme_start_reports_integration_failure(client):
    body = simulate_body(
        initial_state={"S": 1e6, "X": 1e6}, duration=200, num_points=21
    )
    resp = client.post("/api/simulate", json=body)
    assert resp.status_code == 500
    assert resp.json()["code"] == "integration_failed"
