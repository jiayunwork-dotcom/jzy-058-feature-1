"""HTTP 接口端到端测试：稳态解、扫描、冲刷、非法输入与参数档。

每个测试函数使用独立的临时 SQLite 文件，互不污染。
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.profiles import DEMO_PROFILE_NAME


@pytest.fixture()
def client(tmp_path) -> TestClient:
    app = create_app(db_path=str(tmp_path / "test.db"))
    with TestClient(app) as test_client:
        yield test_client


GOOD_PARAMS = {"S0": 100, "D": 0.1, "mu_max": 0.5, "Ks": 10, "Y": 0.5}


# ---------- 系统与示范档 ----------


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_demo_profile_available_on_startup_and_hand_checkable(client):
    resp = client.get("/api/demo")
    assert resp.status_code == 200
    body = resp.json()
    assert body["name"] == DEMO_PROFILE_NAME
    assert body["parameters"] == GOOD_PARAMS

    # 用示范档立刻复算，结果应与手工核算一致：S=2.5，X=48.75
    solved = client.post(f"/api/profiles/{DEMO_PROFILE_NAME}/solve")
    assert solved.status_code == 200
    result = solved.json()
    assert result["S"] == pytest.approx(2.5)
    assert result["X"] == pytest.approx(48.75)
    assert result["is_washout"] is False
    assert result["D"] == pytest.approx(0.1)
    assert result["mu"] == pytest.approx(0.1)


# ---------- 单点求解 ----------


def test_solve_normal_regime(client):
    resp = client.post("/api/solve", json=GOOD_PARAMS)
    assert resp.status_code == 200
    body = resp.json()
    assert body["S"] == pytest.approx(2.5)
    assert body["X"] == pytest.approx(48.75)
    assert body["mu"] == pytest.approx(0.1)
    assert body["mu_max"] == pytest.approx(0.5)
    assert body["is_washout"] is False
    assert body["washout_reason"] is None


def test_solve_critical_dilution(client):
    payload = {**GOOD_PARAMS, "D": 0.5}
    resp = client.post("/api/solve", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["is_washout"] is True
    assert body["X"] == 0
    assert body["S"] == pytest.approx(100)
    assert body["washout_reason"] == "dilution_at_or_above_mu_max"


def test_solve_above_critical_is_clean_washout(client):
    payload = {**GOOD_PARAMS, "D": 0.9}
    body = client.post("/api/solve", json=payload).json()
    assert body["is_washout"] is True
    assert body["X"] == 0
    assert body["S"] == pytest.approx(100)


def test_solve_balance_washout_case(client):
    # S0=5, D=0.3 -> 分式 S=15 ≥ S0 -> 冲刷
    payload = {**GOOD_PARAMS, "S0": 5, "D": 0.3}
    body = client.post("/api/solve", json=payload).json()
    assert body["is_washout"] is True
    assert body["X"] == 0
    assert body["S"] == pytest.approx(5)
    assert body["washout_reason"] == "steady_state_substrate_not_below_influent"


# ---------- 非法输入：结构化拒绝 ----------


def test_invalid_non_positive_params_return_structured_error(client):
    payload = {**GOOD_PARAMS, "D": -0.1, "Ks": 0, "Y": -1}
    resp = client.post("/api/solve", json=payload)
    assert resp.status_code == 422
    body = resp.json()
    assert body["code"] == "invalid_parameters"
    fields = {detail["field"] for detail in body["details"]}
    assert {"D", "Ks", "Y"}.issubset(fields)
    assert body["error"]


def test_negative_s0_rejected(client):
    resp = client.post("/api/solve", json={**GOOD_PARAMS, "S0": -5})
    assert resp.status_code == 422
    fields = {d["field"] for d in resp.json()["details"]}
    assert "S0" in fields


def test_non_numeric_and_missing_fields_rejected(client):
    resp = client.post("/api/solve", json={"S0": "abc", "D": 0.1})
    assert resp.status_code == 422
    body = resp.json()
    assert body["code"] == "invalid_request"


def test_unknown_field_rejected(client):
    resp = client.post("/api/solve", json={**GOOD_PARAMS, "decay": 0.01})
    assert resp.status_code == 422


def test_boolean_as_number_rejected(client):
    resp = client.post("/api/solve", json={**GOOD_PARAMS, "D": True})
    assert resp.status_code == 422


def test_nan_infinity_rejected(client):
    for bad in ("NaN", "Infinity"):
        resp = client.post(
            "/api/solve",
            json={"S0": 100, "D": 0.1, "mu_max": 0.5, "Ks": 10, "Y": bad},
        )
        assert resp.status_code == 422


# ---------- 区间扫描 ----------


def test_scan_returns_full_series(client):
    payload = {
        "parameters": GOOD_PARAMS,
        "range": {"start": 0.1, "stop": 0.6, "step": 0.1},
    }
    resp = client.post("/api/scan", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 6
    points = body["points"]
    assert [round(p["D"], 6) for p in points] == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

    normal = points[:4]
    # 正常区内 S 升、X 降
    substrates = [p["S"] for p in normal]
    biomass = [p["X"] for p in normal]
    assert substrates == sorted(substrates)
    assert biomass == sorted(biomass, reverse=True)

    # 临界与越界点：X 干净归零，绝不出现负值
    for point in points[4:]:
        assert point["is_washout"] is True
        assert point["X"] == 0
        assert point["S"] == pytest.approx(100)
    assert all(p["X"] >= 0 for p in points)
    assert points[4]["washout_reason"] == "dilution_at_or_above_mu_max"


def test_scan_invalid_range_rejected(client):
    payload = {
        "parameters": GOOD_PARAMS,
        "range": {"start": 0.6, "stop": 0.1, "step": 0},
    }
    resp = client.post("/api/scan", json=payload)
    assert resp.status_code == 422
    fields = {d["field"] for d in resp.json()["details"]}
    assert "start" in fields and "step" in fields


# ---------- 具名参数档 ----------


def test_profile_lifecycle(client):
    payload = {"name": "plant-a", "parameters": GOOD_PARAMS}
    created = client.post("/api/profiles", json=payload)
    assert created.status_code == 201
    assert created.json()["name"] == "plant-a"
    assert created.json()["created_at"]

    fetched = client.get("/api/profiles/plant-a")
    assert fetched.status_code == 200
    assert fetched.json()["parameters"] == GOOD_PARAMS

    listing = client.get("/api/profiles")
    names = {p["name"] for p in listing.json()["profiles"]}
    assert {"plant-a", DEMO_PROFILE_NAME}.issubset(names)

    solved = client.post("/api/profiles/plant-a/solve")
    assert solved.json()["S"] == pytest.approx(2.5)

    scan = client.post(
        "/api/profiles/plant-a/scan",
        json={"start": 0.1, "stop": 0.5, "step": 0.2},
    )
    assert scan.status_code == 200
    assert scan.json()["count"] == 3

    assert client.delete("/api/profiles/plant-a").status_code == 204
    assert client.get("/api/profiles/plant-a").status_code == 404
    assert client.delete("/api/profiles/plant-a").status_code == 404


def test_duplicate_profile_conflict(client):
    payload = {"name": "dup", "parameters": GOOD_PARAMS}
    assert client.post("/api/profiles", json=payload).status_code == 201
    resp = client.post("/api/profiles", json=payload)
    assert resp.status_code == 409
    assert resp.json()["code"] == "profile_already_exists"


def test_profile_invalid_name_rejected(client):
    payload = {"name": "bad/name?", "parameters": GOOD_PARAMS}
    resp = client.post("/api/profiles", json=payload)
    assert resp.status_code == 422


def test_profile_invalid_params_rejected(client):
    payload = {"name": "nope", "parameters": {**GOOD_PARAMS, "D": 0}}
    resp = client.post("/api/profiles", json=payload)
    assert resp.status_code == 422
    # 非法参数不得入库
    assert client.get("/api/profiles/nope").status_code == 404


def test_saved_profile_profiles_persist_across_restart(tmp_path):
    db = str(tmp_path / "persist.db")
    client1 = TestClient(create_app(db_path=db))
    client1.post("/api/profiles", json={"name": "keep", "parameters": GOOD_PARAMS})

    client2 = TestClient(create_app(db_path=db))
    resp = client2.get("/api/profiles/keep")
    assert resp.status_code == 200
    assert resp.json()["parameters"] == GOOD_PARAMS


def test_demo_seeding_is_idempotent(tmp_path):
    db = str(tmp_path / "demo.db")
    first = TestClient(create_app(db_path=db))
    ts1 = first.get("/api/demo").json()["created_at"]
    second = TestClient(create_app(db_path=db))
    ts2 = second.get("/api/demo").json()["created_at"]
    assert ts1 == ts2
