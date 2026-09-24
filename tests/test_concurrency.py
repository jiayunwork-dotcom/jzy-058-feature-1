"""并发隔离测试：多个工况档同时在算时，结果只归属各自名下。

不引入 pytest-asyncio：在同步用例内用 asyncio.run 驱动一批并发请求。
"""

from __future__ import annotations

import asyncio

import httpx
import pytest

from app.main import create_app

PARAMS = {"S0": 100, "D": 0.1, "mu_max": 0.5, "Ks": 10, "Y": 0.5}


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture()
def transport_holder(tmp_path):
    app = create_app(db_path=str(tmp_path / "concurrency.db"))
    return app


async def _concurrent_create_and_solve(app, count: int):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test"
    ) as client:
        # 每个并发任务登记一具不同稀释率的档案并立即复算
        async def register_and_solve(i: int) -> tuple[int, int, float, float]:
            dilution = round(0.05 + i * 0.01, 6)
            name = f"case-{i:03d}"
            payload = {"name": name, "parameters": {**PARAMS, "D": dilution}}
            created = await client.post("/api/profiles", json=payload)
            assert created.status_code == 201, created.text
            solved = await client.post(f"/api/profiles/{name}/solve")
            assert solved.status_code == 200
            body = solved.json()
            assert body["D"] == pytest.approx(dilution)
            return i, created.status_code, body["S"], body["X"]

        results = await asyncio.gather(
            *(register_and_solve(i) for i in range(count))
        )

        # 再逐一取回核对：各档参数与结果没有被并发请求互相串改
        for i, _, s, x in results:
            name = f"case-{i:03d}"
            stored = await client.get(f"/api/profiles/{name}")
            assert stored.status_code == 200
            assert stored.json()["parameters"]["D"] == pytest.approx(
                round(0.05 + i * 0.01, 6)
            )
            solved = await client.post(f"/api/profiles/{name}/solve")
            assert solved.json()["S"] == pytest.approx(s)
            assert solved.json()["X"] == pytest.approx(x)
        return results


def test_concurrent_profiles_do_not_cross_contaminate(transport_holder):
    results = _run(_concurrent_create_and_solve(transport_holder, 32))
    # 不同 D -> 不同 (S, X)，且各自 X 为正、互不等同串号
    substrates = {round(r[2], 9) for r in results}
    biomass = {round(r[3], 9) for r in results}
    assert len(substrates) == 32
    assert len(biomass) == 32
    assert all(r[3] > 0 for r in results)


async def _duplicate_name_race(app):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test"
    ) as client:
        payload = {"name": "race", "parameters": PARAMS}
        responses = await asyncio.gather(
            client.post("/api/profiles", json=payload),
            client.post("/api/profiles", json=payload),
            client.post("/api/profiles", json=payload),
        )
        statuses = sorted(r.status_code for r in responses)
        return statuses


def test_duplicate_name_race_yields_single_winner(transport_holder):
    statuses = _run(_duplicate_name_race(transport_holder))
    # 恰好一个创建成功，其余冲突拒绝，库里只有一条
    assert statuses == [201, 409, 409]


async def _concurrent_scan_and_solve(app):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test"
    ) as client:
        await client.post(
            "/api/profiles",
            json={"name": "shared-demo", "parameters": PARAMS},
        )

        async def scan(i: int):
            resp = await client.post(
                "/api/profiles/shared-demo/scan",
                json={"start": 0.05, "stop": 0.5, "step": 0.05},
            )
            assert resp.status_code == 200
            points = resp.json()["points"]
            assert len(points) == 10
            assert points[-1]["X"] == 0
            return points

        scans = await asyncio.gather(*(scan(i) for i in range(16)))
        first = scans[0]
        for other in scans[1:]:
            assert other == first
        return scans


def test_concurrent_scans_are_isolated_and_consistent(transport_holder):
    _run(_concurrent_scan_and_solve(transport_holder))
