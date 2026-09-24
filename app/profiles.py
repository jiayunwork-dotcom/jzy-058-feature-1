"""参数档管理：工况以具名档案登记、持久化、凭名取回复算。

本模块只负责档案的存取与参数反序列化；HTTP 层负责参数校验，
求解仍由 :mod:`app.solver` 完成，取档与求解之间不共享可变状态。
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from decimal import Decimal

from app.database import Database
from app.solver import ProcessParameters

# 服务拉起即可手工核对的示范档：有氧工况，D 明显小于 μmax。
# S = 10·0.1/(0.5−0.1) = 2.5，X = 0.5·(100−2.5) = 48.75
DEMO_PROFILE_NAME = "demo_aerobic"
DEMO_PROFILE: dict[str, object] = {
    "s0": "100",
    "dilution": "0.1",
    "mu_max": "0.5",
    "ks": "10",
    "y": "0.5",
}


class ProfileError(Exception):
    """档案操作失败的基类。"""


class ProfileAlreadyExistsError(ProfileError):
    def __init__(self, name: str):
        self.name = name
        super().__init__(f"工况档案已存在: {name}")


@dataclass(frozen=True)
class StoredProfile:
    """登记在库的一具工况档案。

    ``created_at`` 由数据库在插入时生成；新建内存对象时留空，
    从库里读出的对象才带有时间戳。
    """

    name: str
    parameters: ProcessParameters
    created_at: str | None = None

    def to_storage(self) -> dict[str, str]:
        """转成 Decimal 的规范字符串，避免浮点落库损失输入口径。"""
        p = self.parameters
        return {
            "name": self.name,
            "s0": str(p.s0),
            "dilution": str(p.dilution),
            "mu_max": str(p.mu_max),
            "ks": str(p.ks),
            "y": str(p.y),
        }


def _row_to_profile(row: sqlite3.Row) -> StoredProfile:
    return StoredProfile(
        name=row["name"],
        parameters=ProcessParameters(
            s0=Decimal(row["s0"]),
            dilution=Decimal(row["dilution"]),
            mu_max=Decimal(row["mu_max"]),
            ks=Decimal(row["ks"]),
            y=Decimal(row["y"]),
        ),
        created_at=row["created_at"],
    )


class ProfileManager:
    """具名工况档案的登记、检索、列举、删除。"""

    def __init__(self, database: Database):
        self._db = database

    def create(self, profile: StoredProfile, *, overwrite: bool = False) -> None:
        data = profile.to_storage()
        with self._db.transaction() as conn:
            if overwrite:
                conn.execute(
                    """
                    INSERT INTO profiles (name, s0, dilution, mu_max, ks, y)
                    VALUES (:name, :s0, :dilution, :mu_max, :ks, :y)
                    ON CONFLICT(name) DO UPDATE SET
                        s0 = excluded.s0,
                        dilution = excluded.dilution,
                        mu_max = excluded.mu_max,
                        ks = excluded.ks,
                        y = excluded.y
                    """,
                    data,
                )
                return
            try:
                conn.execute(
                    """
                    INSERT INTO profiles (name, s0, dilution, mu_max, ks, y)
                    VALUES (:name, :s0, :dilution, :mu_max, :ks, :y)
                    """,
                    data,
                )
            except sqlite3.IntegrityError as exc:
                # 只有唯一约束冲突（重名）才按 409 语义处理；
                # 其它完整性错误（非空、类型等）向上抛出，绝不误吞
                if exc.sqlite_errorcode == sqlite3.SQLITE_CONSTRAINT_UNIQUE:
                    raise ProfileAlreadyExistsError(profile.name) from exc
                raise

    def get(self, name: str) -> StoredProfile:
        row = self._db.query_one(
            "SELECT * FROM profiles WHERE name = ?", (name,)
        )
        if row is None:
            raise KeyError(name)
        return _row_to_profile(row)

    def list_all(self) -> list[StoredProfile]:
        rows = self._db.query_all("SELECT * FROM profiles ORDER BY name")
        return [_row_to_profile(row) for row in rows]

    def delete(self, name: str) -> bool:
        with self._db.transaction() as conn:
            cursor = conn.execute("DELETE FROM profiles WHERE name = ?", (name,))
            return cursor.rowcount > 0

    def get_parameters(self, name: str) -> ProcessParameters:
        """凭名取回可直接送进求解器的参数。"""
        return self.get(name).parameters

    def seed_demo(self) -> bool:
        """登记内置示范档（幂等：已存在则保留，绝不覆盖用户改动）。

        返回是否为本次新建。
        """
        existing = self._db.query_one(
            "SELECT 1 FROM profiles WHERE name = ?", (DEMO_PROFILE_NAME,)
        )
        if existing is not None:
            return False
        params = ProcessParameters(
            s0=Decimal(str(DEMO_PROFILE["s0"])),
            dilution=Decimal(str(DEMO_PROFILE["dilution"])),
            mu_max=Decimal(str(DEMO_PROFILE["mu_max"])),
            ks=Decimal(str(DEMO_PROFILE["ks"])),
            y=Decimal(str(DEMO_PROFILE["y"])),
        )
        # 极端竞态下可能被别的请求抢先插入，此时同样保留已有档案
        try:
            self.create(StoredProfile(DEMO_PROFILE_NAME, params))
        except ProfileAlreadyExistsError:
            return False
        return True
