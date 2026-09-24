# 活性污泥 CSTR 稳态求解服务

给市政污水厂工艺组用的常驻稳态求解后端：上游设计程序把一组 Monod 工艺参数
通过 HTTP 丢过来，拿回一份**可复核**的活性污泥稳态解，不必每次现推公式。

## 模型口径（无衰减项）

单级完全混合反应器（CSTR），微生物按 Monod 动力学降解基质：

- 比增长速率：`μ = μmax · S / (Ks + S)`
- 稳态条件：净增长恰好被出流带走，`D = μ`
- 未冲刷稳态出水基质：`S = Ks · D / (μmax − D)`
- 稳态污泥浓度：`X = Y · (S0 − S)`

**冲刷（washout）边界**——求解器自行识别区段，并在结果中明确回报：

1. `D ≥ μmax`：菌种来不及繁殖被冲光（临界点 `D = μmax` 专门兜住，分母绝不取零）；
2. 未冲刷分式解出 `S ≥ S0`：物料衡算给出 `X ≤ 0`，同样落在冲刷边界之外。

进入冲刷后一律取 `X = 0`、`S = S0`、`μ = μmax·S0/(Ks+S0)`，
**绝不把越界的 D 代回分式造成零分母或负污泥量**。

全部内部计算用 `Decimal` 精确比较，`D == μmax` 不会因浮点误差错判区段。

## 模块职责

| 模块 | 职责 |
| --- | --- |
| `app/kinetics.py` | 动力学内核：Monod 比增长速率 |
| `app/solver.py` | 稳态求解、冲刷判定、参数合法性、稀释率扫描 |
| `app/profiles.py` | 具名参数档登记/取回/删除、内置示范档 |
| `app/database.py` | SQLite 持久化（WAL + 短事务 + 每请求独立连接） |
| `app/schemas.py` | HTTP Pydantic 输入/输出模型 |
| `app/main.py` | FastAPI 装配、路由、结构化错误信封 |

## 快速开始（容器）

```bash
docker build -t sludge-steady-state .
docker run --rm -p 8000:8000 sludge-steady-state
# 服务起来后立刻可用：
curl http://127.0.0.1:8000/api/demo
```

镜像构建阶段会跑一遍完整 pytest，任一测试失败则镜像构建中断。
SQLite 数据默认落在容器内 `/data/sludge.db`，可用环境变量
`SLUDGE_DB_PATH` 覆盖；仅通过 HTTP 对外提供能力，无网页界面。

本地开发（Python 3.12）：

```bash
pip install -r requirements-dev.txt
uvicorn app.main:app --reload
pytest
```

## 内置示范档（可手工核对）

`demo_aerobic`：`S0=100, D=0.1, μmax=0.5, Ks=10, Y=0.5`（D 明显小于 μmax）

- `S = 10·0.1/(0.5−0.1) = 2.5`
- `X = 0.5·(100−2.5) = 48.75`

## HTTP 接口

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/health` | 健康检查 |
| GET | `/api/demo` | 取内置示范档 |
| POST | `/api/solve` | 给定一组参数求单点稳态解 |
| POST | `/api/scan` | 沿稀释率区间扫描，返回 `(D, S, X, μ, 是否冲刷)` 点列 |
| POST | `/api/profiles` | 登记具名参数档（重名返回 409） |
| GET | `/api/profiles` | 列举全部档案 |
| GET | `/api/profiles/{name}` | 凭名取回档案 |
| DELETE | `/api/profiles/{name}` | 删除档案（204/404） |
| POST | `/api/profiles/{name}/solve` | 取回档案并复算 |
| POST | `/api/profiles/{name}/scan` | 取回档案并沿稀释率扫描 |

### 单点求解

```bash
curl -s -X POST http://127.0.0.1:8000/api/solve \
  -H 'Content-Type: application/json' \
  -d '{"S0":100,"D":0.1,"mu_max":0.5,"Ks":10,"Y":0.5}'
# {"D":0.1,"S":2.5,"X":48.75,"mu":0.1,"mu_max":0.5,
#  "is_washout":false,"washout_reason":null}
```

### 稀释率区间扫描

```bash
curl -s -X POST http://127.0.0.1:8000/api/scan \
  -H 'Content-Type: application/json' \
  -d '{"parameters":{"S0":100,"D":0.1,"mu_max":0.5,"Ks":10,"Y":0.5},
       "range":{"start":0.1,"stop":0.6,"step":0.1}}'
```

点列随 D 上升呈现 S 升高、X 下降，跨过 μmax 后 X 干净跌到 0。
扫描区间含两端；点数上限 10 000，防止步长过小失控。

### 冲刷响应

```json
{"D":0.5,"S":100.0,"X":0.0,"mu":0.4545,"mu_max":0.5,
 "is_washout":true,"washout_reason":"dilution_at_or_above_mu_max"}
```

`washout_reason` 取值：

- `dilution_at_or_above_mu_max`：D ≥ μmax；
- `steady_state_substrate_not_below_influent`：分式解 S ≥ S0。

### 非法输入：结构化拒绝

`D、μmax、Ks、Y` 任一非正、`S0` 为负、非数值 / NaN / 无穷 / 布尔冒充数值，
一律 `422`，返回逐字段原因，绝不输出"看似正常其实无意义"的数：

```json
{"error":"工况参数不合法","code":"invalid_parameters",
 "details":[{"field":"S0","reason":"进水基质浓度不能为负"},
            {"field":"D","reason":"必须为正数"}]}
```

## 并发隔离

- 稳态计算是无状态纯函数；
- 每个 HTTP 请求各自打开 SQLite 连接、写入走独立短事务（WAL，busy_timeout 30s）；
- 重名登记靠 UNIQUE 约束，只有唯一冲突映射为 409，其它完整性错误不误吞；
- `tests/test_concurrency.py` 用 32 路并发建档+复算、重名竞争、16 路并发扫描固定了隔离行为。

## 测试

```bash
pytest
# tests/test_kinetics.py   Monod 曲线
# tests/test_solver.py     稳态公式、临界点、冲刷、非法输入、扫描、三条交叉关系
# tests/test_api.py        HTTP 正常/边界/错误、档案生命周期、重启持久化
# tests/test_concurrency.py 并发建档、重名竞争、并发扫描隔离
```

被测试钉死的关键关系：

1. 未冲刷区只抬高 D → S 升高、X 下降；
2. 无衰减项时 S 只由 D 决定、与 S0 无关；底物缺口 `(S0−S)` 翻倍 → X 翻倍；
3. D 越过 μmax → X 严格为 0 而非负；
4. D = μmax 临界点专门处理，分母不为零；
5. 任意工况 `X ≥ 0`、`S ≥ 0`。
