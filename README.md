# 活性污泥 CSTR 稳态求解服务

给市政污水厂工艺组用的常驻求解后端：上游设计程序把一组 Monod 工艺参数
通过 HTTP 丢过来，拿回一份**可复核**的活性污泥稳态解，不必每次现推公式；
还能给定初始基质/污泥浓度与仿真时长，沿时间轴积分出反应器从启动、受扰动
到逼近稳态的**整条动态轨迹**。

## 模型口径（无衰减项）

单级完全混合反应器（CSTR），微生物按 Monod 动力学降解基质：

- 比增长速率：`μ = μmax · S / (Ks + S)`
- 稳态条件：净增长恰好被出流带走，`D = μ`
- 未冲刷稳态出水基质：`S = Ks · D / (μmax − D)`
- 稳态污泥浓度：`X = Y · (S0 − S)`

动态（时间推进）方程与稳态同源，时间积分到足够久即收敛到上述稳态解：

- 污泥变化：`dX/dt = (μ(S) − D) · X`（Monod 增殖减去出流稀释）
- 基质变化：`dS/dt = D · (S0 − S) − μ(S) · X / Y`（进水补充减去出流带走与按产率消耗）

**冲刷（washout）边界**——求解器自行识别区段，并在结果中明确回报：

1. `D ≥ μmax`：菌种来不及繁殖被冲光（临界点 `D = μmax` 专门兜住，分母绝不取零）；
2. 未冲刷分式解出 `S ≥ S0`：物料衡算给出 `X ≤ 0`，同样落在冲刷边界之外。

进入冲刷后一律取 `X = 0`、`S = S0`、`μ = μmax·S0/(Ks+S0)`，
**绝不把越界的 D 代回分式造成零分母或负污泥量**。

全部内部计算用 `Decimal` 精确比较，`D == μmax` 不会因浮点误差错判区段。

## 数值推进：自适应 RK45 与非负兜底

定步长显式推进在稀释率贴近临界（慢特征值趋于 0、收敛极慢）或初值离稳态
很远（局部导数很陡）时，会放大误差甚至算出负浓度。动态仿真因此采用
嵌入式 5(4) 阶 Dormand–Prince Runge–Kutta（FSAL，见 `app/integrator.py`）：

- 每步同时给出五阶解与四阶解，差值作为局部误差估计，按 RMS 范数与
  `rtol/atol` 自动放大、缩小步长，误差超差的步作废重推；
- 输出时刻只决定取样位置，每段通过收窄当前步精确落点，不跨大步插值——
  因此把输出点取密只增加观测，不改变末端收敛到的稳态；
- 物理兜底双保险：中间态把基质探成负值 / 出现 NaN、Inf 即作废重推；
  与容差同量级的微小负越界就地夹回 0，整条轨迹 `S、X ≥ 0`；
- 步长缩到数值下限仍不满足误差/物理约束，或内部步数超过 200 000 上限，
  返回 `500 integration_failed`，绝不输出半截或虚假轨迹。

## 模块职责

| 模块 | 职责 |
| --- | --- |
| `app/kinetics.py` | 动力学内核：Monod 比增长速率（稳态与动态共用） |
| `app/solver.py` | 稳态求解、冲刷判定、参数合法性、稀释率扫描 |
| `app/integrator.py` | 自适应 RK45 时间推进内核（误差控制 + 非负兜底），与动力学解耦 |
| `app/dynamics.py` | 动态仿真：装配 ODE 右端、校验初值/时长/粒度、驱动积分、对拍稳态 |
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
| POST | `/api/simulate` | 给定参数、初态、时长做动态仿真，返回随时间轨迹 |
| POST | `/api/profiles` | 登记具名参数档（重名返回 409） |
| GET | `/api/profiles` | 列举全部档案 |
| GET | `/api/profiles/{name}` | 凭名取回档案 |
| DELETE | `/api/profiles/{name}` | 删除档案（204/404） |
| POST | `/api/profiles/{name}/solve` | 取回档案并复算 |
| POST | `/api/profiles/{name}/scan` | 取回档案并沿稀释率扫描 |
| POST | `/api/profiles/{name}/simulate` | 取回档案参数后做动态仿真 |

### 动态仿真

请求体：工艺参数（`/api/simulate` 内联；具名档路由由路径取参数）、
初始状态 `initial_state`、仿真时长 `duration`，以及二选一的输出粒度
`num_points`（等距点数，缺省 200）或 `interval`（等距时间间隔，末端 T
始终包含）。输出点数上限 5 000，内部推进步数上限 200 000。

```bash
curl -s -X POST http://127.0.0.1:8000/api/simulate \
  -H 'Content-Type: application/json' \
  -d '{"parameters":{"S0":100,"D":0.1,"mu_max":0.5,"Ks":10,"Y":0.5},
       "initial_state":{"S":0,"X":10},
       "duration":200,"num_points":41}'
# {"duration":200.0,"count":41,
#  "points":[{"time":0.0,"S":0.0,"X":10.0,"mu":0.0}, ...],
#  "final":{"time":200.0,"S":2.5,"X":48.75,"mu":0.1},
#  "steady_state":{"D":0.1,"S":2.5,"X":48.75,...,"is_washout":false}}
```

`final` 是轨迹末端点，`steady_state` 是**同参数下稳态求解器**的解，随响应
带回便于直接对拍：非冲刷工况仿真足够久，两点必须重合；冲刷工况
（`D ≥ μmax` 或分式解 `S ≥ S0`）下，无论初始投多少污泥，轨迹中 `X`
单调衰减趋零、`S` 回升到 `S0`，`steady_state.is_washout=true`。

凭具名档仿真（参数从库取回，请求体只带初态与设置）：

```bash
curl -s -X POST http://127.0.0.1:8000/api/profiles/demo_aerobic/simulate \
  -H 'Content-Type: application/json' \
  -d '{"initial_state":{"S":0,"X":10},"duration":200,"interval":25}'
```

非法初值、时长、粒度同样走结构化错误信封（422），错误键为 `S_init`、
`X_init`、`duration`、`num_points`、`interval`；积分器在步数预算内无法
完成推进时返回 500 `integration_failed`，不输出负浓度或半截轨迹。

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

- 稳态计算与动态仿真都是无状态纯函数：每次请求的初态、设置与轨迹都只活在
  本次调用的栈上，不写库、不放全局，多个并发仿真彼此不可能串改轨迹；
- 每个 HTTP 请求各自打开 SQLite 连接、写入走独立短事务（WAL，busy_timeout 30s）；
- 重名登记靠 UNIQUE 约束，只有唯一冲突映射为 409，其它完整性错误不误吞；
- `tests/test_concurrency.py` 用 32 路并发建档+复算、重名竞争、16 路并发
  扫描与 16 路并发仿真固定了隔离行为。

## 测试

```bash
pytest
# tests/test_kinetics.py        Monod 曲线
# tests/test_solver.py          稳态公式、临界点、冲刷、非法输入、扫描、三条交叉关系
# tests/test_integrator.py      RK45 解析解精度、非负兜底、粒度无关、步数上限
# tests/test_dynamics.py        动态↔稳态自洽、冲刷单调趋零、临界附近、非法设置
# tests/test_api.py             HTTP 正常/边界/错误、档案生命周期、重启持久化
# tests/test_simulation_api.py  动态仿真 HTTP：收敛、冲刷、粒度、结构化拒绝、500
# tests/test_concurrency.py     并发建档、重名竞争、并发扫描与并发仿真隔离
```

被测试钉死的关键关系：

1. 未冲刷区只抬高 D → S 升高、X 下降；
2. 无衰减项时 S 只由 D 决定、与 S0 无关；底物缺口 `(S0−S)` 翻倍 → X 翻倍；
3. D 越过 μmax → X 严格为 0 而非负；
4. D = μmax 临界点专门处理，分母不为零；
5. 任意工况 `X ≥ 0`、`S ≥ 0`；
6. 非冲刷工况仿真足够久，末端 `(S, X)` 与稳态求解器的解在容差内一致；
7. 冲刷工况（含 D = μmax 与分式解 S ≥ S0）下，任意初始污泥投量都使
   轨迹中的 X 单调不增地趋零、S 回升至 S0；
8. 同一组参数输出点取密/取疏、按点数或按间隔取样，末端收敛点不变；
9. 负初值、非正时长、非法粒度（点数越界、间隔非正/过细、二者同给）一律
   结构化拒绝，积分器不会带着坏输入空转。
