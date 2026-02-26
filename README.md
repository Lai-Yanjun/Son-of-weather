## 这是做什么的

这是 **阶段 0：验证与参数校准** 的小项目（不下单、不接 CLOB 私钥），用于回答：

- neobrother 在做什么市场？集中度如何？
- 订单/成交有什么特征？是否拆单、是否同秒连打、是否来回手？
- 用 Data API 轮询能有多“实时”？建议轮询间隔是多少？
- 用公开数据近似估算最大回撤（MDD），并给出跟单比例 `k` 的建议区间。

输出会写到 `reports/` 下的 markdown 报告与 json 数据。

## 快速开始（Windows / Linux 通用）

1) 创建虚拟环境并装依赖

```bash
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
```

2) 运行分析

```bash
.\.venv\Scripts\python run_report.py --config config.yaml report
```

生成：
- `reports/neobrother_report.md`
- `reports/neobrother_stats.json`

## 实盘测试（极小额，下单需你自行承担风险）

本项目也提供一个“只针对单一市场的实盘测试脚本”（默认 dry-run，不会下单；需要显式 `--live` 才会发单）：

- `live_trade_test_eth_2200_feb26_yes.py`：ETH > 2200（2/26）买 YES

步骤：

1) 安装依赖（会额外安装 `py-clob-client`）

```bash
.\.venv\Scripts\python -m pip install -r requirements.txt
```

2) 配置 `.env`

复制 `.env.example` 为 `.env`，填入 `PRIVATE_KEY`、`SIGNATURE_TYPE`、`FUNDER_ADDRESS`。

3) 先 dry-run 看计划

```bash
.\.venv\Scripts\python live_trade_test_eth_2200_feb26_yes.py --usdc 1
```

4) 真下单（会创建/派生 API creds、并发一笔极小额订单）

```bash
.\.venv\Scripts\python live_trade_test_eth_2200_feb26_yes.py --usdc 1 --live
```

注意：脚本会在 GTC 下单后默认 10 秒撤单；如果你要测试立即成交可用 `--order-type FOK`（风险更高）。

## 生成/派生 API creds（不下单）

如果你只想生成 `.env` 里这三项（不触发任何下单），运行：

```bash
.\.venv\Scripts\python gen_clob_api_creds.py
```

它会打印 `POLY_API_KEY / POLY_SECRET / POLY_PASSPHRASE`，复制到你的 `.env` 即可。

## 配置

编辑 `config.yaml`：
- `activity_max_trades`：拉取多少条成交用于统计（默认 2000）
- `round_trip_window_sec`：来回手识别窗口（默认 10min）
- `mdd_budget_usdc / max_total_cost_exposure_usdc`：用于计算建议 `k` 的预算示例

## 影子交易跑一周（树莓派部署版本，不下单）

你要做的是：在树莓派上跑一周“影子跟随”（只做**计划生成 + 风控 + 账本模拟**），用于观测：
- 跳过比例（too small / slippage / quote failed / 风控触发等）
- 滑点分布（你此刻可成交价 vs 他成交价）
- 敞口曲线（总成本敞口、单市场敞口、现金占用）
- 影子账户的 PnL 轨迹（近似，用 best_sell 估值）

### 1) 配置（默认就按你的要求：400u 初始本金）

`config.yaml` 新增了 `shadow:` 段，默认值已经按“400u 跑一周”给好：
- `initial_cash_usdc: 400`
- `k: 0.25`（按成交成本敞口缩放）
- `min_per_trade_usdc: 5` / `max_per_trade_usdc: 25`
- `max_abs_slippage: 0.015`
- `max_market_exposure_usdc: 120` / `max_total_exposure_usdc: 240`
- `daily_loss_limit_usdc: 60`
- `poll_interval_sec: 5`（带 `jitter_sec`）

说明：
- **首次启动**会默认把 `shadow_last_seen_ts` 设为“当前时间 - 60 秒”，避免把历史成交当作实时信号。
- 如需“从头回放历史”做离线回测：删除 `shadow_state.db` 再跑即可。

### 2) 运行（Windows / Linux 通用）

```bash
python -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python run_shadow.py --days 7 --state-db shadow_state.db --out-dir reports
```

Windows PowerShell 对应：

```bash
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python run_shadow.py --days 7 --state-db shadow_state.db --out-dir reports
```

### 3) 输出文件

运行期间会持续更新：
- `reports/shadow_kpi.md` / `reports/shadow_kpi.json`：当前 KPI（可随时打开看）
- `reports/shadow_<username>_<startTs>.jsonl`：每笔决策（follow/skip）与原因、盘口 quote、影子成交与账本变动
- `shadow_state.db`：断点续跑状态（去重、映射缓存、影子持仓/现金/已实现 PnL、估值快照）

### 4) 迁移到树莓派（推荐 /opt）

假设你把项目放到树莓派的 `/opt/pm-shadow`：

```bash
sudo mkdir -p /opt/pm-shadow
sudo chown -R $USER:$USER /opt/pm-shadow

# 从 Windows 传过去（示例用 scp，亦可用 rsync）
# scp -r ./天气之子 pi@<pi-ip>:/opt/pm-shadow

cd /opt/pm-shadow
python3 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python run_shadow.py --days 7 --state-db shadow_state.db --out-dir reports
```

### 5) systemd 后台常驻（可选）

仓库里提供了 `deploy/pm_shadow.service`，按 `/opt/pm-shadow` 路径写好。

```bash
sudo cp /opt/pm-shadow/deploy/pm_shadow.service /etc/systemd/system/pm_shadow.service
sudo systemctl daemon-reload
sudo systemctl enable --now pm_shadow.service

# 查看日志
journalctl -u pm_shadow.service -f
```


