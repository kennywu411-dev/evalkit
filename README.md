# evalkit

[![CI](https://github.com/kennywu411-dev/evalkit/actions/workflows/ci.yml/badge.svg)](https://github.com/kennywu411-dev/evalkit/actions/workflows/ci.yml) ![Python 3.12](https://img.shields.io/badge/python-3.12-blue) ![Dependencies: none](https://img.shields.io/badge/dependencies-none-success) ![License: MIT](https://img.shields.io/badge/license-MIT-green)

LLM 应用评测平台：评测集管理、打分器、实验运行、配对统计、回归门禁、线上反馈和静态看板，打包成一个零运行时依赖的 Python 包。

两个真实项目的评测已经直接改成调用 evalkit：[DataPilot](https://github.com/kennywu411-dev/datapilot)（NL2SQL 问数）和[合同风险审查](https://github.com/kennywu411-dev/contract-review-agent)。两个项目已有的运行记录都能复算出各自数字登记表里的关键数字：

| 项目 | 任务 | 复算项 | 结果 |
|---|---|---|---|
| DataPilot | NL2SQL 问数（BIRD Mini-Dev 500 题，执行结果比对） | 基线 EX 与置信区间、按难度拆分、12 组配对检验、消融、3 次重复的均值 ± 标准差与稳定性、成本、延迟；指标语义层 6 组 × 3 次的通过题数和 24 组逐次配对检验；分层抽样与 SQL 执行比对器的逐题核对 | 90 / 90 一致 |
| 合同风险审查 | 48 份构造合同的风险审查（规则打分） | 检出率、类型 / 依据准确率、三种口径精确率、干净合同误报、转人工、反问，全部 / 调试小样 / 留出集，5 次全量运行 | 79 / 79 一致 |

完整对照表见 [docs/reproduction.md](docs/reproduction.md)。

## 快速开始

```bash
uv sync
uv run pytest -q
```

入门示例是一个"回答首都"的假应用，有 v1、v2 两个版本，不调用任何模型：

```bash
uv run evalkit run --dataset examples/quickstart/capitals.jsonl \
  --task examples.quickstart.app:answer_v1 --scorer exact_match --variant app --tag v1
uv run evalkit run --dataset examples/quickstart/capitals.jsonl \
  --task examples.quickstart.app:answer_v2 --scorer exact_match --variant app --tag v2

uv run evalkit compare runs/v1-* runs/v2-* --slice region   # 配对比较，按地区拆分
uv run evalkit gate --baseline runs/v1-* --candidate runs/v2-*   # v2 变差，退出码 1
uv run evalkit dashboard runs examples/datapilot/runs examples/contract_review/runs -o site/index.html
```

`site/index.html` 是一个自包含的文件，数据内嵌，直接用浏览器打开即可。

复算两个接入项目的数字：

```bash
uv run python examples/datapilot/reproduce.py
uv run python examples/contract_review/reproduce.py
```

## 能力一览

| 模块 | 做什么 |
|---|---|
| `evalkit.dataset` | JSONL 评测集；语义化版本号 + 内容哈希；开发集 / 留出集划分；按标签分层抽样（最大余数法，固定种子） |
| `evalkit.scorers` | 精确匹配（含归一化、多候选答案）；SQL 执行结果比对（只读连接、预测与标准共用超时预算，口径同 BIRD 官方 EX）；规则打分（多条具名检查） |
| `evalkit.judge` | LLM 当裁判：逐题打分；两两比较（交换位置各判一次，不一致记平局）；与人工标注的一致率和 Cohen's κ 校准 |
| `evalkit.clients` | 裁判模型客户端（OpenAI 兼容接口、Anthropic Messages API），只用 urllib；限流重试、按 token 计价、累计花费上限 |
| `evalkit.labels` | 人工标注导入（CSV / JSONL，可按标注者筛选），可直接当打分器用 |
| `evalkit.runner` | 并发执行、逐题落盘、断点续跑、花费硬上限、配置快照、每题 trace；`rescore` 用已保存的输出重新打分 |
| `evalkit.stats` | Wilson 区间、bootstrap 区间、McNemar 精确检验、配对 bootstrap 差值、均值 ± 标准差、稳定性 |
| `evalkit.report` | 按变体汇总、按子集拆分、成本和延迟分位数、多次运行统计、跨题累加的比例指标（`Ratio`） |
| `evalkit.compare` | 两次运行的配对比较，可按子集拆分 |
| `evalkit.gate` | 回归门禁，一条命令判断有没有显著变差，退出码可直接用于 CI |
| `evalkit.feedback` | 线上反馈：只追加写入，同一目标以最后一次打标为准，按维度汇总；自带最小的 HTTP 接收服务 |
| `evalkit.dashboard` | 静态看板：实验列表、指标对比、失败题逐条回放、成本与延迟 |

## 命令行

| 命令 | 用途 |
|---|---|
| `evalkit run` | 跑一次实验（`--task 模块:函数`，`--scorer` 可多个，`--repeats`、`--max-cost`、`--resume`） |
| `evalkit report` | 汇总一次运行，`--slice` 按标签拆分 |
| `evalkit compare` | 两次运行的配对比较 |
| `evalkit repeats` | 多次运行的均值 ± 标准差和稳定性，可以合并多个运行目录 |
| `evalkit gate` | 回归门禁，不通过时退出码 1 |
| `evalkit rescore` | 用保存的输出重新打分，不调用模型 |
| `evalkit dataset info / validate / diff / split / sample` | 评测集管理 |
| `evalkit calibrate` | LLM 裁判与人工标注的一致率和 κ，`--min-agreement` 不达标时退出码 1 |
| `evalkit feedback summary / add / serve` | 线上反馈：汇总、追加、HTTP 接收服务 |
| `evalkit dashboard` | 生成静态看板 |

## 评测集管理

```bash
evalkit dataset info evals/qa.jsonl                         # 题数、版本、内容哈希、划分、各标签分布
evalkit dataset validate evals/qa.jsonl                     # 缺标准答案、未知划分等问题，有问题退出码 1
evalkit dataset split evals/qa.jsonl --holdout 0.3 --by difficulty --version 1.1.0 -o evals/qa-v1.1.jsonl
evalkit dataset sample evals/qa.jsonl -n 20 --by difficulty -o evals/smoke.jsonl   # 冒烟集
evalkit dataset diff evals/qa.jsonl evals/qa-v1.1.jsonl     # 新增 / 删除 / 改动了哪些题、改了哪些字段
```

评测集是一个 JSONL 文件加一个同名的 `.meta.json`（名称和语义化版本号）。每次运行的配置快照里都记录评测集的内容哈希，以及每题的哈希。

## 接入自己的应用

被评测的应用以一个函数接入：收到一道题，返回输出（以及花费和中间过程）。

```python
import evalkit.dataset as ds
from evalkit.runner import TaskResult, Variant, run
from evalkit.scorers import ExactMatch, RuleScorer, contains_all

data = ds.load("evals/qa.jsonl")  # 每行 {"id", "input", "expected", "tags", "split"}
dev = data.split("dev")  # 调试只看开发集，留出集留到最后


def make_task(model: str):
    async def task(item):
        resp = await my_app.answer(item.input, model=model)
        return TaskResult(output=resp.text, cost_usd=resp.cost, trace=resp.steps)

    return task


result = run(
    dev,
    [
        Variant("haiku", make_task("haiku"), {"model": "haiku", "prompt": "v3"}),
        Variant("sonnet", make_task("sonnet"), {"model": "sonnet", "prompt": "v3"}),
    ],
    [ExactMatch(), RuleScorer("format", [("cites", contains_all(["依据"]))])],
    tag="prompt-v3",
    repeats=3,
    max_cost=2.0,  # 累计花费上限（美元），续跑时已花的也算
)
```

运行目录结构：

```
runs/<run_id>/
  config.json     评测集名称、版本、内容哈希，各变体参数，打分器，题目 id，代码版本
  results.jsonl   每题一行，跑完立即追加
  summary.json    汇总
  traces/         每题一个文件：输入、输出、中间步骤、各打分器结果
```

约定：

- 第一个打分器是主指标，它的 `passed` 写进结果行，配对检验和门禁默认看它。
- 任务可以是同步函数，也可以是异步函数。同步函数放进线程里执行，同样受 `concurrency` 控制。
- 任务抛异常记为 `status=error`，续跑（`resume=<run_id>`）时只重跑这些题；抛 `AbortRun`（密钥失效、额度用完）则立即停止整次运行。
- 续跑可以往评测集里加题。如果已经跑过的题内容变了，就拒绝续跑：标准答案订正后应该用 `rescore`，或者新开一次运行。
- 应用已经有自己的 trace 格式时，`TaskResult(trace=<dict>, trace_id=<自己的 id>)` 会把它原样写进 `traces/<trace_id>.json`，并附上题目、输出和打分。读取旧 trace 的代码不用改。

## LLM 当裁判

裁判模型通过一个函数注入：接收提示词，返回原话或 `(原话, 花费)`。`evalkit.clients` 提供两个现成的实现，只用标准库：

```python
from evalkit.clients import anthropic_client, openai_compatible, with_budget
from evalkit.judge import PairwiseJudge, PointwiseJudge, calibrate
from evalkit.labels import load_labels

llm = with_budget(
    anthropic_client("<模型 id>", price_per_mtok=(1.0, 5.0)),  # (输入, 输出) 每百万 token 美元
    max_cost=2.0,  # 累计花费到上限后抛 AbortRun，评测运行会停下并提示续跑
)
# 也可以用 openai_compatible("<模型 id>", base_url="https://.../v1", api_key_env="...")

judge = PointwiseJudge(client=llm, rubric="答案是否准确、是否引用了依据", pass_score=4)
pair = PairwiseJudge(client=llm, rubric="哪个修改建议更可执行")
out = pair.compare_many([(q, old, new) for q, old, new in samples])
print(out["b_win_rate"], out["position_consistency"])

report = calibrate(judge_labels, load_labels("labels/human.csv"))
print(report["agreement"], report["kappa"], report["disagreements"])
```

两条使用原则：两两比较一律交换位置各判一次，两次结论不一致记平局，并报告位置一致率；裁判和人工的一致率没达标之前，裁判分数只作参考，不进门禁。校准也可以在命令行里做，适合放进 CI：

```bash
evalkit calibrate --judge labels/judge.csv --human labels/human.csv --min-agreement 0.8
```

## 统计口径

- 通过率区间：Wilson 95% 区间。题少时比正态近似可靠，20 题的区间往往宽达 30 个百分点，这本身就是结论：小样本只能测单价，不能下结论。
- 比较两个配置：只比较两边都有的题，用 McNemar 精确检验，只看"A 对 B 错"和"B 对 A 错"的题。这比分别算两个区间、看是否重叠更灵敏。
- 多次运行：同一配置跑多次，报告均值 ± 标准差，以及"每次都对 / 每次都错 / 时对时错"的题数。DataPilot 的数据显示单题会翻转（Sonnet 4.4%、Haiku 7.4% 的题时对时错），但总分波动在 1.5 个百分点以内。
- 跨题累加的指标：检出率、精确率这类指标按题累加分子分母（微平均），而不是对每题的比例取平均。用 `Ratio` 声明，`aggregate` 计算，可以给出按题重抽样的 bootstrap 区间。
- 分位数：P50 / P95 取排序后第 ⌊q·n⌋ 个值，不插值，和两个接入项目的口径一致。

## 回归门禁

```bash
evalkit gate --baseline runs/<基线> --candidate runs/<候选> --max-drop 0.02 --alpha 0.05 \
  [--max-error-rate 0.05] [--max-cost-ratio 1.5] [--summary-file "$GITHUB_STEP_SUMMARY"]
```

下面任何一条触发都不通过（退出码 1）：通过率下降且 McNemar p < alpha；降幅超过 `max-drop`；候选缺了基线里的题；出错率超限；花费超过基线的指定倍数。

显著性检验和降幅阈值两条都要。题少时（比如 20 题的冒烟集）掉 10 个点也可能不显著，只靠检验会放过回归；题多时 1 个点的波动也可能"显著"，只靠阈值又会误报。阈值回答"业务上能容忍多少"，检验回答"这是不是噪声"。

`.github/workflows/ci.yml` 里有完整示例：跑测试、复算两个接入项目的数字，并演示门禁拦下一次变差的发布。

## 线上反馈

```python
from evalkit.feedback import FeedbackStore, summarize

store = FeedbackStore("feedback.jsonl", labels=["useful", "not_useful", "not_risk", "unclear"])
store.add(
    {
        "session_id": sid,
        "review_id": rid,
        "finding_index": 2,
        "label": "not_risk",
        "finding": {"risk_type": "penalty", "level": "高"},
    }
)

summarize(
    store.load(),
    key=["session_id", "review_id", "finding_index"],  # 同一目标以最后一次打标为准
    labels=["useful", "not_useful", "not_risk", "unclear"],
    positive="useful",
    negative="not_risk",  # 采纳率、误报率
    where=lambda r: r["finding"]["level"] in ("高", "中"),
    by="finding.risk_type",
)
```

文件只追加、不改写，改标历史全部保留。只存调用方传进来的字段，建议只记 id、标签和输出摘要，不记用户原文。

没有现成后端时，可以直接启动接收服务，让前端 POST 打标：

```bash
evalkit feedback serve feedback.jsonl --key session_id,review_id,finding_index \
  --labels useful,not_useful,not_risk,unclear --positive useful --negative not_risk --port 8787
# POST /feedback 写入一条打标（标签不在集合里、缺字段返回 422），GET /feedback/summary 读汇总
```

## 看板

`evalkit dashboard <运行目录或其父目录>... -o site/index.html` 生成一个 HTML 文件：

- 实验列表：各运行、各变体的通过率与区间、成本、延迟、状态分布；点开看子集拆分、配置快照和自定义指标（运行目录里的 `metrics.json`）。
- 指标对比：任选同一评测集下的两个实验，浏览器里现算配对检验，按子集拆分；两边判分不一致的题可以一键跳到回放。
- 失败回放：按状态、子集、题号筛选失败题，逐条看判分、模型输出和中间过程。
- 成本与延迟：单题成本、P50 / P95 延迟，以及"通过率 × 成本"散点图。

为了控制体积，只有失败题保留输出和过程。

## 在已有项目里接入

两个项目原来各有一套自己的评测代码，现在都改成调用 evalkit。接入前后各做一次数字快照，逐项比较，全部一致之后才合并。

### DataPilot

- 运行：`evals/runner.py` 由 `evalkit.runner` 负责并发、逐题落盘、续跑和本次花费上限。生成和判分拆成两步：生成是任务，执行结果比对是打分器。
- 结果格式：结果行同时带 evalkit 字段（`item_id`、`variant`、`passed`、`scores`）和原有字段（用 `row_fields` 补），一个文件两边都能读。中间步骤照旧记在结果行里（`save_traces=False`），因为这个项目的运行目录是纳入版本库的，不另写几千个 trace 文件。模型调用出错仍记为 `llm_error`，续跑时重跑（`retry_status`）。
- 统计：分层抽样、Wilson 区间、分位数、McNemar 检验改用 `evalkit.dataset` 和 `evalkit.stats`，与原实现逐位一致。
- 新命令：`evals/compare.py --repeat --slice`（按某次重复配对、按子集拆分），`evals/gate.py --compare`（两次已有运行的配对门禁），`evals/dashboard.py`（静态看板）。
- 验证：接入前后的快照共 311 项，逐项一致。快照内容包括 45 次运行的汇总、256 组配对比较、看板数据、模型对比、指标匹配、数据质量评测和离线门禁；智能路由回放的输出也逐行一致。

### 合同风险审查

`cra eval-review` 直接调用 evalkit：

- 运行：合同 → `Dataset`（正文只记哈希），审查流程 → `Variant`，原有判分规则包成打分器。逐份落盘、续跑、预算闸门、并发都由 `evalkit.runner` 负责。
- 指标：检出率、精确率等 20 多个指标改用 `Ratio` 声明，分位数用 `evalkit.stats.percentile`。
- 兼容：运行目录是 evalkit 格式，同时派生一份原来的 `review_results.jsonl` 和 `summary.json`。trace 文件沿用原格式。依据校验、多轮追问、分析脚本和 Web 运行记录页都不用改。接入之前的旧运行，在续跑、门禁、看板时现场转换。
- 新命令：`cra eval-gate`（回归门禁）、`cra eval-dashboard`（看板），都可以先按当前金标准从 trace 重新判分，让历史运行和新运行用同一口径比较。
- 验证：接入前后各做一次快照，内容包括全部 21 次审查运行的汇总，以及三个分析脚本对 11 次关键运行的输出。逐项比较一致。接入前有几项会报错（早期运行缺少后来才加的字段），现在不再报错：分析脚本可以直接复现"运行时口径"的数字。
- 线上反馈：试用反馈的存储和"按最后一次为准"的汇总改用 `evalkit.feedback`，输出口径不变。

## 两个复算示例

适配器只做一件事：把原项目的运行记录转成 evalkit 的运行目录，然后所有数字都由 evalkit 的通用模块计算。

DataPilot（`examples/datapilot/`）：`item_id` 对应题号，`variant` 对应模型，主打分器是执行结果比对。原项目的 9 次实验运行导入后，复算基线、逐项实验阶梯、12 组配对检验、检索消融（只完成 373 题，在共同题目上比较）、最终配置 3 次重复。原项目的 3 次重复分散在两个运行目录里，导入时用 repeat 偏移把它们接成一个运行。

合同风险审查（`examples/contract_review/`）：一份合同是一道题，判分规则移植成 `ContractScorer`。每份合同产出逐项计数（检出了哪些风险、几条误报、引用对不对），再用 13 个 `Ratio` 指标跨合同累加。两种口径都复算了：运行时记下的计数（验证累加方式），以及用 `rescore` 从保存的审查结果按订正后的金标准重新判分（验证规则打分器逐份一致）。评测集只导出金标准和标签，不含合同正文。

复算发现的一处笔误：DataPilot 登记表里 E1 → E2（Sonnet）的 McNemar p 原来写作 0.59。13 / 17 的精确值是 0.5847，应为 0.58，原项目自己的 `compare.py` 算出的也是这个值，是手工抄录时的舍入笔误，已在原项目里改正。复算脚本保留了 `KNOWN` 机制：查明原因、暂时还不能改原文的差异单独列出并写明原因，而不是放宽比较精度。

## 设计取舍

- 零运行时依赖：核心只用标准库。评测代码会被拷进各个项目、跑在 CI 里，依赖越少越不容易和宿主项目冲突。统计方法都很小，自己实现并用已知数值测试。
- 运行记录是唯一事实来源：所有数字由 `results.jsonl` 现算，报告、门禁、看板都不手抄数字。结果逐题追加，进程中途退出不会丢数据，续跑时同一 key 以最后一行为准。
- 主指标是二值的：配对检验和门禁需要每题一个对错。连续分数（裁判分、规则通过比例）放在 `scores` 里一并保存；需要跨题累加的指标用 `Ratio` 单独声明，不强行塞进"每题一个分数"。
- 配置快照带内容哈希：只记版本号挡不住"同一版本号、不同题目"。快照按题记录哈希，续跑时可以加题，但已经跑过的题被改动就拒绝。
- 重新打分不调用模型：金标准订正、打分规则修改后，用保存的输出重算（`rescore`），结果写到新目录，原始结果不动。合同审查示例里的"订正后口径"就是这样得到的。
- 裁判是可插拔的函数：不绑定任何模型厂商。测试用假函数，接入时包一层真实调用。
- 看板是单个静态文件：不需要服务器和前端构建，可以作为 CI 产物上传，也可以直接发给别人。代价是数据量大时文件会变大，所以通过的题不带输出。
- 适配器放在 examples 而不是核心：SQL、合同这些领域知识不进核心包。核心只认识"题、输出、分数"。

## 目录结构

```
src/evalkit/          核心包（dataset、scorers、judge、clients、labels、runner、stats、report、compare、gate、feedback、dashboard、cli）
tests/                单元测试和集成测试（含两个示例的复算）
examples/quickstart/  入门示例
examples/datapilot/   DataPilot 适配器、复算脚本、导出的运行记录样例
examples/contract_review/  合同审查适配器、规则打分器、复算脚本、评测集（仅金标准）和运行记录样例
docs/reproduction.md  复算对照表
```

## 局限

- 主指标只支持二值；排序类、生成质量类任务需要先用规则或裁判转成通过 / 不通过，或者用 `Ratio` 和 `scores` 自行汇总。
- McNemar 检验假设两次运行的题目是同一批、彼此独立。同一合同里的多处风险不独立，这类指标用按题重抽样的 bootstrap 区间。
- 运行器是单机 asyncio 并发，没有分布式调度和任务队列。
- 看板把数据嵌进 HTML，适合上千道题、几十次运行的规模，更大的规模需要改成按需加载。
