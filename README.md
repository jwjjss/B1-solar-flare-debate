# 挑战杯 · B-1 耀斑触发前兆因果链发现

> 赛道一 方向 B（方向2B）太阳物理假设生成与证据推理 · 题目 1｜耀斑触发前兆因果链发现
> 仓库 https://github.com/jwjjss/B1-solar-flare-debate ｜ 在线可交互前端 https://b1-solar-flare-debate.streamlit.app/
>
> 用多智能体系统自动发现「太阳耀斑触发前兆因果链」并输出可验证科学假设。**无需物理实验**，系统的"实验"就是多智能体辩论过程本身。作品正文见 `TASK_DECLARATION.md`。

---

## 团队

| 成员 | 模块 | 一句话 |
|------|------|--------|
| **钟菁泽** | 辩论调度器 + 数据Agent + Qwen-VL | 系统大脑，全链路已跑通 |
| **吴宜俊** | 文献Agent + 假设生成 + SFT数据集 | 读论文抽因素，造训练数据 |
| **高焕景** | SFT微调（当前主攻）+ 前端/API（W4起） | 百炼训模型，后面做界面 |

> 邹智林已退出，其模块（数据Agent + Qwen-VL）由钟菁泽接手。

---

## 30 秒跑起来

```bash
# 1. 装依赖（一次性）
pip install -r requirements.txt

# 2. 跑全链路 demo（不需要 API Key，mock 数据即可）
python run_demo.py
```

运行后看 `outputs/` 目录，三个产物就是系统最终输出：

```
outputs/
├── transcript.json         # 三角色辩论全过程
├── causal_graph.json       # 因果图谱（节点+边+置信度）
└── hypothesis_report.json  # 11字段标准科学假设报告
```

改完代码后跑一下测试确认没坏：

```bash
python -m pytest tests/ -v
```

---

## 一键复现核心科学输出

本仓库的核心科学输出（候选假设 + 证据链 + 因果图谱 + 辩论记录）可一键复现。已在 `outputs/real_final/` 内附带一份**真实运行**的完整示例输出，评委无需 API Key 即可直接查阅。

**方式 A：无密钥快速验证全链路（mock，CPU-only，< 10 秒）**

```bash
python run_demo.py
```

**方式 B：复现真实科学输出（需百炼 API Key，CPU-only，约 2–5 分钟）**

```bash
# 1. 配置密钥（仅环境变量，切勿硬编码）
export DASHSCOPE_API_KEY="sk-xxxxxxxx"        # Linux/macOS
# set DASHSCOPE_API_KEY=sk-xxxxxxxx           # Windows cmd
# $env:DASHSCOPE_API_KEY="sk-xxxxxxxx"        # Windows PowerShell

# 2. 指定 JW-FD 数据目录（含 JW-FD*.zip 或其解压目录），运行真实链路
python infer.py --real \
  --data_root /path/to/JW-FD \
  --query "太阳耀斑触发前兆因果链发现" \
  --out outputs/reproduce
```

运行结束后，`outputs/reproduce/` 将生成 6 份产物：`literature_facts.json`（前兆因子）、`evidence_report.json`（统计证据 + 反例）、`transcript.json`（三角色辩论全过程）、`causal_graph.json`（因果图谱）、`hypothesis_report.json`（12 字段科学假设报告）、`run_metadata.json`（运行元信息）。

**前端查看**：`cd web && streamlit run app.py`，或直接访问已部署的公网前端 https://b1-solar-flare-debate.streamlit.app/ 。

> 硬件与时间：CPU-only 可完整复现，无需本地 GPU（推理走阿里云百炼云端 API）。mock < 10 秒；真实模式约 2–5 分钟，取决于 API 响应速度。

---

## 目录结构（按你要改的文件标了名）

```
challenge-cup/
│
├── shared/                     ← 【耦合点，改之前必须看】
│   ├── contracts.py            ← 4 个 JSON 格式定义（dataclass），改它必须例会确认
│   ├── config.py               ← 读 config.yaml
│   └── mock_data/              ← 4 个假数据，联调前各跑各的
│
├── debate/                     ← 【钟菁泽】辩论调度器
│   ├── roles.py                ← 三角色提示词（物理学家/质疑者/方法论审查）
│   ├── model_client.py         ← 百炼 Qwen 客户端，带 SFT/VL 切换
│   └── scheduler.py            ← run_debate() 入口，mock+LLM 双模式
│
├── literature/                 ← 【吴宜俊，你改这】文献 Agent
│   └── agent.py                ← 目前 mock 回退，你要接入 arxiv+Qwen+ChromaDB
│
├── data_check/                 ← 【钟菁泽】数据证据 Agent（已做完）
│   └── agent.py                ← JW-FD 加载 → 统计关联 → 输出 evidence_report.json
│
├── web/                        ← 【高焕景，W4 再碰】前端 + API
│   ├── app.py                  ← Streamlit 面板
│   └── api.py                  ← FastAPI 接口
│
├── causal_graph_builder.py     ← 【钟菁泽，已完成】辩论记录 → 因果图谱
├── hypothesis_generator.py     ← 【钟菁泽/吴宜俊】11 字段假设报告生成
├── run_demo.py                 ← 一键跑通全链路
├── run_with_key.py             ← 加载 .env 后再跑（有 Key 时用）
├── config.yaml                 ← 全局开关（SFT 模型、VL、轮次、阈值）
├── tests/                      ← 测试，跑 green 就行
├── outputs/                    ← run_demo.py 产物（gitignore）
├── requirements.txt
├── .env.example
└── README.md                   ← 你正在看
```

---

## 数据怎么流的（看懂这个就懂了整个项目）

```
吴宜俊                          钟菁泽                            高焕景
│                               │                                │
│ literature/agent.py           │  debate/scheduler.py           │
│                               │                                │
│ ① 搜论文、Qwen抽前兆因素       │                                │
│   → literature_facts.json ────►                               │
│                               │  data_check/agent.py           │
│                               │  ② 拿因素去 JW-FD 做统计验证    │
│                               │   → evidence_report.json       │
│                               │                                │
│                               │  ③ 三角色辩论（物理学家↔质疑者   │
│                               │     ↔方法论审查）多轮互怼       │
│                               │   → causal_graph.json          │
│                               │   → hypothesis_report.json     │
│                               │                                │
│                               │  ④ W5 接入 SFT 模型 ←────────── 百炼训练
│                               │                                │
│ ⑤ SFT 数据集（W3-W4）──────────► 高焕景拿去训 ←────────────    │
│                               │                                │
│                               │    ⑥ W4 起做前端 + API ──────► web/
```

---

## 你只需要改的文件

### 吴宜俊：`literature/agent.py`

现状：`run()` 方法直接读 `shared/mock_data/literature_facts.json` 返回。

你要做的事：
1. 接入 arxiv API，搜"太阳耀斑触发前兆"等关键词拉论文摘要
2. 用 Qwen 读摘要/全文，抽取前兆因素（每个因素带论文来源 arXiv ID）
3. 存入 ChromaDB 向量库
4. `run()` 返回的 dict 格式必须和 `shared/mock_data/literature_facts.json` 一模一样

**注意**：`model_client.py` 已经封装好了百炼 Qwen 调用，直接 import 用，不用自己写。

### 高焕景：百炼平台 SFT 微调

不在本地写代码——去百炼控制台操作：

1. 领 300 元代金券（https://university.aliyun.com/action/tiaozhanbei）
2. 熟悉 SFT 训练流程（数据格式、参数配置、计费）
3. 等吴宜俊产出 SFT 数据集后提交训练
4. 训练完成后，在 `config.yaml` 里把 `use_sft_model: true`，钟菁泽这边自动切换

前端 `web/` 先别看，W4 再启动。

### 钟菁泽：别的地方别乱碰

`debate/`、`data_check/`、`causal_graph_builder.py`、`hypothesis_generator.py`、`tests/`、`shared/`——这些我都搞定了，你们正常用就行。有问题找我。

---

## 四大接口契约（全局耦合点）

所有模块之间只通过这 4 个 JSON 传递数据，格式定义在 `shared/contracts.py`：

| 契约 | 谁产出 | 谁消费 | mock 样例 |
|------|--------|--------|-----------|
| `literature_facts.json` | 吴宜俊 | 钟菁泽（辩论器） | `shared/mock_data/` |
| `evidence_report.json` | 钟菁泽（数据Agent） | 钟菁泽（辩论器） | `shared/mock_data/` |
| `causal_graph.json` | 钟菁泽（辩论器） | 高焕景（前端） | `shared/mock_data/` |
| `hypothesis_report.json` | 钟菁泽（报告生成器） | 高焕景（前端） | `shared/mock_data/` |

> **改契约规则**：例会提 → 确认 → 改 `contracts.py` → 改自己代码。不准先斩后奏。

---

## Git 规范

```
git checkout -b feature/你的模块名    # 开分支
# 改代码...
git add -A
git commit -m "[模块] 动作：简述"    # 如 [literature] feat: 接入arxiv检索
git push origin feature/你的模块名    # 推到远端
# 去 Gitee 提 PR → 钟菁泽 review → 合入 master
```

---

## 红线（违反 = 废标）

1. **严禁虚构**文献、数据、参考文献（科学事实准确性 15 分）
2. **必须 Qwen**系列（百炼平台调用），换模型直接违规
3. **前端 + API 是必交项**，不是加分项（W4 启动，不能拖到最后一刻）
4. 允许 SFT 微调 Qwen（百炼平台、代金券可抵扣），训练是加分非必需
5. API Key 放 `.env`，**不准提交到 Git**
6. 大文件（JW-FD 数据集）不入库，放本地 `data/` 并 gitignore

---

## 有问题怎么办

1. 先看 `shared/contracts.py`——四个 dataclass 就是所有模块的接口格式
2. 跑 `python run_demo.py`——看全链路能不能通
3. 跑 `python -m pytest tests/ -v`——看测试绿不绿
4. 还不行就群里问钟菁泽
