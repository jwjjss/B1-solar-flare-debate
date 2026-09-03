"""
可交互前端（Streamlit）· B-1 太阳耀斑触发前兆因果链系统

5 个 tab 对应系统 5 个环节，让评委一眼看懂「文献 → 证据 → 辩论 → 图谱 → 假设」闭环：
  ① 文献因素   ② 数据证据   ③ 三角色辩论   ④ 因果图谱（可视化）  ⑤ 科学假设报告

数据源：只读 outputs/real_final（真实产物）静态 JSON，缺失时回退 mock_data；
绝不调用 LLM、绝不联网，保证评委离线可复现、零 API 费用。

运行：streamlit run web/app.py
"""
from __future__ import annotations

import html
import json

import pandas as pd
import streamlit as st

try:  # streamlit run web/app.py（web/ 在 sys.path）
    from data_source import (  # type: ignore
        DataSourceError, data_dir_label, is_real, load,
    )
except ImportError:  # 以仓库根为工作目录
    from web.data_source import (
        DataSourceError, data_dir_label, is_real, load,
    )

# ───────────────────────── 角色配色与中文名 ─────────────────────────
ROLE_META = {
    "physicist": {"zh": "物理学家", "color": "#1f77b4", "icon": "🔭"},
    "skeptic": {"zh": "质疑者", "color": "#d62728", "icon": "⚖️"},
    "methodologist": {"zh": "方法论审查", "color": "#2ca02c", "icon": "📋"},
}

# 因果图谱节点样式（按 ntype）
NODE_STYLE = {
    "precursor": {"shape": "ellipse", "fill": "#D5E8F0", "color": "#1f77b4", "zh": "前兆"},
    "mechanism": {"shape": "box", "fill": "#FFF2CC", "color": "#d6a400", "zh": "机制"},
    "event": {"shape": "doublecircle", "fill": "#F8CECC", "color": "#b85450", "zh": "事件"},
}

# 边样式（按 relation）
EDGE_STYLE = {
    "triggers": {"style": "solid", "color": "#b85450", "zh": "直接触发"},
    "enables": {"style": "solid", "color": "#d6a400", "zh": "使能/许可"},
    "correlates": {"style": "dashed", "color": "#999999", "zh": "弱关联"},
}

# 假设报告 11 字段 → 中文小标题（按展示顺序）
REPORT_FIELDS = [
    ("problem_statement", "1 · 待研究问题（Problem Statement）"),
    ("rationale", "2 · 解决思路（Rationale）"),
    ("technical_details", "3 · 技术手段（Technical Details）"),
    ("datasets", "4 · 使用的数据集（Datasets）"),
    ("source", "5 · 假设推演依据（Source）"),
    ("target", "6 · 验证目标 / 拟采集数据（Target）"),
    ("paper_title", "7 · 假想论文标题（Paper Title）"),
    ("abstract", "8 · 摘要（Abstract）"),
    ("methods", "9 · 方法论（Methods）"),
    ("experiments", "10 · 实验设计（Experiments）"),
    ("results", "11 · 结果 / 已有证据（Results）"),
]


# ───────────────────────── 工具函数 ─────────────────────────
@st.cache_data(show_spinner=False)
def _load_cached(name: str):
    return load(name)


def _safe_load(name: str):
    try:
        return _load_cached(name)
    except DataSourceError as exc:
        st.error(str(exc))
        return None


def _wrap(text: str, width: int = 28) -> str:
    """把长标签按词折行，供 graphviz 节点显示（用 \\n 换行）。"""
    words = str(text).split()
    lines, cur = [], ""
    for w in words:
        if len(cur) + len(w) + 1 <= width:
            cur = f"{cur} {w}".strip()
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return "\\n".join(lines)


def _build_dot(graph: dict) -> str:
    """手写 Graphviz DOT 字符串（st.graphviz_chart 在浏览器端用 viz.js 渲染，
    无需系统安装 graphviz 可执行文件）。"""
    nodes = {n["id"]: n for n in graph.get("nodes", [])}
    lines = [
        "digraph G {",
        '  graph [rankdir=LR, fontname="Microsoft YaHei", bgcolor="transparent"];',
        '  node  [fontname="Microsoft YaHei", fontsize=11, style="filled"];',
        '  edge  [fontname="Microsoft YaHei", fontsize=9];',
    ]
    for nid, n in nodes.items():
        st_style = NODE_STYLE.get(n.get("ntype", "precursor"), NODE_STYLE["precursor"])
        label = _wrap(n.get("label", nid))
        if n.get("ntype") != "event":
            label = f"{nid}\\n{label}"
        lines.append(
            f'  "{nid}" [label="{label}", shape={st_style["shape"]}, '
            f'fillcolor="{st_style["fill"]}", color="{st_style["color"]}"];'
        )
    for e in graph.get("edges", []):
        es = EDGE_STYLE.get(e.get("relation", "triggers"), EDGE_STYLE["triggers"])
        conf = e.get("confidence", "")
        lines.append(
            f'  "{e["src"]}" -> "{e["dst"]}" '
            f'[label="{e.get("relation", "")}\\nconf={conf}", style={es["style"]}, '
            f'color="{es["color"]}", fontcolor="{es["color"]}"];'
        )
    lines.append("}")
    return "\n".join(lines)


def _legend() -> None:
    """图谱图例。"""
    cols = st.columns(3)
    with cols[0]:
        st.caption("🔵 椭圆 = 前兆（precursor）")
    with cols[1]:
        st.caption("🟡 方框 = 机制（mechanism）")
    with cols[2]:
        st.caption("🔴 双圈 = 耀斑事件（event）")
    st.caption("实线 = triggers/enables（因果主链） ｜ 虚线 = correlates（弱关联，未通过数据门控）")


# ───────────────────────── 页面框架 ─────────────────────────
st.set_page_config(page_title="B-1 耀斑前兆因果链", page_icon="☀️", layout="wide")

st.title("☀️ B-1 太阳耀斑触发前兆因果链发现系统")
badge = "🟢 真实模式产物" if is_real() else "🟠 MOCK 兜底数据"
st.caption(f"多智能体辩论科学假设生成系统 ｜ 数据源：{data_dir_label()} ｜ {badge}")
st.divider()

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📚 文献因素", "📊 数据证据", "💬 三角色辩论", "🕸️ 因果图谱", "📝 科学假设报告"]
)

# ───────────────────────── ① 文献因素 ─────────────────────────
with tab1:
    facts = _safe_load("literature_facts.json")
    if facts:
        st.metric("检索论文数", facts.get("papers_searched", "?"))
        st.caption(f"检索关键词：{facts.get('query', '')}")
        st.write(f"共抽取 **{len(facts.get('factors', []))}** 个前兆候选因子，每个均带真实可查的 arXiv 来源：")
        for f in facts.get("factors", []):
            ftype = f.get("ftype", "precursor")
            with st.expander(f"[{f.get('id')}] {f.get('name')}  ·  {ftype}  ·  conf={f.get('confidence')}"):
                st.markdown(f"**类型**：`{ftype}` ｜ **文献置信度**：{f.get('confidence')} ｜ **来源**：{f.get('source')}")
                st.info(f.get("description", ""))
                st.markdown(f"🔗 核验链接：https://arxiv.org/abs/{str(f.get('source', '')).replace('arXiv:', '')}")

# ───────────────────────── ② 数据证据 ─────────────────────────
with tab2:
    evidence = _safe_load("evidence_report.json")
    if evidence:
        n_feat = evidence.get("n_features")
        head = f"数据集：**{evidence.get('dataset', '?')}**"
        if n_feat:
            head += f" ｜ 特征列数：**{n_feat}**"
        st.markdown(head)
        st.caption("判定规则：p ≤ 0.05 且 |correlation| ≥ 0.3 → 支持（✓）；否则不支持（✗）")
        rows = []
        for e in evidence.get("evidences", []):
            rows.append({
                "因素": e.get("factor_id"),
                "支持": "✓ 支持" if e.get("supported") else "✗ 不支持",
                "相关性 r": e.get("correlation"),
                "p 值": e.get("p_value"),
                "样本量": e.get("sample_size"),
                "反例数": e.get("counter_examples"),
                "说明": e.get("note", ""),
            })
        df = pd.DataFrame(rows)
        st.dataframe(df, width="stretch", hide_index=True)
        # 支持/不支持计数
        n_sup = sum(1 for e in evidence.get("evidences", []) if e.get("supported"))
        c1, c2 = st.columns(2)
        c1.metric("数据支持的因子", n_sup)
        c2.metric("被数据推翻的因子", len(evidence.get("evidences", [])) - n_sup)

# ───────────────────────── ③ 三角色辩论 ─────────────────────────
with tab3:
    transcript = _safe_load("transcript.json")
    if transcript:
        n_rounds = max((m.get("round", 1) for m in transcript), default=1)
        st.markdown(f"共 **{n_rounds}** 轮、**{len(transcript)}** 条发言。三角色：🔭 物理学家（蓝）｜ ⚖️ 质疑者（红）｜ 📋 方法论审查（绿）")
        for r in range(1, n_rounds + 1):
            st.markdown(f"#### 第 {r} 轮")
            msgs = [m for m in transcript if m.get("round") == r]
            for m in msgs:
                meta = ROLE_META.get(m.get("speaker", ""), {"zh": m.get("speaker", "?"), "color": "#666", "icon": "•"})
                content = html.escape(m.get("content", ""))
                bubble = (
                    f'<div style="border-left:5px solid {meta["color"]};'
                    f'background:#fafafa;padding:10px 14px;margin:8px 0;border-radius:4px;">'
                    f'<b style="color:{meta["color"]};">{meta["icon"]} {meta["zh"]}</b>'
                    f'<span style="color:#888;font-size:0.85em;"> ｜ round {r}</span>'
                    f'<div style="white-space:pre-wrap;margin-top:6px;font-size:0.92em;'
                    f'line-height:1.5;">{content}</div></div>'
                )
                with st.expander(f'{meta["icon"]} 第 {r} 轮 · {meta["zh"]}', expanded=(r == 1)):
                    st.markdown(bubble, unsafe_allow_html=True)

# ───────────────────────── ④ 因果图谱 ─────────────────────────
with tab4:
    graph = _safe_load("causal_graph.json")
    if graph:
        st.success(f"**根因结论**：{graph.get('root_cause', '')}")
        st.metric("辩论轮数", graph.get("debate_rounds", "?"))
        st.subheader("因果链可视化")
        _legend()
        try:
            st.graphviz_chart(_build_dot(graph), width="stretch")
        except Exception as exc:  # pragma: no cover - 渲染兜底
            st.warning(f"图形渲染不可用（{exc}），请见下方边明细表。")
        st.subheader("边明细")
        erows = []
        nodes = {n["id"]: n for n in graph.get("nodes", [])}
        for e in graph.get("edges", []):
            erows.append({
                "源": f'{e.get("src")}（{nodes.get(e.get("src"), {}).get("label", "")[:30]}）',
                "目标": e.get("dst"),
                "关系": e.get("relation"),
                "置信度": e.get("confidence"),
                "证据来源": e.get("evidence_ref", ""),
            })
        st.dataframe(pd.DataFrame(erows), width="stretch", hide_index=True)
        with st.expander("查看原始 causal_graph.json"):
            st.json(graph)

# ───────────────────────── ⑤ 科学假设报告 ─────────────────────────
with tab5:
    report = _safe_load("hypothesis_report.json")
    if report:
        st.header(report.get("paper_title", "科学假设报告"))
        st.caption("11 字段《科学假设与研究计划》· 评审硬性要求，缺一不可")
        for key, zh in REPORT_FIELDS:
            st.markdown(f"**{zh}**")
            st.write(report.get(key, ""))
        # 参考文献
        st.markdown("**12 · 参考文献（References，真实可查，严禁虚构）**")
        for ref in report.get("references", []):
            aid = str(ref).replace("arXiv:", "")
            st.markdown(f"- {ref} ｜ 🔗 https://arxiv.org/abs/{aid}")
        st.divider()
        st.download_button(
            "⬇️ 下载完整报告 JSON",
            data=json.dumps(report, ensure_ascii=False, indent=2),
            file_name="hypothesis_report.json",
            mime="application/json",
        )
