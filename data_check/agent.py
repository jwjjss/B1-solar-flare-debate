"""
数据证据 Agent · 邹智林负责
输入：factors（LiteratureFacts 中的前兆因素列表）
输出：EvidenceReport（契约 2）

mock 模式：直接返回 shared/mock_data/evidence_report.json
真实模式：解析 JW-FD → 对每个因素做统计关联（相关性/p值/时序）→ 返回
主线做统计分析；模型训练为加分项（见规范文档），SFT 数据集由吴宜俊构建、高焕景微调。
"""
from __future__ import annotations
import json, os, re, zipfile, sys
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from shared.contracts import (
    LiteratureFacts, EvidenceReport, EvidenceItem,
    evidence_report_from,
)

MOCK_PATH = os.path.join(os.path.dirname(__file__), "..", "shared", "mock_data", "evidence_report.json")

# JW-FD 数据集默认搜索路径（相对于项目根目录的上级）
_PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
_CANDIDATE_ZIPS = [
    os.path.join(_PROJECT_ROOT, "..", "JW-FD数据集示例(1).zip"),
    os.path.join(_PROJECT_ROOT, "..", "JW-FD数据集示例.zip"),
    os.path.join(_PROJECT_ROOT, "data", "JW-FD.zip"),
]

# ─────────────────── factor → feature 关键词映射表 ───────────────────
# 键为关键词（小写），值为 JW-FD CSV 中的列名（或列名前缀）
# 匹配优先级：先精确匹配 name，再模糊匹配 description
_KEYWORD_FEATURE_MAP: List[Tuple[List[str], List[str], str]] = [
    # (关键词列表, 候选列名列表, 说明)
    # F1 磁通量绳 / 磁通量 / flux rope
    (
        ["磁通量", "flux rope", "magnetic flux", "通量绳", "总磁通", "unsigned flux"],
        ["Total unsigned flux", "Total positive flux", "Total signed flux"],
        "磁通量相关特征",
    ),
    # F2 剪切运动 / 光球运动 / shear / gradient
    (
        ["剪切", "shear", "光球", "photoshe", "梯度", "gradient", "运动增强"],
        ["Gradient mean", "Gradient std", "Gradient max"],
        "磁场梯度/剪切运动特征",
    ),
    # F3 紫外 / UV / 辐射 / wavelet
    (
        ["紫外", "UV", "1600", "辐射", "增亮", "异常上升", "wavelet", "小波"],
        ["Wavelet Energy L1", "Wavelet Energy L2", "Wavelet Energy L3"],
        "小波能量/辐射特征",
    ),
    # 中性线 / neutral line / NL
    (
        ["中性线", "neutral line", "NL", " polarity", "磁极"],
        ["NL length", "NL no. fragments", "NL gradient-weighted length"],
        "中性线形态特征",
    ),
    # 曲率 / curvature / bending
    (
        ["曲率", "curvature", "弯曲", "bending"],
        ["NL curvature mean", "NL bending energy mean"],
        "磁场曲率/弯曲能量",
    ),
]

# 耀斑标签优先级：优先用 48hr/72hr 窗口（简略版数据集中 24hr 全为 0）
_LABEL_PRIORITY = [
    "flare_label_M1.0_24hr",
    "flare_label_M1.0_48hr",
    "flare_label_M1.0_72hr",
    "flare_label_C1.0_24hr",
    "flare_label_C1.0_48hr",
    "flare_label_C1.0_72hr",
]


def _find_dataset() -> Optional[str]:
    """在候选路径中查找 JW-FD zip 文件"""
    for path in _CANDIDATE_ZIPS:
        norm = os.path.normpath(path)
        if os.path.isfile(norm):
            return norm
    return None


def _load_jwfd_csv(zip_path: str) -> Tuple[pd.DataFrame, List[str]]:
    """
    从 JW-FD zip 中加载 CSV，返回 (DataFrame, feature_columns)
    feature_columns = 前 29 列（纯数值特征，不含标签和文件名）
    """
    zf = zipfile.ZipFile(zip_path)
    # 找到 fits 版 CSV（更通用）
    csv_name = None
    for name in zf.namelist():
        if name.endswith(".csv") and "fits" in name.lower() and "png" not in name.lower():
            csv_name = name
            break
    if csv_name is None:
        # 退而求其次：任何 CSV
        for name in zf.namelist():
            if name.endswith(".csv"):
                csv_name = name
                break
    if csv_name is None:
        raise FileNotFoundError("zip 中未找到 CSV 文件")

    raw = zf.read(csv_name).decode("utf-8")
    df = pd.read_csv(pd.io.common.StringIO(raw))

    # 数值化：无法转换的值置为 NaN
    numeric_cols = []
    for col in df.columns:
        if col.startswith("flare_") or col in ("image_filename", "image_path"):
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
        numeric_cols.append(col)

    return df, numeric_cols


def _select_label_column(df: pd.DataFrame) -> Tuple[str, np.ndarray]:
    """
    选择最佳耀斑标签列：优先 M 级 > C 级，优先有正样本的窗口
    返回 (列名, 二值标签数组)
    """
    for label_col in _LABEL_PRIORITY:
        if label_col not in df.columns:
            continue
        labels = pd.to_numeric(df[label_col], errors="coerce").fillna(0).values
        if labels.sum() > 0:
            return label_col, labels

    # 尝试从 flare_class_* 列构造二值标签（有耀斑=1，无=0）
    for window in ["48hr", "72hr", "24hr", "12hr"]:
        col = f"flare_class_{window}"
        if col in df.columns:
            labels = (df[col].astype(str).str.strip() != "0").astype(int).values
            if labels.sum() > 0:
                return col, labels

    # 全零：返回第一个可用标签列
    for label_col in _LABEL_PRIORITY:
        if label_col in df.columns:
            return label_col, np.zeros(len(df))

    # 最后的回退
    return "no_flare_label", np.zeros(len(df))


def _match_factor_to_features(
    factor_name: str, factor_desc: str, feature_columns: List[str]
) -> List[str]:
    """
    将文献前兆因素映射到 JW-FD 特征列列表
    策略：先用关键词表匹配，找不到则返回空列表
    """
    text = f"{factor_name} {factor_desc}".lower()
    matched_cols: List[str] = []

    for keywords, candidate_cols, _desc in _KEYWORD_FEATURE_MAP:
        if any(kw.lower() in text for kw in keywords):
            for col in candidate_cols:
                if col in feature_columns and col not in matched_cols:
                    matched_cols.append(col)

    # 额外：如果 factor_name/desc 中直接包含列名（大小写不敏感）
    for col in feature_columns:
        if col.lower() in text and col not in matched_cols:
            matched_cols.append(col)

    return matched_cols


def _compute_evidence_for_factor(
    df: pd.DataFrame,
    feature_cols: List[str],
    labels: np.ndarray,
    factor_name: str,
) -> Dict:
    """
    对单个前兆因素计算统计证据
    返回 {correlation, p_value, sample_size, counter_examples, note, supported}
    """
    n = len(df)
    valid_mask = np.ones(n, dtype=bool)

    # 取映射到的所有特征列，计算综合代理指标（取第一主成分的简化版：均值 z-score）
    feat_arrays = []
    for col in feature_cols:
        vals = pd.to_numeric(df[col], errors="coerce").values.astype(float)
        mask = ~np.isnan(vals)
        valid_mask &= mask
        feat_arrays.append(vals)

    if not feat_arrays or valid_mask.sum() < 5:
        return {
            "correlation": 0.0,
            "p_value": 1.0,
            "sample_size": int(valid_mask.sum()),
            "counter_examples": 0,
            "note": f"因素「{factor_name}」在 JW-FD 中未找到有效代理特征或有效样本不足",
            "supported": False,
        }

    n_valid = int(valid_mask.sum())

    # 综合代理：取所有特征 z-score 后求均值
    z_scores = []
    for vals in feat_arrays:
        v = vals[valid_mask]
        std = np.std(v)
        if std < 1e-12:
            z_scores.append(np.zeros(n_valid))
        else:
            z_scores.append((v - np.mean(v)) / std)
    proxy = np.mean(z_scores, axis=0)

    label_valid = labels[valid_mask]

    # 计算 Pearson 相关
    if np.std(proxy) < 1e-12 or np.std(label_valid) < 1e-12:
        corr, p_val = 0.0, 1.0
    else:
        corr, p_val = pearsonr(proxy, label_valid)

    # 反例计数：proxy 高（> 均值 + 0.5σ）但无耀斑
    threshold = np.mean(proxy) + 0.5 * np.std(proxy)
    high_proxy_no_flare = int(((proxy > threshold) & (label_valid == 0)).sum())

    supported = bool(p_val < 0.05 and abs(corr) > 0.2)
    note_parts = [f"代理特征: {', '.join(feature_cols[:3])}"]
    note_parts.append(f"有效样本 {n_valid}/{n}")
    if supported:
        note_parts.append(f"与耀斑显著{'正' if corr > 0 else '负'}相关")
    else:
        note_parts.append("未达统计显著阈值 (p<0.05, |r|>0.2)")

    return {
        "correlation": round(float(corr), 3),
        "p_value": round(float(p_val), 4),
        "sample_size": n_valid,
        "counter_examples": high_proxy_no_flare,
        "note": "；".join(note_parts),
        "supported": supported,
    }


def _print_eda(df: pd.DataFrame, feature_cols: List[str], label_col: str, labels: np.ndarray):
    """输出探索性数据分析摘要"""
    print(f"  [EDA] JW-FD 数据集概览：")
    print(f"        样本数: {len(df)}")
    print(f"        特征列: {len(feature_cols)}")
    print(f"        标签列: {label_col}（正样本: {int(labels.sum())}/{len(df)}）")

    # 活动区分布
    if "image_filename" in df.columns:
        ar_ids = []
        for fn in df["image_filename"].astype(str):
            m = re.search(r"AR(\d+)", fn)
            if m:
                ar_ids.append(m.group(1))
        ar_counter = Counter(ar_ids)
        ar_str = ", ".join(f"AR{k}({v})" for k, v in sorted(ar_counter.items()))
        print(f"        活动区: {len(ar_counter)} 个 — {ar_str}")

    # 特征缺失情况
    missing_pct = df[feature_cols].isnull().mean()
    cols_with_missing = (missing_pct > 0).sum()
    if cols_with_missing > 0:
        worst = missing_pct.idxmax()
        print(f"        缺失: {int(cols_with_missing)} 列有缺失值，最高缺失率 {missing_pct.max():.1%}（{worst}）")
    else:
        print(f"        缺失: 无")

    # 特征量级
    flux_cols = [c for c in feature_cols if "flux" in c.lower()]
    grad_cols = [c for c in feature_cols if "gradient" in c.lower()]
    if flux_cols:
        avg_flux = df[flux_cols[0]].mean()
        print(f"        磁通量均值: {avg_flux:,.0f}（{flux_cols[0]}）")
    if grad_cols:
        avg_grad = df[grad_cols[0]].mean()
        print(f"        梯度均值: {avg_grad:.1f}（{grad_cols[0]}）")


class DataEvidenceAgent:
    """
    JW-FD 数据证据 Agent
    对文献 Agent 抽取的前兆因素，在 JW-FD 上做统计关联分析，
    输出 EvidenceReport 供辩论调度器消费。
    """

    def __init__(self, dataset_path: Optional[str] = None):
        """
        dataset_path: JW-FD zip 文件路径。
                      若不指定，自动在候选路径中搜索。
        """
        self._dataset_path = dataset_path

    def run(self, factors: LiteratureFacts, use_real: bool = False) -> EvidenceReport:
        """
        主入口。
        use_real=False: mock 模式，直接返回 mock JSON（向后兼容）
        use_real=True:  真实模式，加载 JW-FD 做统计验证
        """
        if not use_real:
            with open(MOCK_PATH, "r", encoding="utf-8") as f:
                return evidence_report_from(json.load(f))

        # ──── 真实模式 ────
        zip_path = self._dataset_path or _find_dataset()
        if zip_path is None:
            print("  [data_check] 未找到 JW-FD 数据集，回退 mock 模式")
            with open(MOCK_PATH, "r", encoding="utf-8") as f:
                return evidence_report_from(json.load(f))

        print(f"  [data_check] 加载 JW-FD: {os.path.basename(zip_path)}")
        df, feature_cols = _load_jwfd_csv(zip_path)

        # 选择最佳耀斑标签
        label_col, labels = _select_label_column(df)
        _print_eda(df, feature_cols, label_col, labels)

        # 对每个 factor 计算证据
        evidences: List[EvidenceItem] = []
        for factor in factors.factors:
            matched = _match_factor_to_features(
                factor.name, factor.description, feature_cols
            )
            if not matched:
                print(f"        {factor.id} ({factor.name}): 未匹配到特征列，标记为不支持")
                evidences.append(EvidenceItem(
                    factor_id=factor.id,
                    supported=False,
                    correlation=0.0,
                    p_value=1.0,
                    sample_size=len(df),
                    counter_examples=0,
                    note=f"因素「{factor.name}」在 JW-FD 中未找到对应代理特征",
                ))
                continue

            result = _compute_evidence_for_factor(
                df, matched, labels, factor.name
            )
            status = "支持" if result["supported"] else "不支持"
            print(f"        {factor.id} ({factor.name}): {status}"
                  f" (r={result['correlation']}, p={result['p_value']})")
            evidences.append(EvidenceItem(
                factor_id=factor.id,
                supported=result["supported"],
                correlation=result["correlation"],
                p_value=result["p_value"],
                sample_size=result["sample_size"],
                counter_examples=result["counter_examples"],
                note=result["note"],
            ))

        return EvidenceReport(
            dataset="JW-FD",
            evidences=evidences,
            n_features=len(feature_cols),
        )

    def extract_visual_features(self, fits_path: str, vision_client=None) -> dict:
        """W5-W6 必做多模态：用 Qwen-VL（config.model.vision_model）读取 FITS 磁场图，
        提取视觉特征（如黑子边界扭曲度、磁通量绳形态），与 CSV 统计特征交叉验证。
        TODO(邹智林): 调 vision_client 解析图，返回视觉特征 dict。
        """
        raise NotImplementedError("Qwen-VL 多模态待实现（W5-W6）")
