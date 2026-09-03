"""
全局配置加载（W0 抽出，便于后期调优）。
无 pyyaml 或文件缺失时回退默认，保证 mock 模式永远能跑。
"""
from __future__ import annotations
import os

DEFAULT_CONFIG = {
    "debate": {"max_rounds": 8, "rounds": 2, "confidence_threshold": 0.5},
    "model": {
        "reasoning_model": "qwen-plus",
        "vision_model": "qwen-vl-max",
        "use_sft_model": False,
        "sft_model": "qwen-plus",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    },
    "flags": {"use_vl": False},
}

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.yaml")


def load_config() -> dict:
    try:
        import yaml
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        merged = {**DEFAULT_CONFIG}
        for k, v in cfg.items():
            if isinstance(v, dict) and isinstance(merged.get(k), dict):
                merged[k] = {**merged[k], **v}
            else:
                merged[k] = v
        return merged
    except Exception:
        # 无 pyyaml / 无文件 → 回退默认，不影响 mock 跑通
        return DEFAULT_CONFIG
