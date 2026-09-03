"""
模型客户端：优先使用百炼平台 Qwen（OpenAI 兼容协议），无 Key 时返回 None 表示 mock 模式。

百炼兼容 OpenAI 接口，base_url 固定为：
  https://dashscope.aliyuncs.com/compatible-mode/v1
Key 从环境变量 DASHSCOPE_API_KEY 读取（也可用 BAILIAN_API_KEY）。
"""
import os


def _load_dotenv() -> None:
    """把项目根目录 .env 里的 KEY=VALUE 读入环境变量（不覆盖已有值）。

    无第三方依赖；文件缺失或行格式不符时静默跳过，保证 mock 模式不受影响。
    """
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env")
    if not os.path.isfile(env_path):
        return
    try:
        with open(env_path, "r", encoding="utf-8") as fh:
            for raw in fh:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
    except OSError:
        return


def get_model_client(config: dict | None = None):
    """返回 AutoGen 的 Qwen 客户端；若无 Key 或导入失败则返回 None（mock 模式）。

    config: 来自 shared.config.load_config()；据 use_sft_model 切换推理/SFT 模型。
    """
    _load_dotenv()
    key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("BAILIAN_API_KEY")
    if not key:
        return None
    cfg = config or {}
    m = cfg.get("model", {})
    use_sft = m.get("use_sft_model", False)
    model = m.get("sft_model") if use_sft else m.get("reasoning_model", "qwen-plus")
    vision_model = m.get("vision_model", "qwen-vl-max")
    vision = (model == vision_model)  # 选中的模型即 VL 模型时开启视觉能力
    base_url = m.get("base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    try:
        from autogen_ext.models.openai import OpenAIChatCompletionClient
        return OpenAIChatCompletionClient(
            model=model,
            api_key=key,
            base_url=base_url,
            model_info={
                "vision": vision,
                "function_calling": True,
                "json_output": True,
                "structured_output": True,
                "family": "qwen",
            },
        )
    except Exception as e:  # pragma: no cover
        print(f"[warn] 无法初始化 Qwen 客户端: {e}，回退 mock 模式")
        return None


def is_mock(client) -> bool:
    return client is None
