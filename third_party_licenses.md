# 第三方依赖许可证清单 (Third-Party Licenses)

本作品在 MIT 许可证下开源。以下为本项目直接依赖的第三方组件及其许可证。
版本以 `requirements.txt` 钉死版本为准（对应仓库 Commit 快照）。

| 组件 | 版本 | 用途 | 许可证 |
|------|------|------|--------|
| autogen-agentchat | 0.7.5 | 多智能体辩论编排（GroupChat） | MIT / CC-BY-4.0(部分文档) |
| autogen-core | 0.7.5 | AutoGen 核心运行时 | MIT |
| autogen-ext[openai] | 0.7.5 | OpenAI 兼容客户端（对接百炼 DashScope） | MIT |
| openai | 2.45.0 | 百炼兼容模式 API 客户端 | Apache-2.0 |
| streamlit | 1.63.0 | 可交互前端面板 | Apache-2.0 |
| fastapi | 0.136.1 | 可调用测试接口 | MIT |
| uvicorn | 0.47.0 | ASGI 服务器 | BSD-3-Clause |
| chromadb | 1.5.9 | 文献向量库 | Apache-2.0 |
| numpy | 2.4.6 | 数值计算 | BSD-3-Clause |
| pandas | 3.0.3 | 数据处理（JW-FD 表格） | BSD-3-Clause |
| scipy | 1.18.0 | 统计关联分析（Pearson 相关 / p 值） | BSD-3-Clause |
| PyYAML | 6.0.3 | config.yaml 解析 | MIT |

## 外部模型与数据服务

| 名称 | 用途 | 许可 / 条款 |
|------|------|------------|
| Qwen 系列（qwen-plus / qwen-vl-max） | 各 Agent 推理引擎，经阿里云百炼（DashScope）调用 | 阿里云百炼服务条款；通义千问模型遵循其开源/商用许可 |
| 自研微调模型 qwen3-4b-instruct-2507（百炼 LoRA 部署） | 补充验证（辩论+假设阶段对照） | 基座 Qwen3 遵循其模型许可；微调权重归属本团队 |
| JW-FD 数据集 | 太阳耀斑预报观测数据（数据证据 Agent） | 竞赛组委会公开数据包，遵循其竞赛使用条款 |
| arXiv 文献 | 文献 Agent 检索与事实抽取 | arXiv 公开访问；各论文版权归原作者，引用遵循 arXiv ID 标识 |

## 说明

- API 密钥（DASHSCOPE_API_KEY 等）仅通过环境变量 / `.env` 注入，`.env` 已在 `.gitignore` 中排除，未硬编码于任何源码或配置。
- 各组件完整许可证文本可在其官方仓库或随包 `LICENSE` 文件中获取。
- 本清单对应提交 Commit 快照，如依赖升级请以对应版本许可证为准。
