"""启动脚本：加载 .env 后运行 demo"""
import os, sys

# 手动加载 .env 文件
env_path = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(env_path):
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                val = val.strip()
                if val:
                    os.environ[key.strip()] = val

# 运行 demo
import run_demo
run_demo.main()
