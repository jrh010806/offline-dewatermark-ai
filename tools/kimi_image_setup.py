"""Local setup helpers. Credentials are entered in Terminal, never in chat."""
from __future__ import annotations

import argparse
import getpass
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def set_image_key() -> None:
    from kimi_image_tools import SECRETS

    print("图像编辑使用独立的 OpenAI API Key；Kimi/DeepSeek 对话 Key 不适用。")
    print("没有此 Key 可以直接回车跳过，Kimi CLI 仍可使用。")
    value = getpass.getpass("OpenAI API Key（输入隐藏）：").strip()
    if not value:
        print("已跳过，没有更改修图密钥。")
        return
    if not value.startswith("sk-") or any(c.isspace() for c in value):
        raise ValueError("Key 格式不正确，未保存。")
    SECRETS.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd = os.open(SECRETS, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as handle:
        handle.write(value)
    SECRETS.chmod(0o600)
    print("已保存到本机私有配置目录，未写入项目或 shell 历史；尚未验证 Key。")


def launch() -> None:
    os.chdir(ROOT)
    # This launcher is explicitly installed for this user's project and tool server.
    os.environ["KIMI_CODE_TRUST_WORKSPACE"] = "1"
    print("Kimi 图片实验入口\n")
    print("输入 /mcp 可查看已连接的 watermark-images 工具。")
    print("先让 Kimi 调用 status 检查，再运行现有检测。")
    print("未配置修图 Key 时只能检测和预演，不能生成修复结果。\n")
    os.execvp("kimi", ["kimi"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["image-key", "launch", "status"])
    args = parser.parse_args()
    if args.action == "image-key":
        set_image_key()
    elif args.action == "launch":
        launch()
    else:
        from kimi_image_tools import status
        print(json.dumps(status(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyboardInterrupt) as exc:
        print(str(exc) or "已取消。", file=sys.stderr)
        raise SystemExit(1)
