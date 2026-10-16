#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
chatcli — 一个漂亮的命令行 AI 聊天客户端（兼容 OpenAI API）

安装依赖：
    pip install openai rich prompt_toolkit

快速开始：
    export OPENAI_API_KEY="sk-..."
    python chatcli.py -m gpt-4o-mini

    # 或者使用任意兼容 OpenAI 协议的服务
    python chatcli.py -u https://api.siliconflow.cn/v1 -m deepseek-ai/DeepSeek-V3

输入 /help 查看全部命令。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# --------------------------------------------------------------------------- #
# 依赖检查
# --------------------------------------------------------------------------- #
try:
    from openai import OpenAI, APIError, APIConnectionError, AuthenticationError
    from rich.console import Console
    from rich.live import Live
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.table import Table
    from rich.theme import Theme
    from prompt_toolkit import PromptSession
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.styles import Style as PromptStyle
except ImportError as exc:  # pragma: no cover
    sys.stderr.write(
        f"\n缺少依赖：{getattr(exc, 'name', exc)}\n\n"
        "请先安装：\n\n    pip install openai rich prompt_toolkit\n\n"
    )
    raise SystemExit(1)


# --------------------------------------------------------------------------- #
# 常量与配置
# --------------------------------------------------------------------------- #
VERSION = "1.0.0"

CONFIG_DIR = Path(__file__).resolve().parent
CONFIG_FILE = CONFIG_DIR / "config.json"
HISTORY_FILE = CONFIG_DIR / "history.txt"

DEFAULTS: dict = {
    "api_key": "",
    "base_url": "https://api.openai.com/v1",
    "model": "gpt-4o-mini",
    "system": "你是一个乐于助人、回答准确的 AI 助手。",
    "temperature": 0.7,
}

THEME = Theme(
    {
        "cli.info": "dim",
        "cli.accent": "bold magenta",
        "cli.warn": "yellow",
        "cli.err": "bold red",
        "cli.dim": "bright_black",
    }
)

PROMPT_STYLE = PromptStyle.from_dict({"prompt": "bold #00afff"})


# --------------------------------------------------------------------------- #
# 配置读写
# --------------------------------------------------------------------------- #
def load_config() -> dict:
    cfg = dict(DEFAULTS)
    if CONFIG_FILE.exists():
        try:
            data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                cfg.update({k: v for k, v in data.items() if k in DEFAULTS})
        except (OSError, json.JSONDecodeError) as exc:
            sys.stderr.write(f"警告：配置文件读取失败（{exc}），已使用默认配置。\n")
    return cfg


def apply_env(cfg: dict) -> dict:
    env_map = {
        "api_key": ("CHATCLI_API_KEY", "OPENAI_API_KEY"),
        "base_url": ("CHATCLI_BASE_URL", "OPENAI_BASE_URL"),
        "model": ("CHATCLI_MODEL",),
        "system": ("CHATCLI_SYSTEM",),
    }
    for key, names in env_map.items():
        for name in names:
            value = os.environ.get(name)
            if value:
                cfg[key] = value
                break
    return cfg


def apply_args(cfg: dict, args: argparse.Namespace) -> dict:
    if args.model:
        cfg["model"] = args.model
    if args.base_url:
        cfg["base_url"] = args.base_url
    if args.api_key:
        cfg["api_key"] = args.api_key
    if args.system is not None:
        cfg["system"] = args.system
    if args.temperature is not None:
        cfg["temperature"] = args.temperature
    return cfg


def save_config(cfg: dict) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    try:
        CONFIG_FILE.chmod(0o600)  # 里面存了 API Key
    except OSError:
        pass


# --------------------------------------------------------------------------- #
# 聊天应用
# --------------------------------------------------------------------------- #
class ChatApp:
    def __init__(self, cfg: dict) -> None:
        self.console = Console(theme=THEME, highlight=False)
        self.client = OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"])
        self.model: str = cfg["model"]
        self.temperature: float = float(cfg["temperature"])

        self.messages: list[dict] = []
        if cfg.get("system"):
            self.messages.append({"role": "system", "content": cfg["system"]})

        self.started_at = time.time()
        self.rounds = 0
        self.chars = 0

        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        self.session: PromptSession = PromptSession(
            history=FileHistory(str(HISTORY_FILE)),
            style=PROMPT_STYLE,
            enable_open_in_editor=True,  # 长文本按 Ctrl+X Ctrl+E 调出编辑器
            complete_while_typing=False,
        )

    # ---------------------------------------------------------------- 界面 --
    def print_banner(self) -> None:
        system_text = self.get_system() or "（未设置）"
        if len(system_text) > 58:
            system_text = system_text[:58] + "…"

        grid = Table.grid(padding=(0, 2))
        grid.add_column(style="cli.dim", justify="right", no_wrap=True)
        grid.add_column(style="white", overflow="fold")
        grid.add_row("model", f"[cli.accent]{self.model}[/cli.accent]")
        grid.add_row("endpoint", str(self.client.base_url))
        grid.add_row("system", system_text)

        self.console.print()
        self.console.print(
            Panel(
                grid,
                title="[bold]💬 chatcli[/bold]",
                subtitle="[cli.dim]/help 查看命令[/cli.dim]",
                border_style="bright_black",
                padding=(1, 2),
            )
        )
        self.console.print()

    def print_help(self) -> None:
        rows = [
            ("/help", "显示本帮助"),
            ("/model [名称]", "查看或切换模型"),
            ("/system [文本]", "查看或设置系统提示词（用 - 清除）"),
            ("/temp [数值]", "查看或设置温度（0 ~ 2）"),
            ("/clear", "清空对话上下文"),
            ("/undo", "删除最后一轮对话"),
            ("/retry", "重新生成上一条回复"),
            ("/save [路径]", "把当前对话保存为 JSON"),
            ("/load <路径>", "从 JSON 载入对话"),
            ("/stats", "显示会话统计"),
            ("/exit", "退出程序"),
        ]
        grid = Table.grid(padding=(0, 3))
        grid.add_column(style="bold cyan", no_wrap=True)
        grid.add_column(style="white")
        for cmd, desc in rows:
            grid.add_row(cmd, desc)

        self.console.print(
            Panel(
                grid,
                title="[bold]命令[/bold]",
                border_style="bright_black",
                padding=(1, 2),
            )
        )
        self.console.print(
            "[cli.dim]提示：Ctrl+C 清空当前输入（连按两次退出） · "
            "Ctrl+D 退出 · Ctrl+X Ctrl+E 打开外部编辑器[/cli.dim]"
        )

    def print_stats(self) -> None:
        duration = int(time.time() - self.started_at)
        hours, rem = divmod(duration, 3600)
        minutes, seconds = divmod(rem, 60)

        grid = Table.grid(padding=(0, 2))
        grid.add_column(style="cli.dim", justify="right")
        grid.add_column(style="white")
        grid.add_row("模型", f"[cli.accent]{self.model}[/cli.accent]")
        grid.add_row("温度", str(self.temperature))
        grid.add_row("上下文消息数", str(len(self.messages)))
        grid.add_row("对话轮次", str(self.rounds))
        grid.add_row("助手回复字符", f"{self.chars:,}")
        grid.add_row("会话时长", f"{hours:02d}:{minutes:02d}:{seconds:02d}")

        self.console.print(
            Panel(
                grid,
                title="会话统计",
                border_style="bright_black",
                padding=(1, 2),
            )
        )

    # ------------------------------------------------------------ 系统提示 --
    def get_system(self) -> str:
        if self.messages and self.messages[0]["role"] == "system":
            return self.messages[0]["content"]
        return ""

    def set_system(self, text: str) -> None:
        if self.messages and self.messages[0]["role"] == "system":
            if text:
                self.messages[0]["content"] = text
            else:
                self.messages.pop(0)
        elif text:
            self.messages.insert(0, {"role": "system", "content": text})

    # ---------------------------------------------------------------- 请求 --
    def stream_reply(self) -> str | None:
        """向 API 发起流式请求并实时渲染，返回完整回复文本（失败返回 None）。"""
        buffer = ""
        status = None
        live = None
        last_paint = 0.0

        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                temperature=self.temperature,
                stream=True,
            )

            status = self.console.status("[cli.info]思考中…[/cli.info]", spinner="dots")
            status.start()

            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                piece = getattr(delta, "content", None)
                if not piece:
                    continue

                if live is None:
                    status.stop()
                    status = None
                    live = Live(
                        Markdown(""),
                        console=self.console,
                        refresh_per_second=12,
                        vertical_overflow="visible",
                    )
                    live.start()

                buffer += piece

                now = time.monotonic()
                if now - last_paint >= 0.05:
                    live.update(Markdown(buffer))
                    last_paint = now

            if live is not None:
                live.update(Markdown(buffer))
            else:
                self.console.print("[cli.warn]（模型没有返回任何内容）[/cli.warn]")
                return None

        except KeyboardInterrupt:
            self.console.print()
            self.console.print("[cli.info]⏹  已中断生成[/cli.info]")
        except APIConnectionError:
            self.console.print()
            self.console.print(
                "[cli.err]✗ 无法连接到 API，请检查 base_url 与网络。[/cli.err]"
            )
            return None
        except AuthenticationError:
            self.console.print()
            self.console.print("[cli.err]✗ 认证失败，请检查 API Key。[/cli.err]")
            return None
        except APIError as exc:
            self.console.print()
            self.console.print(f"[cli.err]✗ API 错误：{exc}[/cli.err]")
            return None
        finally:
            if status is not None:
                status.stop()
            if live is not None:
                live.stop()

        return buffer or None

    def _request(self) -> bool:
        """对当前 messages 发起一次请求；成功则追加 assistant 回复。"""
        self.console.print()
        reply = self.stream_reply()
        self.console.print()
        if reply:
            self.messages.append({"role": "assistant", "content": reply})
            self.rounds += 1
            self.chars += len(reply)
            return True
        return False

    def send(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})
        if not self._request():
            self.messages.pop()
            self.console.print(
                "[cli.info]（本次提问已撤销，按 ↑ 可找回重发）[/cli.info]"
            )

    def retry(self) -> None:
        while self.messages and self.messages[-1]["role"] == "assistant":
            self.messages.pop()
        if not self.messages or self.messages[-1]["role"] != "user":
            self.console.print("[cli.warn]没有可以重新生成的回复[/cli.warn]")
            return
        self._request()

    # ------------------------------------------------------- 保存 / 载入 --
    def save_chat(self, path_str: str) -> None:
        path = Path(
            path_str or f"chat-{time.strftime('%Y%m%d-%H%M%S')}.json"
        ).expanduser()
        payload = {
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model": self.model,
            "temperature": self.temperature,
            "messages": self.messages,
        }
        try:
            path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except OSError as exc:
            self.console.print(f"[cli.err]✗ 保存失败：{exc}[/cli.err]")
            return
        self.console.print(f"[cli.info]✓ 已保存到[/cli.info] {path.resolve()}")

    def load_chat(self, path_str: str) -> None:
        path = Path(path_str).expanduser()
        if not path.exists():
            self.console.print(f"[cli.err]✗ 文件不存在：{path}[/cli.err]")
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            self.console.print(f"[cli.err]✗ 读取失败：{exc}[/cli.err]")
            return

        messages = data.get("messages")
        if not isinstance(messages, list) or not messages:
            self.console.print("[cli.err]✗ 文件格式不正确[/cli.err]")
            return

        self.messages = messages
        if data.get("model"):
            self.model = data["model"]
        self.console.print(
            f"[cli.info]✓ 已载入 {len(messages)} 条消息（模型：{self.model}）[/cli.info]"
        )

    # ---------------------------------------------------------- 命令分发 --
    def handle_command(self, line: str) -> bool:
        """处理 / 开头的命令，返回 False 表示退出程序。"""
        cmd, _, arg = line[1:].partition(" ")
        cmd = cmd.strip().lower()
        arg = arg.strip()

        if cmd in ("exit", "quit", "q"):
            return False

        if cmd in ("help", "h", "?"):
            self.print_help()

        elif cmd == "model":
            if arg:
                self.model = arg
                self.console.print(f"[cli.info]✓ 已切换模型 →[/cli.info] [cli.accent]{arg}[/cli.accent]")
            else:
                self.console.print(f"[cli.info]当前模型：[/cli.info][cli.accent]{self.model}[/cli.accent]")

        elif cmd == "system":
            if not arg:
                current = self.get_system()
                self.console.print(
                    Panel(
                        current or "[cli.dim]（未设置）[/cli.dim]",
                        title="system prompt",
                        border_style="bright_black",
                        padding=(1, 2),
                    )
                )
            elif arg == "-":
                self.set_system("")
                self.console.print("[cli.info]✓ 已清除系统提示词[/cli.info]")
            else:
                self.set_system(arg)
                self.console.print("[cli.info]✓ 已更新系统提示词[/cli.info]")

        elif cmd in ("temp", "temperature"):
            if not arg:
                self.console.print(f"[cli.info]temperature = {self.temperature}[/cli.info]")
            else:
                try:
                    value = float(arg)
                    if not 0.0 <= value <= 2.0:
                        raise ValueError
                except ValueError:
                    self.console.print("[cli.warn]请输入 0 ~ 2 之间的数字[/cli.warn]")
                else:
                    self.temperature = value
                    self.console.print(f"[cli.info]✓ 温度已设为 {value}[/cli.info]")

        elif cmd in ("clear", "cls"):
            self.messages = []
            if self.get_system():
                pass
            self.console.print("[cli.info]✓ 已清空上下文[/cli.info]")

        elif cmd == "undo":
            removed = 0
            while self.messages and self.messages[-1]["role"] != "system":
                msg = self.messages.pop()
                removed += 1
                if msg["role"] == "user":
                    break
            if removed:
                self.console.print(f"[cli.info]✓ 已删除 {removed} 条消息[/cli.info]")
            else:
                self.console.print("[cli.warn]没有可删除的内容[/cli.warn]")

        elif cmd == "retry":
            self.retry()

        elif cmd == "save":
            self.save_chat(arg)

        elif cmd == "load":
            if not arg:
                self.console.print("[cli.warn]用法：/load <路径>[/cli.warn]")
            else:
                self.load_chat(arg)

        elif cmd == "stats":
            self.print_stats()

        else:
            self.console.print(
                f"[cli.warn]未知命令：/{cmd}（输入 /help 查看全部命令）[/cli.warn]"
            )

        return True

    # ---------------------------------------------------------------- 主循环 --
    def run(self) -> None:
        self.print_banner()
        last_interrupt = 0.0

        while True:
            try:
                line = self.session.prompt([("class:prompt", "❯ ")])
            except KeyboardInterrupt:
                now = time.time()
                if now - last_interrupt < 1.5:
                    self.console.print()
                    break
                last_interrupt = now
                self.console.print(
                    "[cli.dim]再按一次 Ctrl+C 退出，或输入 /exit[/cli.dim]"
                )
                continue
            except EOFError:
                self.console.print()
                break

            line = line.strip()
            if not line:
                continue

            if line.startswith("/"):
                if not self.handle_command(line):
                    break
                continue

            self.send(line)

        self.console.print("[cli.info]再见 👋[/cli.info]")


# --------------------------------------------------------------------------- #
# 命令行入口
# --------------------------------------------------------------------------- #
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="chatcli",
        description="一个漂亮的命令行 AI 聊天客户端（兼容 OpenAI API）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  chatcli                                      # 使用配置文件 / 环境变量\n"
            "  chatcli -m gpt-4o                            # 指定模型\n"
            "  chatcli -u https://api.siliconflow.cn/v1 \\\n"
            "          -m deepseek-ai/DeepSeek-V3           # 使用第三方兼容服务\n"
            "  chatcli -s '你是一个诗人' -t 1.2              # 自定义系统提示与温度\n"
            "  chatcli --save-config                        # 保存当前配置\n"
            f"\n配置文件: {CONFIG_FILE}\n"
            f"输入历史: {HISTORY_FILE}\n"
        ),
    )
    parser.add_argument("-m", "--model", help="模型名称")
    parser.add_argument("-u", "--base-url", help="API 基础地址（默认 https://api.openai.com/v1）")
    parser.add_argument("-k", "--api-key", help="API Key（优先使用环境变量）")
    parser.add_argument("-s", "--system", default=None, help="系统提示词")
    parser.add_argument("-t", "--temperature", type=float, default=None, help="采样温度 0~2")
    parser.add_argument("--save-config", action="store_true", help="把当前配置写入配置文件后退出")
    parser.add_argument("--version", action="version", version=f"chatcli {VERSION}")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    cfg = load_config()
    cfg = apply_env(cfg)
    cfg = apply_args(cfg, args)

    if args.save_config:
        save_config(cfg)
        print(f"✓ 已保存配置到 {CONFIG_FILE}")
        return 0

    console = Console(theme=THEME, highlight=False)
    if not cfg["api_key"]:
        console.print("[cli.err]✗ 未找到 API Key。[/cli.err]\n")
        console.print("可以通过以下任意一种方式提供：\n")
        console.print("  1. 环境变量  [cyan]export OPENAI_API_KEY=\"sk-...\"[/cyan]")
        console.print("  2. 命令行    [cyan]chatcli -k sk-...[/cyan]")
        console.print(f"  3. 配置文件  [cyan]{CONFIG_FILE}[/cyan]\n")
        console.print("配置文件格式示例：")
        console.print_json(
            json.dumps(
                {
                    "api_key": "sk-...",
                    "base_url": "https://api.openai.com/v1",
                    "model": "gpt-4o-mini",
                    "system": "你是一个乐于助人的 AI 助手。",
                    "temperature": 0.7,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 1

    try:
        ChatApp(cfg).run()
    except KeyboardInterrupt:
        console.print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())