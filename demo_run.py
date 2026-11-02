"""一键演示：默认跑“去B机房巡检”完整链路，面试可直接执行。

用法：
    python demo_run.py
    python demo_run.py --fail              # 注入导航失败，演示自动查知识库
    python demo_run.py --text "机器人现在在哪"
    python demo_run.py --http              # 改走已启动的 FastAPI 服务
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

import httpx

from config import get_settings, resolve_nav_backend, setup_logging
from graph import RobotNavAgent
from state import AgentResponse


def _configure_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


def _banner(settings, backend: str, text: str, fail: bool) -> None:
    print("=" * 72)
    print("LLM 语义导航 Agent · 本地 Demo")
    print(f"导航后端：{backend}（auto 模式下无 rclpy 自动使用 mock）")
    print(f"LLM：{settings.llm_model} @ {settings.llm_base_url}")
    print(f"外部 RAG：{settings.rag_base_url}{settings.rag_qa_path}")
    print(f"场景：{text}" + ("（注入导航失败）" if fail else ""))
    print("=" * 72)


def _print_response(response: AgentResponse) -> None:
    print("-" * 72)
    print(f"trace_id : {response.trace_id}")
    print(f"决策     : {response.intent}" + (f" / {response.tool}" if response.tool else ""))
    print(f"状态     : {response.status}")
    print(f"回复     : {response.answer}")
    if response.diagnosis:
        print(
            f"自动诊断 : code={response.diagnosis.get('code')}, "
            f"ok={response.diagnosis.get('ok')}"
        )
    if response.data:
        print("数据     : " + json.dumps(response.data, ensure_ascii=False, default=str))
    print("-" * 72)


def _check_llm(settings) -> bool:
    if not settings.llm_api_key:
        print("[错误] 未配置 LLM_API_KEY")
        print("       请复制 .env.example 为 .env，填入你的模型 API Key。")
        return False
    return True


async def _run_direct(args, settings) -> int:
    if not _check_llm(settings):
        return 2

    agent = RobotNavAgent(settings)
    try:
        response = await agent.run(
            args.text, session_id=args.session, mock_fail_nav=args.fail
        )
    finally:
        await agent.close()

    _print_response(response)
    return 0 if response.status in ("success", "failed", "ask_user", "rejected") else 1


def _run_http(args, settings) -> int:
    host = "127.0.0.1" if settings.api_host in ("0.0.0.0", "::") else settings.api_host
    url = f"http://{host}:{settings.api_port}/agent/chat"
    print(f"[HTTP] POST {url}")
    try:
        with httpx.Client(timeout=200.0) as client:
            response = client.post(
                url, json={"session_id": args.session, "text": args.text}
            )
            response.raise_for_status()
            agent_response = AgentResponse.model_validate(response.json())
    except httpx.ConnectError:
        print("[错误] 无法连接 FastAPI，请先运行：python main.py")
        return 2
    except Exception as exc:
        print(f"[错误] HTTP 调用失败：{type(exc).__name__}: {exc}")
        return 1

    _print_response(agent_response)
    return 0 if agent_response.status in ("success", "failed", "ask_user", "rejected") else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="LLM 语义导航 Agent 一键演示")
    parser.add_argument("--text", default="去B机房巡检", help="用户指令")
    parser.add_argument("--session", default="demo", help="会话 ID")
    parser.add_argument(
        "--fail", action="store_true", help="mock 模式下注入导航失败，演示失败闭环"
    )
    parser.add_argument("--http", action="store_true", help="改走 FastAPI 服务")
    args = parser.parse_args()

    _configure_console()
    settings = get_settings()
    setup_logging(settings)
    backend = resolve_nav_backend(settings)

    _banner(settings, backend, args.text, args.fail)
    if args.fail and backend == "ros":
        print("[提示] 当前为 ros 后端，--fail 只对 mock 生效；真实失败请在 Gazebo 中制造障碍。")

    if args.http:
        return _run_http(args, settings)
    return asyncio.run(_run_direct(args, settings))


if __name__ == "__main__":
    sys.exit(main())
