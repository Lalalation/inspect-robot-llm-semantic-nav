"""知识工具：HTTP 调用外部 RAG 运维平台。

本项目不实现知识库、向量检索、文档入库或 SQL 生成，
只负责发送关键词并归一化对方返回的答案与来源。
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

import httpx
from loguru import logger

from state import ToolResult

if TYPE_CHECKING:
    from tools import ToolContext


class RagHttpTool:
    name = "search_knowledge"
    description = "调用外部 RAG 运维平台，检索故障原因、报警与排查建议"
    args_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {"keyword": {"type": "string", "maxLength": 100}},
        "required": ["keyword"],
    }

    async def run(self, args: Dict[str, Any], ctx: "ToolContext") -> ToolResult:
        started = time.perf_counter()
        log = logger.bind(trace_id=ctx.trace_id)
        keyword = str(args.get("keyword", "")).strip()[:100]

        if not keyword:
            return ToolResult(
                ok=False,
                code="INVALID_KEYWORD",
                message="检索关键词为空",
                latency_ms=(time.perf_counter() - started) * 1000,
            )

        url = ctx.settings.rag_base_url.rstrip("/") + ctx.settings.rag_qa_path
        headers = {"Content-Type": "application/json", "X-Trace-Id": ctx.trace_id}
        if ctx.settings.rag_api_key:
            headers["X-API-Key"] = ctx.settings.rag_api_key
        payload = {"query": keyword, "top_k": ctx.settings.rag_top_k}

        attempts = 1 + max(0, ctx.settings.rag_retries)
        timeout = httpx.Timeout(
            ctx.settings.rag_timeout_s, connect=min(3.0, ctx.settings.rag_timeout_s)
        )
        last: Dict[str, Any] = {}

        log.info(f"调用外部 RAG：{url}，关键词：{keyword}")
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                for attempt in range(1, attempts + 1):
                    try:
                        response = await client.post(url, json=payload, headers=headers)
                        if response.status_code >= 500 and attempt < attempts:
                            last = {
                                "code": "RAG_HTTP_ERROR",
                                "message": f"RAG 服务返回 HTTP {response.status_code}",
                            }
                            await asyncio.sleep(0.3)
                            continue
                        response.raise_for_status()
                        answer, sources = self._normalize(response.json())
                    except httpx.TimeoutException:
                        last = {
                            "code": "RAG_TIMEOUT",
                            "message": f"RAG 服务响应超时（{ctx.settings.rag_timeout_s:.0f}s）",
                        }
                        if attempt < attempts:
                            await asyncio.sleep(0.3)
                            continue
                    except httpx.HTTPStatusError as exc:
                        last = {
                            "code": "RAG_HTTP_ERROR",
                            "message": f"RAG 服务返回 HTTP {exc.response.status_code}",
                        }
                    except (httpx.RequestError, ValueError) as exc:
                        last = {
                            "code": "RAG_HTTP_ERROR",
                            "message": f"RAG 调用失败：{type(exc).__name__}: {exc}",
                        }
                    else:
                        if not answer:
                            return ToolResult(
                                ok=False,
                                code="RAG_EMPTY",
                                message="知识库未检索到相关内容",
                                data={"keyword": keyword, "sources": sources},
                                latency_ms=(time.perf_counter() - started) * 1000,
                            )
                        log.info(f"RAG 返回 {len(sources)} 条参考")
                        return ToolResult(
                            ok=True,
                            code="OK",
                            message=f"知识库检索完成，返回 {len(sources)} 条参考。",
                            data={
                                "keyword": keyword,
                                "answer": answer,
                                "sources": sources,
                            },
                            latency_ms=(time.perf_counter() - started) * 1000,
                        )
        except Exception as exc:  # 网络栈之外的兜底
            last = {
                "code": "RAG_HTTP_ERROR",
                "message": f"RAG 调用异常：{type(exc).__name__}: {exc}",
            }

        log.warning(f"RAG 最终失败：{last.get('code')}，{last.get('message')}")
        return ToolResult(
            ok=False,
            code=str(last.get("code", "RAG_HTTP_ERROR")),
            message=str(last.get("message", "知识库不可用")),
            data={"keyword": keyword},
            latency_ms=(time.perf_counter() - started) * 1000,
        )

    @staticmethod
    def _normalize(payload: Any) -> Tuple[str, List[Dict[str, Any]]]:
        """兼容 {answer,...} 与 {code,data:{answer,...}} 两种外部返回结构。"""

        if not isinstance(payload, dict):
            raise ValueError("RAG 返回不是 JSON 对象")

        code = payload.get("code")
        if code not in (None, 0, 200, "0"):
            raise ValueError(f"RAG 返回业务错误码 {code}")

        data = payload.get("data") if isinstance(payload.get("data"), dict) else payload
        answer = (
            data.get("answer")
            or data.get("content")
            or payload.get("answer")
            or payload.get("content")
            or ""
        )
        raw_sources = data.get("sources") or data.get("references") or []

        sources: List[Dict[str, Any]] = []
        if isinstance(raw_sources, list):
            for item in raw_sources[:5]:
                if isinstance(item, dict):
                    sources.append(
                        {
                            "title": str(item.get("title") or item.get("source") or ""),
                            "snippet": str(item.get("snippet") or item.get("content") or "")[:200],
                            "score": item.get("score"),
                        }
                    )
                elif isinstance(item, str):
                    sources.append({"title": item[:80], "snippet": "", "score": None})

        return str(answer).strip(), sources
