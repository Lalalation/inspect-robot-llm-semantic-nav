"""LangGraph 编排：安全前置 → LLM 决策 → 工具执行 → 失败诊断 → 结构化回复。"""

from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from typing import Any, Dict, List, Optional

import httpx
from langgraph.graph import END, START, StateGraph
from loguru import logger
from pydantic import TypeAdapter, ValidationError

from config import POINT_POSES, Settings, get_settings, resolve_nav_backend, setup_logging
from llm_prompts import REPAIR_PROMPT_TEMPLATE, SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from state import (
    AgentResponse,
    AgentState,
    PlanDecision,
    RejectPlan,
    ToolResult,
)
from tools import TOOL_REGISTRY, ToolContext

_PLAN_ADAPTER = TypeAdapter(PlanDecision)

_DANGEROUS_PATTERNS = (
    re.compile(r"(关闭|禁用|解除|屏蔽|绕过|跳过).{0,8}(急停|避障|限速|安全|防撞|保护)"),
    re.compile(r"(修改|篡改|删除|清空|重建).{0,6}(地图|建图|代价地图)"),
    re.compile(r"(/cmd_vel|cmd_vel|速度指令|直接控制.{0,4}(速度|电机))", re.IGNORECASE),
    re.compile(r"(闯|进入|开进).{0,6}(禁区|危险区|非授权|限制区)"),
    re.compile(r"(忽略|无视).{0,8}(之前|以上|系统|所有).{0,6}(指令|提示|规则)"),
)


class LLMUnavailableError(RuntimeError):
    """LLM API 未配置或调用失败，由入口层转成结构化 error 响应。"""


def extract_json_object(text: str) -> Dict[str, Any]:
    """从模型输出中提取第一个合法 JSON 对象，容忍代码围栏和前后噪声。"""

    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()

    try:
        value = json.loads(cleaned)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    if start < 0:
        raise ValueError("未找到 JSON 对象")

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(cleaned)):
        char = cleaned[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                value = json.loads(cleaned[start : index + 1])
                if not isinstance(value, dict):
                    raise ValueError("JSON 顶层不是对象")
                return value
    raise ValueError("JSON 括号未闭合")


def _parse_plan(raw: str) -> PlanDecision:
    obj = extract_json_object(raw)
    return _PLAN_ADAPTER.validate_python(obj)


def _plain_text(value: Any) -> str:
    """去掉外部文本里的 markdown 痕迹，保证最终回复是纯文本。"""

    text = str(value or "")
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = re.sub(r"[*_`#>]+", "", text)
    text = re.sub(r"^\s*[-+]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


class SessionStore:
    """Redis 会话记忆；连接失败时自动降级为进程内内存，不阻断主链路。"""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._memory: Dict[str, List[Dict[str, str]]] = {}
        self._redis = None
        self._redis_down = False
        try:
            import redis.asyncio as redis_async

            self._redis = redis_async.from_url(
                settings.redis_url,
                decode_responses=True,
                socket_connect_timeout=1.0,
                socket_timeout=1.5,
            )
        except Exception as exc:
            logger.warning(f"Redis 初始化失败，会话记忆降级为内存：{exc}")

    @staticmethod
    def _key(session_id: str) -> str:
        return f"nav_agent:session:{session_id}"

    async def load(self, session_id: str) -> List[Dict[str, str]]:
        if self._redis is not None and not self._redis_down:
            try:
                raw = await self._redis.get(self._key(session_id))
                if raw:
                    value = json.loads(raw)
                    if isinstance(value, list):
                        return value
            except Exception as exc:
                self._redis_down = True
                logger.warning(f"Redis 读取失败，会话降级为内存：{exc}")
        return list(self._memory.get(session_id, []))

    async def save(self, session_id: str, history: List[Dict[str, str]]) -> None:
        self._memory[session_id] = history
        if self._redis is not None and not self._redis_down:
            try:
                await self._redis.setex(
                    self._key(session_id),
                    self._settings.redis_session_ttl_s,
                    json.dumps(history, ensure_ascii=False),
                )
            except Exception as exc:
                self._redis_down = True
                logger.warning(f"Redis 写入失败，会话降级为内存：{exc}")

    async def append(self, session_id: str, user_text: str, answer: str) -> None:
        history = await self.load(session_id)
        history.extend(
            [
                {"role": "user", "content": user_text[:500]},
                {"role": "assistant", "content": answer[:500]},
            ]
        )
        history = history[-self._settings.session_history_turns * 2 :]
        await self.save(session_id, history)

    async def ping(self) -> bool:
        if self._redis is None or self._redis_down:
            return False
        try:
            await asyncio.wait_for(self._redis.ping(), timeout=1.5)
            return True
        except Exception:
            self._redis_down = True
            return False

    async def close(self) -> None:
        if self._redis is not None:
            try:
                closer = getattr(self._redis, "aclose", None) or getattr(self._redis, "close", None)
                if closer:
                    await closer()
            except Exception:
                pass


class RobotNavAgent:
    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        setup_logging(self.settings)
        self.backend = resolve_nav_backend(self.settings)
        self.sessions = SessionStore(self.settings)
        self.llm = httpx.AsyncClient(
            base_url=self.settings.llm_base_url.rstrip("/"),
            timeout=self.settings.llm_timeout_s,
            headers=(
                {"Authorization": f"Bearer {self.settings.llm_api_key}"}
                if self.settings.llm_api_key
                else {}
            ),
        )
        self.graph = self._build_graph()
        logger.info(
            f"Agent 初始化完成：backend={self.backend}, model={self.settings.llm_model}"
        )

    # ---------- 图编排 ----------

    def _build_graph(self):
        builder = StateGraph(AgentState)
        builder.add_node("guard_input", self._node_guard_input)
        builder.add_node("plan", self._node_plan)
        builder.add_node("validate_plan", self._node_validate_plan)
        builder.add_node("execute_tool", self._node_execute_tool)
        builder.add_node("diagnose", self._node_diagnose)
        builder.add_node("compose_answer", self._node_compose_answer)
        builder.add_node("ask_user_end", self._node_ask_user)
        builder.add_node("reject_end", self._node_reject)

        builder.add_edge(START, "guard_input")
        builder.add_conditional_edges(
            "guard_input",
            self._route_after_guard,
            {"plan": "plan", "reject_end": "reject_end"},
        )
        builder.add_edge("plan", "validate_plan")
        builder.add_conditional_edges(
            "validate_plan",
            self._route_after_validate,
            {
                "execute_tool": "execute_tool",
                "ask_user_end": "ask_user_end",
                "reject_end": "reject_end",
            },
        )
        builder.add_conditional_edges(
            "execute_tool",
            self._route_after_execute,
            {"diagnose": "diagnose", "compose_answer": "compose_answer"},
        )
        builder.add_edge("diagnose", "compose_answer")
        builder.add_edge("compose_answer", END)
        builder.add_edge("ask_user_end", END)
        builder.add_edge("reject_end", END)
        return builder.compile()

    @staticmethod
    def _route_after_guard(state: AgentState) -> str:
        return "reject_end" if state.get("guard_reject") else "plan"

    @staticmethod
    def _route_after_validate(state: AgentState) -> str:
        plan = state.get("plan") or {}
        action = plan.get("action")
        if action == "tool_call":
            return "execute_tool"
        if action == "ask_user":
            return "ask_user_end"
        return "reject_end"

    @staticmethod
    def _route_after_execute(state: AgentState) -> str:
        return "diagnose" if state.get("nav_failure") else "compose_answer"

    # ---------- 节点 ----------

    async def _node_guard_input(self, state: AgentState) -> Dict[str, Any]:
        text = (state.get("user_input") or "").strip()
        log = logger.bind(trace_id=state.get("trace_id", "-"))

        if not text:
            return {"guard_reject": "请输入具体指令，例如：去B机房巡检。"}
        if len(text) > self.settings.max_input_chars:
            return {
                "guard_reject": f"输入过长（{len(text)} 字），请控制在 {self.settings.max_input_chars} 字以内。"
            }
        for pattern in _DANGEROUS_PATTERNS:
            if pattern.search(text):
                log.warning(f"安全拦截命中危险规则：{pattern.pattern}")
                return {
                    "guard_reject": "检测到危险或越权指令，已拦截。本 Agent 只执行授权点位内的巡检导航、状态查询和知识检索。"
                }
        log.info("前置安全校验通过")
        return {}

    async def _node_plan(self, state: AgentState) -> Dict[str, Any]:
        text = state.get("user_input", "")
        log = logger.bind(trace_id=state.get("trace_id", "-"))
        history = state.get("history") or []

        messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(history[-self.settings.session_history_turns * 2 :])
        messages.append(
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(text=text)}
        )

        raw = await self._chat(messages)
        log.info(f"LLM 原始输出：{raw[:300]}")
        try:
            plan = _parse_plan(raw)
        except (ValueError, ValidationError) as exc:
            log.warning(f"LLM 输出解析失败，触发一次修复：{exc}")
            repair_messages = messages + [
                {"role": "assistant", "content": raw[:800]},
                {
                    "role": "user",
                    "content": REPAIR_PROMPT_TEMPLATE.format(
                        error=str(exc)[:200], raw=raw[:500]
                    ),
                },
            ]
            raw2 = await self._chat(repair_messages)
            log.info(f"LLM 修复输出：{raw2[:300]}")
            try:
                plan = _parse_plan(raw2)
                raw = raw2
            except (ValueError, ValidationError) as exc2:
                log.error(f"LLM 修复后仍不合法：{exc2}")
                plan = RejectPlan(reason="模型输出格式异常，请换一种说法重试。")
                raw = raw2

        return {"plan": plan.model_dump(), "raw_llm": raw[:1000]}

    async def _node_validate_plan(self, state: AgentState) -> Dict[str, Any]:
        plan = state.get("plan") or {}
        action = plan.get("action")
        log = logger.bind(trace_id=state.get("trace_id", "-"))

        if action == "tool_call":
            tool_name = str(plan.get("tool", "")).strip()
            args = plan.get("args") or {}
            if tool_name not in TOOL_REGISTRY:
                reason = f"不支持的工具：{tool_name or '空'}，已拒绝执行。"
                return {"plan": RejectPlan(reason=reason).model_dump()}
            if not isinstance(args, dict):
                return {"plan": RejectPlan(reason="工具参数格式错误，已拒绝执行。").model_dump()}

            if tool_name == "navigate_to_point":
                point_name = str(args.get("point_name", "")).strip()
                if point_name not in POINT_POSES:
                    allowed = "、".join(POINT_POSES.keys())
                    return {
                        "plan": RejectPlan(
                            reason=f"点位“{point_name or '空'}”不在授权白名单内，可选：{allowed}。"
                        ).model_dump()
                    }
                args = {"point_name": point_name}
            elif tool_name == "search_knowledge":
                keyword = str(args.get("keyword", "")).strip()
                if not keyword:
                    return {"plan": RejectPlan(reason="知识检索关键词为空，请补充具体故障现象。").model_dump()}
                args = {"keyword": keyword[:100]}
            elif tool_name == "query_robot_status":
                args = {}

            log.info(f"决策校验通过：tool={tool_name}, args={args}")
            normalized: Dict[str, Any] = dict(plan)
            normalized["tool"] = tool_name
            normalized["args"] = args
            return {"plan": normalized, "tool_name": tool_name, "tool_args": args}

        if action == "ask_user":
            question = str(plan.get("question", "")).strip()
            if not question:
                return {"plan": RejectPlan(reason="需要补充信息，但问题为空，请重新描述。").model_dump()}
            options = [str(item).strip() for item in plan.get("options", []) if str(item).strip()][:6]
            compact_points = {re.sub(r"[\s:：,，。.、]", "", name): name for name in POINT_POSES}
            if any(re.sub(r"[\s:：,，。.、]", "", item) in compact_points for item in options):
                # 小模型容易写错点位名，选项统一回填权威白名单
                options = list(POINT_POSES)
            return {
                "plan": {
                    "action": "ask_user",
                    "question": question,
                    "options": options,
                }
            }

        if action == "reject":
            reason = str(plan.get("reason", "")).strip() or "该请求不在授权范围内。"
            return {"plan": RejectPlan(reason=reason).model_dump()}

        return {"plan": RejectPlan(reason="模型未给出有效决策，请重新描述。").model_dump()}

    async def _node_execute_tool(self, state: AgentState) -> Dict[str, Any]:
        tool_name = state.get("tool_name", "")
        args = state.get("tool_args") or {}
        tool = TOOL_REGISTRY.get(tool_name)
        log = logger.bind(trace_id=state.get("trace_id", "-"))

        if tool is None:
            result = ToolResult(
                ok=False,
                code="TOOL_NOT_FOUND",
                message=f"工具未注册：{tool_name}",
            )
        else:
            ctx = ToolContext(
                settings=self.settings,
                backend=self.backend,
                trace_id=state.get("trace_id", "-"),
                mock_fail_nav=bool(state.get("mock_fail_nav")),
            )
            started = time.perf_counter()
            log.info(f"执行工具 {tool_name}，参数：{args}")
            try:
                result = await tool.run(args, ctx)
            except Exception as exc:
                log.exception(f"工具 {tool_name} 异常")
                result = ToolResult(
                    ok=False,
                    code="TOOL_EXCEPTION",
                    message=f"工具异常：{type(exc).__name__}: {exc}",
                    latency_ms=(time.perf_counter() - started) * 1000,
                )
            log.info(
                f"工具返回：code={result.code}, ok={result.ok}, "
                f"latency={result.latency_ms:.0f}ms"
            )

        return {
            "tool_result": result.model_dump(),
            "nav_failure": tool_name == "navigate_to_point" and not result.ok,
        }

    async def _node_diagnose(self, state: AgentState) -> Dict[str, Any]:
        log = logger.bind(trace_id=state.get("trace_id", "-"))
        point_name = str((state.get("tool_args") or {}).get("point_name", "目标点"))
        code = str((state.get("tool_result") or {}).get("code", "UNKNOWN"))
        keyword = f"{point_name} 导航失败 {code}"[:100]
        log.warning(f"导航失败，自动调用 search_knowledge 诊断：{keyword}")

        ctx = ToolContext(
            settings=self.settings,
            backend=self.backend,
            trace_id=state.get("trace_id", "-"),
        )
        try:
            result = await TOOL_REGISTRY["search_knowledge"].run({"keyword": keyword}, ctx)
        except Exception as exc:
            log.exception("自动诊断调用知识库异常")
            result = ToolResult(
                ok=False,
                code="TOOL_EXCEPTION",
                message=f"自动诊断异常：{type(exc).__name__}: {exc}",
            )
        log.info(f"自动诊断结果：code={result.code}, ok={result.ok}")
        return {"diagnosis": result.model_dump()}

    async def _node_compose_answer(self, state: AgentState) -> Dict[str, Any]:
        tool_result = state.get("tool_result") or {}
        tool_name = state.get("tool_name", "")
        diagnosis = state.get("diagnosis")
        answer = self._build_answer(tool_name, tool_result, diagnosis)

        response = AgentResponse(
            session_id=state.get("session_id", "default"),
            trace_id=state.get("trace_id", "-"),
            status="success" if tool_result.get("ok") else "failed",
            intent="tool_call",
            tool=tool_name,
            answer=answer,
            data=tool_result.get("data") or {},
            diagnosis=diagnosis,
        )
        logger.bind(trace_id=state.get("trace_id", "-")).info(f"最终回复：{answer[:120]}")
        return {"final": response.model_dump()}

    async def _node_ask_user(self, state: AgentState) -> Dict[str, Any]:
        plan = state.get("plan") or {}
        response = AgentResponse(
            session_id=state.get("session_id", "default"),
            trace_id=state.get("trace_id", "-"),
            status="ask_user",
            intent="ask_user",
            answer=str(plan.get("question", "请补充目标点位。")),
            data={"options": plan.get("options", [])},
        )
        return {"final": response.model_dump()}

    async def _node_reject(self, state: AgentState) -> Dict[str, Any]:
        plan = state.get("plan") or {}
        reason = str(state.get("guard_reject") or plan.get("reason") or "该请求不在授权范围内。")
        response = AgentResponse(
            session_id=state.get("session_id", "default"),
            trace_id=state.get("trace_id", "-"),
            status="rejected",
            intent="reject",
            answer=reason,
        )
        logger.bind(trace_id=state.get("trace_id", "-")).warning(f"拒绝请求：{reason}")
        return {"final": response.model_dump()}

    # ---------- LLM ----------

    async def _chat(self, messages: List[Dict[str, str]]) -> str:
        if not self.settings.llm_api_key:
            raise LLMUnavailableError(
                "未配置 LLM_API_KEY，请在 .env 中填写模型 API Key。"
            )

        payload: Dict[str, Any] = {
            "model": self.settings.llm_model,
            "messages": messages,
            "temperature": self.settings.llm_temperature,
            "max_tokens": self.settings.llm_max_tokens,
            "stream": False,
            "response_format": {"type": "json_object"},
        }
        max_attempts = 2
        content = ""
        for attempt in range(1, max_attempts + 1):
            try:
                response = await self.llm.post("/chat/completions", json=payload)
                if response.status_code == 400:
                    # 部分 OpenAI 兼容服务不支持 response_format，去掉后重试一次
                    payload.pop("response_format", None)
                    response = await self.llm.post("/chat/completions", json=payload)
                response.raise_for_status()
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                break
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                if status in (429, 500, 502, 503, 504) and attempt < max_attempts:
                    logger.warning(f"LLM 返回 HTTP {status}，第 {attempt} 次尝试失败，准备重试")
                    await asyncio.sleep(0.8 * attempt)
                    continue
                detail = exc.response.text[:300]
                raise LLMUnavailableError(
                    f"LLM API 返回 HTTP {exc.response.status_code}: {detail}"
                ) from exc
            except httpx.TransportError as exc:
                if attempt < max_attempts:
                    logger.warning(
                        f"LLM 连接失败：{type(exc).__name__}，第 {attempt} 次尝试，准备重试"
                    )
                    await asyncio.sleep(0.6 * attempt)
                    continue
                raise LLMUnavailableError(
                    f"LLM API 连接失败：{type(exc).__name__}: {exc}"
                ) from exc
            except Exception as exc:
                raise LLMUnavailableError(
                    f"LLM API 调用失败：{type(exc).__name__}: {exc}"
                ) from exc
        return str(content or "").strip()

    # ---------- 回复组装（代码模板，不让 LLM 自由生成） ----------

    def _build_answer(
        self,
        tool_name: str,
        tool_result: Dict[str, Any],
        diagnosis: Optional[Dict[str, Any]],
    ) -> str:
        ok = bool(tool_result.get("ok"))
        message = _plain_text(tool_result.get("message", ""))
        data = tool_result.get("data") or {}
        code = tool_result.get("code", "UNKNOWN")

        if tool_name == "navigate_to_point":
            point_name = data.get("point_name", "目标点")
            if ok:
                pose = data.get("final_pose") or {}
                if pose:
                    return (
                        f"已到达{point_name}，当前位置 ({pose.get('x', 0.0):.2f}, "
                        f"{pose.get('y', 0.0):.2f})，朝向 {pose.get('yaw', 0.0):.2f} rad。"
                    )
                return f"已到达{point_name}。"
            parts = [f"导航到{point_name}失败：{message}（故障码 {code}）。"]
            if diagnosis:
                diag_data = diagnosis.get("data") or {}
                if diagnosis.get("ok"):
                    rag_answer = _plain_text(diag_data.get("answer", ""))[:600]
                    parts.append(f"已自动检索运维知识库：{rag_answer}")
                    titles = [
                        _plain_text(item.get("title", ""))
                        for item in diag_data.get("sources", [])
                        if item.get("title")
                    ]
                    if titles:
                        parts.append("参考来源：" + "、".join(titles[:3]) + "。")
                else:
                    parts.append(
                        f"知识库暂不可用（{diagnosis.get('code', 'UNKNOWN')}），"
                        f"建议人工排查，故障码 {code}。"
                    )
            return " ".join(parts)

        if tool_name == "query_robot_status":
            if ok:
                return message or "已获取机器人状态。"
            return f"无法获取机器人状态：{message}"

        if tool_name == "search_knowledge":
            if ok:
                answer = _plain_text(data.get("answer", ""))[:600]
                sources = [
                    _plain_text(item.get("title", ""))
                    for item in data.get("sources", [])
                    if item.get("title")
                ]
                text = f"知识库检索结果：{answer}"
                if sources:
                    text += " 参考来源：" + "、".join(sources[:3]) + "。"
                return text
            return f"知识库检索失败（{code}）：{message}"

        return message or "操作已完成。"

    # ---------- 对外入口 ----------

    async def run(
        self,
        user_input: str,
        session_id: str = "default",
        mock_fail_nav: bool = False,
    ) -> AgentResponse:
        trace_id = uuid.uuid4().hex[:12]
        log = logger.bind(trace_id=trace_id)
        log.info(f"收到指令：{user_input}")

        history = await self.sessions.load(session_id)
        init_state: AgentState = {
            "trace_id": trace_id,
            "session_id": session_id,
            "user_input": user_input,
            "history": history,
            "mock_fail_nav": mock_fail_nav,
        }

        try:
            final_state = await self.graph.ainvoke(
                init_state, config={"recursion_limit": 8}
            )
        except LLMUnavailableError as exc:
            log.error(str(exc))
            return AgentResponse(
                session_id=session_id,
                trace_id=trace_id,
                status="error",
                intent="error",
                answer="LLM 服务不可用，请检查 LLM_API_KEY 与 LLM_BASE_URL 配置。",
                data={"detail": str(exc)},
            )
        except Exception as exc:
            log.exception("Agent 执行异常")
            return AgentResponse(
                session_id=session_id,
                trace_id=trace_id,
                status="error",
                intent="error",
                answer="Agent 内部异常，请查看日志。",
                data={"detail": f"{type(exc).__name__}: {exc}"},
            )

        final = final_state.get("final")
        if not final:
            log.error("图执行结束但未生成 final 响应")
            return AgentResponse(
                session_id=session_id,
                trace_id=trace_id,
                status="error",
                intent="error",
                answer="Agent 未生成有效回复，请重试。",
            )

        response = AgentResponse.model_validate(final)
        await self.sessions.append(session_id, user_input, response.answer)
        log.info(
            f"本轮完成：status={response.status}, tool={response.tool}, "
            f"answer={response.answer[:80]}"
        )
        return response

    async def health(self) -> Dict[str, Any]:
        configured = bool(self.settings.llm_api_key)
        llm_ok = configured
        llm_error = "" if configured else "未配置 LLM_API_KEY"
        if configured:
            try:
                response = await self.llm.get("/models")
                if response.status_code in (401, 403) or response.status_code >= 500:
                    llm_ok = False
                    llm_error = f"HTTP {response.status_code}"
            except Exception as exc:
                llm_ok = False
                llm_error = f"{type(exc).__name__}: {exc}"

        redis_ok = await self.sessions.ping()
        return {
            "status": "ok" if llm_ok else "degraded",
            "llm": {
                "ok": llm_ok,
                "configured": configured,
                "base_url": self.settings.llm_base_url,
                "model": self.settings.llm_model,
                "error": llm_error,
            },
            "redis": {"ok": redis_ok},
            "nav_backend": self.backend,
            "rag": {
                "base_url": self.settings.rag_base_url,
                "path": self.settings.rag_qa_path,
            },
        }

    async def close(self) -> None:
        await self.sessions.close()
        await self.llm.aclose()
