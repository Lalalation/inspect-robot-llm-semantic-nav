"""LangGraph 状态与全链路数据契约。

LLM 只允许输出 ToolCallPlan / AskUserPlan / RejectPlan 三种 JSON；
工具统一返回 ToolResult；对外统一返回 AgentResponse。
"""

from __future__ import annotations

from typing import Annotated, Any, Dict, List, Literal, Optional, TypedDict, Union

from pydantic import BaseModel, ConfigDict, Field


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)


class ToolCallPlan(_StrictModel):
    action: Literal["tool_call"] = "tool_call"
    tool: str
    args: Dict[str, Any] = Field(default_factory=dict)
    reason: str = ""


class AskUserPlan(_StrictModel):
    action: Literal["ask_user"] = "ask_user"
    question: str
    options: List[str] = Field(default_factory=list)


class RejectPlan(_StrictModel):
    action: Literal["reject"] = "reject"
    reason: str


PlanDecision = Annotated[
    Union[ToolCallPlan, AskUserPlan, RejectPlan],
    Field(discriminator="action"),
]


class ToolResult(BaseModel):
    """所有工具的唯一返回信封，决策层只依赖 ok / code。"""

    ok: bool
    code: str
    message: str
    data: Dict[str, Any] = Field(default_factory=dict)
    latency_ms: float = 0.0


class AgentRequest(BaseModel):
    session_id: str = "default"
    text: str = Field(min_length=1, max_length=2000)
    mock_fail_nav: bool = False


class AgentResponse(BaseModel):
    session_id: str
    trace_id: str
    status: Literal["success", "failed", "ask_user", "rejected", "error"]
    intent: Literal["tool_call", "ask_user", "reject", "error"]
    tool: Optional[str] = None
    answer: str
    data: Dict[str, Any] = Field(default_factory=dict)
    diagnosis: Optional[Dict[str, Any]] = None


class AgentState(TypedDict, total=False):
    trace_id: str
    session_id: str
    user_input: str
    history: List[Dict[str, str]]
    mock_fail_nav: bool

    guard_reject: str
    raw_llm: str
    plan: Optional[Dict[str, Any]]

    tool_name: str
    tool_args: Dict[str, Any]
    tool_result: Optional[Dict[str, Any]]
    nav_failure: bool
    diagnosis: Optional[Dict[str, Any]]
    retry_count: int

    final: Optional[Dict[str, Any]]
