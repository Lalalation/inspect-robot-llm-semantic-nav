"""状态工具：查询仿真机器人位置、电量与运行状态。

ros 模式读取共享 RosRuntime 缓存（/amcl_pose、/odom、/battery_state）；
mock 模式返回固定状态，仅用于无 ROS 环境的 Agent 演示。
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any, Dict

from loguru import logger

from state import ToolResult
from tools.ros_nav_tool import get_ros_runtime

if TYPE_CHECKING:
    from tools import ToolContext


class StatusTool:
    name = "query_robot_status"
    description = "查询仿真机器人当前位置、电量和运行状态"
    args_schema: Dict[str, Any] = {"type": "object", "properties": {}}

    async def run(self, args: Dict[str, Any], ctx: "ToolContext") -> ToolResult:
        started = time.perf_counter()
        log = logger.bind(trace_id=ctx.trace_id)

        if ctx.backend == "mock":
            battery = round(ctx.settings.mock_battery_pct, 1)
            data = {
                "x": 0.0,
                "y": 0.0,
                "yaw": 0.0,
                "battery_pct": battery,
                "state": "idle",
                "source": "mock",
                "backend": "mock",
            }
            log.info(f"[mock] 机器人状态：{data}")
            return ToolResult(
                ok=True,
                code="OK",
                message=f"机器人位于 (0.00, 0.00)，电量 {battery:.1f}%，运行状态 idle。",
                data=data,
                latency_ms=(time.perf_counter() - started) * 1000,
            )

        try:
            runtime = await asyncio.to_thread(get_ros_runtime, ctx.settings)
            snapshot = await asyncio.to_thread(runtime.get_status_snapshot, 2.0)
        except Exception as exc:
            log.error(f"读取 ROS 状态失败：{type(exc).__name__}: {exc}")
            return ToolResult(
                ok=False,
                code="ROBOT_STATUS_UNAVAILABLE",
                message=f"无法获取机器人状态：{type(exc).__name__}: {exc}",
                data={"backend": "ros"},
                latency_ms=(time.perf_counter() - started) * 1000,
            )

        if snapshot is None:
            return ToolResult(
                ok=False,
                code="ROBOT_STATUS_UNAVAILABLE",
                message="尚未收到 /amcl_pose 或 /odom，定位可能未初始化",
                data={"backend": "ros"},
                latency_ms=(time.perf_counter() - started) * 1000,
            )

        battery = snapshot.get("battery_pct")
        battery_text = "未知" if battery is None else f"{battery:.1f}%"
        message = (
            f"机器人位于 ({snapshot['x']:.2f}, {snapshot['y']:.2f})，"
            f"朝向 {snapshot['yaw']:.2f} rad，电量 {battery_text}，"
            f"运行状态 {snapshot['state']}。"
        )
        return ToolResult(
            ok=True,
            code="OK",
            message=message,
            data=snapshot,
            latency_ms=(time.perf_counter() - started) * 1000,
        )
