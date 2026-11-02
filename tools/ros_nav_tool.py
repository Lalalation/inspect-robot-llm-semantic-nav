"""导航工具：rclpy 上层 Action 客户端，向 Nav2 下发 NavigateToPose。

本文件同时承载 nav / status 共用的 RosRuntime（单节点 + 后台 spin），
避免两个工具各自初始化 rclpy 造成重复节点。
"""

from __future__ import annotations

import asyncio
import math
import os
import threading
import time
from typing import TYPE_CHECKING, Any, Dict, Optional

from loguru import logger

from config import POINT_POSES, PointPose
from state import ToolResult

if TYPE_CHECKING:
    from config import Settings
    from tools import ToolContext

GOAL_STATUS_SUCCEEDED = 4
GOAL_STATUS_CANCELED = 5
GOAL_STATUS_ABORTED = 6

_RUNTIME: Optional["RosRuntime"] = None
_RUNTIME_LOCK = threading.Lock()
_RUNTIME_INIT_LOCK = threading.Lock()


def _yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def _quaternion_from_yaw(yaw: float) -> tuple[float, float, float, float]:
    half = yaw / 2.0
    return 0.0, 0.0, math.sin(half), math.cos(half)


class RosRuntime:
    """一个 rclpy 节点，后台线程 spin；异步侧通过 Future 桥接等待结果。"""

    def __init__(self, settings: "Settings") -> None:
        self._settings = settings
        self._started = False
        self._ready = threading.Event()
        self._start_error: Optional[str] = None
        self._thread: Optional[threading.Thread] = None
        self._node = None
        self._executor = None
        self._action_client = None

        self._pose_amcl: Optional[tuple[float, float, float, float]] = None
        self._pose_odom: Optional[tuple[float, float, float, float]] = None
        self._battery: Optional[tuple[float, float]] = None
        self._busy = False
        self._trace_id = "-"
        self._last_feedback_log = 0.0

    # ---------- 生命周期 ----------

    def start(self) -> None:
        with _RUNTIME_LOCK:
            if not self._started:
                self._started = True
                self._thread = threading.Thread(
                    target=self._thread_main, name="ros-nav-runtime", daemon=True
                )
                self._thread.start()
        if not self._ready.wait(timeout=20.0):
            raise RuntimeError("ROS 运行时初始化超时")
        if self._start_error:
            raise RuntimeError(self._start_error)

    def _thread_main(self) -> None:
        try:
            if self._settings.ros_domain_id is not None:
                os.environ["ROS_DOMAIN_ID"] = str(self._settings.ros_domain_id)

            import rclpy
            from rclpy.action import ActionClient
            from rclpy.executors import MultiThreadedExecutor
            from rclpy.node import Node
            from nav2_msgs.action import NavigateToPose
            from nav_msgs.msg import Odometry
            from sensor_msgs.msg import BatteryState
            from geometry_msgs.msg import PoseWithCovarianceStamped

            if not rclpy.ok():
                rclpy.init(args=None)

            node = Node("robot_nav_agent")
            node.create_subscription(Odometry, self._settings.odom_topic, self._on_odom, 10)
            node.create_subscription(
                PoseWithCovarianceStamped, self._settings.amcl_topic, self._on_amcl, 10
            )
            node.create_subscription(
                BatteryState, self._settings.battery_topic, self._on_battery, 10
            )
            self._action_client = ActionClient(
                node, NavigateToPose, self._settings.nav_action_name
            )

            executor = MultiThreadedExecutor(num_threads=4)
            executor.add_node(node)
            self._node = node
            self._executor = executor
            self._ready.set()
            executor.spin()
        except Exception as exc:  # rclpy / 消息包缺失、DDS 异常都落到这里
            self._start_error = f"{type(exc).__name__}: {exc}"
            self._ready.set()
            logger.exception("ROS 运行时启动失败，请检查 ROS2 环境与 Nav2 话题")

    # ---------- 订阅回调 ----------

    def _on_odom(self, msg: Any) -> None:
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self._pose_odom = (
            float(p.x),
            float(p.y),
            _yaw_from_quaternion(float(q.x), float(q.y), float(q.z), float(q.w)),
            time.monotonic(),
        )

    def _on_amcl(self, msg: Any) -> None:
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self._pose_amcl = (
            float(p.x),
            float(p.y),
            _yaw_from_quaternion(float(q.x), float(q.y), float(q.z), float(q.w)),
            time.monotonic(),
        )

    def _on_battery(self, msg: Any) -> None:
        pct = float(getattr(msg, "percentage", float("nan")))
        if math.isnan(pct) or pct < 0.0:
            return
        self._battery = (min(100.0, max(0.0, pct * 100.0)), time.monotonic())

    def _on_feedback(self, msg: Any) -> None:
        try:
            remaining = float(msg.feedback.distance_remaining)
        except Exception:
            return
        now = time.monotonic()
        if now - self._last_feedback_log >= self._settings.nav_feedback_log_interval_s:
            self._last_feedback_log = now
            logger.bind(trace_id=self._trace_id).info(
                f"Nav2 feedback：距离目标还剩 {remaining:.2f} m"
            )

    # ---------- 状态查询 ----------

    def get_status_snapshot(self, wait_s: float = 2.0) -> Optional[Dict[str, Any]]:
        if not self._ready.is_set() or self._start_error:
            return None
        deadline = time.monotonic() + max(0.0, wait_s)
        while time.monotonic() < deadline:
            if self._pose_amcl or self._pose_odom:
                break
            time.sleep(0.05)

        pose = self._pose_amcl or self._pose_odom
        if pose is None:
            return None
        x, y, yaw, ts = pose
        age = time.monotonic() - ts
        state = "navigating" if self._busy else ("error" if age > 5.0 else "idle")
        battery = self._battery[0] if self._battery else None
        return {
            "x": round(x, 3),
            "y": round(y, 3),
            "yaw": round(yaw, 3),
            "battery_pct": None if battery is None else round(battery, 1),
            "state": state,
            "source": "amcl" if self._pose_amcl else "odom",
            "pose_age_s": round(age, 2),
            "backend": "ros",
        }

    # ---------- 导航 ----------

    async def navigate(
        self, point_name: str, pose: PointPose, timeout_s: float, trace_id: str
    ) -> ToolResult:
        if self._start_error:
            return ToolResult(
                ok=False,
                code="ROS_UNAVAILABLE",
                message=f"ROS 运行时不可用：{self._start_error}",
                data={"backend": "ros"},
            )
        if self._busy:
            return ToolResult(
                ok=False,
                code="NAV_BUSY",
                message="机器人正在执行上一个导航任务",
                data={"backend": "ros"},
            )

        from nav2_msgs.action import NavigateToPose
        from geometry_msgs.msg import PoseStamped

        self._busy = True
        self._trace_id = trace_id
        goal_handle = None
        client = self._action_client
        log = logger.bind(trace_id=trace_id)

        try:
            available = await asyncio.to_thread(
                client.wait_for_server, self._settings.nav_server_wait_s
            )
            if not available:
                return ToolResult(
                    ok=False,
                    code="NAV_SERVER_UNAVAILABLE",
                    message="Nav2 NavigateToPose Action 服务不可用",
                    data={"backend": "ros"},
                )

            goal = NavigateToPose.Goal()
            goal.pose = PoseStamped()
            goal.pose.header.frame_id = self._settings.nav_frame_id
            goal.pose.header.stamp = self._node.get_clock().now().to_msg()
            goal.pose.pose.position.x = pose.x
            goal.pose.pose.position.y = pose.y
            qx, qy, qz, qw = _quaternion_from_yaw(pose.yaw)
            goal.pose.pose.orientation.x = qx
            goal.pose.pose.orientation.y = qy
            goal.pose.pose.orientation.z = qz
            goal.pose.pose.orientation.w = qw

            log.info(f"向 Nav2 下发导航：{point_name} -> ({pose.x:.2f}, {pose.y:.2f})")
            goal_handle = await asyncio.wait_for(
                self._await_ros_future(client.send_goal_async(goal, feedback_callback=self._on_feedback)),
                timeout=15.0,
            )
            if not goal_handle.accepted:
                return ToolResult(
                    ok=False,
                    code="NAV_REJECTED",
                    message="Nav2 拒绝了本次导航目标",
                    data={"backend": "ros"},
                )

            result = await asyncio.wait_for(
                self._await_ros_future(goal_handle.get_result_async()), timeout=timeout_s
            )
            status = int(result.status)
            if status == GOAL_STATUS_SUCCEEDED:
                snapshot = self.get_status_snapshot(0.3)
                return ToolResult(
                    ok=True,
                    code="OK",
                    message=f"已到达{point_name}",
                    data={
                        "point_name": point_name,
                        "final_pose": (
                            {k: snapshot[k] for k in ("x", "y", "yaw")} if snapshot else None
                        ),
                        "nav_status": status,
                        "backend": "ros",
                    },
                )
            if status == GOAL_STATUS_CANCELED:
                return ToolResult(
                    ok=False,
                    code="NAV_CANCELED",
                    message="导航目标被取消",
                    data={"nav_status": status, "backend": "ros"},
                )
            if status == GOAL_STATUS_ABORTED:
                return ToolResult(
                    ok=False,
                    code="NAV_ABORTED",
                    message="Nav2 中止导航，可能遇到障碍或规划失败",
                    data={"nav_status": status, "backend": "ros"},
                )
            return ToolResult(
                ok=False,
                code="NAV_FAILED",
                message=f"导航结束但状态异常：{status}",
                data={"nav_status": status, "backend": "ros"},
            )
        except asyncio.TimeoutError:
            log.warning(f"导航超时（{timeout_s:.0f}s），取消目标并准备重试")
            if goal_handle is not None:
                await self._cancel_goal(goal_handle)
            return ToolResult(
                ok=False,
                code="NAV_TIMEOUT",
                message=f"导航超时（{timeout_s:.0f}s），目标未到达",
                data={"backend": "ros", "timeout_s": timeout_s},
            )
        except Exception as exc:
            log.exception("导航执行异常")
            return ToolResult(
                ok=False,
                code="ROS_ERROR",
                message=f"导航执行异常：{type(exc).__name__}: {exc}",
                data={"backend": "ros"},
            )
        finally:
            self._busy = False

    async def _cancel_goal(self, goal_handle: Any) -> None:
        try:
            await asyncio.wait_for(
                self._await_ros_future(goal_handle.cancel_goal_async()), timeout=3.0
            )
        except Exception as exc:
            logger.warning(f"取消导航目标失败：{type(exc).__name__}: {exc}")

    @staticmethod
    async def _await_ros_future(ros_future: Any) -> Any:
        """把 rclpy Future 桥接成 asyncio Future（rclpy 在后台线程 spin）。"""

        loop = asyncio.get_running_loop()
        waiter: asyncio.Future = loop.create_future()

        def _on_done(done_future: Any) -> None:
            try:
                if done_future.cancelled():
                    exc, value = asyncio.CancelledError(), None
                else:
                    exc, value = done_future.exception(), None
                    if exc is None:
                        value = done_future.result()
            except Exception as err:  # pragma: no cover - rclpy 内部异常兜底
                exc, value = err, None

            def _deliver() -> None:
                if waiter.done():
                    return
                if exc is not None:
                    waiter.set_exception(exc)
                else:
                    waiter.set_result(value)

            loop.call_soon_threadsafe(_deliver)

        ros_future.add_done_callback(_on_done)
        return await waiter


def get_ros_runtime(settings: "Settings") -> RosRuntime:
    """进程级单例；首次调用时启动 rclpy 节点并等待就绪。"""

    global _RUNTIME
    with _RUNTIME_INIT_LOCK:
        if _RUNTIME is None:
            candidate = RosRuntime(settings)
            candidate.start()
            _RUNTIME = candidate
        return _RUNTIME


class RosNavTool:
    name = "navigate_to_point"
    description = "导航到授权点位（A栋门口、B机房、茶水间、北门岗亭）"
    args_schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "point_name": {"type": "string", "enum": list(POINT_POSES.keys())}
        },
        "required": ["point_name"],
    }

    async def run(self, args: Dict[str, Any], ctx: "ToolContext") -> ToolResult:
        started = time.perf_counter()
        point_name = str(args.get("point_name", "")).strip()
        pose = POINT_POSES.get(point_name)
        log = logger.bind(trace_id=ctx.trace_id)

        if pose is None:
            allowed = "、".join(POINT_POSES.keys())
            return ToolResult(
                ok=False,
                code="INVALID_POINT",
                message=f"点位不在白名单：{point_name or '空'}。可选：{allowed}",
                data={"allowed_points": list(POINT_POSES.keys())},
                latency_ms=(time.perf_counter() - started) * 1000,
            )

        attempts = 1 + max(0, ctx.settings.nav_retries)
        last: Optional[ToolResult] = None

        for attempt in range(1, attempts + 1):
            log.info(f"导航尝试 {attempt}/{attempts}：{point_name}")
            if ctx.backend == "mock":
                result = await self._mock_navigate(point_name, pose, ctx)
            else:
                try:
                    runtime = await asyncio.to_thread(get_ros_runtime, ctx.settings)
                except Exception as exc:
                    result = ToolResult(
                        ok=False,
                        code="ROS_UNAVAILABLE",
                        message=f"ROS 运行时启动失败：{type(exc).__name__}: {exc}",
                        data={"backend": "ros"},
                    )
                else:
                    result = await runtime.navigate(
                        point_name, pose, ctx.settings.nav_timeout_s, ctx.trace_id
                    )

            result.data.update(
                {"point_name": point_name, "attempt": attempt, "attempts": attempts}
            )
            if result.ok:
                result.latency_ms = (time.perf_counter() - started) * 1000
                log.info(f"导航成功：{point_name}，耗时 {result.latency_ms:.0f} ms")
                return result

            last = result
            if result.code in ("ROS_UNAVAILABLE", "NAV_BUSY", "NAV_SERVER_UNAVAILABLE"):
                break
            if attempt < attempts:
                log.warning(f"第 {attempt} 次导航失败（{result.code}），准备重试")
                await asyncio.sleep(0.5)

        assert last is not None
        last.latency_ms = (time.perf_counter() - started) * 1000
        log.warning(f"导航最终失败：{last.code}，{last.message}")
        return last

    async def _mock_navigate(
        self, point_name: str, pose: PointPose, ctx: "ToolContext"
    ) -> ToolResult:
        """无 ROS 环境时模拟 Nav2 反馈，仅用于演示 Agent 层。"""

        log = logger.bind(trace_id=ctx.trace_id)
        duration = max(0.3, ctx.settings.mock_nav_duration_s)
        steps = 3
        for index in range(1, steps + 1):
            await asyncio.sleep(duration / steps)
            remaining = round(4.2 * (1.0 - index / steps), 2)
            log.info(f"[mock] Nav2 feedback：距离目标还剩 {remaining:.2f} m")

        should_fail = ctx.mock_fail_nav or ctx.settings.mock_nav_result == "fail"
        if should_fail:
            return ToolResult(
                ok=False,
                code="NAV_TIMEOUT",
                message=f"仿真导航超时，未到达{point_name}",
                data={
                    "backend": "mock",
                    "nav_status": GOAL_STATUS_ABORTED,
                    "simulated": True,
                },
            )
        return ToolResult(
            ok=True,
            code="OK",
            message=f"仿真导航完成，已到达{point_name}",
            data={
                "backend": "mock",
                "final_pose": {"x": pose.x, "y": pose.y, "yaw": pose.yaw},
                "nav_status": GOAL_STATUS_SUCCEEDED,
                "simulated": True,
            },
        )
