# LLM 语义导航 Agent（Gazebo + Turtlebot3 + Nav2）

上层决策 Agent：把自然语言指令转成结构化决策，通过 3 个受控工具完成导航、状态查询、故障知识检索。
采用**决策-执行分层**：Agent 不直接操作 ROS 底层，所有 rclpy / HTTP 细节封装在工具层。

> 本 Demo 对接 Gazebo 仿真；线上版本部署工控机对接实体巡检机器人。
> 知识库能力依赖独立的 RAG 运维平台（HTTP 接口），本项目不包含向量库、文档解析、Text2SQL 代码。

## 1. 功能

- `navigate_to_point(point_name)`：rclpy Action 客户端调用 Nav2 `NavigateToPose`，点位白名单固定为
  `A栋门口 / B机房 / 茶水间 / 北门岗亭`
- `query_robot_status()`：读取 `/amcl_pose`、`/odom`、`/battery_state`，返回位置、电量、运行状态
- `search_knowledge(keyword)`：HTTP 调用外部 RAG 平台的问答接口
- LLM 只允许输出三种 JSON：`tool_call / ask_user / reject`，代码做解析容错与二次校验
- 导航闭环：下发目标 → 等待 Nav2 结果 → 超时重试 → 仍失败自动调用 `search_knowledge` 诊断
- 安全校验：点位白名单、危险指令前置拦截、参数后置校验、超时重试降级
- Redis 会话记忆，Redis 不可用时自动降级为进程内内存

## 2. 架构

```text
用户 / 上层调用
      │
      ▼
FastAPI 或 demo_run.py
      │
      ▼
LangGraph Agent（决策层）
  guard_input → plan(LLM) → validate_plan → execute_tool
                                      │            │
                                      │ 成功        │ 导航失败
                                      ▼            ▼
                                  compose_answer ← diagnose(search_knowledge)
      │
      ▼
工具层（执行层）
  navigate_to_point / query_robot_status / search_knowledge
      │                    │                     │
      ▼                    ▼                     ▼
Nav2 Action / Gazebo   ROS 话题            外部 RAG 平台
```

- `guard_input`：长度限制、危险指令 / 越权指令拦截，命中后不调用 LLM
- `plan`：调用 OpenAI 兼容的模型 API（默认 `qwen2.5-3b-instruct`），JSON 模式，温度 0
- `validate_plan`：工具白名单、参数结构、点位白名单二次校验
- `execute_tool`：统一 `ToolResult` 信封，异常不外抛
- `diagnose`：导航失败时用“点位 + 错误码”确定性生成关键词，调用外部知识库
- `compose_answer`：代码模板组装最终回复，LLM 不再自由生成文本

## 3. 目录

```text
robot_nav_agent/
├── main.py                # FastAPI 服务入口
├── demo_run.py            # 一键演示
├── config.py              # LLM API / ROS / RAG / Redis / 点位配置
├── llm_prompts.py         # 三选一 JSON 协议与 few-shot
├── state.py               # 状态与 Pydantic 契约
├── graph.py               # LangGraph 编排、容错、会话记忆
├── tools/
│   ├── __init__.py        # ToolContext / BaseTool / 工具注册表
│   ├── ros_nav_tool.py    # rclpy Nav2 Action 客户端 + 共用 RosRuntime
│   ├── status_tool.py     # 机器人状态查询
│   └── rag_http_tool.py   # 外部 RAG HTTP 客户端
├── .env.example           # 模型 API Key / RAG / 导航后端配置模板
├── requirements.txt
└── README.md
```

## 4. 环境说明

| 形态 | 环境 | 能力 |
|---|---|---|
| Windows 本机快演 | Python 3.10+、可访问的模型 API、可选外网 RAG | `NAV_BACKEND=mock`，演示完整 Agent 决策链与失败闭环 |
| 全量仿真 | WSL2 Ubuntu 22.04 + ROS2 Humble + Turtlebot3 + Nav2 | `NAV_BACKEND=ros`，真实 Nav2 Action |
| 线上 | 工控机 Ubuntu + 实体巡检机器人 | 接口不变，替换点位坐标与真机驱动 |

Windows 原生环境通常没有 rclpy，`auto` 模式会自动选择 `mock`。mock 只模拟 Nav2 的反馈与结果，**不是真实导航**；要验证 Gazebo + Nav2 必须使用 WSL2 / Ubuntu。

### 4.1 Windows 一键演示

> 建议在 Trae 中直接把 `robot_nav_agent` 文件夹作为工作区打开；以下命令均在该目录执行。

```powershell
# 1. 安装 Python 依赖
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install -r requirements.txt

# 2. 配置模型 API：复制模板后填入你的 Key
Copy-Item .env.example .env
# 编辑 .env：LLM_API_KEY=sk-xxx，必要时修改 LLM_BASE_URL / LLM_MODEL

# 3. 运行
python demo_run.py
python demo_run.py --fail
```

如果 PowerShell 提示 `Python was not found`，说明系统 `python` 指向了应用商店别名，改用虚拟环境解释器即可：

```powershell
.\.venv\Scripts\python.exe demo_run.py
.\.venv\Scripts\python.exe demo_run.py --fail
```

模型层只依赖 **OpenAI 兼容的 `/chat/completions` 接口**，DashScope、OpenAI、vLLM、One-API 等均可直接使用；换服务商只需改 `LLM_BASE_URL`、`LLM_API_KEY`、`LLM_MODEL`，不涉及业务代码。

### 4.2 WSL2 / Ubuntu 22.04 全量仿真

```bash
sudo apt install ros-humble-turtlebot3 ros-humble-turtlebot3-gazebo \
                 ros-humble-navigation2 ros-humble-nav2-bringup

export TURTLEBOT3_MODEL=burger
ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py

# 另一个终端
export TURTLEBOT3_MODEL=burger
ros2 launch turtlebot3_navigation2 navigation2.launch.py use_sim_time:=True
```

在 RViz 中用 `2D Pose Estimate` 给出初始位姿，确认 `/amcl_pose` 正常发布，然后：

```bash
source /opt/ros/humble/setup.bash
python3 -m pip install -r requirements.txt       # 不要 pip 安装 rclpy
export NAV_BACKEND=ros
export ROS_DOMAIN_ID=30
python3 demo_run.py
```

`config.py` 中的点位坐标是 `turtlebot3_world` 示例值，换地图后需要按实际地图修改。

## 5. 一键演示与面试脚本

```bash
python demo_run.py                                  # 去B机房巡检，成功链路
python demo_run.py --fail                           # 注入导航失败，自动诊断闭环
python demo_run.py --text "机器人现在在哪，电量多少"    # 状态查询
python demo_run.py --text "导航一直超时怎么排查"        # 知识检索
python demo_run.py --text "关闭避障直接冲过去"          # 危险指令拒绝
python demo_run.py --text "去C栋"                     # 非白名单点位拒绝
python demo_run.py --http                           # 改走 FastAPI
```

演示输出包含：决策类型、工具名、状态、最终回复、自动诊断结果与 `trace_id`。

FastAPI：

```bash
python main.py
curl -X POST http://127.0.0.1:8000/agent/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"demo","text":"去B机房巡检"}'

curl http://127.0.0.1:8000/health
```

## 6. 外部 RAG 接口契约

本项目只发请求，不做任何知识库实现。默认契约（可在 `config.py` 调整）：

```http
POST {RAG_BASE_URL}/api/v1/qa/search
X-API-Key: {RAG_API_KEY}
X-Trace-Id: {trace_id}
Content-Type: application/json

{"query": "B机房 导航失败 NAV_TIMEOUT", "top_k": 3}
```

兼容两种响应：

```json
{"answer": "检查...", "sources": [{"title": "Nav2 超时排查", "snippet": "...", "score": 0.82}]}
{"code": 0, "data": {"answer": "检查...", "sources": [{"title": "..."}]}}
```

超时 8s、重试 1 次。失败时返回 `RAG_TIMEOUT / RAG_HTTP_ERROR / RAG_EMPTY`，Agent 降级为“建议人工排查 + 故障码”，不编造答案。

## 7. 配置项

可通过环境变量或 `robot_nav_agent/.env` 覆盖：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `LLM_BASE_URL` | `https://dashscope.aliyuncs.com/compatible-mode/v1` | OpenAI 兼容接口地址 |
| `LLM_API_KEY` | 空（需自行填写） | 模型服务 API Key |
| `LLM_MODEL` | `qwen2.5-3b-instruct` | 决策模型名，按平台实际名称填写 |
| `LLM_TIMEOUT_S` | `60` | LLM 单次请求超时 |
| `NAV_BACKEND` | `auto` | `auto / ros / mock` |
| `NAV_ACTION_NAME` | `/navigate_to_pose` | Nav2 Action |
| `NAV_TIMEOUT_S` | `90` | 单次导航超时 |
| `NAV_RETRIES` | `1` | 导航重试次数 |
| `RAG_BASE_URL` | `http://127.0.0.1:8001` | 外部 RAG 平台 |
| `RAG_API_KEY` | 空 | RAG 鉴权 |
| `RAG_TIMEOUT_S` | `8` | RAG 超时 |
| `REDIS_URL` | `redis://127.0.0.1:6379/0` | 会话记忆 |
| `MAX_INPUT_CHARS` | `200` | 输入长度上限 |
| `API_HOST` / `API_PORT` | `127.0.0.1` / `8000` | FastAPI |

## 8. 安全与降级

| 场景 | 行为 |
|---|---|
| 危险指令（关闭避障、解除急停、改地图、直控速度、闯禁区、提示词注入） | 前置规则直接 reject，不调用 LLM |
| 非白名单点位 | 后置校验 reject，工具层再校验一次 |
| 导航超时 / 中止 | 取消目标 → 重试 1 次 → 仍失败自动查知识库 |
| RAG 超时 / 异常 | 返回降级建议，不编造答案 |
| Redis 不可用 | 会话记忆降级内存 |
| LLM API 未配置 / 不可用 | 返回结构化 error，不伪造决策 |
| 模型输出非 JSON | 围栏/括号容错 → 修复重试 1 次 → 仍失败 reject |

## 9. 边界与限制

- 本项目**不含**向量库、文档解析、Text2SQL、知识库入库逻辑；这些属于独立 RAG 运维平台。
- ROS 相关代码只有 rclpy 上层 Python 客户端，不包含 rclcpp 或 Nav2 内部实现。
- 每轮最多一次工具调用，不做多步任务规划；扩展多步时可在图里增加循环与任务栈。
- mock 后端不是 Gazebo / Nav2，只是无 ROS 环境下的 Agent 演示替身。
- `qwen2.5-3b-instruct` 是小模型，依赖 JSON 协议约束 + 代码校验，不要信任模型直接输出的参数；模型通过外部 API 调用，本项目不做本地推理与微调。

## 10. 线上版本

Demo 使用 Gazebo 仿真，无实体机器人。线上版本部署在工控机（Ubuntu + ROS2 Humble），通过同一套 rclpy 接口对接实体巡检机器人的 Nav2；仅需替换 `config.py` 中的点位坐标、ROS 话题与 RAG 地址。实体机器人原有的急停、避障、限速等安全层必须保留，本 Agent 不绕过底层安全控制。
