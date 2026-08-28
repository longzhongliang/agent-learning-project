# Coding Agent V2 — 工程化 LangChain 智能体

一个从单文件脚本迁移出的**生产级工程化** LangChain Coding Agent。
具备会话持久化、长期记忆、RAG 知识库、自动任务规划、人工审批（human-in-the-loop）等完整能力。

> 原单文件版本 `langchain_quickstart_01.py`（1882 行）不被修改；本项目是它的工程化重构。

## 核心能力

| 能力 | 说明 |
|---|---|
| 会话管理 | SQLite 持久化，支持会话列表 / 新建 / 删除 / 自动标题 |
| 长期记忆 | LangGraph `store` 跨会话记住用户偏好 |
| 断点恢复 | `checkpointer` 快照 + 崩溃后工具调用恢复 |
| 知识库 RAG | 支持 md/txt/pdf/docx/csv/xlsx/html，向量检索 + 文件指纹增量重建 |
| 自动规划 | `StateGraph` 判断任务复杂度，复杂任务先生成执行计划 |
| 人工审批 | 写入/运行文件必须人工确认；拒绝也能正常恢复对话 |
| Gradio UI | 支持浏览器界面中的会话选择、计划确认、工具审批和运行 trace |
| MCP 扩展 | 可选接入过滤后的 GitHub 读取类工具和网页 fetch 工具 |
| 工具集 | 文件读写搜索、运行 Python、天气查询、偏好记忆 |

## 技术亮点（面试可讲）

- **分层架构**：工具 / 配置 / 持久化 / 组装 / 交互 各司其职，模块间单向依赖，避免循环导入。
- **依赖注入**：`build_agent(settings, checkpointer, store)` 显式传依赖，便于测试与替换。
- **配置即代码**：`Settings` dataclass 从 `.env` 读取并校验，API Key 不落代码。
- **RAG 增量重建**：用文件 SHA-256 指纹清单判断文档是否变化，未变化时直接加载已有向量库，避免每次全量重建。
- **可测试**：`tests/` 覆盖路径穿越防护、文本搜索、文件读写、会话持久化，不依赖模型即可回归。
- **鲁棒的人机协作**：所有 interrupt 都会恢复（含拒绝），不会留下"卡死"的暂停状态。

## 目录结构

```
app/
├── agent.py          # Agent 组装：模型 + 工具 + 系统提示词
├── cli.py            # 终端交互与人工审批
├── config.py         # 环境变量配置与校验
├── persistence.py    # SQLite checkpoint / store / 会话元数据
├── planning.py       # 自动计划 StateGraph
├── rag.py            # RAG 知识库（加载 / 向量库 / 检索）
├── mcp_tools.py      # 可选 MCP 工具接入（按白名单过滤）
├── ui.py             # Gradio 浏览器界面
└── tools/
    ├── execution.py      # 运行 Python 文件
    ├── filesystem.py     # 文件读写搜索（受限于工作目录）
    ├── preferences.py    # 长期偏好记忆
    └── weather.py        # 真实天气 API
tests/                # 无模型回归测试
```

## 首次运行（PowerShell）

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
Copy-Item .env.example .env
```

编辑 `.env`，填入**已轮换**的 DeepSeek API Key，然后：

```powershell
python -m unittest discover -s tests -v
python -m app.cli
```

对话内支持：
- `退出` / `exit` / `quit`：结束会话
- `/plan <任务>`：强制先生成计划再执行
- `/trace on` / `/trace off`：控制是否打印本轮工具调用过程

启动浏览器界面：

```powershell
python -m app.cli --ui
```

如需启用 MCP 外部工具，在 `.env` 中设置：

```dotenv
AGENT_ENABLE_MCP_TOOLS=true
```

MCP 工具会按白名单过滤，只暴露 GitHub 查找/读取类工具和网页 fetch 工具，不把仓库写操作交给 Agent。
GitHub MCP 依赖本机可用的 `npx`；fetch MCP 使用当前 Python 环境中的 `mcp-server-fetch`。

## 安全边界

- API Key 只放 `.env`，已被 `.gitignore` 排除。
- 从原始单文件同步时，不迁移任何硬编码密钥；发现硬编码密钥请先轮换。
- 读、写、搜索、列目录只允许访问 `AGENT_WORKSPACE_DIR`（含路径穿越防护测试）。
- 写入文件、运行 Python 都会暂停，必须人工确认；输入 `no` 会把拒绝结果恢复给 Agent。
- `run_python_file` 仅限制路径、超时并移除常见 API Key；它**不是安全沙箱**。接入不可信任务前，应进一步使用 Docker 或受限执行环境。
- 知识库文档存放于 `<workspace>/knowledge/`，请只放入可信资料。
