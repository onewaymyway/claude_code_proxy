# NVIDIA API Proxy for Claude Code

将 [Claude Code](https://claude.ai/code) 的请求转发到 [NVIDIA NIM API](https://build.nvidia.com)，让 Claude Code 使用第三方大模型（如 MiniMax、LLaMA 等）作为推理后端。

```
Claude Code  ──(Anthropic API 格式)──►  本代理  ──(OpenAI 格式)──►  NVIDIA API
             ◄──(Anthropic API 格式)──           ◄──(OpenAI 格式)──
```

---

## 功能特性

- **协议转换**：Anthropic Messages API ↔ OpenAI Chat Completions API，支持流式与非流式
- **工具调用转换**：将多种格式的模型工具调用输出统一转换为 Anthropic 标准 `tool_use` 事件，Claude Code 可直接执行
- **格式约定注入**：自动将工具调用格式规范注入 system prompt，显著提升模型输出格式的稳定性
- **JSON 容错修复**：集成 `json_repair`，自动修复模型输出中畸形的 JSON（缺括号、多余逗号等）
- **请求日志**：每个请求生成独立 JSON 日志文件，记录原始请求、转换后请求、模型原始输出、最终响应
- **调试模式**：彩色终端实时打印请求/响应详情，支持运行时热切换
- **在途请求监控**：接口查看当前正在进行的请求列表及其状态
- **配置热重载**：无需重启即可重新加载 `config.json`
- **错误透传**：上游 HTTP 错误码（504、403 等）原样透传给 Claude Code，不再静默吞掉

---

## 文件结构

```
nvidia_proxy/
├── proxy.py              # 主代理服务
├── tool_call_parser.py   # 工具调用解析模块（独立，可单独测试）
├── config.json           # 配置文件
├── requirements.txt      # Python 依赖
└── start_proxy.sh        # 启动脚本（Linux/macOS）
```

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
pip install json-repair   # 可选，安装后自动启用 JSON 容错修复
```

### 2. 配置

编辑 `config.json`，填写 NVIDIA API Key：

```json
{
  "nvidia": {
    "api_key": "nvapi-xxxxxxxxxx",
    "base_url": "https://integrate.api.nvidia.com/v1",
    "model": "minimaxai/minimax-m2.7",
    "timeout": 120
  },
  "proxy": {
    "host": "0.0.0.0",
    "port": 8082
  }
}
```

API Key 在 [build.nvidia.com](https://build.nvidia.com) 注册后获取。

### 3. 启动代理

```bash
python proxy.py

# 或指定配置文件路径
python proxy.py --config /path/to/config.json
```

### 4. 配置 Claude Code

**方式 A：项目级配置（推荐，不会提交到 git）**

在项目根目录创建 `.claude/settings.local.json`：

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://127.0.0.1:8082",
    "ANTHROPIC_API_KEY": "dummy-key"
  }
}
```

**方式 B：临时环境变量**

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:8082
export ANTHROPIC_API_KEY=dummy-key
claude
```

---

## 配置说明

```jsonc
{
  "nvidia": {
    "api_key":  "nvapi-xxx",                          // NVIDIA API Key（必填）
    "base_url": "https://integrate.api.nvidia.com/v1", // API 地址
    "model":    "minimaxai/minimax-m2.7",              // 使用的模型
    "timeout":  120                                    // 请求超时（秒）
  },
  "proxy": {
    "host": "0.0.0.0",   // 监听地址（仅重启后生效）
    "port": 8082          // 监听端口（仅重启后生效）
  },
  "defaults": {
    "temperature": 1.0,   // 默认温度（可被请求覆盖）
    "top_p":       0.95,
    "max_tokens":  8192
  },
  "log": {
    "level":  "INFO",                                 // DEBUG / INFO / WARNING
    "format": "%(asctime)s [%(levelname)s] %(message)s"
  },
  "debug": {
    "enabled":             false,  // 启动时开启调试打印
    "print_request":       true,   // 打印入站请求（messages、tools）
    "print_response":      true,   // 打印响应文本
    "print_stream_chunks": false,  // 流式时逐字实时输出
    "print_raw_body":      false,  // 打印 Claude Code 原始请求体（含 tools 定义）
    "max_content_length":  500,    // 内容截断长度（0 = 不截断）
    "file_log": {
      "enabled": false,            // 开启请求日志文件
      "log_dir": "./proxy_log"     // 日志目录
    }
  }
}
```

---

## 管理接口

代理运行后，以下接口均可通过 HTTP 调用：

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/health` | 健康检查，返回当前模型和调试状态 |
| `GET` | `/config` | 查看当前生效配置（API Key 脱敏） |
| `POST` | `/config/reload` | 从文件热重载配置，无需重启 |
| `GET` | `/requests` | 查看当前在途（未完成）的请求列表 |
| `GET` | `/debug` | 查看调试开关状态 |
| `POST` | `/debug/on` | 开启实时调试打印 |
| `POST` | `/debug/off` | 关闭实时调试打印 |
| `POST` | `/debug/config` | 细粒度调整调试选项 |

### 常用示例

```bash
# 开启调试，同时开启逐字打印和原始请求体打印
curl -X POST http://127.0.0.1:8082/debug/config \
  -H "Content-Type: application/json" \
  -d '{"enabled": true, "print_stream_chunks": true, "print_raw_body": true}'

# 修改配置后热重载（无需重启）
curl -X POST http://127.0.0.1:8082/config/reload

# 查看当前正在处理的请求
curl http://127.0.0.1:8082/requests
```

`/requests` 返回示例：

```json
{
  "count": 1,
  "requests": [
    {
      "req_id": 5,
      "started_at": "2026-05-10T15:09:09",
      "elapsed_seconds": 3.2,
      "stream": true,
      "tools": 12,
      "messages": 8,
      "last_user_msg": "帮我写第137章...",
      "phase": "upstream"
    }
  ]
}
```

`phase` 字段：`upstream`（等待上游响应）/ `parsing`（解析工具调用）

---

## 工具调用支持

### 格式注入

代理会自动将工具调用规范注入 system prompt，要求模型统一使用 `<tool_use>` 格式输出：

```xml
<tool_use>
{"name": "ToolName", "input": {"key": "value"}}
</tool_use>
```

### 多格式解析（tool_call_parser.py）

模型实际输出格式不稳定时，`tool_call_parser` 支持 6 种格式的自动识别与解析：

| 格式 | 示例 | 来源 |
|------|------|------|
| F1 `<tool_use>` JSON | `<tool_use>{"name":...}</tool_use>` | 本代理约定格式（优先） |
| F2 `<tool_call>` + `<invoke name>` | `<invoke name="Tool"><parameter name="k">v</parameter>` | 通用 XML |
| F3 `<tool_call>` + `<function=Name>` | `<function=Bash><parameter=command>ls</parameter>` | 实际出现格式 |
| F4 `<function_calls>` + `<invoke>` | `<function_calls><invoke name="Tool">` | Anthropic legacy |
| F5 `<minimax:tool_call>` | `<invoke name="Tool" arg="..." skill="..."/>` | MiniMax |
| F6 Markdown 代码块 | `` ```json\n{"name":...}\n``` `` | 部分模型 |

所有格式均支持 JSON 两级降级解析：先 `json.loads`，失败后用 `json_repair` 自动修复。

### 单独测试解析器

```bash
python tool_call_parser.py   # 运行内置 9 个测试用例
```

---

## 请求日志

开启 `debug.file_log.enabled` 后，每个请求在 `./proxy_log/` 目录生成一个 JSON 文件：

**文件名格式**：`20260510_150909_123_req005.json`

**文件结构**（分两阶段写入）：

```json
{
  "meta": {
    "req_id": 5,
    "started_at": "2026-05-10T15:09:09",    // 请求到达时立即写入
    "finished_at": "2026-05-10T15:09:12",   // 完成后更新
    "elapsed_seconds": 2.841,
    "stream": true,
    "model": "minimaxai/minimax-m2.7",
    "tool_calls_count": 1,
    "status": "success",                    // in_progress / success / error
    "error": null
  },
  "raw_request":      {},   // Claude Code 发来的原始 Anthropic 格式请求
  "openai_request":   {},   // 转换后发给 NVIDIA 的 OpenAI 格式请求（含注入的 system prompt）
  "raw_model_output": "",   // 模型返回的原始字符串（含 <tool_use> 标记）
  "tool_calls":       [],   // 解析出的工具调用列表
  "response":         {}    // 最终返回给 Claude Code 的响应
}
```

请求到达时立即创建文件（`status: in_progress`），完成后更新。若代理崩溃，`status` 保持 `in_progress`，便于排查未完成的请求。

---

## 常见问题

**Q：出现 `TLS/SSL connection has been closed (EOF)` 错误**

代理已配置 `verify=False` 和 `trust_env=False`（禁用系统代理），通常不受影响。若仍出现，检查是否有防火墙或安全软件拦截了 HTTPS 连接。

**Q：出现 504 Gateway Timeout**

NVIDIA API 上部分模型为按需加载，首次请求需要等待冷启动（30~120 秒）。解决方法：
1. 调大 `config.json` 中的 `nvidia.timeout`（建议 300）
2. 用 curl 直接测试是否上游超时：`curl -N https://integrate.api.nvidia.com/v1/chat/completions ...`
3. 换一个更稳定的模型，如 `meta/llama-3.1-8b-instruct`

**Q：工具调用没有执行**

1. 开启调试模式查看模型原始输出：`curl -X POST http://127.0.0.1:8082/debug/on`
2. 检查终端输出中 `🔧 工具调用 x N` 是否出现
3. 查看请求日志中的 `raw_model_output` 字段，确认模型是否输出了 `<tool_use>` 块
4. 若模型输出了其他格式（如 XML），`tool_call_parser` 会自动尝试识别，终端会打印匹配到的格式（F1~F6）

---

## 开发说明

### 新增工具调用格式

在 `tool_call_parser.py` 中：

1. 添加正则和解析函数 `_parse_fN(text) -> list[ToolCall]`
2. 将其注册到 `_PARSERS` 列表（按优先级排列）
3. 在 `_ALL_STRIP_PATTERNS` 中添加对应正则，用于清除标记提取纯文字

`proxy.py` 无需改动。

### 切换模型

修改 `config.json` 中的 `nvidia.model`，然后调用热重载接口：

```bash
curl -X POST http://127.0.0.1:8082/config/reload
```

可用模型列表参见 [build.nvidia.com](https://build.nvidia.com)。
