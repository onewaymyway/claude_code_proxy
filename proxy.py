#!/usr/bin/env python3
"""
NVIDIA API Proxy for Claude Code
将 Anthropic API 格式的请求转换为 OpenAI 格式，并转发到英伟达 API

两大核心功能：
1. 将 Claude Code 携带的 tools 定义注入到 system prompt，
   要求模型严格按照 JSON 格式输出工具调用，格式稳定可解析。
2. 流式/非流式响应结束后，解析模型输出的 JSON tool_call，
   转换为 Anthropic 标准 tool_use SSE 事件，Claude Code 可直接执行。
"""

import sys
import re
import json
import time
import uuid
import argparse
import logging
from pathlib import Path
from typing import AsyncGenerator, Optional

import json_repair
import httpx
import tool_call_parser
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn

# ── 配置加载 ──────────────────────────────────────────────────────────────────

_DEFAULTS = {
    "nvidia": {
        "api_key":  "",
        "base_url": "https://integrate.api.nvidia.com/v1",
        "model":    "minimaxai/minimax-m2.7",
        "timeout":  120,
    },
    "proxy": {
        "host": "0.0.0.0",
        "port": 8082,
    },
    "defaults": {
        "temperature": 1.0,
        "top_p":       0.95,
        "max_tokens":  8192,
    },
    "log": {
        "level":  "INFO",
        "format": "%(asctime)s [%(levelname)s] %(message)s",
    },
    "debug": {
        "enabled":             False,
        "print_request":       True,
        "print_response":      True,
        "print_stream_chunks": False,
        "print_raw_body":      False,
        "max_content_length":  500,
        "file_log": {
            "enabled":  False,
            "log_dir":  "./proxy_log",
        },
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    result = base.copy()
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def load_config(config_path: str) -> dict:
    path = Path(config_path)
    if not path.exists():
        path.write_text(
            json.dumps(_DEFAULTS, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"✅ 已生成示例配置文件：{path.resolve()}")
        print("   请填写 nvidia.api_key 后重新启动。")
        sys.exit(0)
    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
    except json.JSONDecodeError as e:
        print(f"❌ 配置文件解析失败：{e}")
        sys.exit(1)
    return _deep_merge(_DEFAULTS, raw)


# ── 命令行 & 配置 ─────────────────────────────────────────────────────────────

_parser = argparse.ArgumentParser(description="NVIDIA API Proxy for Claude Code")
_parser.add_argument(
    "--config", "-c",
    default=str(Path(__file__).parent / "config.json"),
    help="配置文件路径（默认：同目录下的 config.json）",
)
_args, _ = _parser.parse_known_args()
CFG = load_config(_args.config)

logging.basicConfig(
    level=getattr(logging, CFG["log"]["level"].upper(), logging.INFO),
    format=CFG["log"]["format"],
)
logger = logging.getLogger(__name__)
logger.info(f"📄 已加载配置文件：{Path(_args.config).resolve()}")

NVIDIA_API_KEY  = CFG["nvidia"]["api_key"]
NVIDIA_BASE_URL = CFG["nvidia"]["base_url"].rstrip("/")
NVIDIA_TIMEOUT  = CFG["nvidia"]["timeout"]
DEFAULT_MODEL   = CFG["nvidia"]["model"]
PROXY_HOST      = CFG["proxy"]["host"]
PROXY_PORT      = int(CFG["proxy"]["port"])
DEF_TEMPERATURE = CFG["defaults"]["temperature"]
DEF_TOP_P       = CFG["defaults"]["top_p"]
DEF_MAX_TOKENS  = CFG["defaults"]["max_tokens"]

# ── 调试状态 ──────────────────────────────────────────────────────────────────

class DebugState:
    def __init__(self, cfg: dict):
        d = cfg["debug"]
        self.enabled             = d["enabled"]
        self.print_request       = d["print_request"]
        self.print_response      = d["print_response"]
        self.print_stream_chunks = d["print_stream_chunks"]
        self.print_raw_body      = d.get("print_raw_body", False)
        self.max_content_length  = d["max_content_length"]
        fl = d.get("file_log", {})
        self.file_log_enabled    = fl.get("enabled", False)
        self.file_log_dir        = fl.get("log_dir", "./proxy_log")
        self._counter            = 0

    def next_id(self) -> int:
        self._counter += 1
        return self._counter

debug = DebugState(CFG)

# ── 在途请求追踪 ──────────────────────────────────────────────────────────────

class ActiveRequests:
    """线程安全的在途请求注册表"""

    def __init__(self):
        self._lock = __import__("threading").Lock()
        self._reqs: dict = {}   # req_id → info dict

    def register(self, req_id: int, body: dict, stream: bool) -> None:
        msgs = body.get("messages", [])
        # 取最后一条 user 消息作为摘要
        last_user = ""
        for m in reversed(msgs):
            if m.get("role") == "user":
                c = m.get("content", "")
                if isinstance(c, list):
                    c = " ".join(b.get("text","") for b in c if isinstance(b,dict) and b.get("type")=="text")
                last_user = str(c)[:120]
                break
        with self._lock:
            self._reqs[req_id] = {
                "req_id":     req_id,
                "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "started_ts": time.time(),
                "stream":     stream,
                "tools":      len(body.get("tools") or []),
                "messages":   len(msgs),
                "last_user_msg": last_user,
                "phase":      "upstream",   # upstream → parsing → done
            }

    def update_phase(self, req_id: int, phase: str) -> None:
        with self._lock:
            if req_id in self._reqs:
                self._reqs[req_id]["phase"] = phase

    def unregister(self, req_id: int) -> None:
        with self._lock:
            self._reqs.pop(req_id, None)

    def snapshot(self) -> list:
        now = time.time()
        with self._lock:
            result = []
            for info in self._reqs.values():
                r = dict(info)
                r["elapsed_seconds"] = round(now - r.pop("started_ts"), 1)
                result.append(r)
        return sorted(result, key=lambda x: x["req_id"])

active_requests = ActiveRequests()

# ── 请求文件日志 ─────────────────────────────────────────────────────────────

class FileLogger:
    """
    每个请求对应一个 JSON 日志文件，分两阶段写入：
      begin()  — 请求到达时立即创建文件，文件名含 [pending]
      finish() — 请求结束时更新内容，并将文件重命名为 [success] 或 [error]

    文件名示例：
      20260510_143512_345_req003_[pending].json   ← 请求进行中
      20260510_143512_345_req003_[success].json   ← 请求成功
      20260510_143512_345_req003_[error].json     ← 请求失败
    代理崩溃时文件名保持 [pending]，便于识别未完成的请求。
    """

    # 状态标签
    _S_PENDING = "[pending]"
    _S_SUCCESS = "[success]"
    _S_ERROR   = "[error]"

    def __init__(self, state: "DebugState"):
        self._state = state
        self._lock  = __import__("threading").Lock()
        self._paths: dict[int, Path] = {}   # req_id → 当前文件路径

    @property
    def enabled(self) -> bool:
        return self._state.file_log_enabled

    def _log_dir(self) -> Path:
        d = Path(self._state.file_log_dir)
        if not d.is_absolute():
            d = Path(__file__).parent / d
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _make_path(self, req_id: int, status_tag: str) -> Path:
        ts = time.strftime("%Y%m%d_%H%M%S")
        ms = int((time.time() % 1) * 1000)
        return self._log_dir() / f"{ts}_{ms:03d}_req{req_id:03d}_{status_tag}.json"

    def _rename(self, old_path: Path, new_tag: str) -> Path:
        """将文件名中的状态标签替换为新标签，返回新路径。"""
        name = old_path.name
        for tag in (self._S_PENDING, self._S_SUCCESS, self._S_ERROR):
            name = name.replace(f"_{tag}", "")
        new_path = old_path.parent / (name.replace(".json", f"_{new_tag}.json"))
        try:
            old_path.rename(new_path)
        except Exception as e:
            logger.warning(f"日志重命名失败 {old_path.name} → {new_path.name}: {e}")
            return old_path   # 降级：保留原路径继续写
        return new_path

    # ── 阶段 1：请求开始 ─────────────────────────────────────────────────────
    def begin(self, *, req_id: int, raw_request: dict, openai_req: dict,
              stream: bool) -> None:
        """请求到达时立即创建 [pending] 文件"""
        if not self.enabled:
            return
        path = self._make_path(req_id, self._S_PENDING)
        with self._lock:
            self._paths[req_id] = path
        record = {
            "meta": {
                "req_id":           req_id,
                "started_at":       time.strftime("%Y-%m-%dT%H:%M:%S"),
                "finished_at":      None,
                "elapsed_seconds":  None,
                "stream":           stream,
                "model":            openai_req.get("model", ""),
                "tool_calls_count": None,
                "status":           "pending",
                "error":            None,
            },
            "raw_request":       raw_request,
            "openai_request":    openai_req,
            "raw_model_output":  None,
            "tool_calls":        None,
            "response":          None,
        }
        try:
            path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
            logger.debug(f"[#{req_id}] 日志已创建: {path.name}")
        except Exception as e:
            logger.warning(f"[#{req_id}] 日志创建失败: {e}")

    # ── 阶段 2：请求结束 ─────────────────────────────────────────────────────
    def finish(self, *, req_id: int, response, elapsed: float,
               tool_calls: list, error: str = None,
               raw_model_output: str = None) -> None:
        """更新日志内容，并将文件重命名为 [success] 或 [error]"""
        if not self.enabled:
            return
        with self._lock:
            path = self._paths.pop(req_id, None)
        if path is None:
            logger.warning(f"[#{req_id}] finish 找不到对应日志文件，跳过")
            return
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"[#{req_id}] 读取日志文件失败: {e}")
            return

        status = "error" if error else "success"
        record["meta"].update({
            "finished_at":      time.strftime("%Y-%m-%dT%H:%M:%S"),
            "elapsed_seconds":  round(elapsed, 3),
            "tool_calls_count": len(tool_calls),
            "status":           status,
            "error":            error,
        })
        record["tool_calls"]       = tool_calls
        record["response"]         = response
        record["raw_model_output"] = raw_model_output

        # 先重命名，再写内容（重命名轻量，内容写入较慢，顺序更安全）
        new_tag  = self._S_ERROR if error else self._S_SUCCESS
        new_path = self._rename(path, new_tag)
        try:
            new_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
            logger.debug(f"[#{req_id}] 日志已完成: {new_path.name}")
        except Exception as e:
            logger.warning(f"[#{req_id}] 日志完成写入失败: {e}")

file_logger = FileLogger(debug)

# ── 终端彩色输出 ──────────────────────────────────────────────────────────────
R   = "\033[0m";  B   = "\033[1m";  CYN = "\033[96m"; GRN = "\033[92m"
YLW = "\033[93m"; MAG = "\033[95m"; RED = "\033[91m";  DIM = "\033[2m"
BLU = "\033[94m"

def _clip(text: str, limit: int) -> str:
    if limit > 0 and len(text) > limit:
        return text[:limit] + f"  {DIM}…(+{len(text)-limit}){R}"
    return text

def _sep(ch="─", w=72, c=DIM):
    return f"{c}{ch*w}{R}"

def dbg_request(req_id: int, body: dict, openai_req: dict):
    if not (debug.enabled and debug.print_request):
        return
    lim = debug.max_content_length
    print()
    print(_sep("═", c=CYN))
    print(f"{B}{CYN}▶ REQUEST #{req_id}{R}  {DIM}{time.strftime('%H:%M:%S')}{R}  "
          f"stream={YLW}{body.get('stream', False)}{R}  model={MAG}{openai_req['model']}{R}")

    # 原始 tools 字段
    if debug.print_raw_body:
        print(f"{DIM}  [raw body]{R}")
        print("  " + json.dumps(body, ensure_ascii=False, indent=2)[:2000])
    elif body.get("tools"):
        tools = body["tools"]
        names = [t.get("name", "?") for t in tools]
        print(f"{DIM}  tools({len(tools)}): {', '.join(names)}{R}")

    print(_sep(c=CYN))
    msgs = openai_req.get("messages", [])
    print(f"{B}Messages ({len(msgs)}){R}")
    for i, msg in enumerate(msgs):
        role = msg.get("role", "?")
        text = msg.get("content", "")
        col  = BLU if role == "system" else (GRN if role == "user" else YLW)
        tag  = f"{col}{B}[{role}]{R}"
        snip = _clip(str(text), lim)
        pad  = "        "
        body_str = f"\n{pad}".join(snip.split("\n"))
        print(f"  {i+1:>2}. {tag}  {body_str}")
    print(f"{DIM}  params: temperature={openai_req.get('temperature')}  "
          f"top_p={openai_req.get('top_p')}  max_tokens={openai_req.get('max_tokens')}{R}")
    print(_sep(c=CYN))

def dbg_response(req_id: int, text: str, elapsed: float, tokens_out: int):
    if not (debug.enabled and debug.print_response):
        return
    print()
    print(_sep(c=GRN))
    print(f"{B}{GRN}◀ RESPONSE #{req_id}{R}  {DIM}{elapsed:.2f}s{R}  output_tokens={YLW}{tokens_out}{R}")
    print(_sep(c=GRN))
    print(f"  {_clip(text, debug.max_content_length)}")
    print(_sep(c=GRN))

def dbg_stream_start(req_id: int):
    if not debug.enabled:
        return
    print()
    print(_sep(c=GRN))
    print(f"{B}{GRN}◀ STREAM #{req_id}{R}  {DIM}{time.strftime('%H:%M:%S')}{R}")
    if debug.print_stream_chunks:
        print(f"{DIM}  ── content ──{R}")

def dbg_stream_chunk(text: str):
    if debug.enabled and debug.print_stream_chunks:
        print(f"{GRN}{text}{R}", end="", flush=True)

def dbg_stream_end(req_id: int, elapsed: float, chars: int, tool_calls: list):
    if not debug.enabled:
        return
    if debug.print_stream_chunks:
        print()
    print(f"{DIM}  ▸ done #{req_id} | {elapsed:.2f}s | ~{chars} chars{R}")
    if tool_calls:
        print(f"  {YLW}🔧 工具调用 x{len(tool_calls)}（已转换为 Anthropic 格式）{R}")
        for tc in tool_calls:
            inp = json.dumps(tc["input"], ensure_ascii=False)
            print(f"     · {MAG}{tc['name']}{R}  {DIM}{_clip(inp, 120)}{R}")
    print(_sep(c=GRN))

def dbg_error(req_id: int, status: int, detail: str):
    if not debug.enabled:
        return
    print()
    print(_sep(c=RED))
    print(f"{B}{RED}✗ ERROR #{req_id}{R}  status={status}")
    print(f"  {_clip(detail, 300)}")
    print(_sep(c=RED))


app = FastAPI(title="NVIDIA API Proxy for Claude Code")

# ── httpx 客户端工厂 ──────────────────────────────────────────────────────────
def _http_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=NVIDIA_TIMEOUT, verify=False, trust_env=False)


# ══════════════════════════════════════════════════════════════════════════════
# 工具调用：格式注入 & 解析
# ══════════════════════════════════════════════════════════════════════════════

def build_tool_system_prompt(tools: list) -> str:
    """
    把 Claude Code 传来的 tools 定义转成 system prompt 追加内容，
    要求模型严格用 JSON 格式输出工具调用，格式唯一且稳定。

    约定的输出格式（模型必须遵守）：
    <tool_use>
    {"name": "ToolName", "input": {"key": "value"}}
    </tool_use>

    选择这个格式的理由：
    - 用唯一的 <tool_use> 标签包裹，正则匹配不会误判
    - 内部是标准 JSON，参数解析零歧义
    - 同一消息可以包含多个 <tool_use> 块（多工具调用）
    - 不依赖模型对 XML 属性语法的理解
    """
    tool_descs = []
    for t in tools:
        name   = t.get("name", "")
        desc   = t.get("description", "")
        schema = t.get("input_schema") or t.get("parameters") or {}
        tool_descs.append(
            f"- **{name}**: {desc}\n"
            f"  Input schema: {json.dumps(schema, ensure_ascii=False)}"
        )

    tools_block = "\n".join(tool_descs)

    return f"""
## Available Tools

You have access to the following tools:

{tools_block}

## Tool Call Format (STRICT)

When you need to call a tool, output it using EXACTLY this format — no other format is accepted:

<tool_use>
{{"name": "ToolName", "input": {{"param1": "value1", "param2": "value2"}}}}
</tool_use>

### Examples

**Example 1 — Read a file:**
<tool_use>
{{"name": "Read", "input": {{"file_path": "/project/src/main.py"}}}}
</tool_use>

**Example 2 — Run a shell command:**
<tool_use>
{{"name": "Bash", "input": {{"command": "ls -la /project", "description": "List project files"}}}}
</tool_use>

**Example 3 — Call a Skill (novel writing assistant):**
<tool_use>
{{"name": "Skill", "input": {{"skill": "skill_name","args": "xxxx"}}}}
</tool_use>

**Example 4 — Write content to a file:**
<tool_use>
{{"name": "Write", "input": {{"file_path": "/project/output.md", "content": "# Title\n\nContent here."}}}}
</tool_use>

**Example 5 — Multiple tool calls in one response (output sequentially, one block per call):**
Let me first check the directory, then read the file.
<tool_use>
{{"name": "Bash", "input": {{"command": "ls /project/docs"}}}}
</tool_use>
<tool_use>
{{"name": "Read", "input": {{"file_path": "/project/docs/README.md"}}}}
</tool_use>

### Rules

- The content inside `<tool_use>...</tool_use>` must be **valid JSON** with exactly two keys: `"name"` and `"input"`.
- `"name"` must exactly match one of the tool names listed above.
- `"input"` must be a JSON object `{{}}` — never a string or array.
- For multiple tool calls, output multiple `<tool_use>` blocks sequentially (not nested).
- Do **NOT** use XML attributes, markdown code fences, or any other wrapper format.
- Explanatory text before or after `<tool_use>` blocks is allowed.
- If no tool is needed, respond normally without any `<tool_use>` block.
""".strip()


# 解析委托给独立模块 tool_call_parser（支持 F1~F6 六种格式）
def split_text_and_tools(full_text: str) -> tuple[str, list[dict]]:
    """把完整输出拆成 (纯文字, 工具调用列表)，委托给 tool_call_parser"""
    return tool_call_parser.parse_to_dicts(full_text)


def build_tool_use_sse_events(tool_calls: list[dict], start_index: int = 1) -> list[str]:
    """
    工具调用列表 → Anthropic SSE 事件序列
    每个工具调用：content_block_start + content_block_delta + content_block_stop
    """
    events = []
    for i, tc in enumerate(tool_calls):
        idx = start_index + i
        events.append(
            f"event: content_block_start\ndata: {json.dumps({'type':'content_block_start','index':idx,'content_block':{'type':'tool_use','id':tc['id'],'name':tc['name'],'input':{}}}, ensure_ascii=False)}\n\n"
        )
        input_json = json.dumps(tc["input"], ensure_ascii=False)
        events.append(
            f"event: content_block_delta\ndata: {json.dumps({'type':'content_block_delta','index':idx,'delta':{'type':'input_json_delta','partial_json':input_json}}, ensure_ascii=False)}\n\n"
        )
        events.append(
            f"event: content_block_stop\ndata: {json.dumps({'type':'content_block_stop','index':idx})}\n\n"
        )
    return events


# ── 格式转换 ──────────────────────────────────────────────────────────────────

def anthropic_to_openai_messages(msgs: list) -> list:
    result = []
    for msg in msgs:
        role    = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, list):
            # 处理 tool_result 消息（Claude Code 回传工具结果）
            parts = []
            for b in content:
                if not isinstance(b, dict):
                    continue
                btype = b.get("type", "")
                if btype == "text":
                    parts.append(b.get("text", ""))
                elif btype == "tool_result":
                    # 把工具结果转成文字，拼回 user 消息
                    tool_content = b.get("content", "")
                    if isinstance(tool_content, list):
                        tool_content = "\n".join(
                            x.get("text", "") for x in tool_content
                            if isinstance(x, dict) and x.get("type") == "text"
                        )
                    tool_id = b.get("tool_use_id", "")
                    parts.append(f"[Tool result for {tool_id}]\n{tool_content}")
                elif btype == "tool_use":
                    # assistant 消息里的 tool_use 块转成文字（历史记录）
                    name  = b.get("name", "")
                    inp   = json.dumps(b.get("input", {}), ensure_ascii=False)
                    parts.append(f"<tool_use>\n{{\"name\": \"{name}\", \"input\": {inp}}}\n</tool_use>")
            content = "\n".join(parts)
        result.append({"role": role, "content": content})
    return result


def build_openai_request(body: dict, model: str) -> dict:
    messages = anthropic_to_openai_messages(body.get("messages", []))

    # 收集 system prompt
    system_parts = []
    system = body.get("system")
    if system:
        sys_text = (
            "\n".join(b.get("text", "") for b in system if isinstance(b, dict))
            if isinstance(system, list) else str(system)
        )
        system_parts.append(sys_text)

    # 如果请求携带 tools，追加工具格式说明到 system prompt
    tools = body.get("tools") or []
    if tools:
        system_parts.append(build_tool_system_prompt(tools))
        logger.debug(f"注入 tools system prompt，共 {len(tools)} 个工具")

    if system_parts:
        messages.insert(0, {"role": "system", "content": "\n\n".join(system_parts)})

    req = {
        "model":       model,
        "messages":    messages,
        "stream":      body.get("stream", False),
        "temperature": body.get("temperature", DEF_TEMPERATURE),
        "top_p":       body.get("top_p",       DEF_TOP_P),
        "max_tokens":  body.get("max_tokens",  DEF_MAX_TOKENS),
    }
    if body.get("stop_sequences"):
        req["stop"] = body["stop_sequences"]
    return req


def openai_to_anthropic(openai_resp: dict, model: str) -> dict:
    """非流式响应转换，含工具调用解析"""
    msg_id   = f"msg_{uuid.uuid4().hex[:24]}"
    choices  = openai_resp.get("choices", [])
    raw_text = choices[0]["message"]["content"] if choices else ""
    usage    = openai_resp.get("usage", {})

    pure_text, tool_calls = split_text_and_tools(raw_text)

    content_blocks = []
    if pure_text:
        content_blocks.append({"type": "text", "text": pure_text})
    for tc in tool_calls:
        content_blocks.append({
            "type": "tool_use", "id": tc["id"],
            "name": tc["name"], "input": tc["input"],
        })

    return {
        "id": msg_id, "type": "message", "role": "assistant",
        "content": content_blocks,
        "model": model,
        "stop_reason":   "tool_use" if tool_calls else "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens":  usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
    }


# ── 调试开关路由 ──────────────────────────────────────────────────────────────

@app.get("/debug")
async def get_debug():
    return {
        "enabled":             debug.enabled,
        "print_request":       debug.print_request,
        "print_response":      debug.print_response,
        "print_stream_chunks": debug.print_stream_chunks,
        "print_raw_body":      debug.print_raw_body,
        "max_content_length":  debug.max_content_length,
        "file_log_enabled":    debug.file_log_enabled,
        "file_log_dir":        debug.file_log_dir,
        "tip": "POST /debug/on | /debug/off | /debug/config",
    }

@app.post("/debug/on")
async def debug_on():
    debug.enabled = True
    logger.info("🔍 调试模式已开启")
    print(f"\n{B}{YLW}🔍 调试模式已开启{R}\n")
    return {"enabled": True}

@app.post("/debug/off")
async def debug_off():
    debug.enabled = False
    logger.info("🔕 调试模式已关闭")
    print(f"\n{B}{DIM}🔕 调试模式已关闭{R}\n")
    return {"enabled": False}

@app.post("/debug/config")
async def debug_config(request: Request):
    """
    可调节字段：enabled / print_request / print_response /
                print_stream_chunks / print_raw_body / max_content_length
    print_raw_body=true 会打印 Claude Code 发来的完整原始请求体（含 tools 定义）
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="无效的 JSON")
    if "enabled"             in body: debug.enabled             = bool(body["enabled"])
    if "print_request"       in body: debug.print_request       = bool(body["print_request"])
    if "print_response"      in body: debug.print_response      = bool(body["print_response"])
    if "print_stream_chunks" in body: debug.print_stream_chunks = bool(body["print_stream_chunks"])
    if "print_raw_body"      in body: debug.print_raw_body      = bool(body["print_raw_body"])
    if "max_content_length"  in body: debug.max_content_length  = int(body["max_content_length"])
    if "file_log_enabled"    in body:
        debug.file_log_enabled = bool(body["file_log_enabled"])
        if debug.file_log_enabled:
            log_dir = Path(debug.file_log_dir)
            if not log_dir.is_absolute():
                log_dir = Path(__file__).parent / log_dir
            log_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n{B}{YLW}📁 请求文件日志已开启 → {log_dir.resolve()}{R}\n")
    if "file_log_dir"        in body: debug.file_log_dir = str(body["file_log_dir"])
    logger.info(f"🔧 调试配置更新: {body}")
    return await get_debug()


# ── 功能路由 ──────────────────────────────────────────────────────────────────

@app.api_route("/", methods=["GET", "HEAD"])
async def root():
    return JSONResponse({"status": "ok"})

@app.get("/requests")
async def get_active_requests():
    """
    返回当前所有在途（尚未完成）的代理请求。
    phase 字段含义：
      upstream  正在等待 / 接收 NVIDIA API 响应
      parsing   流式接收完毕，正在解析工具调用
    """
    reqs = active_requests.snapshot()
    return JSONResponse({
        "count":    len(reqs),
        "requests": reqs,
    })

@app.get("/health")
async def health():
    return {"status": "ok", "model": DEFAULT_MODEL,
            "base_url": NVIDIA_BASE_URL, "debug_enabled": debug.enabled}

@app.get("/config")
async def show_config():
    safe = _deep_merge(CFG, {})
    key  = safe["nvidia"]["api_key"]
    if key:
        safe["nvidia"]["api_key"] = (key[:8] + "****" + key[-4:]) if len(key) > 12 else "****"
    return JSONResponse(safe)


@app.post("/config/reload")
async def reload_config():
    """
    从配置文件重新加载所有配置，无需重启进程。

    可热更新的项：nvidia.api_key / base_url / model / timeout，
    defaults.temperature / top_p / max_tokens，debug.* 所有字段。

    不可热更新的项：proxy.host / proxy.port（uvicorn 已绑定，重启才能生效）。
    """
    global CFG
    global NVIDIA_API_KEY, NVIDIA_BASE_URL, NVIDIA_TIMEOUT, DEFAULT_MODEL
    global DEF_TEMPERATURE, DEF_TOP_P, DEF_MAX_TOKENS

    config_path = _args.config
    try:
        with open(config_path, encoding="utf-8") as f:
            raw = json.load(f)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"配置文件不存在: {config_path}")
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"配置文件 JSON 解析失败: {e}")

    new_cfg = _deep_merge(_DEFAULTS, raw)

    # 更新全局配置快照
    CFG = new_cfg

    # 更新 NVIDIA / defaults 快捷变量
    NVIDIA_API_KEY  = new_cfg["nvidia"]["api_key"]
    NVIDIA_BASE_URL = new_cfg["nvidia"]["base_url"].rstrip("/")
    NVIDIA_TIMEOUT  = new_cfg["nvidia"]["timeout"]
    DEFAULT_MODEL   = new_cfg["nvidia"]["model"]
    DEF_TEMPERATURE = new_cfg["defaults"]["temperature"]
    DEF_TOP_P       = new_cfg["defaults"]["top_p"]
    DEF_MAX_TOKENS  = new_cfg["defaults"]["max_tokens"]

    # 更新 debug 状态（保留运行时计数器，不重置 req_id）
    d = new_cfg["debug"]
    fl = d.get("file_log", {})
    debug.enabled             = d["enabled"]
    debug.print_request       = d["print_request"]
    debug.print_response      = d["print_response"]
    debug.print_stream_chunks = d["print_stream_chunks"]
    debug.print_raw_body      = d.get("print_raw_body", False)
    debug.max_content_length  = d["max_content_length"]
    debug.file_log_enabled    = fl.get("enabled", False)
    debug.file_log_dir        = fl.get("log_dir", "./proxy_log")

    # 更新日志级别
    new_level = getattr(logging, new_cfg["log"]["level"].upper(), logging.INFO)
    logging.getLogger().setLevel(new_level)

    logger.info(f"🔄 配置已从 {Path(config_path).resolve()} 重新加载")
    print(f"\n{B}{GRN}🔄 配置已重新加载{R}  {DIM}{Path(config_path).name}{R}\n")

    # 返回脱敏后的新配置
    safe = _deep_merge(new_cfg, {})
    key  = safe["nvidia"]["api_key"]
    if key:
        safe["nvidia"]["api_key"] = (key[:8] + "****" + key[-4:]) if len(key) > 12 else "****"

    skipped = ["proxy.host", "proxy.port"]
    return JSONResponse({
        "reloaded":      True,
        "config_file":   str(Path(config_path).resolve()),
        "skipped_keys":  skipped,
        "active_config": safe,
    })

@app.get("/v1/models")
async def list_models():
    return JSONResponse({
        "data": [{"id": DEFAULT_MODEL, "object": "model",
                  "created": int(time.time()), "owned_by": "nvidia"}],
        "object": "list",
    })

@app.post("/v1/messages")
async def messages(request: Request):
    if not NVIDIA_API_KEY:
        raise HTTPException(status_code=500, detail="config.json 中 nvidia.api_key 未设置")
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="无效的 JSON 请求体")

    req_id     = debug.next_id()
    model      = DEFAULT_MODEL
    openai_req = build_openai_request(body, model)

    logger.info(f"[#{req_id}] 收到请求 → stream={body.get('stream', False)}, tools={len(body.get('tools') or [])}")
    active_requests.register(req_id, body, stream=bool(openai_req.get("stream")))
    file_logger.begin(req_id=req_id, raw_request=body, openai_req=openai_req,
                      stream=bool(openai_req.get("stream")))
    dbg_request(req_id, body, openai_req)

    headers = {"Authorization": f"Bearer {NVIDIA_API_KEY}", "Content-Type": "application/json"}
    url     = f"{NVIDIA_BASE_URL}/chat/completions"

    # ── 流式 ──────────────────────────────────────────────────────────────────
    if openai_req.get("stream"):
        msg_id = f"msg_{uuid.uuid4().hex[:24]}"

        # ── 生成器外部先建连接并检查 status_code ─────────────────────────────
        # HTTP 200 发出之前可以 raise HTTPException，错误码完整透传给 Claude Code
        try:
            _stream_client = _http_client()
            _stream_resp = await _stream_client.send(
                _stream_client.build_request("POST", url, json=openai_req, headers=headers),
                stream=True,
            )
        except Exception as e:
            await _stream_client.aclose()
            err_msg = f"[代理异常] {type(e).__name__}: {e}"
            logger.error(f"[#{req_id}] 流式连接失败: {e}", exc_info=True)
            dbg_error(req_id, 0, str(e))
            file_logger.finish(req_id=req_id, response=None, elapsed=0,
                               tool_calls=[], error=err_msg, raw_model_output=None)
            active_requests.unregister(req_id)
            raise HTTPException(status_code=502, detail=err_msg)

        if _stream_resp.status_code != 200:
            err = (await _stream_resp.aread()).decode()
            await _stream_client.aclose()
            err_msg = f"上游 API 错误 {_stream_resp.status_code}: {err or '(无响应体)'}"
            logger.error(f"[#{req_id}] {err_msg}")
            dbg_error(req_id, _stream_resp.status_code, err)
            file_logger.finish(req_id=req_id, response=None, elapsed=0,
                               tool_calls=[], error=err_msg, raw_model_output=err)
            active_requests.unregister(req_id)
            raise HTTPException(status_code=_stream_resp.status_code, detail=err_msg)

        async def event_gen() -> AsyncGenerator[str, None]:
            start = {"type": "message_start", "message": {
                "id": msg_id, "type": "message", "role": "assistant",
                "content": [], "model": model,
                "stop_reason": None, "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            }}
            yield f"event: message_start\ndata: {json.dumps(start)}\n\n"
            yield (f"event: content_block_start\ndata: "
                   f"{json.dumps({'type':'content_block_start','index':0,'content_block':{'type':'text','text':''}})}\n\n")
            yield "event: ping\ndata: {\"type\": \"ping\"}\n\n"

            t0          = time.time()
            full_text   = []
            chars       = 0
            chunk_total = 0
            chunk_ok    = 0
            dbg_stream_start(req_id)

            try:
                async with _stream_resp:
                    async for line in _stream_resp.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        ds = line[5:].strip()
                        if ds == "[DONE]":
                            break
                        chunk_total += 1
                        try:
                            chunk = json.loads(ds)
                            chunk_ok += 1
                        except json.JSONDecodeError:
                            logger.debug(f"[#{req_id}] chunk JSON 解析失败，跳过: {ds[:80]}")
                            continue

                        choices = chunk.get("choices", [])
                        if not choices:
                            continue
                        content = choices[0].get("delta", {}).get("content", "")
                        if content:
                            full_text.append(content)
                            chars += len(content)
                            dbg_stream_chunk(content)
                            ev = {"type": "content_block_delta", "index": 0,
                                  "delta": {"type": "text_delta", "text": content}}
                            yield f"event: content_block_delta\ndata: {json.dumps(ev, ensure_ascii=False)}\n\n"

            except Exception as e:
                # HTTP 200 已发出，无法再改状态码，记日志后断流，客户端会感知连接中断
                logger.error(f"[#{req_id}] stream 读取异常: {e}", exc_info=True)
                dbg_error(req_id, 0, str(e))
                err_msg = f"[代理异常] {type(e).__name__}: {e}"
                file_logger.finish(req_id=req_id, response=None,
                                   elapsed=time.time()-t0, tool_calls=[],
                                   error=err_msg, raw_model_output="".join(full_text) or None)
                active_requests.unregister(req_id)
                await _stream_client.aclose()
                return

            await _stream_client.aclose()

            # ── chunk 全部解析失败：同上，HTTP 200 已发出，断流 ───────────────
            if chunk_total > 0 and chunk_ok == 0 and not full_text:
                err_msg = f"[代理错误] 上游返回了 {chunk_total} 个无法解析的 SSE chunk"
                logger.error(f"[#{req_id}] {err_msg}")
                dbg_error(req_id, 0, err_msg)
                file_logger.finish(req_id=req_id, response=None, elapsed=time.time()-t0,
                                   tool_calls=[], error=err_msg, raw_model_output=None)
                active_requests.unregister(req_id)
                return

            # ── 流结束后解析工具调用 ──────────────────────────────────────────
            active_requests.update_phase(req_id, "parsing")
            combined  = "".join(full_text)
            pure_text, tool_calls = split_text_and_tools(combined)

            yield "event: content_block_stop\ndata: {\"type\": \"content_block_stop\", \"index\": 0}\n\n"

            if tool_calls:
                logger.info(f"[#{req_id}] 检测到 {len(tool_calls)} 个工具调用，注入 tool_use 事件")
                for ev in build_tool_use_sse_events(tool_calls, start_index=1):
                    yield ev
                stop_ev = {"type": "message_delta",
                           "delta": {"stop_reason": "tool_use", "stop_sequence": None},
                           "usage": {"output_tokens": chars}}
            else:
                stop_ev = {"type": "message_delta",
                           "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                           "usage": {"output_tokens": chars}}

            yield f"event: message_delta\ndata: {json.dumps(stop_ev)}\n\n"
            yield "event: message_stop\ndata: {\"type\": \"message_stop\"}\n\n"

            elapsed = time.time() - t0
            dbg_stream_end(req_id, elapsed, chars, tool_calls)
            file_logger.finish(
                req_id=req_id,
                response={"text": pure_text, "tool_calls": tool_calls},
                elapsed=elapsed, tool_calls=tool_calls,
                raw_model_output=combined,
            )
            active_requests.unregister(req_id)

        return StreamingResponse(event_gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    # ── 非流式 ────────────────────────────────────────────────────────────────
    t0 = time.time()
    try:
        async with _http_client() as client:
            resp = await client.post(url, json=openai_req, headers=headers)
    except Exception as e:
        err_msg = f"[代理异常] {type(e).__name__}: {e}"
        logger.error(f"[#{req_id}] 非流式请求异常: {e}", exc_info=True)
        dbg_error(req_id, 0, str(e))
        file_logger.finish(req_id=req_id, response=None, elapsed=time.time()-t0,
                           tool_calls=[], error=err_msg, raw_model_output=None)
        active_requests.unregister(req_id)
        raise HTTPException(status_code=502, detail=err_msg)

    if resp.status_code != 200:
        err_msg = f"上游 API 错误 {resp.status_code}: {resp.text or '(无响应体)'}"
        logger.error(f"[#{req_id}] {err_msg}")
        dbg_error(req_id, resp.status_code, resp.text)
        file_logger.finish(
            req_id=req_id, response=None,
            elapsed=time.time()-t0, tool_calls=[], error=err_msg,
            raw_model_output=resp.text or None,
        )
        active_requests.unregister(req_id)
        raise HTTPException(status_code=resp.status_code, detail=resp.text)

    openai_resp = resp.json()
    nr_choices  = openai_resp.get("choices", [])
    raw_model_output_nr = nr_choices[0]["message"]["content"] if nr_choices else ""
    ar = openai_to_anthropic(openai_resp, model)
    summary = " | ".join(
        b.get("text", b.get("name", "")) for b in ar["content"]
    )
    elapsed = time.time() - t0
    tool_calls_nr = [b for b in ar["content"] if b.get("type") == "tool_use"]
    logger.info(f"[#{req_id}] 完成，stop_reason={ar['stop_reason']}，output_tokens={ar['usage']['output_tokens']}")
    active_requests.update_phase(req_id, "parsing")
    dbg_response(req_id, summary, elapsed, ar["usage"]["output_tokens"])
    file_logger.finish(
        req_id=req_id, response=ar,
        elapsed=elapsed, tool_calls=tool_calls_nr,
        raw_model_output=raw_model_output_nr,
    )
    active_requests.unregister(req_id)
    return JSONResponse(ar)


# ── 主入口 ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not NVIDIA_API_KEY:
        logger.warning("⚠️  nvidia.api_key 未填写，请求将会失败")
    logger.info(f"🚀 代理启动: http://{PROXY_HOST}:{PROXY_PORT}")
    logger.info(f"   转发目标: {NVIDIA_BASE_URL}")
    logger.info(f"   默认模型: {DEFAULT_MODEL}")
    logger.info(f"   超时时间: {NVIDIA_TIMEOUT}s")
    logger.info(f"   调试模式: {'开启 🔍' if debug.enabled else '关闭（POST /debug/on 可开启）'}")
    if debug.file_log_enabled:
        from pathlib import Path as _P
        ld = _P(debug.file_log_dir)
        if not ld.is_absolute(): ld = _P(__file__).parent / ld
        logger.info(f"   请求日志: 开启 📁 → {ld.resolve()}")
    else:
        logger.info(f"   请求日志: 关闭（POST /debug/config {{file_log_enabled:true}} 可开启）")
    logger.info(f"   配置查看: http://127.0.0.1:{PROXY_PORT}/config")
    uvicorn.run(app, host=PROXY_HOST, port=PROXY_PORT, log_level=CFG["log"]["level"].lower())
