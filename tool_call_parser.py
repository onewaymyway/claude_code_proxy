"""
tool_call_parser.py
═══════════════════
解析模型输出文本中的工具调用，支持多种格式，统一返回结构化列表。

返回格式（始终一致，无论输入是哪种格式）：
    [{"id": "toolu_xxx...", "name": "ToolName", "input": {"key": "value"}}]

支持的格式
──────────
FORMAT 1 — <tool_use> JSON（本代理约定格式，优先匹配）
    <tool_use>
    {"name": "ToolName", "input": {"key": "value"}}
    </tool_use>

FORMAT 2 — <tool_call> + <invoke> + <parameter> XML（常见格式）
    <tool_call>
    <invoke name="ToolName">
        <parameter name="key">value</parameter>
    </invoke>
    </tool_call>

FORMAT 3 — <tool_call> + <function=Name> + <parameter> XML（问题中实际出现的格式）
    <tool_call>
    <function=Bash>
        <parameter=command>ls -la</parameter=command>
        <parameter=description>检查目录</parameter=description>
    </function>
    </tool_call>

FORMAT 4 — <function_calls> + <invoke> + <parameter> XML（Anthropic legacy 格式）
    <function_calls>
    <invoke name="ToolName">
        <parameter name="key">value</parameter>
    </invoke>
    </function_calls>

FORMAT 5 — minimax <minimax:tool_call> 格式
    <minimax:tool_call>
    <invoke name="ToolName" arg="..." skill="..."/>
    </minimax:tool_call>

FORMAT 6 — Markdown 代码块内的 JSON（部分模型会输出）
    ```json
    {"name": "ToolName", "input": {"key": "value"}}
    ```
    或
    ```tool_use
    {"name": "ToolName", "input": {"key": "value"}}
    ```

对于每种格式，JSON 值均经过两级降级解析：
  Level 1: json.loads（标准，严格）
  Level 2: json_repair（修复缺括号、多余逗号、markdown 链接等畸形 JSON）
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from typing import Optional

try:
    import json_repair
    _HAS_JSON_REPAIR = True
except ImportError:
    _HAS_JSON_REPAIR = False

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# 数据结构
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ToolCall:
    """单个工具调用"""
    name:   str
    input:  dict
    id:     str = field(default_factory=lambda: f"toolu_{uuid.uuid4().hex[:24]}")
    fmt:    str = ""    # 匹配到的格式名称，调试用

    def to_dict(self) -> dict:
        return {"id": self.id, "name": self.name, "input": self.input}


@dataclass
class ParseResult:
    """解析结果"""
    tool_calls: list[ToolCall]
    pure_text:  str     # 去掉所有 tool_call 标记后的纯文字

    @property
    def has_tools(self) -> bool:
        return bool(self.tool_calls)

    def to_dicts(self) -> list[dict]:
        return [tc.to_dict() for tc in self.tool_calls]


# ══════════════════════════════════════════════════════════════════════════════
# JSON 两级降级解析
# ══════════════════════════════════════════════════════════════════════════════

def _try_parse_json(raw: str, context: str = "") -> Optional[dict]:
    """
    两级降级解析 JSON 字符串 → dict。
    Level 1: json.loads
    Level 2: json_repair（若已安装）
    失败返回 None。
    """
    raw = raw.strip()
    if not raw:
        return None

    # Level 1
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError as e1:
        pass
    else:
        return None  # 解析成功但不是 dict

    # Level 2
    if _HAS_JSON_REPAIR:
        try:
            obj = json_repair.repair_json(raw, return_objects=True)
            if isinstance(obj, dict):
                logger.info(f"[tool_parser] JSON 已修复 {context}: {raw[:80]!r}")
                return obj
        except Exception as e2:
            logger.debug(f"[tool_parser] json_repair 失败 {context}: {e2}")

    logger.warning(f"[tool_parser] JSON 解析失败 {context}: {raw[:120]!r}")
    return None


# ══════════════════════════════════════════════════════════════════════════════
# XML 参数提取公共工具
# ══════════════════════════════════════════════════════════════════════════════

# <parameter name="key">value</parameter>
_PARAM_NAME_RE = re.compile(
    r"<parameter\s+name=['\"]([^'\"]+)['\"]>(.*?)</parameter>",
    re.DOTALL,
)

# <parameter=key>value</parameter=key> 或 <parameter=key>value</parameter>
_PARAM_EQ_RE = re.compile(
    r"<parameter=([^>]+)>(.*?)(?:</parameter=[^>]*>|</parameter>)",
    re.DOTALL,
)


def _extract_params_name_attr(xml_body: str) -> dict:
    """解析 <parameter name="key">value</parameter> 风格"""
    result = {}
    for m in _PARAM_NAME_RE.finditer(xml_body):
        key = m.group(1).strip()
        val = m.group(2).strip()
        result[key] = _coerce_value(val)
    return result


def _extract_params_eq_style(xml_body: str) -> dict:
    """解析 <parameter=key>value</parameter=key> 风格"""
    result = {}
    for m in _PARAM_EQ_RE.finditer(xml_body):
        key = m.group(1).strip()
        val = m.group(2).strip()
        result[key] = _coerce_value(val)
    return result


def _coerce_value(val: str):
    """尝试把字符串值转为 Python 原生类型（int/float/bool/dict/list），否则保留字符串"""
    v = val.strip()
    if v.lower() == "true":  return True
    if v.lower() == "false": return False
    try:    return int(v)
    except: pass
    try:    return float(v)
    except: pass
    if (v.startswith("{") and v.endswith("}")) or (v.startswith("[") and v.endswith("]")):
        obj = _try_parse_json(v)
        if obj is not None:
            return obj
    return val


# ══════════════════════════════════════════════════════════════════════════════
# 各格式解析器
# ══════════════════════════════════════════════════════════════════════════════

# ── FORMAT 1：<tool_use> JSON ─────────────────────────────────────────────────

_F1_RE = re.compile(r"<tool_use>\s*(\{.*?\})\s*</tool_use>", re.DOTALL)

def _parse_f1(text: str) -> list[ToolCall]:
    calls = []
    for m in _F1_RE.finditer(text):
        obj = _try_parse_json(m.group(1), "F1")
        if not obj:
            continue
        name = obj.get("name", "")
        if not name:
            continue
        inp = obj.get("input", {})
        calls.append(ToolCall(name=name, input=inp if isinstance(inp, dict) else {}, fmt="F1"))
    return calls


# ── FORMAT 2：<tool_call><invoke name="..."><parameter name="k">v</parameter></invoke></tool_call>

_F2_OUTER_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_F2_INVOKE_RE = re.compile(r"<invoke\s+name=['\"]([^'\"]+)['\"]>(.*?)</invoke>", re.DOTALL)

def _parse_f2(text: str) -> list[ToolCall]:
    calls = []
    for outer in _F2_OUTER_RE.finditer(text):
        body = outer.group(1)
        for m in _F2_INVOKE_RE.finditer(body):
            name   = m.group(1).strip()
            params = _extract_params_name_attr(m.group(2))
            if not params:
                # 尝试 eq 风格
                params = _extract_params_eq_style(m.group(2))
            if name:
                calls.append(ToolCall(name=name, input=params, fmt="F2"))
    return calls


# ── FORMAT 3：<tool_call><function=Name><parameter=k>v</parameter></function></tool_call>

# <function=Name> ... </function>  (closing tag may be </function=Name> or </function>)
_F3_FUNC_RE = re.compile(
    r"<function=([^>\s]+)\s*>(.*?)(?:</function=[^>]*>|</function>)",
    re.DOTALL,
)

def _parse_f3(text: str) -> list[ToolCall]:
    calls = []
    for outer in _F2_OUTER_RE.finditer(text):   # 外层同样是 <tool_call>
        body = outer.group(1)
        for m in _F3_FUNC_RE.finditer(body):
            name   = m.group(1).strip()
            params = _extract_params_eq_style(m.group(2))
            if not params:
                params = _extract_params_name_attr(m.group(2))
            if name:
                calls.append(ToolCall(name=name, input=params, fmt="F3"))
    return calls


# ── FORMAT 4：<function_calls><invoke name="...">...</invoke></function_calls>

_F4_OUTER_RE = re.compile(r"<function_calls>(.*?)</function_calls>", re.DOTALL)

def _parse_f4(text: str) -> list[ToolCall]:
    calls = []
    for outer in _F4_OUTER_RE.finditer(text):
        body = outer.group(1)
        for m in _F2_INVOKE_RE.finditer(body):
            name   = m.group(1).strip()
            params = _extract_params_name_attr(m.group(2))
            if not params:
                params = _extract_params_eq_style(m.group(2))
            if name:
                calls.append(ToolCall(name=name, input=params, fmt="F4"))
    return calls


# ── FORMAT 5：<minimax:tool_call><invoke name="..." arg="..." skill="..."/></minimax:tool_call>

_F5_OUTER_RE = re.compile(r"<minimax:tool_call>(.*?)</minimax:tool_call>", re.DOTALL)
# self-closing <invoke name="X" k="v" .../>
_F5_INVOKE_RE = re.compile(r"<invoke\s+([^>]+?)/>", re.DOTALL)
_F5_ATTR_RE   = re.compile(r'(\w+)=["\']([^"\']*)["\']')

def _parse_f5(text: str) -> list[ToolCall]:
    calls = []
    for outer in _F5_OUTER_RE.finditer(text):
        body = outer.group(1)
        for m in _F5_INVOKE_RE.finditer(body):
            attrs = dict(_F5_ATTR_RE.findall(m.group(1)))
            name  = attrs.pop("name", "")
            if name:
                calls.append(ToolCall(name=name, input=attrs, fmt="F5"))
    return calls


# ── FORMAT 6：Markdown 代码块内的 JSON ────────────────────────────────────────

_F6_RE = re.compile(
    r"```(?:json|tool_use|tool_call)?\s*\n(\{.*?\})\s*\n```",
    re.DOTALL,
)

def _parse_f6(text: str) -> list[ToolCall]:
    calls = []
    for m in _F6_RE.finditer(text):
        obj = _try_parse_json(m.group(1), "F6")
        if not obj:
            continue
        name = obj.get("name", "")
        if not name:
            continue
        inp = obj.get("input", {})
        calls.append(ToolCall(name=name, input=inp if isinstance(inp, dict) else {}, fmt="F6"))
    return calls


# ══════════════════════════════════════════════════════════════════════════════
# 去除 tool_call 标记的纯文字提取
# ══════════════════════════════════════════════════════════════════════════════

_ALL_STRIP_PATTERNS = [
    _F1_RE,
    _F2_OUTER_RE,
    _F4_OUTER_RE,
    _F5_OUTER_RE,
    _F6_RE,
]

def _strip_tool_markup(text: str) -> str:
    """移除所有 tool_call 标记，返回纯文字"""
    for pat in _ALL_STRIP_PATTERNS:
        text = pat.sub("", text)
    return text.strip()


# ══════════════════════════════════════════════════════════════════════════════
# 主入口
# ══════════════════════════════════════════════════════════════════════════════

# 解析器列表，按优先级排列（F1 最严格最优先，F6 最宽松兜底）
_PARSERS = [
    ("F1", _parse_f1),
    ("F2", _parse_f2),
    ("F3", _parse_f3),
    ("F4", _parse_f4),
    ("F5", _parse_f5),
    ("F6", _parse_f6),
]


def parse(text: str) -> ParseResult:
    """
    主解析函数。
    按优先级尝试所有格式，去重（同名+同参数视为重复），返回 ParseResult。
    """
    all_calls: list[ToolCall] = []
    seen: set[tuple] = set()   # 去重 key = (name, frozenset(input.items()))

    for fmt_name, parser in _PARSERS:
        try:
            found = parser(text)
        except Exception as e:
            logger.debug(f"[tool_parser] {fmt_name} 解析异常: {e}")
            continue

        for tc in found:
            try:
                dedup_key = (tc.name, json.dumps(tc.input, sort_keys=True, ensure_ascii=False))
            except Exception:
                dedup_key = (tc.name, str(tc.input))

            if dedup_key not in seen:
                seen.add(dedup_key)
                all_calls.append(tc)
                logger.debug(f"[tool_parser] {fmt_name} 匹配: {tc.name}  input={str(tc.input)[:80]}")

    pure_text = _strip_tool_markup(text)
    return ParseResult(tool_calls=all_calls, pure_text=pure_text)


def parse_to_dicts(text: str) -> tuple[str, list[dict]]:
    """
    便捷函数，直接返回 (pure_text, tool_calls_as_dicts)。
    与原 split_text_and_tools 接口一致，方便 proxy.py 无缝切换。
    """
    result = parse(text)
    return result.pure_text, result.to_dicts()


# ══════════════════════════════════════════════════════════════════════════════
# 自测（python tool_call_parser.py 直接运行）
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(message)s")

    CASES = [
        # ── F1
        ("F1 标准 JSON", """
思考一下。
<tool_use>
{"name": "Read", "input": {"file_path": "/tmp/test.md"}}
</tool_use>
"""),
        # ── F1 畸形（缺结尾括号）
        ("F1 畸形 JSON（缺括号）", """
<tool_use>
{"name": "Read", "input": {"file_path": "E:/小说/第 136 章 [联盟.md](http://联盟.md)"}
</tool_use>
"""),
        # ── F2
        ("F2 <invoke name> + <parameter name>", """
<tool_call>
<invoke name="Skill">
<parameter name="Skill_Arguments">使用 novel-director，分析进度</parameter>
</invoke>
</tool_call>
"""),
        # ── F3（问题中实际出现的格式）
        ("F3 <function=Name> + <parameter=key>", """
我来帮你开始写第137章。首先让我检查一下工作目录和当前项目状态。<tool_call>
<function=Bash>
<parameter=command>
ls -la "小说/超级人工智能系统小说_01"
</parameter=command>
<parameter=description>
检查小说项目目录结构
</parameter=description>
</function>
</tool_call><tool_call>
<function=Read>
<parameter=file_path>
E:\\codes\\apicall\\小说\\超级人工智能系统小说_01\\正文\\第 136 章 联盟.md
</parameter=file_path>
</function>
</tool_call>
"""),
        # ── F4
        ("F4 <function_calls>", """
<function_calls>
<invoke name="Write">
<parameter name="path">/tmp/out.txt</parameter>
<parameter name="content">hello world</parameter>
</invoke>
</function_calls>
"""),
        # ── F5 minimax
        ("F5 minimax:tool_call", """
<minimax:tool_call>
<invoke name="Skill" arg="分析当前小说进度" skill="novel-director"/>
</minimax:tool_call>
"""),
        # ── F6 markdown
        ("F6 markdown json block", """
```json
{"name": "Search", "input": {"query": "量子计算"}}
```
"""),
        # ── 多工具调用混合
        ("混合：F3 两个工具", """
先看目录再读文件。<tool_call>
<function=Bash>
<parameter=command>ls /tmp</parameter=command>
</function>
</tool_call>然后：<tool_call>
<function=Read>
<parameter=file_path>/tmp/note.txt</parameter=file_path>
</function>
</tool_call>
"""),
        # ── 无工具调用
        ("无工具调用", "这是一段普通回复，不包含任何工具调用。"),
    ]

    W = 70
    passed = failed = 0
    for title, text in CASES:
        result = parse(text)
        ok = True
        if "无工具调用" in title:
            ok = len(result.tool_calls) == 0
        else:
            ok = len(result.tool_calls) >= 1 and all(tc.name for tc in result.tool_calls)

        status = "✅" if ok else "❌"
        if ok: passed += 1
        else:  failed += 1

        print(f"\n{status} {title}")
        print(f"   pure_text: {result.pure_text[:60]!r}{'...' if len(result.pure_text)>60 else ''}")
        for tc in result.tool_calls:
            print(f"   [{tc.fmt}] {tc.name}  input={json.dumps(tc.input, ensure_ascii=False)[:80]}")

    print(f"\n{'─'*W}")
    print(f"  结果：{passed} 通过  {failed} 失败  共 {passed+failed} 用例")
    print(f"{'─'*W}")
