"""Checkpoint-owned Qwen chat templates and function-call wire conversion.

This module runs in the hosting CPython control process. Tokenization and all
model computation remain in the native Garnet worker.
"""
import copy
import json
import re
import uuid
from pathlib import Path

from jinja2 import StrictUndefined
from jinja2.sandbox import ImmutableSandboxedEnvironment
from jsonschema import Draft202012Validator


class ChatTemplate:
    def __init__(self, checkpoint):
        self.path = Path(checkpoint) / "chat_template.jinja"
        environment = ImmutableSandboxedEnvironment(
            trim_blocks=True, lstrip_blocks=True, undefined=StrictUndefined
        )
        # Match the Transformers chat-template JSON filter, including Unicode.
        environment.filters["tojson"] = lambda value, **kw: json.dumps(value, ensure_ascii=False, **kw)
        environment.globals["raise_exception"] = self._raise
        self.template = environment.from_string(self.path.read_text(encoding="utf8"))

    @staticmethod
    def _raise(message):
        raise ValueError(message)

    def render(self, request):
        supported = {"model", "messages", "tools", "tool_choice", "parallel_tool_calls", "stream",
                     "stream_options", "max_tokens", "max_completion_tokens", "n", "temperature", "user"}
        if set(request) - supported:
            raise ValueError("Unsupported native chat options: " + ", ".join(sorted(set(request) - supported)))
        if request.get("temperature", 0) != 0 or request.get("n", 1) != 1:
            raise ValueError("This native profile supports greedy decoding and n=1")
        if request.get("tool_choice", "auto") not in ("auto", "none"):
            raise ValueError("Only auto and none tool choices are qualified")
        stream_options = request.get("stream_options") or {}
        if set(stream_options) - {"include_usage"}:
            raise ValueError("Unsupported stream options")
        messages = copy.deepcopy(request["messages"])
        tools = request.get("tools") or []
        if request.get("tool_choice") == "none":
            tools = []
        for tool in tools:
            if tool.get("type") != "function" or not isinstance(tool.get("function", {}).get("name"), str):
                raise ValueError("Only named function tools are supported")
            if "strict" in tool["function"] and type(tool["function"]["strict"]) is not bool:
                raise ValueError("Function strict must be a boolean")
            schema = tool["function"].get("parameters", {"type": "object"})
            self._local_references(schema)
            Draft202012Validator.check_schema(schema)
        seen_calls = {}
        for message in messages:
            role = message.get("role")
            if role not in ("system", "user", "assistant", "tool"):
                raise ValueError("Unsupported checkpoint message role")
            content = message.get("content")
            if isinstance(content, list):
                if any(part.get("type") != "text" or not isinstance(part.get("text"), str) for part in content):
                    raise ValueError("This chat profile currently accepts text content only")
                message["content"] = "\n".join(part["text"] for part in content)
            elif content is not None and not isinstance(content, str):
                raise ValueError("Invalid message content")
            message.setdefault("reasoning_content", "")
            message.setdefault("tool_calls", [])
            for call in message["tool_calls"] or []:
                if role != "assistant" or call.get("type") != "function":
                    raise ValueError("Invalid function-call history")
                function = call["function"]
                arguments = function["arguments"]
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                if not isinstance(arguments, dict):
                    raise ValueError("Function arguments must be an object")
                function["arguments"] = arguments
                call_id = call.get("id")
                if not isinstance(call_id, str) or call_id in seen_calls:
                    raise ValueError("Missing or duplicate tool-call ID")
                seen_calls[call_id] = function["name"]
            if role == "tool" and message.get("tool_call_id") not in seen_calls:
                raise ValueError("Tool response has no preceding matching function call")
        # Missing optional keys are normal in the checkpoint's own template.
        from jinja2 import Undefined
        self.template.environment.undefined = Undefined
        return self.template.render(messages=messages, tools=tools, add_generation_prompt=True,
                                    enable_thinking=False, add_vision_id=False)

    @staticmethod
    def _local_references(value):
        if isinstance(value, dict):
            if "$ref" in value and not str(value["$ref"]).startswith("#"):
                raise ValueError("Remote function-schema references are not permitted")
            for child in value.values():
                ChatTemplate._local_references(child)
        elif isinstance(value, list):
            for child in value:
                ChatTemplate._local_references(child)


class ToolDecoder:
    """Emit text prefixes immediately; expose a tool only after valid completion."""
    def __init__(self, tools):
        self.tools = {tool["function"]["name"]: tool["function"] for tool in (tools or [])}
        self.text = ""
        self.sent = ""

    def push(self, text):
        if not text.startswith(self.text):
            raise ValueError("Native tokenizer revised an already decoded prefix")
        self.text = text
        boundary = text.find("<tool_call>")
        safe = text[:boundary] if boundary >= 0 else text
        if boundary < 0:
            for size in range(1, len("<tool_call>")):
                if safe.endswith("<tool_call>"[:size]):
                    safe = safe[:-size]
                    break
        if not safe.startswith(self.sent):
            raise ValueError("Native tool prefix revised emitted text")
        delta = safe[len(self.sent):]
        self.sent = safe
        return delta

    def finish(self, text, truncated=False):
        if not text.startswith(self.text):
            raise ValueError("Final native text revised its prefix")
        self.text = text
        boundary = text.find("<tool_call>")
        if boundary < 0:
            return {"role": "assistant", "content": text}, []
        if truncated:
            raise ValueError("Native generation truncated a function call")
        remainder = text[boundary:]
        blocks = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", remainder, re.S)
        if not blocks or re.sub(r"<tool_call>.*?</tool_call>", "", remainder, flags=re.S).strip():
            raise ValueError("Malformed or incomplete native tool call")
        calls = []
        for block in blocks:
            match = re.fullmatch(r"<function=([A-Za-z_][\w.:-]*)>\s*(.*?)\s*</function>", block, re.S)
            if not match or match[1] not in self.tools:
                raise ValueError("Native model selected an unknown function")
            schema = self.tools[match[1]].get("parameters", {})
            strict = self.tools[match[1]].get("strict") is True
            properties = schema.get("properties", {})
            arguments = {}
            parameters = re.findall(r"<parameter=([A-Za-z_][\w.:-]*)>\n?(.*?)\n?</parameter>", match[2], re.S)
            if re.sub(r"<parameter=[^>]+>.*?</parameter>", "", match[2], flags=re.S).strip():
                raise ValueError("Malformed native function arguments")
            for name, value in parameters:
                if name in arguments:
                    raise ValueError("Duplicate native function argument")
                if strict and name not in properties:
                    raise ValueError("Unknown native function argument")
                expected = properties.get(name, {}).get("type")
                if expected == "string":
                    arguments[name] = value
                else:
                    try:
                        arguments[name] = json.loads(value)
                    except ValueError:
                        if expected not in (None, ["string", "null"]):
                            raise
                        arguments[name] = value
            missing = set(schema.get("required", [])) - set(arguments)
            if strict and missing:
                raise ValueError("Native function call omitted required arguments")
            ChatTemplate._local_references(schema)
            # Chat Completions is best-effort unless strict:true was requested.
            # Preserve supplied JSON arguments for the client's own validation
            # and tool-error continuation. Never fill missing arguments or run
            # a function here. Structural/JSON/truncation checks remain mandatory.
            if strict:
                Draft202012Validator(schema).validate(arguments)
            encoded = json.dumps(arguments, ensure_ascii=False, allow_nan=False)
            encoded.encode("utf8", "strict")
            calls.append({"id": "call_" + uuid.uuid4().hex, "type": "function",
                          "function": {"name": match[1], "arguments": encoded}})
        return {"role": "assistant", "content": text[:boundary] or None, "tool_calls": calls}, calls
