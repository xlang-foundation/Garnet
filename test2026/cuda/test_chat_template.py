"""Pure contract tests; no model or provider replacements."""
import importlib.util
from pathlib import Path

import pytest

source = Path(__file__).resolve().parents[2] / "runtime/chat_template.py"
spec = importlib.util.spec_from_file_location("garnet_chat_template", source)
chat = importlib.util.module_from_spec(spec)
spec.loader.exec_module(chat)


def tool(strict=True):
    return {"type": "function", "function": {"name": "write_file", "strict": strict, "parameters": {
        "type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
        "required": ["path", "content"]}}}


def test_qwen_checkpoint_function_call_is_converted_without_executing_it():
    decoder = chat.ToolDecoder([tool()])
    raw = '<tool_call>\n<function=write_file>\n<parameter=path>\nexample.txt\n</parameter>\n<parameter=content>\nhello\nworld\n</parameter>\n</function>\n</tool_call>'
    message, calls = decoder.finish(raw)
    import json
    assert message["content"] is None
    assert json.loads(calls[0]["function"]["arguments"]) == {"path": "example.txt", "content": "hello\nworld"}


@pytest.mark.parametrize("raw", [
    '<tool_call><function=unknown></function></tool_call>',
    '<tool_call><function=write_file><parameter=path>a</parameter></function></tool_call>',
    '<tool_call><function=write_file>',
])
def test_malformed_or_unknown_tool_calls_fail_closed(raw):
    with pytest.raises(ValueError):
        chat.ToolDecoder([tool()]).finish(raw)


def test_truncated_function_call_is_never_returned_as_success():
    with pytest.raises(ValueError, match="truncated"):
        chat.ToolDecoder([tool()]).finish('<tool_call>', truncated=True)


@pytest.mark.parametrize("strict", [False, None])
def test_non_strict_arguments_preserve_missing_fields_for_client_validation(strict):
    import json
    definition = tool(strict)
    if strict is None:
        definition["function"].pop("strict")
    raw = '<tool_call><function=write_file><parameter=path>a.txt</parameter></function></tool_call>'
    _, calls = chat.ToolDecoder([definition]).finish(raw)
    assert json.loads(calls[0]["function"]["arguments"]) == {"path": "a.txt"}
    assert "content" not in json.loads(calls[0]["function"]["arguments"])


@pytest.mark.parametrize("strict", [False, True])
def test_duplicate_arguments_are_never_silently_overwritten(strict):
    raw = '<tool_call><function=write_file><parameter=path>a</parameter><parameter=path>b</parameter></function></tool_call>'
    with pytest.raises(ValueError, match="Duplicate"):
        chat.ToolDecoder([tool(strict)]).finish(raw)


def test_captured_native_non_strict_call_keeps_client_error_recovery_possible():
    import copy
    import json
    capture = json.loads(Path(__file__).with_name("fixtures").joinpath("non_strict_edit_call.json").read_text())
    definition = capture["tool"]
    _, calls = chat.ToolDecoder([definition]).finish(capture["text"])
    arguments = json.loads(calls[0]["function"]["arguments"])
    assert set(arguments) == {"folder_id", "new_text", "old_text"}
    assert "path" not in arguments  # Never infer or repair the missing path.
    from jsonschema import ValidationError
    with pytest.raises(ValidationError, match="path"):
        chat.Draft202012Validator(definition["function"]["parameters"]).validate(arguments)
    strict = copy.deepcopy(definition)
    strict["function"]["strict"] = True
    with pytest.raises(ValueError, match="required"):
        chat.ToolDecoder([strict]).finish(capture["text"])
