from agentflow.specs import AgentKind
from agentflow.traces import create_trace_parser


def test_codex_trace_parser_extracts_assistant_message():
    parser = create_trace_parser(AgentKind.CODEX, "plan")
    events = parser.feed('{"type":"response.output_item.done","item":{"type":"message","role":"assistant","content":[{"type":"output_text","text":"codex ok"}]}}')
    assert events[0].kind == "assistant_message"
    assert parser.finalize() == "codex ok"


def test_codex_trace_parser_ignores_unstable_feature_warning():
    parser = create_trace_parser(AgentKind.CODEX, "plan")

    assert parser.feed('{"type":"item.completed","item":{"id":"item_0","type":"error","message":"Under-development features enabled: responses_websockets_v2. To suppress this warning, set suppress_unstable_features_warning = true in /home/shou/.codex/config.toml."}}') == []

    events = parser.feed('{"type":"response.output_item.done","item":{"type":"message","role":"assistant","content":[{"type":"output_text","text":"codex ok"}]}}')

    assert events[0].kind == "assistant_message"
    assert parser.finalize() == "codex ok"


def test_codex_trace_parser_keeps_real_error_items():
    parser = create_trace_parser(AgentKind.CODEX, "plan")

    events = parser.feed('{"type":"item.completed","item":{"id":"item_0","type":"error","message":"permission denied"}}')

    assert events[0].kind == "item_completed"
    assert events[0].title == "Item completed: error"
    assert events[0].content == "permission denied"


def test_claude_trace_parser_extracts_result():
    parser = create_trace_parser(AgentKind.CLAUDE, "implement")
    parser.feed('{"type":"assistant","message":{"content":[{"type":"text","text":"working"}]}}')
    parser.feed('{"type":"result","result":"done"}')
    assert parser.finalize() == "working\ndone"


def test_claude_trace_parser_dedupes_matching_result():
    parser = create_trace_parser(AgentKind.CLAUDE, "implement")
    parser.feed('{"type":"assistant","message":{"content":[{"type":"text","text":"working"}]}}')
    parser.feed('{"type":"result","result":"working"}')
    assert parser.finalize() == "working"


def test_claude_trace_parser_ignores_hook_chatter():
    parser = create_trace_parser(AgentKind.CLAUDE, "implement")

    assert parser.feed('{"type":"system","subtype":"hook_started","hook_name":"SessionStart:startup"}') == []
    assert parser.feed('{"type":"system","subtype":"hook_response","hook_name":"SessionStart:startup","output":"very large startup payload"}') == []

    events = parser.feed('{"type":"assistant","message":{"content":[{"type":"text","text":"working"}]}}')

    assert events[0].kind == "assistant_message"
    assert parser.finalize() == "working"


def test_claude_trace_parser_keeps_hook_failures():
    parser = create_trace_parser(AgentKind.CLAUDE, "implement")

    events = parser.feed('{"type":"system","subtype":"hook_failed","hook_name":"SessionStart:startup","stderr":"hook exploded"}')

    assert events[0].kind == "hook_error"
    assert events[0].title == "Hook failed: SessionStart:startup"
    assert events[0].content == "hook exploded"


def test_kimi_trace_parser_extracts_text_part():
    parser = create_trace_parser(AgentKind.KIMI, "review")
    parser.feed('{"jsonrpc":"2.0","method":"event","params":{"type":"ContentPart","payload":{"type":"text","text":"kimi trace"}}}')
    assert parser.finalize() == "kimi trace"


def test_pi_trace_parser_extracts_final_assistant_text_from_agent_end():
    parser = create_trace_parser(AgentKind.PI, "scan")
    # Realistic Pi event sequence: session / agent_start / turn_start / message_start
    # (user) / message_end (user) / message_start (assistant) / message_update deltas /
    # message_end (assistant) / turn_end / agent_end.
    parser.feed('{"type":"session","id":"abc","cwd":"/tmp"}')
    parser.feed('{"type":"agent_start"}')
    parser.feed('{"type":"turn_start"}')
    parser.feed('{"type":"message_update","assistantMessageEvent":{"type":"text_delta","delta":"Hello"}}')
    parser.feed('{"type":"message_update","assistantMessageEvent":{"type":"text_delta","delta":" there"}}')
    parser.feed(
        '{"type":"message_end","message":{"role":"assistant",'
        '"content":[{"type":"text","text":"Hello there"}]}}'
    )
    parser.feed(
        '{"type":"agent_end","messages":['
        '{"role":"user","content":[{"type":"text","text":"say hi"}]},'
        '{"role":"assistant","content":[{"type":"text","text":"Hello there"}]}'
        "]}"
    )
    assert parser.finalize() == "Hello there"


def test_pi_trace_parser_emits_delta_events():
    parser = create_trace_parser(AgentKind.PI, "scan")
    events = parser.feed(
        '{"type":"message_update","assistantMessageEvent":{"type":"text_delta","delta":"partial"}}'
    )
    assert len(events) == 1
    assert events[0].kind == "assistant_delta"
    assert events[0].content == "partial"


def test_pi_trace_parser_prefers_agent_end_when_present():
    parser = create_trace_parser(AgentKind.PI, "scan")
    # Only feed agent_end with a single assistant message.
    parser.feed(
        '{"type":"agent_end","messages":['
        '{"role":"assistant","content":[{"type":"text","text":"final answer"}]}'
        "]}"
    )
    assert parser.finalize() == "final answer"


_AGY_INIT = '{"event":"init","conversation_id":"c1","init":{"model":"gemini-4-neon-low","cwd":"/w","tools":["view_file"],"permission_mode":"always-proceed"}}'


def test_antigravity_trace_parser_uses_result_response_and_usage():
    parser = create_trace_parser(AgentKind.ANTIGRAVITY, "ping")
    parser.feed(_AGY_INIT)
    parser.feed('{"event":"step_update","step_update":{"conversation_id":"c1","step_index":0,"state":"DONE","step_type":"user_input"}}')
    deltas = parser.feed('{"event":"step_update","step_update":{"conversation_id":"c1","step_index":1,"state":"ACTIVE","step_type":"agent_response","text_delta":"PO"}}')
    parser.feed('{"event":"step_update","step_update":{"conversation_id":"c1","step_index":1,"state":"DONE","step_type":"agent_response","text_delta":"NG\\n","usage":{"input_tokens":9474,"output_tokens":19}}}')
    events = parser.feed(
        '{"event":"result","result":{"conversation_id":"c1","status":"SUCCESS","response":"PONG\\n","duration_seconds":6.2,"num_turns":1,'
        '"usage":{"input_tokens":9474,"output_tokens":19,"thinking_tokens":16,"cache_read_tokens":0,"total_tokens":9493}}}'
    )

    assert deltas[0].kind == "assistant_delta"
    assert deltas[0].content == "PO"
    assert [event.kind for event in events] == ["result", "usage"]
    assert events[1].raw["total_tokens"] == 9493
    assert parser.usage["thinking_tokens"] == 16
    assert parser.failure is None
    assert parser.finalize() == "PONG"


def test_antigravity_trace_parser_reports_tool_steps():
    parser = create_trace_parser(AgentKind.ANTIGRAVITY, "scan")
    call = parser.feed(
        '{"event":"step_update","step_update":{"step_index":2,"state":"ACTIVE","step_type":"tool","tool_name":"read_file",'
        '"tool_info":{"name":"read_file","parameters":{"Path":"/w/note.txt"}}}}'
    )
    done = parser.feed(
        '{"event":"step_update","step_update":{"step_index":2,"state":"DONE","step_type":"tool","tool_name":"read_file",'
        '"tool_info":{"name":"read_file","parameters":{"Path":"/w/note.txt"},"output":"1 lines, 13 bytes"}}}'
    )

    assert call[0].kind == "tool_call"
    assert call[0].title == "Tool call: read_file"
    assert "/w/note.txt" in call[0].content
    assert done[0].kind == "tool_result"
    assert done[0].content == "1 lines, 13 bytes"


def test_antigravity_trace_parser_marks_print_timeout_as_failure():
    parser = create_trace_parser(AgentKind.ANTIGRAVITY, "slow")
    parser.feed(_AGY_INIT)
    parser.feed('{"event":"step_update","step_update":{"step_index":1,"state":"ACTIVE","step_type":"agent_response","text_delta":"1\\n2\\n"}}')
    events = parser.feed(
        '{"event":"result","result":{"conversation_id":"c1","status":"ERROR","response":"",'
        '"error":"print timeout after 2s with the turn still in progress; the response is partial",'
        '"duration_seconds":0,"num_turns":1,"usage":{"input_tokens":0,"output_tokens":0,"total_tokens":0},"finished_reason":"print_timeout"}}'
    )

    assert events[0].kind == "error"
    assert parser.failure == (
        "print timeout after 2s with the turn still in progress; the response is partial (finished_reason=print_timeout)"
    )
    # Partial streamed text is kept as the final response.
    assert parser.finalize() == "1\n2"


def test_antigravity_trace_parser_marks_error_status_without_error_text():
    parser = create_trace_parser(AgentKind.ANTIGRAVITY, "bad")
    parser.feed('{"event":"result","result":{"status":"CANCELLED","response":""}}')

    assert parser.failure == "agy finished with status CANCELLED"
    assert parser.finalize() == ""


def test_antigravity_trace_parser_uses_structured_output_when_response_missing():
    parser = create_trace_parser(AgentKind.ANTIGRAVITY, "capital")
    parser.feed('{"event":"result","result":{"status":"SUCCESS","response":"","structured_output":{"capital":"Paris"}}}')

    assert parser.failure is None
    assert parser.finalize() == '{"capital": "Paris"}'


def test_antigravity_trace_parser_resets_between_attempts():
    parser = create_trace_parser(AgentKind.ANTIGRAVITY, "retry")
    parser.feed('{"event":"result","result":{"status":"ERROR","response":"","error":"boom"}}')
    assert parser.failure == "boom"

    parser.start_attempt(2)
    parser.feed('{"event":"result","result":{"status":"SUCCESS","response":"ok\\n"}}')
    assert parser.failure is None
    assert parser.finalize() == "ok"


def test_antigravity_trace_parser_reports_tool_errors_and_denied_actions():
    parser = create_trace_parser(AgentKind.ANTIGRAVITY, "scan")
    error = parser.feed(
        '{"event":"step_update","step_update":{"step_index":2,"state":"ERROR","step_type":"tool","tool_name":"write_file",'
        '"tool_info":{"name":"write_file","error":{"type":"TOOL_ERROR","message":"permission check failed for write_file"}}}}'
    )
    events = parser.feed(
        '{"event":"result","result":{"status":"SUCCESS","response":"",'
        '"denied_actions":[{"action":"write_file","display_name":"Edit"}]}}'
    )

    assert error[0].kind == "tool_error"
    assert "permission check failed" in error[0].content
    assert [event.kind for event in events] == ["result", "denied_actions"]
    assert events[1].content == "write_file"
    assert parser.failure is None
