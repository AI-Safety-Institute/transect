"""One-shot generator for tests/fixtures/openclaw/mini_telemetry.jsonl.

Hand-designed miniature of an OpenClaw telemetry-hal capture (schema B,
cron orchestrator). Every importer rule gets one exercising record; the
expected values in tests/test_ingestion_selection.py are hand-computed
from here.
"""

import json
from pathlib import Path

ORCH = "agent:main:cron:sess-cron-0001"
SUB_A = "agent:main:subagent:sub-aaaa"
SUB_B = "agent:main:subagent:sub-bbbb"
MODEL = "claude-opus-4"


def usage(inp, out, cr, cw, tot):
    return {
        "input": inp,
        "output": out,
        "cacheRead": cr,
        "cacheWrite": cw,
        "totalTokens": tot,
        "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": 0},
    }


U1 = {
    "role": "user",
    "idempotencyKey": "idem-1",
    "timestamp": 1750000000000,
    "content": "Start the experiment run.",
}
A1 = {
    "role": "assistant",
    "responseId": "resp-1",
    "timestamp": 1750000010000,
    "api": "anthropic-messages",
    "provider": "anthropic",
    "model": MODEL,
    "responseModel": MODEL,
    "stopReason": "toolUse",
    "usage": usage(100, 50, 0, 900, 1050),
    "content": [
        {"type": "text", "text": "Spawning a literature-review sub-agent."},
        {
            "type": "toolCall",
            "id": "tc-1",
            "name": "sessions_spawn",
            "arguments": {
                "label": "lit-review",
                "task": "Survey related work on RL generalisation",
            },
        },
    ],
}
TR1 = {
    "role": "toolResult",
    "toolCallId": "tc-1",
    "toolName": "sessions_spawn",
    "timestamp": 1750000012000,
    "isError": False,
    "content": json.dumps({"childSessionKey": SUB_A, "status": "accepted"}),
}
A2 = {
    "role": "assistant",
    "responseId": "resp-2",
    "timestamp": 1750000070000,
    "api": "anthropic-messages",
    "provider": "anthropic",
    "model": MODEL,
    "responseModel": MODEL,
    "stopReason": "toolUse",
    "usage": usage(20, 40, 950, 30, 1040),
    "content": [
        {"type": "text", "text": "Checking workspace."},
        {
            "type": "toolCall",
            "id": "tc-2",
            "name": "exec",
            "arguments": {"command": "ls"},
        },
    ],
}
TR2 = {
    "role": "toolResult",
    "toolCallId": "tc-2",
    "toolName": "exec",
    "timestamp": 1750000071500,
    "isError": False,
    "content": "file1\nfile2",
}
# Scaffold roll-up aggregate: rid-less, tool-less 'stop', usage = sums of the
# A1+A2 block, totalTokens FROZEN at A2's -> must be excluded at parse.
ROLLUP = {
    "role": "assistant",
    "timestamp": 1750000071600,
    "model": MODEL,
    "api": "anthropic-messages",
    "provider": "anthropic",
    "stopReason": "stop",
    "usage": usage(120, 90, 950, 930, 1040),
    "content": [{"type": "text", "text": "Checking workspace."}],
}
U2 = {
    "role": "user",
    "idempotencyKey": "idem-2",
    "timestamp": 1750000105000,
    "content": "Also check the baselines.",
}
# Zero-usage provider-failure placeholder: kept, usage NOT fabricated.
A3 = {
    "role": "assistant",
    "responseId": "resp-3",
    "timestamp": 1750000190000,
    "api": "anthropic-messages",
    "provider": "anthropic",
    "model": MODEL,
    "responseModel": MODEL,
    "stopReason": "error",
    "errorMessage": "provider overloaded",
    "usage": usage(0, 0, 0, 0, 0),
    "content": [
        {"type": "text", "text": "[assistant turn failed: provider overloaded]"}
    ],
}
COMPACTION = {
    "role": "compactionSummary",
    "summary": "## Goal\nShort summary.",
    "tokensBefore": 180000,
    "timestamp": 1750000220000,
}
A4 = {
    "role": "assistant",
    "responseId": "resp-4",
    "timestamp": 1750000230000,
    "api": "anthropic-messages",
    "provider": "anthropic",
    "model": MODEL,
    "responseModel": MODEL,
    "stopReason": "stop",
    "usage": usage(200, 80, 100, 50, 430),
    "content": [{"type": "text", "text": "Wrapping up."}],
}

BOILER = (
    "[Subagent Context] You are running as a subagent (depth 1/1). "
    "Results auto-announce to your requester; do not busy-poll for status."
    "\n\n[Subagent Task]\n\n"
)

lines = [
    # cumulative orchestrator snapshots (turns recur -> dedup exercised)
    {"type": "agent.start", "sessionKey": ORCH, "agentId": "main", "messages": [U1]},
    {"type": "agent.end", "sessionKey": ORCH, "agentId": "main", "messages": [U1, A1]},
    # sub-agent A: spawn-linked via TR1's childSessionKey; schema-B activity
    {
        "type": "agent.start",
        "sessionKey": SUB_A,
        "agentId": "main",
        "prompt": BOILER + "Survey related work on RL generalisation.",
        "promptLength": 150,
    },
    {
        "type": "tool.start",
        "toolName": "web_search",
        "params": {"q": "RL generalisation"},
        "sessionKey": SUB_A,
        "agentId": "main",
        "ts": 1750000015000,
    },
    {
        "type": "tool.end",
        "toolName": "web_search",
        "durationMs": 95000,
        "success": True,
        "sessionKey": SUB_A,
        "agentId": "main",
        "ts": 1750000110000,
    },
    {
        "type": "tool.start",
        "toolName": "read_file",
        "params": {"path": "notes.md"},
        "sessionKey": SUB_A,
        "agentId": "main",
    },
    {
        "type": "tool.end",
        "toolName": "read_file",
        "durationMs": 30000,
        "success": False,
        "error": "file not found",
        "sessionKey": SUB_A,
        "agentId": "main",
    },
    {"type": "agent.end", "sessionKey": SUB_A, "agentId": "main"},
    # orchestrator's own tool.* events (main kind): tolerated, no lane
    {
        "type": "tool.start",
        "toolName": "exec",
        "params": {"command": "date"},
        "sessionKey": "agent:main:main",
        "agentId": "main",
    },
    {
        "type": "tool.end",
        "toolName": "exec",
        "durationMs": 371,
        "success": True,
        "sessionKey": "agent:main:main",
        "agentId": "main",
    },
    {
        "type": "agent.end",
        "sessionKey": ORCH,
        "agentId": "main",
        "messages": [U1, A1, TR1, A2, TR2, ROLLUP],
    },
    # sub-agent B: NO spawn call anywhere -> anchor-placed (anchor = A2's ts)
    {
        "type": "agent.start",
        "sessionKey": SUB_B,
        "agentId": "main",
        "prompt": BOILER + "Run the ablation sweep for setting X.",
        "promptLength": 150,
    },
    {
        "type": "tool.start",
        "toolName": "exec",
        "params": {"command": "python ablate.py"},
        "sessionKey": SUB_B,
        "agentId": "main",
        "ts": 1750000075000,
    },
    {
        "type": "tool.end",
        "toolName": "exec",
        "durationMs": 10000,
        "success": True,
        "sessionKey": SUB_B,
        "agentId": "main",
        "ts": 1750000085000,
    },
    {"type": "agent.end", "sessionKey": SUB_B, "agentId": "main"},
    # operator channel: matched (marks U2) + unmatched (own message)
    {
        "type": "message.in",
        "channel": "telegram",
        "from": "telegram:[REDACTED]",
        "content": "Also check the baselines.",
    },
    {
        "type": "message.in",
        "channel": "telegram",
        "from": "telegram:[REDACTED]",
        "content": "Please prioritise the ablation experiment.",
    },
    {
        "type": "agent.end",
        "sessionKey": ORCH,
        "agentId": "main",
        "messages": [U1, A1, TR1, A2, TR2, ROLLUP, U2, A3, COMPACTION, A4],
    },
]

out = Path(__file__).parent / "openclaw" / "mini_telemetry.jsonl"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text("".join(json.dumps(line) + "\n" for line in lines))
print(f"wrote {out} ({len(lines)} lines)")
