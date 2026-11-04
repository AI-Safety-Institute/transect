# Synthetic nested Petri audit

`nested_audit.json` is a readable Inspect eval log with one sample and epoch.
It contains only synthetic content from the repository's public, scripted
bakery pricing audit plus a synthetic seed-design wrapper. No private
transcripts, credentials, provider requests or real-world actions are used.

The outer writer makes a tool-calling seed-design turn, launches an inner
audit, receives separate classifier feedback, and writes its final review:

```text
outer-solver (solver)
  seed-writer (writer): 2 contestant model turns, including run_trial
  trial-launch (launch)
    audit-loop (audit_loop)
      auditor: 6 model turns
      target: 4 model turns over 2 trajectories, with 1 rollback
    feedback (launch): 1 classifier model turn
```

Parent IDs record ownership; the writer's span remains open while the trial
runs. Custom wrapper types exercise Inspect's flattened timeline shape.
Stored timelines use sample-prefixed names (`nested-audit-target` and
`nested-audit-auditor`). The final sample messages belong to the outer
writer, so the target's surviving path is deliberately unavailable; its
recorded branches and rollback remain available.

Regenerate from the committed public demo without model calls:

```bash
uv run python tests/fixtures/generate_nested_petri_eval.py
```

The lengthy auditor system prompt and repeated tool schemas are replaced with
a short synthetic prompt and empty schemas; tool calls, results, message IDs,
branch anchors and the remaining conversation history are retained.

Regeneration is byte-for-byte stable against the committed source fixture.
If that source is intentionally regenerated, regenerate this fixture too.

Read it through Inspect like any Inspect JSON log:

```python
from inspect_ai.log import read_eval_log

log = read_eval_log("tests/fixtures/nested_petri/nested_audit.json")
```

For Scout ingestion, first write a compressed copy because its log discovery
expects the `.eval` suffix. This leaves the text fixture unchanged:

```python
from inspect_ai.log import read_eval_log, write_eval_log

log = read_eval_log("tests/fixtures/nested_petri/nested_audit.json")
write_eval_log(log, "/tmp/nested_audit.eval")
```
