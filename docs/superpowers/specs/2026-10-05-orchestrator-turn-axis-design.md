# Orchestrator turn axis: design

Resolves issue #50 (phase axis carries main-agent-lane turns only;
sub-agents as overlaid swimlanes) and, through it, issue #49
(phase-card excerpt rows include sub-agent turns). Issue #6 (the
total-coverage projection painting turns outside declared ranges) is
adjacent and stays separate.

Decisions recorded here were taken in conversation on 2026-10-05. No
backward compatibility with existing scan stores is kept: the package
is pre-release and the committed fixture stores are regenerated.

## 1. The contract

**`turn` means orchestrator turn, everywhere.** The orchestrator (main
lane) is the span Scout's `timeline_messages(..., depth=1)` would scan:
the outermost non-utility span with at least one direct model event,
container spans being transparent. `turn` is the 0-based ordinal of
that span's model events with output, in event order. Nothing else is
called a turn: not an all-lane index, not a message index. The old
all-lane event-order index is dropped entirely; the only link to the
Scout viewer is the per-phase deep link by event uuid, which is
unchanged.

The contract holds at every level at once:

- the stored scan values of every built-in scanner;
- the frames (`token_timeline.turn`, `phase_turns.turn`,
  `phases.turn_start/turn_end`, `turn_groups`, `phase_turn_votes.turn`,
  `flushes.turn`, `interventions.turn`, `lane_activity.turn`,
  `turn_tags.turn`, a custom layer's `turn`);
- the judge-facing digest indices (segmentation, verifier, narrator),
  which become contiguous orchestrator ordinals;
- every chart's x axis, every card's turn range, every excerpt row.

Sub-agent model calls have no `turn`. They carry `agent_span_id`,
`agent_lane`, and `lane_turn` (0-based ordinal within their own lane),
and their placement on the orchestrator axis is a derived position
(section 3).

### Main-lane resolution

`scanners.helpers.main_span` keeps its walk and its docstring cites the
Scout definition above. The id wrapper no longer swallows exceptions. A
transcript whose resolved span holds no model events, or whose
container has two or more non-utility agent children with model
events, has no single orchestrator: the scanner raises a `ValueError`
naming the candidate spans, and the error surfaces per transcript in
the scan status section like any other scanner error. No fallback, no
stamp.

### The public helper

`transect.orchestrator_turns(transcript)` yields
`(turn, model_event, tool_calls)` for the orchestrator's model turns,
and is the one sanctioned way for a custom mechanical scanner to number
turns. It resolves the main lane itself (building the timeline when the
store carries none), so a user's `@scanner` content declaration needs
`events=["model", "span_begin", "span_end", "tool"]` and nothing else.
The existing all-lane iterator stays internal, renamed to say so, and
is used only where a scanner must see every lane (token usage, sub-agent
activity).

## 2. Stored values (scanners)

### `token_timeline`

`value["timeline"]`: one entry per model turn in any lane, in event
order. Per entry: `turn` (orchestrator ordinal, or `None` on a
sub-agent turn), `lane_turn`, `agent_lane`, `agent_span_id`,
`n_tool_calls`, `timestamp` and `completed` (ISO strings, `None` when
the event records none), and the usage fields as today.

`value["spans"]`: one entry per sub-agent span (the shared sub-agent
definition in `helpers.subagent_span_begins`), replacing `span_ends`:

- `agent_span_id`, `agent_lane`;
- `spawn_turn`: the orchestrator turn preceding the span's `span_begin`
  in event order (`-1` when none precedes it, clamped to 0 by the
  frame). On Inspect handoffs and linked OpenClaw spawns this is the
  turn that issued the spawn; on unlinked OpenClaw sessions it is the
  importer's file-order anchor. One rule, no linkage logic;
- `first_at`, `last_at`: timestamps of the span's first and last model
  or tool event (`None` when unrecorded);
- `end_at`, `end_recorded`: the `span_end` event's timestamp and
  whether the source recorded the end (OpenClaw synthesises ends at
  last activity, so `end_recorded` is false there, as today);
- `event_order_end_turn`: the orchestrator turn preceding the
  `span_end` event, the fallback end when timestamps are missing.

`value["lane_activity"]`: as today, with `turn` now the orchestrator
turn preceding the tool event in event order. The frame docstring's
claim that it is a main-lane turn becomes true.

### `decision_phases`

`turn_digests` enumerates orchestrator turns, so digest indices are
contiguous orchestrator ordinals; delegation lines still fold into the
last digest-eligible turn at or before the span's spawn turn. `n_turns`
is the orchestrator count and the dense `turns` surface covers exactly
those. No judge prompt changes beyond the indices. Verifier item ids
(`:turn-{n}`) renumber with the digests.

### `context_flush`

`turn` counts orchestrator turns preceding the event. Each entry gains
`agent_span_id` (`None` on the orchestrator): a compaction recorded
inside a sub-agent's conversation is kept in the frame rather than
dropped, and is distinguishable.

### `human_intervention`

The first-seen map is built over orchestrator turns only. The
message-thread walk advances the axis only on assistant messages that
are not a sub-agent model event's output (an Inspect handoff appends
the sub-agent's messages to the parent thread). Input and approval
events anchor by orchestrator count. The rule reads as before: the
first orchestrator turn whose input saw the message.

### `reasoning_turns`, `subagent_classification`

No code change. The loader inherits the renumbered digests; the span
classifier is keyed by span id and never touched the axis.

## 3. Positions on the orchestrator axis (frames)

Orchestrator turn `m` occupies the cell `[m - 0.5, m + 0.5]`, which
stands for the wall-clock interval from the start of model call `m` to
the start of call `m + 1` (the call plus the tool execution that
followed it). For a timestamp `t` with `t_m <= t < t_(m+1)`:

```
pos(t) = m - 0.5 + (t - t_m) / (t_(m+1) - t_m)
```

The last cell's right edge is the last call's `completed` time when
recorded, else its `timestamp` plus the median cell width. A time
before the first call clamps to `-0.5`; a time after the last cell
clamps to the axis's right edge, and the span's tooltip says activity
continued after the last orchestrator turn. A zero-width cell (two
calls at one instant) is skipped: a time at that instant belongs to the
next non-empty cell. Wall-clock `timestamp` is used, never Inspect's
working time.

A span's derived coordinates, computed in `subagents_df` from the
`spans` record and the orchestrator rows' timestamps:

- `start_pos = pos(first_at)`, `end_pos = pos(end_at if end_recorded
  else last_at)`, floats;
- `anchor_turn`, `end_turn`: the cells holding those positions, ints;
- `position_source`: `"timestamp"` when every needed timestamp exists,
  else `"event_order"`, in which case `start_pos = end_pos =
  anchor_turn = spawn_turn` and `end_turn = event_order_end_turn`.

Phase membership for the per-phase sub-agent count and the delegated
spend uses `spawn_turn`, never the timestamp anchor, so the two chips
describe the same population and a span is a member of exactly one
phase.

## 4. Frames

`SCHEMA_VERSION` bumps. Changes per frame (full contracts in the
module docstrings):

- **token_timeline**: `turn` becomes nullable `Int64`; adds `lane_turn`
  (`Int64`) and `timestamp` (`string`, ISO). Derived token views stay
  per lane. Frame order is event order.
- **flushes**: adds `agent_span_id`. The `tokens_after` inference reads
  orchestrator rows (`agent_span_id` null), not the most frequent lane.
  Synthesized drop detection runs on the orchestrator context series
  only; the per-lane scan goes away with the mixed series.
- **interventions**: schema unchanged.
- **lane_activity**: schema unchanged; `turn` semantics as in section 2.
- **phase_turns**, **turn_groups**, **phase_turn_votes**: schema
  unchanged, orchestrator coordinates.
- **phases**: `new_work_tokens` becomes orchestrator-only (the dense
  map over orchestrator turns joined to orchestrator token rows). Adds
  `delegated_new_work_tokens` (sum of `subagents.new_work` over spans
  whose `spawn_turn` lies in the phase's range; `NA` when no span
  carries usage, never 0 for tool-only lanes) and `n_subagents`
  (spans spawned in range). `phases_df` therefore takes the subagents
  structural spine as an input; `api.transect`/`load` build the
  subagents frame first.
- **subagents**: `span_start_turn`, `span_last_turn`, `span_end_turn`,
  `span_end_recorded` are replaced by `spawn_turn`, `anchor_turn`,
  `end_turn`, `start_pos`, `end_pos`, `position_source`,
  `end_recorded`; `started_at` stays and `ended_at` is added. Token
  rollups and the judged half are unchanged.
- **turn_tags** and a custom layer's `turns_frame`: unchanged schema,
  orchestrator `turn`.

## 5. Report

- **x axis**: `n_turns` is the orchestrator count (`token_timeline.turn`
  max plus one over non-null rows). Shared width and domain as today.
- **Phase band and agreement strip**: unchanged code paths over the
  renumbered frames. The band tiles orchestrator turns; sub-agent turns
  no longer exist on it. (The "attributed" basis remains for tool-only
  orchestrator turns; its colour is issue #6's business.)
- **Token telemetry**: bars and the context-window step read
  orchestrator rows only; flush rules draw orchestrator flushes only.
  Delegated spend has its homes in the per-label spend chart and the
  swimlane tooltips.
- **Sub-agent activity**: stays where it is in the page order. The
  swimlanes draw from `subagents` (`start_pos`..`end_pos`), packed by
  the existing greedy layout over float extents, one row block per
  classification label as today. Drawing mode follows
  `position_source`: a box for timestamp-placed spans, with the `×` end
  marker when `end_recorded`; the start tick for event-order spans. The
  notes text says what a box is: the span's wall-clock extent mapped
  onto the orchestrator turns active at the time, not a turn count.
  Row labels gain the block's peak concurrency.
- **Phase cards**: excerpt rows are orchestrator turns only (the
  excerpt reader enumerates `orchestrator_turns`), the lane label is
  dropped from rows, and "N more" counts orchestrator turns. Tags:
  `turns a–b`, `N reasoning turn(s)`, `N orchestrator new-work tokens`,
  `N delegated new-work tokens (sub-agents spawned here)` (omitted when
  `NA`), `N tool call(s)` (orchestrator turns), `N sub-agent(s)
  spawned`, compaction, intervention, agreement. The card's
  `data-spend` sort attribute is the orchestrator figure.
- **Token spend by phase**: each phase bar carries two segments,
  orchestrator and delegated, with a legend, so the chart and the chips
  never disagree. The "unjudged" bucket stays orchestrator-only.
- **Custom layers**: `SectionContext.n_turns` is the orchestrator count.
  `TurnBand`, `TurnChart`, `EventMarks` and the tag families raise a
  `ValueError` naming the layer and the offending values when a frame
  carries a `turn` outside `0..n_turns-1`, the signature of a scanner
  that enumerated every lane.

## 6. Tests and fixtures

- **New fixture**: `tests/fixtures/logs/<stamp>_parallel-subagents_*.eval`
  from `tests/fixtures/generate_parallel_eval.py`, a mockllm run of
  `deepagent(background=True)` with two sub-agents dispatched in the
  background while the orchestrator keeps working, outputs routed by
  content (the demo generator's pattern) so concurrency cannot
  reorder the script. Event timestamps are rewritten to a
  deterministic schedule (the demo generator's `stretch_wallclock`
  pattern) so the two sub-agents overlap each other and several
  orchestrator turns. Test-only; the examples keep the demo log.
- **Regenerated**: `tests/fixtures/demo_scan` and
  `tests/fixtures/demo_scan_cohort` (value schema change).
- **Pinned by tests** (parametrised, one sentence each): the
  orchestrator enumeration on the demo log and the parallel fixture;
  `pos(t)` edge cases (inside a cell, before first, after last,
  zero-width, missing timestamp fallback); renumbered digests and dense
  turns; flushes' lane column and orchestrator-only inference and
  synthesis; interventions on a thread with sub-agent assistant
  messages; phases' two rollups and `n_subagents` by spawn turn;
  subagents' coordinates on both fixtures; excerpt rows orchestrator
  only; the custom-layer out-of-range failure; main-lane resolution
  raising on an ambiguous tree.
- **Browser check**: `transect()` on the demo log and on the parallel
  fixture with no judges, plus `load()` + `render()` of the regenerated
  stores, per the AGENTS.md report rules.

## 7. Docs

AGENTS.md gains a load-bearing contract bullet for the orchestrator
axis; README's report walkthrough, frames list and mermaid diagram
follow the new columns and joins; `add-a-layer` documents
`orchestrator_turns` as the only turn source and the fail-loud rule;
`custom-ui`'s grain table loses the viewer-numbering remark;
`using-transect` is checked for sub-agent-turn wording. Frame module
docstrings are the column contracts and change with the columns.

## 8. Delivery

Stacked PRs, each shippable, in this order:

1. `feat: orchestrator turns are the turn axis` - sections 1 to 4, the
   fixture regeneration, the new parallel fixture, the report changes
   needed to keep it rendering (axis count, orchestrator-only
   telemetry, swimlanes from the new columns in box/tick mode), docs.
2. `feat: sub-agent swimlanes placed by wall-clock on the orchestrator
   axis` - section 5's swimlane drawing and notes, peak concurrency
   labels, browser-verified on both fixtures.
3. `feat: custom layers number orchestrator turns` - the public helper,
   block validation, skill updates.
4. `feat: phase cards show orchestrator work and spawned sub-agents` -
   section 5's cards and the two-segment spend chart (closes #49).

Each PR bumps `pyproject.toml`; the first is a minor bump (frame
contract change), the rest patch. Issue #6 follows on its own.

## Out of scope

Scout-style same-name envelopes with a count badge inside a label
block (the packing already bounds height at peak concurrency); per-row
viewer deep links on excerpt rows; markers for sub-agent compactions on
swimlane bars; any change to the OpenClaw importer (timestamps make its
block placement irrelevant to positioning).
