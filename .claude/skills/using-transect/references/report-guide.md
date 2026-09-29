# Reading a Transect report, section by section

The report is one HTML file (charts load from a CDN, so it reads
online), all sections sharing one turn axis (the same turn sits in the same pixel column in every
chart). Deep links open the Scout viewer at the exact event when a
viewer is wired. Global reading rules:

- **Provenance everywhere.** Every judged surface names its judge
  model(s) and carries confidence/agreement. `label_source` says who
  decided: `single_judge`, `majority_vote`, or `verifier`.
- **Grey means "not judged"**, never "fine"; "no data" means the
  source never recorded it, never zero.
- **Charts fit the window by default** - a long run compresses many
  turns per pixel, which reads as "squashed", not broken. The
  "zoomed-in charts (scroll horizontally)" checkbox at the top
  restores full width in a horizontal scroll box; the chart controls
  stay pinned while scrolling.
- **Turn indices are 0-based** model-turn indices, the same number in
  every chart, tooltip, card, and frame (the Scout viewer numbers
  turns from 1 - the report footer line says so).

## Analysis status

A red flag at the very top appears only when the stored scan carries
execution failures; it links to the run-wide "Scan execution & coverage"
section at the bottom (transcripts completed per scanner, recorded errors,
stored-but-unmounted scanners), which covers the whole stored scan
including epochs outside the selected report. Execution completion does
not establish usable labels: judgement quality (abstentions, filled
labels, member and verifier degradation, the unjudged-units flag) is the
reliability audit's territory. Empty sections do not establish an
absence of activity. Original review coverage is also shown next to
verifier rates in the audit.

## 0. Eval setup

The report's front matter: a one-sentence deterministic intro under
the transcript heading (template-composed from recorded facts, no
LLM prose), then three default-collapsed expandables:

- **Core setup**: model, task/sample/epoch, run date, source log,
  the agent scaffold and its full arguments as run (scaffold
  prompt, tool roster, attempts, submit, compaction, truncation,
  approval), and the verbatim initial prompts (system message +
  first task message the model actually saw) as sub-expandables.
  For Inspect `.eval` logs, two compaction cards follow: the
  summarization prompt and the save-to-memory nudge issued before
  compaction, each distinct recorded text once (per-event text is in
  `flushes`). The cards appear only when the log recorded a compaction
  (or, with none, just the configured template if one was set). Absent
  text reads "not recorded by source" - expected when the run had no
  `memory` tool (no nudge is issued) or used native provider compaction
  (no prompt is recorded). OpenClaw exports carry neither, so those
  cards do not appear.
  A recorded compaction threshold also appears here, as a token count or
  a percentage of the context window. The threshold row is absent when
  unrecorded.
- **Additional config details**: task args/file/version, the
  eval-level limits as configured, epochs + reducer, fail_on_error,
  sandbox, generation config, model roles, dataset shape, scorers.
- **Run summary**: messages, wall-clock time, total tokens, score/
  success, error, terminating limit.

Three absence wordings, deliberately distinct: "not set" means the
source log's header was read and the option was simply not
configured; "scaffold default" means the agent scaffold's arguments
were recorded but this one was left to the scaffold's own default
(the log records only configured arguments, not resolved defaults);
"not recorded by source" means the source never recorded the fact at
all - the whole config block reads that way on OpenClaw imports (no
.eval header exists). The compaction row describes the configured
setting only; the flushes on the token chart are detected from the
transcript regardless.

## 1. Phase timeline

A colored band: one rect per contiguous run of the per-turn phase
assignment. Hover for phase, turn range, confidence; click a rect to
jump to its phase card. The summary line above the chart names the
judge roster, the cohort shape, and the verifier model when one was
armed (the Sub-agent activity summary line does the same).

- Three different greys exist; when a user asks about "the grey",
  find out which one they see:
  1. a grey chunk in the band = unjudged turns (refusal / no answer /
     missing turn, or no matching phase - the tooltip names which).
     An unjudged turn splits an otherwise-judged phase on purpose.
  2. a grey cell in the agreement strip below = no vote agreement for
     that turn (single voter, a verifier re-label that superseded the
     votes, or a turn no judge saw) - the band above it can still be
     coloured.
  3. dimming from the confidence-opacity checkbox or the phase
     filter - a view state, not a data state.
  Grey means "not judged" one way only: filled and attributed turns
  (label inherited or taken from the surrounding phase) are painted
  their phase's colour - the basis says how a turn got its label,
  the paint does not.
- The **agreement strip** below, labelled "agreement" in the margin
  (only when several voters judged; a solo run states in its place
  that a single judge has no one to agree with): green cells, darker
  = higher per-turn agreement among judges; grey
  = no vote agreement (single voter, verifier re-label, or not
  judged). Its tooltip: one "member votes" row joining every member's
  own vote + confidence (members named "roll-N" on a k-roll run - the
  summary line above the chart names the model - and by model name in
  a cohort), the confidence with its provenance suffixed ("0.90
  (mean)" when a vote decided, "(verifier)" after an overturn, bare
  for a single judge), the decided label source, and the turn's basis
  (how it got its label): `judged` directly; `filled` (inherits the
  previous label); `attributed` (tool-only/failed turn); `refusal` /
  `no_answer` / `missing_turn` (unjudged). On a turn whose phase the
  verifier re-labelled, agreement reads "n/a (verifier re-label)" -
  the member ballots still show, they just no longer decide.
- Reading a long grey or pale stretch in the strip: when the band
  above stays coloured, the usual cause is one cohort member silently
  producing no answer for a stretch of turns - the turns fall to a
  single voter and read "n/a (single voter)", which is missing
  coverage, not disagreement. Confirm in the audit's Member coverage
  row (votes produced vs asked, longest missed stretch) before
  reading anything into the labels there. Grey strip runs can also
  legitimately cross phase boundaries (leading/trailing tool-only
  turns belong to a phase without being judged).
- Legend chips count phases/turns per label. Reserved buckets (ops,
  none_of_the_above) chip even at zero - an unused escape hatch is
  information.

## 2. Human interventions

Solid navy rules on their own strip - mid-run human interactions. Hover
for channel, a preview and the outcome; the expandable lists every
message in full (long ones scroll). Two shapes, and the channel names
the source - explain them when asked:

- Human-initiated, shown as "message (human)":
  - **operator**: a message steering the running agent from outside
    (inspect's operator channel: ACP remote steering, OpenClaw inbound
    messages including slash commands such as `/stop`).
  - **input**: a human typed into the session mid-run. The run's first
    input message is the task prompt itself and is deliberately not
    counted - only later ones are interventions.
- Agent-initiated, shown as "asked (agent)" (the question or tool
  call) and "answered (human)" (the reply), with the outcome on the
  header line when the source recorded one:
  - **input_event**: an ask-user / request-input question and its
    answer (accepted / declined / cancelled); a declined or cancelled
    request shows "no answer recorded". A console input recording has
    no separate question, so it shows as "recorded (human)". Older
    logs record no outcome, and stores scanned before transect 0.1.8
    carry no question or outcome at all.
  - **approval**: a tool call decided by inspect's built-in human
    approver (approve / modify / reject / escalate / terminate), with
    the explanation; a custom approver under another name is not
    counted, and a modify shows the original call.

Detection is structural (inspect's own message-source field), so
scaffold-generated user messages - handoff boundaries, react
continue-prompts - can never be miscounted as human. Cross-reference
interventions with phase changes right after them (the phase cards
have an "at/after a human intervention" filter for exactly that).

## 3. Token telemetry

Per-turn bars over a separate always-visible context-window chart.
The measure radio picks what the bars show - in the simplest terms:

- **per-turn total**: everything the turn's model call processed -
  uncached input, cache reads/writes, and output (including reasoning). "How big was
  this call."
- **per-turn new work**: a heuristic for new content: uncached input,
  output, and cache writes capped at growth in the lane's context.
  It does not measure cognitive work or dollar cost.
- **cumulative (excluding cache reads)**: uncached input, output, and
  full cache-write tokens summed over turns. Cache reads are excluded;
  token types are not price-weighted. This is not monetary spend.
  The dataframe column is named `billable`.
- **linear/log scale**: linear for comparing turns at a glance; log
  when a few huge turns flatten everything else - it makes the small
  turns readable again without hiding the big ones. Dashed red rules mark context flushes (compactions);
the flush list expandable gives each one's turn, type/source, and
tokens before -> after. A context-window sawtooth drop at a flush is
the compaction doing its job.

A recorded absolute compaction threshold appears as a dotted purple
horizontal line on the context-window chart, initially visible, with its
own legend line under the chart next to the flush legend. The checkbox
above that chart toggles it (the vertical scale follows).
The line and checkbox are absent when no token threshold is recorded.
A percentage alone stays in Core setup: the report does not look up a
model capacity to convert it. The run's total token limit and the size
observed before a flush are not substitutes for a configured threshold.

The raw ModelUsage counters ride the frame beside the derived views,
under Inspect's normalized contract (`input_tokens` excludes cache
reads/writes; `output_tokens` includes reasoning) - custom importers
must normalize to it before analysis. Missing optional breakdowns
count as zero in derived views, and a differing raw `total_tokens`
is retained rather than silently reconciled.

Flushes are detected two ways, and the type/source cells say which -
explain these when a user asks what they mean:

- **Recorded**: the source emitted an explicit compaction event;
  `type` is the compaction kind as recorded (e.g. `summary`) and
  `source` is who recorded it. When the event omits the post-flush
  size, `tokens_after` is inferred from the first non-gap turn after
  the flush and marked "(inferred)".
- **Synthesized** (`type = synthesized`, `source = context_drop`): no
  event was recorded, but a lane's context series fell below 0.6x the
  previous non-gap turn and stayed there - a compaction the source
  never reported, detected from its footprint. Trust it as "the
  window shrank here", not as a statement of how or by whom.

## 4. Sub-agent activity

Swimlanes of spawned sub-agents, one row-block per classified role,
colored by role. Rows within a block are packing, not identity: each
mark is one distinct sub-agent span, and spans stack onto a shared
row only when their extents do not overlap (saving vertical space) -
two marks on the same row are not the same sub-agent resuming. The
tooltip's lane name is the span's own identity.

Two honest drawing modes: boxes spanning first-to-last observed
activity when the source records span ends (an x marks a recorded
completion); start-ticks only when it does not (a box would fake a
duration). Tooltip per span: lane, turns, classification with its
confidence (provenance suffixed, as on the agreement strip), vote
agreement and label source, a verifier line
when the span was reviewed ("overturned (was X)" / "reviewed, not
overturned"), one joined "member votes" row, then tool calls and
busy time - "no data" where the source recorded nothing, and
reliability rows absent where the fact does not apply. Per-label
token rollups (new work, output, and tokens excluding cache reads) live on the Token spend
section's "By sub-agent label" bars' tooltip, not here. Spawn
prompts expandable shows what each sub-agent
was asked to do; label definitions expandable shows the rubric the
classifier held.

Read the labels for what they are: the classifier judges the spawn
prompt - the task the orchestrator handed over - not the sub-agent's
actual activity. So a role label states the orchestrator's intent for
the span; a sub-agent that drifted from its brief keeps the label of
the brief, undetected. To check what a sub-agent actually did, use
its lane's turns in the token timeline and the transcript deep links,
not the label. Three classification states are distinguished: not
run (grey note), joined (labels), and ran-but-joined-nothing (loud
warning - treat as a bug).

## 5. Token spend

Two bar groups from the frames' own rollups: new-work tokens by phase
label (an "unjudged" bucket holds spend no phase owns) and by
sub-agent role. When sub-agent spend cannot be computed (tool-events-
only lanes carry no usage) the report says so instead of drawing a
zero.

Computing new-work tokens per turn from frames: `phases.n_turns` counts only
reasoning-bearing turns (judged and filled), but `new_work_tokens` folds
in sub-agent-lane turns attributed into the phase - so
`new_work_tokens / n_turns` mixes different turn populations for any phase that
delegates. For the phase's own reasoning, group `phase_turns` by phase
and `basis` (reasoning-bearing vs attributed) instead of dividing by
`n_turns`.

## 6. Phase cards

One expandable card per phase, chronological - the drill-down for the
timeline. Headlines and summaries retain the complete generated text. A note on
the collapsed card says the phase is shown as one turn group and why (the
narrator supplied no groups, its groups did not line up with the phase's
turns, or no narration was available). This is separate from classification warnings. Each card: narrated headline and summary (LLM narrator
output - descriptive, not a verdict), the class-box (the judge's
classification + mean confidence in one container - neutral when
healthy, red with a warning glyph and the issue text when a
reliability issue fired), tag line (turn range, reasoning turns,
spend, tool calls, sub-agents, compaction/intervention-during-phase),
per-member judge ballots with support, turn-group excerpts from the
transcript (a turn whose text is Inspect's compaction summary is marked
as the summarizer's call, not the agent's), and a deep link. The class-box flags on: a verifier
overturn (naming the pre-overturn label), low mean confidence
(<= 0.6), or the confidently-split case (high confidence, low
agreement). When any of these surface, bring in the transect-diagnostics
skill and encourage the user to iterate - on the spec (sharper label
descriptions where the judges split) and/or the judge setup (regime,
verifier, models) - and assist them through that loop rather than
just reporting the chip.

The filter bar sorts and filters cards: by label, spend, tool calls,
confidence, sub-agent deployments, and "at/after a context flush" /
"at/after a human intervention" (the phase containing the event plus
the one right after it - the "during" tags on the card mark only the
containing phase).

## 7. Reliability and provenance audit

Always rendered, last - the provenance surface. One block each for
Phases and Sub-agents, whose text metric rows sit in a collapsed
"Run-level details" expandable: turn coverage split by basis, judge
regime, verifier state, verifier selection/outcome counts (both
surfaces are exact: the phases counts read the scanner's own audit
block, so no-verdict reviews and relabels merged away by
re-stitching are counted; the sub-agent counts are per-span rows,
spans never merge), and the regime-appropriate indicator -
including, on voting regimes, the alpha and AC1 comparison (labelled
"Chance-corrected self-consistency" on k-roll, where it is
intra-rater test-retest over the rolls, never "inter-judge").

Each block then carries up to three per-classification maps - plain
HTML tables whose cells are coloured orange (low/unhealthy) to pale
to blue (high/healthy), values always in the cell text, brackets =
Wilson 95% CIs in a smaller font, one "Labels (N = decided units)"
group header spanning the classification columns. Columns cover the
full declared vocabulary - a label with zero judgements still gets
a column (an unused label is information). Every metric row carries
a (?) hover tooltip explaining all of the cell's numbers (rate,
interval, fraction) plus the concerning threshold where one exists;
the same texts sit in a collapsed "What these rows mean" legend
under each table. In order:

- **Provenance map** (first - a provenance problem is a parent
  cause of the reliability issues below it): share (count) of each
  classification's decided units by the deciding label source, on a
  neutral intensity ramp (shares have no good/bad polarity). Rows
  show only the deciding mechanisms the run's judge setup makes
  possible - an impossible source is no row, not an empty one.
- **Reliability map**: rows for mean agreement and mean confidence
  on the units the label decided, verifier re-label rate and random
  spot-check overturns per original label, and how often the label
  appears as a minority vote.
- **Member coverage**: one row per cohort member (model, or roll-N
  on k-roll runs): units that produced a vote, with per-reason miss
  counts.

The Flags block is headed "Flags (overall scanner assessment)" -
run-level flags, distinct from the per-row thresholds the map
tooltips name. Both sections also surface their own red/amber flags
inline, so a flag is visible where the reader already is, not only
in this audit.
Flags render collapsed (headline = metric and value; explanation and
remediation unfold), grouped in the audit under two sub-headings -
"Phase segmentation and labelling" and "Sub-agent labelling" - and
unprefixed inline in their own sections. When any flag fires, the
audit's Flags block ends with a pointer at the transect-diagnostics skill
for the iteration loop (a clean audit renders "No metric here
crosses a heuristic threshold" and no pointer). The thresholds are
heuristic conventions, not validated cutoffs:

| Indicator | Bands |
|---|---|
| k-roll self-consistency (mean per-turn agreement) | amber below 0.80 and above 0.95; inspect cases, effective sampling settings and cache provenance before attributing a cause |
| Cohort agreement | Krippendorff's alpha (red below 0.66, amber from 0.66 up to 0.80, clean at 0.80 and above) shown with Gwet's AC1 - the two use different chance models and are not bounds on correctness; flags threshold on alpha |
| Verifier re-label rate | amber at or above 0.20 (overall and per original label); a same-model verifier measures self-revision under a different prompt, not a bound on independent review |
| Random spot-check overturn | red on any hit - a randomly sampled case received an applied relabel; review the source to assess correctness |
| Confidence | mean +/- 95% CI; the CI is omitted below N=8 rather than overstated |

Every count states its denominator.

The Verifier row's "off" is exact: the scanner stamps the armed
state, so off means the verifier was not armed. A cohort run says
why (`off (cohort regime: the majority vote is the correction
mechanism)` - a cohort never runs a verifier), and an armed verifier
that never triggered shows as on, with a Verifier selection row of
zero counts - armed-and-quiet, not off.

## Interpretation discipline

Before quoting any judged number, glance at the judge roster the
audit names: `mockllm/*` models mean a scripted demo or test store -
its reliability figures reflect scripted behavior, which may deliberately
include disagreement or failed reviews. Identify them as demonstrations, not
empirical judge performance. Phases and roles are judge labels with stated
confidence/agreement, headlines are narrator prose, and reliability
flags are heuristic screens - so say "the judges labelled turns 40-95
as model_development (mean confidence 0.91, alpha 1.0)", not "the
agent did model development". Where the reliability audit is amber or
red, lead with that before quoting any label it covers - and treat it
the way a flagged class-box is treated: bring in transect-diagnostics and
offer to work the spec / judge-setup iteration with the user, not
just to narrate the flag.

Intervals are descriptive calculations under sampling assumptions, not calibrated
uncertainty guarantees for dependent turns, shared batches or correlated judges.
Later phase chunks share prior-consensus hints: vote agreement is conditional on
that procedure, not agreement between independent end-to-end segmentations.
Changing a rubric to fit selected disagreements is development work; assess a
frozen revision on appropriately selected untouched material before claiming
better labels. Report confidence is the judge's stated confidence, not a verified
probability of correctness.

Narrative turn groups retain their supplied ranges only when they form a complete
partition. New scans replace invalid partitions with a neutral phase group rather
than stretching model-written descriptions to other turns. Reloading a scan
preserves its stored groups; this check does not establish narrative correctness.

`phases.narration_group_status` records `complete`, `invalid_partition`,
`empty_groups`, `no_narrative`, or `not_run`. Complete checks partition
coordinates, not factual accuracy.
