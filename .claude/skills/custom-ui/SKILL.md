---
name: custom-ui
description: >
  Use when building a custom presentation of Transect results
  outside the shipped report: a dashboard, a paper or slide figure,
  an executive one-pager, a cross-run comparison, an export into
  another document. Teaches the TransectResults data contract (frames,
  grains, join keys, judged columns) and the reliability and
  provenance rules any rendering must keep.
---

# Custom UIs over Transect results

The shipped report is the reference rendering, not the ceiling. It
shows what an honest rendering of these results looks like; this
skill is for when its layout, branding, or scope is not enough.
The one promise that carries over: the reliability and provenance
discipline travels with the data. It is a property of the results,
not of our HTML - a custom UI that drops it is misrepresenting the
scan.

Use the shipped report for the same scan as a presentation reference
(`transect.render`, guide in the using-transect skill), and check the source
for any claims the custom artifact will make. For a new evaluation, read the
using-transect skill's "Adapting to a new evaluation" section.
The shipped report can contain incomplete or mistaken judgments; its presence
does not validate a custom presentation.

## 1. The data contract

`TransectResults` (from `transect.transect` or an exact-scan `transect.load`)
carries frames and status. Source excerpts and navigation also depend on the
recorded transcript location:

- `results.frames()`: the built-in frames, plain pandas. The column
  contract for each is its module docstring; the
  [frame sources](https://github.com/AI-Safety-Institute/transect/tree/main/src/transect/frames)
  and [README diagram](https://github.com/AI-Safety-Institute/transect/blob/main/README.md)
  map the joins.
- `results.layer_frames[name]` / `results.turn_tags`: custom
  layers' frames and their tag families.
- `results.scan_location` / `results.transcripts_location`: the
  exact scan and its transcripts (Scout viewer deep links).
- `results.scan_status`: the run-wide execution record (scanned vs total
  transcripts, recorded errors). Keep its scope explicit when the UI
  filters epochs or units; judgement quality is the reliability audit's
  territory.

Frames can carry task/setup text, intervention text, spawn instructions and
custom metadata; they are not anonymized exports. Map source fields separately
to judge questions and recipient-visible content, and inspect exports for the
intended sharing scope. Preserve native values behind any display formatting.

Grains and join keys, in one table:

| grain | frame(s) | key |
|---|---|---|
| transcript | transcript_info | `transcript_id` (every frame carries the identity prefix: `sample_id`, `task_set`, `epoch`, `transcript_id`, `agent`) |
| model turn | token_timeline, phase_turns, turn_tags | `turn` (0-based; the Scout viewer numbers from 1) |
| phase | phases, turn_groups | `phase_index`; `phase` is the label |
| sub-agent span | subagents, lane_activity | `agent_span_id` |
| judge ballot | phase_turn_votes, subagent_votes, `transect.member_ballots(frame, unit_col)` | (unit, `model`, `roll`) |
| loader item / batch unit | a custom layer's `turns_frame` | `item`, plus the loader's declared facts (e.g. `turn`) |

The judged vocabulary to respect everywhere: `label_source` (who
decided: single_judge / majority_vote / verifier), `basis` (how a
turn's label arose: judged / filled / attributed / unjudged
reasons), the flattened verifier columns (`verifier_selected`,
`overturned`, `original_label`, `verifier_trigger`, ...), the raw
`members` column (per-member ballots), and `label_definitions` (the
declared rubric per surface - unused labels included).

## 2. The rules any rendering must keep

- **Failure status survives filtering.** Show run-wide incomplete execution,
  missing requested work and unassessed coverage even when an empty section
  is omitted. A completed scanner can still contain failed item judgments.
- **Provenance beside every label.** A judged label shown without
  who decided it, at what confidence, with what agreement (where
  voting ran) presents a judgement as a fact. Inline, in a tooltip,
  or one caption away - but present.
- **Denominators stated.** Agreement is voters-on-the-modal-label
  over those who voted, and the denominator shrinks when a member
  errs, refuses, or skips - write "1.00 (2 of 3 voted)", never a
  bare 1.00. Rates carry k/n and a Wilson interval.
- **Absences are honest, never filled.** Unjudged turns, `no_answer`
  units, refusals, and errors stay visible (grey, or an explicit
  count) - never dropped, and never silently renormalized away. If
  a chart covers judged units only, its caption says so.
- **The roster renders whole.** An unused label and
  `none_of_the_above` are information; hiding them inflates what
  remains.
- **Numbers come from `transect.reliability`.** Agreement coefficients
  (alpha and AC1 under different chance models), label stats, member
  coverage, re-label rates, Wilson intervals - all public, all
  keyed per transcript. Keep transcript scope explicit rather than silently
  pooling units.
- Preserve complete narration text and surface neutral-group reasons from
  `phases.narration_group_status`; do not infer acceptance or fallback from
  prose. See the report guide for the values.
- **Descriptive, not validated.** Labels and narration are an LLM
  judge's output against the user's vocabulary. Keep a framing
  sentence to that effect; the shipped report's explanation lines
  (templates under `src/transect/report/templates/notes.html.j2`) are
  copy you may reuse.
- **Grey means unjudged.** Reserve one neutral color for absence
  and never reuse it for a real label; color follows the entity
  across every view of the same artifact.

## 3. The reliability annex (mandatory for standalone artifacts)

An artifact that leaves the shipped report behind takes the audit
with it. Any standalone deliverable (a PDF, a figure bundle, an
embedded page) must include a compact annex stating:

- judge regime and roster (`transect.reliability.detect_regime`), and
  the verifier's model and arming state;
- unit coverage: judged / filled / unjudged counts per surface;
- agreement and confidence with their Ns (and the ± only where
  N supports it);
- verifier selection and outcomes (triggered, spot-checked,
  overturned);
- any fired reliability flags;
- run-wide execution and coverage from `results.scan_status`, distinguished
  from the artifact's selected units and any unresolved custom coverage;
- provenance of the run itself: Transect and its version
  (`transect.__version__`), the scan location or id, and when it ran.

The shipped report's "Reliability & provenance audit" section is
the reference layout for this annex.

## 4. Attribution

A custom UI must reference Transect as the source of the analysis - at
minimum "Produced with Transect <version>" (plus the scan id where
provenance matters).

## 5. What to reuse from the package

Public, stable, use freely: `results.frames()` and the frame
column contracts, `transect.reliability` (every statistic the audit
shows), `transect.member_ballots`, `transect.judge_identity`, `transect.load` /
`transect.render`.

Private but readable: the report's own presentation logic. The following
`src/` paths require a matching source checkout; they are not installed skill
resources. Browse the public
[report sources](https://github.com/AI-Safety-Institute/transect/tree/main/src/transect/report)
or inspect the installed Python modules. They carry no stability contract;
decide whether to import or copy deliberately:

- `src/transect/report/reliability.py`: the flag thresholds and wording
  (`build_flags`), `confidence_tiers`, `describe_regime`,
  `abstention_counts`, `format_confidence` - reuse these rather
  than re-deriving "amber below 0.80" by hand.
- `src/transect/report/colors.py`: label/phase color assignment
  (`_label_colors`, `_phase_colors`), the unjudged grey
  (`_UNJUDGED_GREY`), the audit cell ramps (`health_color`,
  `share_color`), chip tints.
- `src/transect/report/display.py`: humanized vocabularies -
  `MEMBER_NO_VOTE` (why a member produced no vote),
  `member_display` / `roster` / `judge_label`.
- `src/transect/report/sections.py` + `templates/`: the reference
  wording for explanations, legends, and the audit.

## 6. Ship checklist

Before delivering a custom artifact:

- run-wide failures and unknown coverage remain visible after filtering;
- every judged label has provenance within reach;
- every agreement/rate states its denominator and N;
- unjudged/no_answer mass is visible or explicitly counted;
- the full declared roster appears (unused labels included);
- statistics computed via `transect.reliability`, keyed per transcript;
- the descriptive-not-validated framing sentence is present;
- the reliability annex is embedded (standalone artifacts);
- Transect is referenced with its version;
- rendered values and source links have been checked at their actual
  destinations, including Boolean/status cells and events with similar text.

A working file link, populated chart or installed skill directory does not
establish native agent discovery, source alignment or label correctness.
Verify those properties separately; preserve any workaround and its provenance.

Anti-patterns, each a misrepresentation: a label bar chart with no
coverage statement; confidence averaged across pooled transcripts;
`none_of_the_above` dropped "for clarity"; unjudged turns recolored
or omitted; a bare 1.00 agreement off a shrunk denominator; the
narrator's headlines quoted as what happened.
