---
name: transect-diagnostics
description: >
  Use when a Transect report's reliability audit flags amber/red or a
  phase card's class-box turns red, when a user asks how much to
  trust judged labels, or when iterating the judge setup (regime,
  k_rolls, verifier, models) or the spec's rubric off reliability
  signals: judge self-consistency, cohort agreement, a same-model
  verifier, verifier re-label rates, disagreement drill-downs,
  scrambled-vocabulary checks. The sibling using-transect skill covers
  running the default path and reading the report; this skill owns
  the metrics-to-knobs iteration loop.
---

# Transect diagnostics: from reliability signals to setup changes

The job: a user has a Transect run whose reliability surface (the
report's audit section, the per-card class-boxes, or the frames) says
something is off - or suspiciously perfect - and you help them decide
what to change: the spec's rubric, the judge setup, or nothing.
Ground every step in this run's own numbers (load the frames), walk
the remediation ladder for the setup they ran, and offer to make the
change and re-run rather than just narrating the flag.

Two framing rules, before any number:

- **A high score is a prompt to check, not a green light.** The
  common failure is correlated bias: judges leaning on the same label
  wording agree strongly and are wrong together. Reliability means
  repeatable; provenance means traceable; neither means correct -
  correctness needs expert labels, which nothing here supplies.
- **Thresholds are conventions, not validated cutoffs.** Treat
  0.8/0.66/0.2 as conversation starters.
- **Check the roster first**: a `mockllm/*` judge means a scripted
  demo or test store - its reliability figures are fabricated by
  construction; diagnose mechanics, not quality.

## 1. Identify the setup, pick the ladder

The audit's "Judge regime" row (or the scanner-stamped
`phases.judge_regime` / `judge_models` / `n_models` / `k_rolls`
columns in the frames) tells you which indicator applies. Setups can compose: a k-roll run with the verifier on gets
both the self-consistency and re-label indicators. A cohort never
has a verifier (the majority vote is its correction mechanism), so
cohort runs show the agreement indicator alone.

| Setup | Indicator | Healthy |
|---|---|---|
| solo + verifier (the default) | verifier re-label rate (self-revision disclosure) | < 0.20 per original label - but read the interval before the point estimate; at small N it spans most of [0, 1] |
| one model, `k_rolls` > 1 | k-roll self-consistency (mean per-turn agreement) | >= 0.80, and not suspiciously ~1.0 |
| >= 2 distinct models | cohort agreement (Krippendorff's alpha) | 0.80 and above; amber from 0.66 up to 0.80; red below 0.66 |

## 2. The remediation ladders

**Solo + verifier (re-label rate):**

Reading "examined N, overturned 0" as a composite - the most common
trust question: it establishes the labels are traceable and lightly
self-checked on N phases; with a same-model verifier it is
self-review (same blind spots); at small N the interval says nothing
("0/1, CI 0-79%" is consistent with flawless and with
wrong-four-in-five); and even a clean pass at scale means repeatable,
never correct.

1. Rate very low with `verifier_model` = judge model: that is a
   self-consistency lower bound, not an independent check (same
   blind spots) - the cheapest upgrade is a different
   `verifier_model` and re-run (re-spends only the verifier subset).
   Pick the verifier at least as capable as the judge: a stronger
   model, or a peer-capability model from a different provider (a
   weaker verifier mostly adds noise to the labels it reviews).
   Read the upgraded run as: the same non-overturns, now an
   independent second opinion - the interval is unchanged, so if it
   was too thin to conclude before, it still is; the upgrade changes
   the evidence's kind, not its amount.
2. Intervals very wide, or several labels hovering at the threshold:
   the verifier only saw doubtful phases + the random sample
   (default max(5%, 3), capped by the phase count - a 1-phase run
   examines 1) - too few to conclude. Raise `verify_sample` and
   re-run: it is the share of judged units spot-checked (1.0 =
   review everything); the judge pass replays from cache, so the
   extra spend is the verifier's. For spans that is exactly the
   newly sampled calls (one review call per span); for phases the
   verifier reviews in chunks, so newly sampled phases recompose
   the chunk prompts and previously reviewed phases can re-spend
   with them. It applies to sub-agent spans too (per-span
   probability, same default 5%).
3. Rate >= 0.20 overall or for a label: tighten the rubric, starting
   with the worst per-label row in the audit - sharpen that label's
   `description` in the spec, then re-run.
4. Any random spot-check overturn (red): confident mislabelling -
   treat this run's labels with caution and re-run after the rubric
   fix rather than reading past it.

**K-roll (self-consistency):**
1. Below 0.80: check the judge's sampling settings first (see
   "Tuning the judge's generation settings" below) - lower
   temperature/top_p where the model exposes them; on reasoning
   models, which often fix or ignore temperature, the dial is
   `reasoning_effort` / `reasoning_tokens` (more deliberation is
   usually more consistent). If inconsistency remains, tighten the
   rubric where the per-turn votes split (see the drill-down
   recipes).
2. At temperature 0 the indicator is vacuous - rolls near-replay, so
   agreement ~1.0 is a tautology, not health (the scanner warns at
   setup time when the judge is a Model whose config pins
   temperature 0; a provider default of 0 is not visible to it).
   Say so; do not report it as a green light.
3. Suspiciously high (> 0.95 at temperature > 0): run the
   scrambled-vocabulary check
   ([references/procedures.md](references/procedures.md)) -
   extreme self-consistency can be name-anchoring or a plumbing
   artifact, not quality.

**Cohort (alpha):**
1. Red/amber: find *which* label drives the disagreement (per-label
   breakdown recipe in the reference) and sharpen that label's
   description - definitional vagueness is the usual cause; then
   re-run.
2. Widening the cohort helps only when the interval is wide, and
   past ~5 judges it stops helping - fix definitions before adding
   judges.
3. High alpha: worth one scrambled-vocabulary check before leaning
   on it (rule out shared name-anchoring).
4. The audit prints alpha and Gwet's AC1 together - read the
   bracket: both >= 0.8 = healthy (still
   run the high-side check above); both low or negative =
   disagreement is real - the one-label-dominates excuse for a low
   alpha does not apply; bracket straddling the bands = prevalence
   skew is in play - trust the per-label pair breakdown over either
   summary number.
5. Scale honesty: on a short run with few labels, a dramatic
   negative alpha usually just means one systematic dissenter - go
   straight to the per-member dissent recipe rather than reading the
   magnitude.

### Tuning the judge's generation settings

Generation settings ride on the judge `Model` object - build it with
a `GenerateConfig` and hand it to `transect()`; Transect treats the judge's
own config as authoritative and only layers cache scoping on top:

```python
from inspect_ai.model import GenerateConfig, get_model
import transect

judge = get_model(
    "anthropic/claude-sonnet-4-6",
    config=GenerateConfig(temperature=0.3),  # classic sampler
    # config=GenerateConfig(reasoning_effort="high"),  # reasoning model
)
results = transect.transect(..., judge_models=judge, k_rolls=3)
```

Which dial exists depends on the model: classic samplers expose
`temperature` / `top_p` / `top_k` / `seed`; frontier reasoning models
often fix or ignore temperature and tune with `reasoning_effort` or
`reasoning_tokens` instead. Passing a setting the model ignores is
silent - verify the knob applies to the model in hand before
attributing a consistency change to it.

## 3. Drill down before changing anything

Locate the disagreement before prescribing: the reference file's
recipes find the split turns, the confusable label pairs, each
member's ballots, and the per-member dissent rates; then read those
turns' excerpts / deep links to see *why* the judges split. A rubric
fix written against actual split turns beats a blind rewording.
Solo runs have no votes (`phase_turn_votes` is empty and the recipes
above return nothing) - there the only drill-down surface is the
verifier columns on `phases`.

## 4. Iteration mechanics (cost-aware)

- Spec/rubric text changes alter the judge prompts: every judged call
  misses cache and re-spends. Scan into a fresh `scans_dir`; keep the
  previous store for before/after comparison (load both, diff the frames).
- The judge cache is machine-global (inspect-ai's on-disk cache under
  `XDG_CACHE_HOME`, shared across projects and venvs). Cached calls
  are replays, not independent draws - a stability measurement needs
  a fresh `XDG_CACHE_HOME` or `cache=False`, or it re-reads old
  responses while looking like a re-run.
- Adding cohort members or rolls: new members/rolls hit the API.
  Existing sub-agent members replay from their roll-scoped cache
  (per-span, independent calls). Phase members only partly replay:
  each chunk's prompt embeds the running consensus, so wherever a
  new member flips a chunk's consensus tail, every member's later
  chunks re-spend. Budget for partial re-spend on multi-chunk phase
  runs.
- Changing only `verifier_model`: re-spends only the verifier's
  subset, not the whole pass.
- Compare runs on the frames, not the HTML: per-turn label overlap,
  agreement distributions, re-label rates before vs after.
- **Before reading any before/after number, prove the change reached
  the run**: diff the two stores' `label_definitions` frames (the
  rubric is recorded in the scan at scan time). Identical rubric +
  identical ballots means the judged calls replayed from the model
  cache (or the runner was handed the old spec path) - that is a
  wiring failure, not a null result of the fix. Recipe in the
  reference.

## 5. What is not built (say so, don't improvise)

- The per-label / per-member *agreement* breakdowns (confusable
  pairs, dissent rates) are not in the audit - compute them with
  the reference recipes. The audit does carry per-label re-label
  rows, member coverage rows, percent agreement, alpha, and AC1.
- The scrambled-vocabulary check has helpers
  (`transect.diagnostics.scramble_spec` / `descramble`) but no `scramble`
  config and no diagnostic banner: a scrambled run's report must
  never be presented as results - labels there are deliberately
  meaningless. Say it explicitly; quarantine the store.
- The interval on the re-label rate is Wilson, a stand-in for
  Clopper-Pearson.
- A planted-wrong-label catch-rate harness (the confirmatory
  experiment for verifier responsiveness) does not exist yet
  (roadmapped).
