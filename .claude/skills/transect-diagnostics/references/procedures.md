# Diagnostic procedures and recipes

All of these are $0 post-passes over an existing scan store, except
the scrambled-vocabulary check, which compares paid conditions and repeats. Load
frames with `f = transect.load(exact_scan_location).frames()`. The snippets below
assume one transcript/sample/epoch; filter to it first, or add transcript identity
to every grouping and join. New judged runs require per-call sizing and approval.

## Find the disagreement (before prescribing anything)

These vote-based recipes need a voting regime: on a solo run
`phase_turn_votes` is empty and they return nothing - there, the
drill-down surface is the verifier columns on `phases` (below).

Split turns and each member's ballot on them:

```python
t = f["phase_turns"]
split = t[(t.basis == "judged") & (t.judge_agreement < 1.0)]
v = f["phase_turn_votes"]
ballots = v[v.turn.isin(split.turn)].sort_values(["turn", "model", "roll"])
ballots[["turn", "model", "roll", "phase", "confidence"]]
```

Disagreeing label pairs: which labels members assign to the same turn.
Use this breakdown to select cases for source review:

```python
import itertools
from collections import Counter

pairs = Counter()
for turn, grp in v[v.phase.notna()].groupby("turn"):
    for a, b in itertools.combinations(sorted(set(grp.phase)), 2):
        pairs[(a, b)] += 1
pairs.most_common()  # [(('baseline', 'model_dev'), 12), ...]
```

Per-member dissent: who disagrees with the decided labels, and how often.
Read individual ballots before attributing a low alpha to a particular member.
Filter to actual votes first: a member's
refusals/no-answers carry `phase = NA`, and NA != decided would
count a miss as dissent (coverage is the Member coverage rows' job,
not this one's):

```python
decided = t[["turn", "phase"]].rename(columns={"phase": "decided"})
j = v[v.phase.notna()].merge(decided, on="turn")
j.assign(dissent=j.phase != j.decided).groupby(
    ["model", "roll"]
).dissent.mean().sort_values(ascending=False)  # dissent rate per member
```

On a verifier-armed run the decided labels include verifier
overturns, so every roll shows "dissent" on an overturned phase's
turns even where the rolls agreed with each other - read those rows
as the verifier's doing, not a member's.

Use the most frequent disagreeing pair to select source examples for review.
If those examples reveal an unclear distinction, revise the two labels'
descriptions; disagreement alone does not establish a rubric defect.
Sub-agent equivalents: `subagent_votes` grouped by `agent_span_id`.

Disagreement across the same kind of turn, such as delegation or tool errors,
can suggest an attribution convention. Read those cases and compare fresh
responses from the unchanged setup before attributing the pattern. If the
evaluation needs an explicit convention, record it as a Spec decision in phase
`context` and test it on development cases.

## Reading alpha and AC1

The audit reports percent agreement, Krippendorff's alpha and Gwet's AC1.
The chance-corrected coefficients use different chance models; they are not
bounds on a latent true agreement. Differences can reflect label prevalence and
weighting. Read coverage, label distributions and the actual disagreements before
attributing a cause. Flag thresholds are reporting conventions, not validity tests.

## Verifier signal, split by source

```python
from transect.reliability import review_units
reviews = review_units(f["phases"])
reviews[["original_phase_index", "original_label", "verifier_label",
         "verifier_completed", "overturned", "verifier_trigger"]]
```

These are original review units, which can outnumber the final merged phases.
`verifier_selected` means a review record exists (failed attempts included);
`verifier_completed` requires a usable verdict - relabel rates condition on
completed verdicts and are not accuracy estimates. The phase row's singular
`verifier` is a representative projection for display, never the population.

Small-N honesty: with a handful examined, the Wilson interval on the
re-label rate spans most of [0, 1] - say "too few examined to
conclude", not the point estimate.

## Before/after comparison (any knob change)

Scan the changed setup into a fresh `scans_dir`. Then, before reading
any number, **prove the change reached the run** - the rubric is
recorded in each scan, so diff it:

```python
a = transect.load("scans_v1/").frames()
b = transect.load("scans_v2/").frames()
keys = ["surface", "label"]
d = a["label_definitions"][keys + ["description"]].merge(
    b["label_definitions"][keys + ["description"]],
    on=keys,
    suffixes=("_v1", "_v2"),
    how="outer",
)
changed = d[d.description_v1 != d.description_v2]
```

This compares recorded label descriptions only: changes to context, prompts,
models or runtime settings require their own provenance checks. If a planned
description change is absent, inspect which Spec the runner loaded. Identical
ballots do not establish cache replay; changed prompts can yield identical labels.

Then diff the outcome:

```python
m = a["phase_turns"][["turn", "phase"]].merge(
    b["phase_turns"][["turn", "phase"]], on="turn", suffixes=("_v1", "_v2")
)
m = m.dropna(subset=["phase_v1", "phase_v2"])  # unjudged turns are
# not label changes: NaN == NaN is False and would deflate the overlap
(m.phase_v1 == m.phase_v2).mean()  # per-turn label overlap
```

(One sample/epoch per store here; when comparing across samples or
epochs, merge on the identity prefix + `turn`, not `turn` alone.)

Also compare: agreement distributions (`judge_agreement.describe()`),
re-label rates, and which turns changed label (`m[m.phase_v1 !=
m.phase_v2]`) - then read those turns to judge whether the change
is source-supported. Cases used in this loop are development cases. Freeze a
revision and assess appropriately selected untouched cases before generalizing;
record exposure and separate chosen conventions from corrected errors.

## Scrambled-vocabulary check (manual procedure)

Purpose: measure sensitivity to label names under a changed prompt. The
helper preserves descriptions/context byte-for-byte, including any old label-name
references: inspect the resulting prompt for coherence before using the comparison.

1. Create and retain the mapping:

   ```python
   from transect.diagnostics import scramble_spec
   scrambled, mapping = scramble_spec(spec, seed=7)
   ```

2. Size each condition's actual calls, including unchanged-setup repeats, and
   obtain required spending approval. Use separate stores and fresh local response
   caches where new draws are intended; unchanged calls may otherwise replay.
3. Restore original names before matching units:

   ```python
   from transect.diagnostics import descramble
   sb = transect.load(exact_scrambled_scan_location).frames()["phase_turns"]
   back = descramble(sb.phase, mapping["phases"])
   ```

4. Compare aligned labels, coverage and their distributions. Account for ordinary
   repeat variation, separate category-order controls, prompt coherence and prevalence. High
   overlap shows stability under this intervention, not proven definition-reading;
   low overlap shows sensitivity, not proven name anchoring. A uniform `1/K`
   baseline is inappropriate unless its assumptions hold.
5. Mark outputs as diagnostic: no automatic banner is added. Retain the original
   mapping, code and stores; do not quote neutral labels as transcript findings.
