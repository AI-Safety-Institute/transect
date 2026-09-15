# Diagnostic procedures and recipes

All of these are $0 post-passes over an existing scan store, except
the scrambled-vocabulary check, which is one extra judged run. Load
frames with `f = transect.load(scans_dir).frames()`.

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

Confusable label pairs - which two labels the members trade on the
same turn (the per-label breakdown that points at the vague
definition):

```python
import itertools
from collections import Counter

pairs = Counter()
for turn, grp in v[v.phase.notna()].groupby("turn"):
    for a, b in itertools.combinations(sorted(set(grp.phase)), 2):
        pairs[(a, b)] += 1
pairs.most_common()  # [(('baseline', 'model_dev'), 12), ...]
```

Per-member dissent - who disagrees with the decided labels, and how
often (a dramatic negative alpha on a short two-label run is usually
one systematic dissenter). Filter to actual votes first: a member's
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

The top pair names the rubric sentence to sharpen: rewrite those two
labels' `description`s to explicitly discriminate the cases the
judges traded on (read the split turns' excerpts / deep links first).
Sub-agent equivalents: `subagent_votes` grouped by `agent_span_id`.

When sharpening won't help: attribution conventions. If one member
dissents across turns of the same *kind* (every delegation/handoff
utterance, every tool-error turn) rather than across one label *pair*,
suspect a turn-attribution convention rather than a vague description -
sharpening then resolves the flagged turns but the same dissent tends to
reappear on the next turns of that kind, leaving alpha roughly where it
started. Before concluding, rule out (a) one over-broad label clause a
targeted `description` edit would fix, and (b) roll-to-roll noise, by
re-rolling the same spec (k>1) to confirm the same-kind dissent is
stable; read the dissenter's split-turn excerpts to check they share a
kind (there is no event-type column - use `basis` and the sub-agent
lane). The fix is then a spec-level attribution rule (e.g. a delegation
utterance belongs to the phase it delegates into), which belongs in the
global `context`, not any per-label `description` - raise it as a spec
decision.

## Reading the alpha/AC1 bracket

The audit prints percent agreement, Krippendorff's alpha, and Gwet's
AC1 together (compute them yourself with
`transect.reliability.cohort_agreement(f["phase_turn_votes"],
"turn", "phase")` when working from frames). Alpha reads low and AC1
high when one label dominates - the two bracket the true agreement.

Reading the alpha/AC1 bracket: both >= 0.8 = healthy (still consider
the high-side scramble check); both low or negative = the
disagreement is real and prevalence skew is not the excuse; bracket
straddling the bands = skew is distorting one of them - decide from
the confusable-pair breakdown, not from either summary number.

## Verifier signal, split by source

```python
p = f["phases"]
reviewed = p[p.verifier_reviewed.fillna(False)]
reviewed[
    [
        "phase_index",
        "original_label",
        "phase",
        "overturned",
        "verifier_trigger",
        "confidence_source",
    ]
]
```

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

An empty `changed` after a rubric edit means the edit never reached
the run: identical prompts replay from the model cache, so identical
ballots are a wiring failure (edited file not saved, runner pointed
at the old spec path), never a "the fix changed nothing" result.
Fix the wiring and re-run before concluding anything.

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
was an improvement or churn.

## Scrambled-vocabulary check (manual procedure)

Purpose: distinguish definition-reading from name-anchoring. Rename
the labels, keep the descriptions; a judge reading definitions gives
the same partition under scrambled names - consistency **under**
scramble is the healthy outcome. Run it when agreement or
self-consistency is suspiciously high, or before leaning on a high
alpha.

1. Build the scrambled spec with the helper - neutral tokens,
   descriptions/context byte-identical, deterministic:

   ```python
   from transect.diagnostics import scramble_spec

   scrambled, mapping = scramble_spec(spec, seed=7)
   # save mapping to JSON next to the scrambled scans dir
   ```

2. Run the same judge setup on `scrambled` into a fresh scans dir.
   Cost: one full judged pass (prompts differ, nothing replays).
3. Descramble and compare - map the scrambled run's per-turn labels
   back, then per-turn overlap against the original run (the
   before/after recipe above):

   ```python
   from transect.diagnostics import descramble

   sb = transect.load("scans_scrambled/").frames()["phase_turns"]
   back = descramble(sb.phase, mapping["phases"])
   ```
4. Read the result: high overlap = the judge follows definitions
   (healthy); the partition tracking label *names* instead - e.g.
   labels collapsing onto tokens whose names happen to read
   meaningfully, or overlap near chance - = name-anchoring or a
   plumbing artifact; the original run's high agreement was not
   evidence of quality. Anchor "near chance" at roughly 1/(number of
   labels) and "high" well above it - there is no validated cutoff,
   so report the number with both anchors.
5. **The scrambled run's report and store are diagnostic artifacts
   only** - there is no automatic banner (no `scramble` config
   exists), so you must say it explicitly and never quote its labels
   as findings. Delete or clearly quarantine the scrambled store
   afterwards.
