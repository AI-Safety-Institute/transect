---
name: add-a-layer
description: >
  Use when adding a custom layer to a Transect run: a user-defined
  judged classification (own loader + vocabulary via
  cohort_llm_scanner), a mechanical scanner (a grep or metric over
  turns, no judge), a bring-your-own-data frame, custom report
  sections (charts, bands, tables, markdown), phase-card tags, a
  token-spend grouping, or a custom entity block in the reliability
  audit. Covers the Layer contract end to end, the supported judged
  path, vocabulary rules (always include a none_of_the_above escape
  hatch), and how to verify the result. Running the default pipeline
  and reading the report live in the sibling using-transect skill.
---

# Adding a custom layer

A `Layer` is the one injection surface for user additions:

```python
import transect

results = transect.transect(logs, spec, judge_models=..., extra_layers=[MY_LAYER])
```

Each layer bundles an optional scanner (joins the scan batch), an
optional frame, typed report blocks, phase-card tags, and an audit
declaration. Every field is optional but the layer must do something.
For evaluation-specific choices, first read the using-transect skill's
"Adapting to a new evaluation" section. The worked reference is
[transect_custom_layer.py](https://github.com/AI-Safety-Institute/transect/blob/main/examples/transect_custom_layer.py).

| Layer field | What it does |
|---|---|
| `name` | mount key (`results.layer_frames[name]`), the provenance label on everything the layer renders, and its section key in `section_order`; must not collide with a built-in frame name (`transect.frames.results.builtin_frame_names()`) or a report section key (`transect.report.SECTION_KEYS`) |
| `scanner` | any `@scanner`-decorated Scout scanner - a mechanical extractor (a grep over turns, a per-turn metric) needs no judge and costs nothing; a judged classification builds on `cohort_llm_scanner` (the path in section 1). An instance, or the un-invoked factory |
| `scanner_args` | extra keyword arguments `transect()` passes to a factory-form scanner (settings the factory needs that are neither judge arguments nor in the spec); refused with an instance or no scanner |
| `frame` | fn over the scanner's raw results (use `transect.turns_frame` on the supported path), or a ready DataFrame (data-only layer); omitted with a scanner = `transect.frames.user.generic_flatten` (identity columns + the value's top-level keys, one row per result) |
| `section` | typed report blocks, rendered as a badge-marked section |
| `tags` | phase-card tag families from the layer's per-turn frame, e.g. `{"skill": "label"}` (family name -> frame column); also feeds the Token spend grouping selector |
| `audit` | `(unit_col, label_col)` - opt in to an entity block in the reliability audit |

## 1. The supported judged path

A judged layer is a scanner built on `transect.cohort_llm_scanner` behind
your own `@scanner`-decorated factory, with its own loader and closed
vocabulary. That path stamps the judge-identity block and the loader
item's metadata on every result, so frames, tags, the audit, and the
reliability functions work unchanged.

```python
from inspect_scout import Scanner, Transcript, scanner
from transect import Layer, cohort_llm_scanner, reasoning_turns, turns_frame
from transect.report import Markdown, TurnBand

@scanner(loader=reasoning_turns())
def research_skills(spec, judge_models=None, k_rolls=1, verify=None):
    # the rubric lives in the spec's extra block: {label: description},
    # always including a none_of_the_above escape hatch
    skills = spec.extra["research_skills"]
    question = (
        "Above is one reasoning turn of an agent. Label the turn with the "
        "best-fitting category:\n"
        + "\n".join(f"- {label}: {text}" for label, text in skills.items())
    )
    return cohort_llm_scanner(
        question=question,
        answer=list(skills),
        models=judge_models,
        k_rolls=k_rolls,
        verify=verify,
        vocabulary=skills,
    )


MY_LAYER = Layer(
    name="research_skills",
    scanner=research_skills,  # the factory, un-invoked
    frame=turns_frame,
    tags={"skill": "label"},
    audit=("turn", "label"),
    section=[
        Markdown("#### One skill label per reasoning turn"),
        TurnBand(label="label"),
    ],
)
```

The pieces, and the rules that make them work:

- **Loader**: the loader is how you choose the unit of judgement,
  not a requirement. `transect.reasoning_turns()` yields one item per
  main-lane reasoning turn (reasoning blocks inlined behind a
  `[THINKING]` marker when the source records them, sub-agent spawn
  tasks behind a `[DELEGATES]` marker) - the right unit for a second
  per-turn facet, already numbered on the orchestrator turn axis. For
  another unit shape, write your own `@loader`; whatever metadata you
  put on each yielded item (e.g. `{"turn": n}`) rides into the frame
  as columns, and the item id becomes the frame's `item` column. A
  `turn` fact must be the orchestrator ordinal from
  `transect.scanners.helpers.orchestrator_turns(transcript)` (section
  4): a sub-agent's model calls are off the axis, so numbering every
  model event in the transcript puts your rows on turns the report
  does not have. No loader is valid too: `@scanner(messages="all")`
  hands the judge the whole transcript as one item - one call, one
  label per transcript - and the frame, ballots, definitions, and
  audit (`audit=("item", "label")`) work unchanged. Only the
  turn-anchored surfaces (tags, TurnBand/TurnChart, the spend
  grouping) need a per-turn unit.
- **Batching**: `reasoning_turns(batch=N)` +
  `cohort_llm_scanner(..., batch=True)` judges up to N turns per
  call instead of one - the big cost lever on long runs, and the
  judge sees neighbouring turns as context. Results stay per turn
  (votes, verifier reviews, and the spot-check draw are all
  per-turn; a turn the answer skipped gets an explicit `no_answer`
  status, never a filled label), so the frame, tags, and audit are
  identical to the unbatched shape. The batching itself lives in
  the loader, not the scanner: `batch=True` only teaches the
  scanner to answer per marked unit, so the loader must yield batch
  items. `batch=True` with no loader (or a loader that declares no
  units) fails at scan time as a per-transcript scan error naming
  the fix - it never judges the whole transcript as one blob.

  Batching is also the memory lever. A per-turn layer holds one
  result row per item in memory until the loader finishes a
  transcript, and each row stores that item's scan events (the judge
  calls with their full prompts), so a long run's memory and scan
  store grow with the item count. `batch=N` divides the item count
  by N.

  Batching is unit-neutral. A custom loader batches by doing
  exactly three things per yielded item:

  ```python
  from transect.scanners.cohort import batch_item_content


  @loader(messages="all")
  def my_batches() -> Loader[Transcript]:
      async def load(transcript: Transcript) -> AsyncIterator[Transcript]:
          units = my_units(transcript)  # your unit texts + facts
          for window in chunked(units, 10):
              yield Transcript(
                  transcript_id=f"{transcript.transcript_id}:w-{window[0].key}",
                  # 1. content: one [ITEM n] marker per unit, in order
                  messages=[
                      ChatMessageUser(
                          content=batch_item_content([u.text for u in window])
                      )
                  ],
                  # 2. one fact dict per unit, in the same order -
                  #    the facts become the frame's columns
                  # 3. optional "id" per unit: keys the verifier's
                  #    single-unit view and the spot-check seed
                  metadata={
                      "items": [
                          {"span_id": u.key, "id": f"{transcript.transcript_id}:{u.key}"}
                          for u in window
                      ]
                  },
              )

      return load
  ```

  Content markers and the `metadata["items"]` list must align
  one-to-one in order - the answer addresses units by the marker's
  1-based ordinal, so unit identity never depends on the model
  echoing an id back.
- **Factory form**: pass the un-invoked factory as `Layer.scanner`.
  It is called at `transect()` with the subset of the judge arguments
  its signature declares (`judge_models` / `k_rolls` / `verify` /
  `verifier_model` / `verify_sample`) - the signature is the opt-in,
  and a declared parameter always takes the entry point's value.
  Hardcode layer-local judge choices in the factory body instead of
  declaring the parameter. Declaring `judge_models` when `transect()`
  got none is refused loudly. Two more inputs reach a factory: declare
  `spec` to receive the loaded `Spec` (rubrics in `spec.extra`, the
  worked example's shape), and put anything else it needs - settings
  that are not eval knowledge - in `Layer(scanner_args={...})`, passed
  through as keyword arguments (a key the factory cannot take, or one
  shadowing a judge argument or `spec`, is refused). `load()` never
  calls a factory, so a factory-form layer remounts from a stored scan
  with no credentials. A ready instance (the factory called by you) is
  also accepted and receives nothing. To bound the judges' connection
  count, retries or timeouts, pass `Model` instances built with
  `get_model(name, config=GenerateConfig(...))` as `judge_models`.
- **Frame fn inputs**: a frame fn receives the raw results table only.
- **Vocabulary**: `answer` is the closed label list the judge picks
  from; `vocabulary` records the rubric in the scan store as
  `label_vocab` provenance (never shown to the judge - put rubric
  text the judge should see in `question`). It accepts a mapping
  (label -> description), a list of dicts, or bare label strings.
  Always include a `none_of_the_above` escape-hatch category, exactly
  as the built-in scanners do: a judge forced to pick a substantive
  label when none fits inflates every category. With descriptions
  recorded, the layer's report section closes with a "Label
  definitions" expandable, and the audit uses the declared roster to
  order its per-label maps (an unused label still gets a column -
  that absence is information). Prefer keeping the vocabulary in the
  spec rather than inline: the spec's `extra` block is a
  validated-opaque dict for exactly this
  (`extra: {research_skills: {label: description, ...}}` in the
  YAML, `load_spec(...).extra["research_skills"]` in the module) -
  eval knowledge stays in one file. The worked example does this.
- **Question and display inputs**: map each requirement to the actual
  consumer. `Spec.context` reaches phase prompts only; `Spec.extra` and
  `vocabulary` do not inject text into your custom judge. Build `question`
  explicitly and capture the prepared prompt in a deterministic check. Keep
  source-derived facts and judge associations distinct in display columns.
- **Frame**: `transect.turns_frame` projects one row per loader item: the
  item id and metadata, the decided `label`, and the judged columns
  (confidence, agreement, label_source, verifier review, judge
  identity). Do not read the stored item payload; item facts belong
  in loader metadata.
- **One row per attempted item**: the audit reads coverage from the
  frame, so a hand-rolled frame fn must keep every item judging
  attempted, with a null `label` where no judgement was delivered -
  filtering to labelled rows makes failed or refused items invisible
  to the audit and its unjudged-units flag. `turns_frame` satisfies
  this by construction.

## 2. Report blocks

`Layer.section` takes typed blocks from `transect.report` - never raw
HTML. Each renders inside the layer's own badge-marked section, on
the shared turn axis where applicable:

| Block | Renders | Needs |
|---|---|---|
| `Markdown(text)` | prose (HTML escaped, headings demoted) | - |
| `TurnBand(label=col)` | contiguous same-label runs as coloured chunks, with filter select, legend chips, judge tooltip rows, and grey fillers over missing turns | per-turn frame with `turn` + the label column |
| `TurnChart(y=col, kind=...)` | bar / line / step / area per turn | `turn` + a numeric column |
| `SpanLanes(start, end, label)` | one lane per row spanning start..end | the three columns |
| `EventMarks(label)` | dashed rules at the frame's turns | `turn` |
| `Table(columns=...)` | sortable table of the frame (scrolls internally) | - |
| `Component(fn)` | escape hatch: your fn gets a `SectionContext` (transcript_id, the layer's frame slice, n_turns) and returns an inspect_viz component | - |

Block shapes are validated loudly at `transect()` - the error names the
block, layer, and column. Every viz block also takes a ``frame=``
DataFrame override (default: the layer's own frame), cut to the
transcript when it carries ``transcript_id`` and broadcast as-is
otherwise.

## 3. Tags, spend grouping, audit

- `tags={"family": "column"}` puts `family=value` chips on each phase
  card's turn range (per-family tint, provenance title naming the
  layer), adds one filter select per family to the card controls, and
  adds the family to the Token spend section's "Group by:" selector
  (per-turn new_work summed per label, an "untagged" bucket keeping
  totals honest). Requirements: a per-turn frame (`turn` int;
  `transcript_id` optional - absent means broadcast to every
  transcript), string/categorical value columns (numerics are
  refused - fence 2), and no reserved bookkeeping columns as families
  (fence 1; an explicit list/dict overrides fence 1, never fence 2).
- `audit=("turn", "label")` adds the layer to the closing reliability
  audit: judge regime, verifier rows, confidence tiers, and the
  reliability / member-coverage / provenance maps, under the layer's
  name with the custom badge. The full judged projection is checked
  loudly at load - the declared unit/label columns plus judge
  identity, verifier review, and agreement / confidence /
  label_source, exactly what `turns_frame` over a
  `cohort_llm_scanner` scanner carries - so audit on a mechanical or
  hand-built frame is refused with the fix named. With the columns
  present but the judge never run, the block is quietly absent.
- `section_order=[...]` on `transect()` / `render()` places the
  layer's section anywhere: keys are `transect.report.SECTION_KEYS` plus
  each layer's name; listed sections render first in the given
  order, unlisted follow in default order, the audit always last.

## 4. Mechanical ($0) layers

A layer needs no judge. Two zero-LLM-call shapes:

- **Mechanical scanner**: any `@scanner` that extracts structure - a
  regex/grep over turns, a per-turn metric, event counts. Number turns
  with `transect.scanners.helpers.orchestrator_turns(transcript)`
  (declare `events=["model", "span_begin", "span_end", "tool"]` so it
  can resolve the main lane): `turn` means orchestrator turn everywhere,
  and enumerating every lane's model events would misalign your rows
  with the axis. Pack per-turn entries as a ``value["turns"]`` list
  (one result per transcript) and `transect.turns_frame` explodes them
  into one row per turn - `token_timeline`'s own convention - ready for
  tags and turn-anchored blocks; any other value shape mounts via
  `generic_flatten` or your own frame fn.
- **No scanner at all**: pass a ready per-turn DataFrame - e.g.
  `Layer(name="review", frame=my_df, tags=True)`. `tags=True`
  selects every non-reserved string column as a family. A scanner
  and a ready DataFrame together are refused (a ready frame cannot
  be fed by a scanner).

## 5. Notebook analysis and reliability

`transect.load(results.scan_location, extra_layers=[MY_LAYER])` remounts the layer's
frame from the stored scan without re-scanning (the layer must be
passed again - the store alone does not know your frame fn). The
reliability functions run over the layer's frame directly
(`detect_regime`, `label_stats`, `relabel_rate`, ...); for the
votes-shaped ones, `transect.member_ballots(frame, "turn")` explodes the
frame's raw `members` column into the long-format ballots
(`cohort_agreement`, `member_coverage`, `label_stats`'s votes
argument). Empty outside the voting regimes - a solo judge records
no member ballots. Read each installed `transect.reliability` function's
docstring for its input columns and units, or the matching source
[reliability module](https://github.com/AI-Safety-Institute/transect/blob/main/src/transect/reliability.py).
Missing required columns raise a `KeyError` naming the fix.

Before scaling a custom layer, check source identities through loader, stored
result, frame, tags and source link. Test dropped, repeated and reordered
batch answers; a matching row count alone does not establish alignment.
Preserve distinct actions within one turn, or document an explicit aggregation
rule. Inspect `results.scan_status` and per-item statuses after loading the exact
scan; successful execution does not establish complete or correct labels.
