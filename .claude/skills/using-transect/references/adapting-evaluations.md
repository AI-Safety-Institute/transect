# Adapting an evaluation

Use this recipe when the original task, scaffold, scoring or review question
needs more than the default report. It guides configuration and calibration;
it supplies neither a domain taxonomy nor evidence that a judge is valid.

## Inventory the selected material

Record the exact transcript file/hash, sample and epoch, evaluation family,
partition, task definition, scaffold configuration, native scorer and intended
reference material. Record their revisions or hashes separately from the
installed Transect, Inspect and Scout versions. A harness revision does not
establish the dependency versions that executed a stored run.

Distinguish original source material from previous generated specs, labels and
reports. Establish which sources the user permits for development and which
are held out. Keep held-out outcomes and intended solutions out of prompt
iteration unless the agreed evaluation design explicitly permits them.

Inspect the selected source's actual schema and ordering before deriving
counts or captions. A native scalar, a list of observed events, a checkpoint
roster and a graph's nodes can measure different things. Preserve native
values and identify their units. If source revisions or runtime schemas
conflict, record the unresolved correspondence; do not invent a version pin
or replace the selected definitions with a newer ordering.

Prepare a bounded author packet using an explicit field whitelist: task,
scaffold, relevant configuration and compact schema examples. Keep full event
histories, retry payloads and unnecessary secrets outside it. Inspect the
actual packet's size and fields before supplying it to an author or judge.
Prefer the installed public log readers. If one fails, bound the attempt and
record the failure; any format-aware fallback must verify the selected member,
sample/epoch and source hash rather than silently choosing another sample.

## Map requirements to consumers

For each requested output, record its source, unit, transformation, consumer
and acceptance check. For example:

| Requirement | Implementation route | Check at the consumer |
|---|---|---|
| Describe activity phases | `Spec.phases`, descriptions and phase-only `Spec.context` | Capture the prepared phase prompt and inspect source-associated labels |
| Classify delegated tasks | `Spec.subagent_labels` | Inspect the prepared span task and its resulting classification; global context is not injected |
| Add another judged facet | Custom loader and `cohort_llm_scanner` question | Verify unit identity, question text, vocabulary and explicit failure statuses |
| Show a native outcome | Mechanical extraction or a data-only `Layer` | Compare displayed values and units with the selected native record |
| Present a custom view | Typed layer blocks or a separate UI over frames | Check joins, coverage, provenance and exact source-link destinations |

`Spec.extra` stores configuration for your code; it does not create a layer or
send text to a model. `vocabulary` records rubric provenance; put definitions
the judge must see into the custom `question`. Map judge-visible source fields
separately from recipient-visible fields. Setup text, interventions, spawn
tasks and custom metadata can survive into frames and reports.

Choose the unit from the evidence. Multiple actions in one turn need separate
identities or an explicit aggregation rule. A single primary label on a
compound action can omit secondary methods from an exact-label filter; decide
whether the user's retrieval task needs primary labels, multiple labels or
separate attributes before designing the layer.

## Start from a fresh configuration

This minimal configuration enables built-in structural extraction only. It
does not inherit a previous evaluation's labels or configure a custom judge:

```python
import transect

spec = transect.Spec.model_validate(
    {"phases": [], "subagent_labels": [], "context": "", "extra": {}}
)
results = transect.transect(
    logs="selected.eval",
    spec=spec,
    sample="selected-sample",
    epochs=1,
    judge_models=None,
    scans_dir="scans/intake",
    viewer=False,
    open_report=False,
)
saved_scan = results.scan_location
reloaded = transect.load(saved_scan)
```

Replace the file, sample and epoch with the inspected selection; OpenClaw
telemetry imports use their own source identity and epoch convention. Derive
any enabled vocabularies from the task and review question, with descriptions
and uncertainty/escape paths. Record the configuration and layer-code versions.
Use the sibling add-a-layer skill for the implementation route. Reload custom
frames with the same layer definitions and the exact saved scan location.

## Check the deterministic path and calibrate

Before paying for labels, trace representative identities through source,
loader, stored result, frame and display. Capture prepared prompts with a
deterministic model substitute to verify that the actual rubric and context
reach the intended consumer. Resolve attachments as the runtime loader does;
distinguish source truncation from any additional excerpt limit. Resolving an
attachment cannot recover content the source never retained.

Use negative controls for dropped, repeated and reordered batch answers. Check
per-unit joins and `no_answer`/error/refusal states, not only total row counts.
Open source links at their actual event destinations. Check rendered Boolean
and status cells rather than inferring visibility from values in a frame.
If a filename workaround requires a copy, verify byte equality and record the
copy's provenance; do not overwrite the original failure record.

Choose a fixed development set with source-grounded reference judgments and
record who reviewed them. Include comments versus executed commands, plans or
retrospective claims versus current actions, attempted operations versus
observed outcomes, and compound actions. Define the evidence policy for labels
and narration: an agent's assertion of success is not an observed result.
Inspect changed cases and regressions when revising prompts or vocabularies;
retain the earlier outputs. Confidence, agreement, a populated report and
improved wording alone do not demonstrate correctness. Keep held-out material
out of this tuning loop and report the limits of any later assessment.

## Size and check the runtime

Prepare a per-call ledger before a paid run: scanner/purpose, batch or unit
identities, model and roll, actual prepared input size, output cap, current
provider prices and allowed retries. Include narration and verifier work;
their counts can depend on earlier labels, so state the assumptions or bounds.
Include custom layers and attachment-resolved inputs. Distinguish fresh calls,
local response-cache replay and provider-side cached-token pricing. An estimate
is not an enforced spending ceiling or authorization to spend.

A configured `Model` is not necessarily the final request configuration:
Scout scan settings and model wrappers can override model defaults. Verify
effective concurrency, retry and output settings at the provider dispatch
boundary using a deterministic provider substitute where possible. A mock of
only the outer `Model.generate` call can miss later overrides. Any live check
belongs in the approved call budget. Check the installed interfaces; do not
invent `model_config`, `max_transcripts` or `cache=False` arguments on public
`transect()`. A lower-level Scout runner or runtime adapter is separate code
whose precedence and limits need their own checks.

On a guard rejection, unexpected effective settings, or exhausted allowance,
stop new dispatches and retain the ledger and store before investigating.
Calling `transect()` again creates a new scan. `load(exact_scan)` only reads
stored results; a parent path selects the newest scan. OpenClaw invocations
retain separate parsed snapshots, while inspect-ai's response cache may replay
matching requests. A replay-assisted new run is not native resume and is not
an independent repeated judgment. For a cold response-cache run, set
`INSPECT_CACHE_DIR` to a new empty directory before starting the process;
preserve previous caches and snapshots as evidence.

## Check completion and deliver the bounded result

After each staged run, inspect `results.scan_status`, item statuses, expected
identities, usable labels, narrator fallback and exact-scan reload. Outer
completion can coexist with failed custom items; unknown custom coverage is
not success. Check the report in a browser for the selected consumer paths.
Carry run-wide failures and scope into custom artifacts, even when their views
filter epochs or units. Distinguish engineering checks from source-grounded
label review, held-out performance and human usefulness.

Record workarounds, unresolved provenance, unreviewed cases and cache reuse.
Manual reading or successful installation of these skills does not establish
native discovery by an agent; test that separately in the target environment.
