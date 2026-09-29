"""Generate examples/logs/house_price_demo.eval (dummy transcript).

A lead engineer agent works on the Kaggle house-price regression
task and hands off three subtasks to sub-agents (EDA, an alternative
model family, a final review). Every model output is scripted
through the ``mockllm/model`` provider, so the run is deterministic,
free, and regenerable:

    uv run python tests/fixtures/generate_demo_eval.py
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Any, cast

from inspect_ai import Task, eval as inspect_eval, task
from inspect_ai.agent import handoff, react
from inspect_ai.dataset import Sample
from inspect_ai.event import InputEvent
from inspect_ai.log import EvalLog, read_eval_log, write_eval_log
from inspect_ai.model import ChatMessageUser, CompactionSummary, ModelOutput, get_model
from inspect_ai.tool import Tool, tool

MOCK = "mockllm/model"

TASK_PROMPT = (
    "Predict house sale prices for the Kaggle House Prices dataset. "
    "Training data is at data/train.csv (1460 rows, 81 columns, target "
    "SalePrice) and the held-out set at data/test.csv. Produce "
    "submission.csv with columns Id,SalePrice. You are evaluated on "
    "RMSE of log SalePrice. Work as a lead engineer: delegate "
    "independent subtasks to your sub-agents where practical."
)

BASH_OUTPUTS = {
    "pip install -q scikit-learn pandas": "",
    "ls data/": "test.csv\ntrain.csv\n",
    "head -3 data/train.csv": (
        "Id,MSSubClass,MSZoning,LotFrontage,LotArea,...,SalePrice\n"
        "1,60,RL,65,8450,...,208500\n"
        "2,20,RL,80,9600,...,181500\n"
    ),
    "wc -l data/train.csv data/test.csv": (
        "  1461 data/train.csv\n  1460 data/test.csv\n  2921 total\n"
    ),
    "head -3 submission.csv": ("Id,SalePrice\n1461,127523.41\n1462,158342.09\n"),
    "wc -l submission.csv": "1460 submission.csv\n",
}

PYTHON_OUTPUTS = {
    "profile": (
        "shape: (1460, 81)\n"
        "SalePrice: mean 180921, median 163000, skew 1.88\n"
        "log1p(SalePrice): skew 0.12\n"
        "missing top-5: PoolQC 99.5%, MiscFeature 96.3%, Alley 93.8%, "
        "Fence 80.8%, FireplaceQu 47.3%\n"
        "top correlations with SalePrice: OverallQual 0.79, GrLivArea "
        "0.71, GarageCars 0.64, TotalBsmtSF 0.61\n"
    ),
    "ridge_baseline": (
        "ridge alpha=10 on 28 numeric features, 5-fold CV\n"
        "fold RMSLE: [0.142, 0.139, 0.151, 0.144, 0.138]\n"
        "mean CV RMSLE: 0.1428\n"
    ),
    "gbm": (
        "gradient boosting (500 trees, lr=0.05, depth=4), 5-fold CV\n"
        "fold RMSLE: [0.128, 0.124, 0.133, 0.126, 0.121]\n"
        "mean CV RMSLE: 0.1264\n"
    ),
    "blend": (
        "blend 0.6*gbm + 0.4*ridge, 5-fold CV\n"
        "mean CV RMSLE: 0.1231\n"
        "wrote submission.csv (1459 predictions)\n"
    ),
    "check_submission": (
        "submission.csv: 1459 rows + header, no NaN, no negatives\n"
        "prediction range: [52k, 745k]; train range: [35k, 755k]\n"
        "distribution KS vs train: 0.04 (ok)\n"
    ),
}


@tool
def bash() -> Tool:
    async def execute(cmd: str) -> str:
        """Run a shell command in the workspace.

        Args:
            cmd: The command to run.
        """
        return BASH_OUTPUTS.get(cmd, "ok\n")

    return execute


@tool
def python_exec() -> Tool:
    async def execute(label: str, code: str) -> str:
        """Run a python snippet in the workspace.

        Args:
            label: Short name for the snippet.
            code: The python source to run.
        """
        return PYTHON_OUTPUTS.get(label, "ok\n")

    return execute


@tool
def memory() -> Tool:
    async def execute(command: str, path: str, file_text: str = "") -> str:
        """Save or read a note in the workspace memory directory.

        Args:
            command: create, view or str_replace.
            path: The note's path under /memories.
            file_text: The note's content, for create.
        """
        return "ok\n"

    return execute


def says(content: str) -> ModelOutput:
    return ModelOutput.from_content(model=MOCK, content=content)


def calls(name: str, content: str, **arguments: object) -> ModelOutput:
    return ModelOutput.for_tool_call(
        model=MOCK, tool_name=name, tool_arguments=dict(arguments), content=content
    )


def submits(answer: str, content: str) -> ModelOutput:
    return calls("submit", content, answer=answer)


EDA_SCRIPT = [
    calls(
        "python_exec",
        "Profiling the training data: shape, target distribution, "
        "missingness, and correlations with SalePrice.",
        label="profile",
        code="import pandas as pd; df = pd.read_csv('data/train.csv'); ...",
    ),
    submits(
        "EDA summary: 1460x81; SalePrice is right-skewed (1.88), "
        "log1p transform normalises it (skew 0.12). Five columns are "
        ">47% missing and should be dropped or flagged. Strongest "
        "predictors: OverallQual, GrLivArea, GarageCars, TotalBsmtSF. "
        "Recommend modelling log1p(SalePrice).",
        "Profiling complete - summarising findings for the lead.",
    ),
]

ALT_MODEL_SCRIPT = [
    calls(
        "python_exec",
        "Training a gradient-boosting model as the alternative family, "
        "same 5-fold split as the ridge baseline for comparability.",
        label="gbm",
        code="from sklearn.ensemble import GradientBoostingRegressor; ...",
    ),
    submits(
        "Gradient boosting reaches mean CV RMSLE 0.1264 vs the ridge "
        "baseline 0.1428 - a clear improvement. Same folds, so the "
        "comparison is fair. Suggest blending both rather than "
        "dropping ridge outright.",
        "The boosted model beats the baseline - reporting back.",
    ),
]

REVIEWER_SCRIPT = [
    calls(
        "python_exec",
        "Validating the submission file: row count, NaNs, negatives, "
        "and prediction distribution versus training prices.",
        label="check_submission",
        code="import pandas as pd; sub = pd.read_csv('submission.csv'); ...",
    ),
    calls(
        "bash",
        "Double-checking the file shape on disk.",
        cmd="head -3 submission.csv",
    ),
    submits(
        "Submission verified: 1459 predictions, no NaN or negative "
        "values, distribution consistent with training prices "
        "(KS 0.04). The blend's CV RMSLE 0.1231 is the best result "
        "produced; ready to submit.",
        "Checks pass - approving the submission.",
    ),
]

LEAD_SCRIPT = [
    calls(
        "bash",
        "Setting up the environment before touching the data: "
        "installing the libraries this task needs.",
        cmd="pip install -q scikit-learn pandas",
    ),
    calls(
        "bash",
        "Starting with a look at the workspace layout. Before touching "
        "anything, a working hypothesis: sale prices in datasets like "
        "this are usually right-skewed, so modelling log SalePrice "
        "should beat modelling raw prices.",
        cmd="ls data/",
    ),
    calls(
        "bash",
        "Checking the schema and scale of the data. Published solutions "
        "for this dataset consistently pair a log-price target with "
        "regularised linear and boosted-tree models - grounding the "
        "plan in that prior work.",
        cmd="head -3 data/train.csv",
    ),
    calls(
        "transfer_to_eda",
        "The head rows show a wide price range; my hypothesis is that "
        "SalePrice is right-skewed and a log1p transform will stabilise "
        "it. Delegating exploratory data analysis to profile the target "
        "distribution, missingness, and feature correlations so we can "
        "confirm or reject that.",
    ),
    calls(
        "python_exec",
        "The EDA confirms the skew hypothesis. Designing the baseline "
        "experiment: ridge on log1p(SalePrice) over the 28 numeric "
        "features, a fixed 5-fold split, RMSLE as the metric - every "
        "later model must reuse these folds so comparisons stay fair.",
        label="ridge_baseline",
        code="from sklearn.linear_model import Ridge; ...",
    ),
    calls(
        "transfer_to_alt_model",
        "Delegating an alternative model family: train a "
        "gradient-boosting regressor on exactly my ridge baseline's "
        "folds (CV RMSLE 0.1428) and report whether it improves.",
    ),
    calls(
        "python_exec",
        "Blending the ridge and gradient-boosting predictions 0.6/0.4 "
        "and writing submission.csv. Reading the fold results: gbm "
        "beats ridge on every fold and the blend beats both - the two "
        "models' errors are partly independent, which is what makes "
        "the blend worth keeping.",
        label="blend",
        code="pred = 0.6 * gbm_pred + 0.4 * ridge_pred; ...",
    ),
    calls(
        "transfer_to_reviewer",
        "Before submitting, delegating an adversarial review: actively "
        "try to break submission.csv - hunt for NaNs, out-of-range "
        "predictions, row-count drift, any reason the blend's CV RMSLE "
        "0.1231 would not hold up. Only if it survives do we submit.",
    ),
    submits(
        "Final submission: 0.6/0.4 blend of gradient boosting and ridge "
        "on log1p(SalePrice), CV RMSLE 0.1231, submission.csv validated "
        "by review.",
        "Stepping back before finishing: the plan held end to end - log "
        "target, fair-fold baseline, one alternative family, a blend; "
        "the one dead end was hoping weight tuning would add more than "
        "it did. Nothing left that changes the answer - submitting.",
    ),
]


def scripted(outputs: list[ModelOutput]):
    return get_model(MOCK, memoize=False, custom_outputs=outputs)


# The lead compacts once, between the alt_model handoff and the blend. The
# threshold sits between the lead's context at the alt_model delegation
# call and at the blend call, so the memory warning band (90% of it)
# catches the delegation call and the log records Inspect's real
# summarization prompt and memory nudge. The summarizer is its own scripted
# model so the lead's script is not consumed; a second compaction would
# exhaust it and fail the run.
COMPACTION_THRESHOLD = 1800
COMPACTION_INSTRUCTIONS = "Keep the fold protocol and every CV RMSLE figure."
LEAD_SUMMARY = (
    "- Task Overview\nPredict Kaggle house SalePrice; evaluated on RMSE of "
    "log SalePrice; produce submission.csv (Id, SalePrice). Working as lead "
    "with EDA, alt-model and reviewer sub-agents.\n\n- Current State\n"
    "Installed scikit-learn and pandas; inspected data/train.csv (1460 rows, "
    "81 columns). EDA confirmed SalePrice is right-skewed, so the target is "
    "log1p(SalePrice). Ridge baseline over 28 numeric features, fixed 5-fold "
    "split, CV RMSLE 0.1428. alt_model trained gradient boosting on the same "
    "folds and beat ridge on every fold (CV RMSLE 0.1302).\n\n"
    "- Important Discoveries\nThe two models' fold errors are only partly "
    "correlated, so a blend should beat both. Every comparison must reuse the "
    "baseline's folds.\n\n- Next Steps\nBlend ridge and gbm predictions, "
    "write submission.csv, hand it to the reviewer, then submit.\n\n"
    "- Context to Preserve\nOperator is time-boxed: blend the two existing "
    "models rather than trying a third family."
)


@task
def house_price_demo() -> Task:
    eda = react(
        name="eda",
        description="Exploratory data analysis specialist.",
        prompt="You are a data analysis specialist.",
        tools=[bash(), python_exec()],
        model=scripted(EDA_SCRIPT),
    )
    alt_model = react(
        name="alt_model",
        description="Tries alternative model families.",
        prompt="You are a modelling specialist.",
        tools=[bash(), python_exec()],
        model=scripted(ALT_MODEL_SCRIPT),
    )
    reviewer = react(
        name="reviewer",
        description="Reviews results before submission.",
        prompt="You are a meticulous reviewer.",
        tools=[bash(), python_exec()],
        model=scripted(REVIEWER_SCRIPT),
    )
    lead = react(
        prompt="You are the lead engineer on a Kaggle regression task.",
        tools=[
            bash(),
            python_exec(),
            memory(),
            handoff(eda),
            handoff(alt_model),
            handoff(reviewer),
        ],
        model=scripted(LEAD_SCRIPT),
        compaction=CompactionSummary(
            threshold=COMPACTION_THRESHOLD,
            instructions=COMPACTION_INSTRUCTIONS,
            model=scripted([says(LEAD_SUMMARY)]),
        ),
    )
    return Task(
        name="house_price_demo",
        dataset=[Sample(id="house-price", input=TASK_PROMPT)],
        solver=lead,
    )


OPERATOR_NOTE = (
    "Operator note: we are time-boxed - blend the two models you "
    "have rather than trying a third family."
)

INPUT_ANSWER = "y"


def check_compaction(sample: Any) -> None:
    """Fail loudly if the run did not compact exactly once, between the
    alt_model handoff and the blend, with the summarization prompt and a
    preceding memory nudge recorded."""
    events = sample.events

    def model_index(needle: str) -> int:
        return next(
            i
            for i, e in enumerate(events)
            if e.event == "model" and needle in (e.output.completion or "")
        )

    flushes = [i for i, e in enumerate(events) if e.event == "compaction"]
    assert len(flushes) == 1, f"expected one compaction, got {len(flushes)}"
    (flush,) = flushes
    summarizer = events[flush - 1]
    assert summarizer.event == "model" and LEAD_SUMMARY in (
        summarizer.output.completion or ""
    ), "the event before the compaction is not the summarization call"
    assert COMPACTION_INSTRUCTIONS in summarizer.input[-1].text
    assert model_index("alternative model family") < flush < model_index("Blending"), (
        "compaction must land between the alt_model handoff and the blend"
    )
    assert any(
        m.role == "user" and m.text.startswith("Context compaction approaching")
        for e in events[:flush]
        if e.event == "model"
        for m in e.input
    ), "no memory nudge recorded before the compaction"


def plant_operator_events(path: Path) -> None:
    """Insert two human interventions into the log and check its compaction."""
    log = read_eval_log(str(path), resolve_attachments=True)
    assert log.samples
    sample = log.samples[0]
    check_compaction(sample)

    def model_index(needle: str) -> int:
        for index, event in enumerate(sample.events):
            if event.event == "model" and needle in (
                event.output.choices[0].message.text or ""
            ):
                return index
        raise LookupError(needle)

    submit = model_index("Stepping back")
    sample.events.insert(
        submit,
        InputEvent(
            input=INPUT_ANSWER,
            input_ansi=INPUT_ANSWER,
            message="Submit the blended predictions as the final answer?",
            timestamp=sample.events[submit - 1].timestamp,
        ),
    )
    for index, message in enumerate(sample.messages):
        if message.role == "assistant" and "Blending" in (message.text or ""):
            sample.messages.insert(
                index, ChatMessageUser(content=OPERATOR_NOTE, source="operator")
            )
            break
    else:
        raise LookupError("blend assistant message")
    stretch_wallclock(log)
    write_eval_log(log, str(path))


# synthetic per-event durations (seconds)
_DURATIONS = {"model": 34, "tool": 11, "compaction": 6, "input": 18}


def stretch_wallclock(log: EvalLog) -> None:
    """Spread the instant mockllm run over a plausible half hour.

    Rewrites event timestamps to a deterministic schedule and sets the
    matching sample/eval durations.
    """
    assert log.samples
    sample = log.samples[0]
    clock = sample.events[0].timestamp
    elapsed = 0.0
    # the schedule rewrite duck-types across the event union
    for index, event in enumerate(cast("list[Any]", sample.events)):
        event.timestamp = clock + timedelta(seconds=elapsed)
        event.working_start = elapsed
        step = _DURATIONS.get(event.event, 2) + (index * 7) % 23
        if hasattr(event, "completed") and event.completed is not None:
            event.completed = event.timestamp + timedelta(seconds=step)
            event.working_time = float(step)
        elapsed += step
    sample.total_time = elapsed
    sample.working_time = elapsed
    log.stats.completed_at = (clock + timedelta(seconds=elapsed)).isoformat()


if __name__ == "__main__":
    log_dir = Path(__file__).parents[2] / "examples" / "logs"
    logs = inspect_eval(
        house_price_demo(),
        model=MOCK,
        log_dir=str(log_dir),
        display="plain",
    )
    # stable filename so the committed log does not churn on regenerate
    written = Path(logs[0].location.removeprefix("file://"))
    target = log_dir / "house_price_demo.eval"
    written.replace(target)
    plant_operator_events(target)
    print(f"wrote {target}")
