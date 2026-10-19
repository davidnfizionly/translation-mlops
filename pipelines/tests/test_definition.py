"""Offline checks of the planned SageMaker Pipeline definition.

Reads the definition from a saved Terraform plan (build/ml-plan.json, from
`terraform -chdir=infra/ml show -json tfplan`) and checks the wiring that a
plan cannot: step references, the gate path against gate.py's real output,
the test split kept out of training, and the run limits.
"""

import json
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
PLAN = REPO / "build" / "ml-plan.json"
sys.path.insert(0, str(REPO / "evaluation"))
import gate  # noqa: E402


def load_definition():
    if not PLAN.exists():
        pytest.skip("build/ml-plan.json not found: run terraform plan and show -json first")
    plan = json.loads(PLAN.read_text())
    for rc in plan["resource_changes"]:
        if rc["type"] == "aws_sagemaker_pipeline":
            return json.loads(rc["change"]["after"]["pipeline_definition"])
    raise AssertionError("no aws_sagemaker_pipeline in the plan")


@pytest.fixture(scope="module")
def definition():
    return load_definition()


def all_steps(steps):
    for step in steps:
        yield step
        if step["Type"] == "Condition":
            yield from all_steps(step["Arguments"]["IfSteps"])
            yield from all_steps(step["Arguments"]["ElseSteps"])


def gets(node):
    if isinstance(node, dict):
        for k, v in node.items():
            if k == "Get":
                yield v
            else:
                yield from gets(v)
    elif isinstance(node, list):
        for v in node:
            yield from gets(v)


def test_step_graph(definition):
    names = {s["Name"]: s["Type"] for s in all_steps(definition["Steps"])}
    assert names == {"Train": "Training", "Evaluate": "Processing", "ChrfGate": "Condition", "Register": "RegisterModel", "GateNotPassed": "Fail"}
    assert definition["Version"] == "2020-12-01"


def test_every_reference_resolves(definition):
    steps = {s["Name"] for s in all_steps(definition["Steps"])}
    params = {p["Name"] for p in definition["Parameters"]}
    for ref in gets(definition):
        if ref.startswith("Steps."):
            assert ref.split(".")[1] in steps, ref
        elif ref.startswith("Parameters."):
            assert ref.split(".")[1] in params, ref
        else:
            assert ref in ("Execution.PipelineExecutionId",), ref


def test_gate_reads_the_key_gate_py_writes(definition):
    cond = next(s for s in definition["Steps"] if s["Name"] == "ChrfGate")["Arguments"]["Conditions"][0]
    path = cond["LeftValue"]["Std:JsonGet"]["Path"]
    assert cond["Type"] == "GreaterThanOrEqualTo" and cond["RightValue"] == 1
    evaluate = next(s for s in definition["Steps"] if s["Name"] == "Evaluate")
    pf = evaluate["PropertyFiles"][0]
    assert cond["LeftValue"]["Std:JsonGet"]["PropertyFile"]["Get"] == f"Steps.Evaluate.PropertyFiles.{pf['PropertyFileName']}"
    assert pf["FilePath"] == "evaluation.json"
    assert pf["OutputName"] in {o["OutputName"] for o in evaluate["Arguments"]["ProcessingOutputConfig"]["Outputs"]}
    # Follow the JSON path through a real gate.py result.
    report = {"test_file_sha256": "x", "test_rows": 1, "limited": False, "decoding": {}, "directions": {"d": {"bleu": 1.0, "chrf": 1.0}}}
    node = gate.compare(report, report)
    for key in path.split("."):
        node = node[key]
    assert node in (0, 1)
    script = (REPO / "evaluation" / "pipeline_eval.sh").read_text()
    assert "--out /opt/ml/processing/evaluation/evaluation.json" in script


def test_register_only_behind_the_gate(definition):
    top = {s["Name"] for s in definition["Steps"]}
    assert "Register" not in top
    cond = next(s for s in definition["Steps"] if s["Name"] == "ChrfGate")["Arguments"]
    assert [s["Name"] for s in cond["IfSteps"]] == ["Register"]
    assert [s["Type"] for s in cond["ElseSteps"]] == ["Fail"]
    reg = cond["IfSteps"][0]["Arguments"]
    assert reg["ModelPackageGroupName"] == "translation-mlops-nllb-en-din-nus"
    assert reg["InferenceSpecification"]["Containers"][0]["ModelDataUrl"] == {"Get": "Steps.Train.ModelArtifacts.S3ModelArtifacts"}
    assert "huggingface-pytorch-inference" in reg["InferenceSpecification"]["Containers"][0]["Image"]


def test_training_never_sees_the_test_split(definition):
    train = next(s for s in definition["Steps"] if s["Name"] == "Train")["Arguments"]
    uris = [c["DataSource"]["S3DataSource"]["S3Uri"] for c in train["InputDataConfig"]]
    assert {c["ChannelName"] for c in train["InputDataConfig"]} == {"train", "valid"}
    assert all("split=test" not in u for u in uris)
    evaluate = next(s for s in definition["Steps"] if s["Name"] == "Evaluate")["Arguments"]
    assert any("split=test" in i["S3Input"]["S3Uri"] for i in evaluate["ProcessingInputs"] if isinstance(i["S3Input"]["S3Uri"], str))


def test_limits_and_defaults(definition):
    params = {p["Name"]: p["DefaultValue"] for p in definition["Parameters"]}
    assert params["MaxSteps"] == "3000" and params["TrainRows"] == "50000"
    assert re.search(r"evaluation/tm-eval-base-\d{8}-\d{6}/metrics\.json$", params["BaselineMetricsUri"])
    train = next(s for s in definition["Steps"] if s["Name"] == "Train")["Arguments"]
    assert train["StoppingCondition"]["MaxRuntimeInSeconds"] == 7200
    assert train["ResourceConfig"]["InstanceType"] == "ml.g4dn.xlarge" and train["ResourceConfig"]["InstanceCount"] == 1
    assert train["CheckpointConfig"]["S3Uri"]["Std:Join"]["Values"][1] == {"Get": "Execution.PipelineExecutionId"}
    # Memory settings after the out-of-memory failure of run 1 (effective batch stays 32).
    hp = train["HyperParameters"]
    assert (hp["per_device_batch_size"], hp["gradient_accumulation_steps"]) == ("8", "4")
    assert train["Environment"]["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"
    evaluate = next(s for s in definition["Steps"] if s["Name"] == "Evaluate")["Arguments"]
    assert evaluate["StoppingCondition"]["MaxRuntimeInSeconds"] == 2700
    assert evaluate["ProcessingResources"]["ClusterConfig"]["InstanceType"] == "ml.g4dn.xlarge"


def test_same_image_as_the_baseline_job(definition):
    launcher = (REPO / "scripts" / "launch_evaluation.py").read_text()
    image = next(s for s in definition["Steps"] if s["Name"] == "Evaluate")["Arguments"]["AppSpecification"]["ImageUri"]
    tag = image.split(":")[-1]
    assert tag in launcher
