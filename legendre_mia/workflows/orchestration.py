from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from ..utils.config import PACKAGE_ROOT, REPO_ROOT


MODEL_ROOT = PACKAGE_ROOT / "legendre_mia" / "models"
WORKFLOW_ROOT = PACKAGE_ROOT / "legendre_mia" / "workflows"


def _process_environment() -> Dict[str, str]:
    environment = os.environ.copy()
    package_path = str(PACKAGE_ROOT)
    existing = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = (
        package_path if not existing else os.pathsep.join((package_path, existing))
    )
    return environment


class StageRunner:
    def __init__(
        self,
        output_root: Path,
        *,
        dry_run: bool = False,
        resume: bool = True,
        run_signature: str = "",
    ) -> None:
        self.output_root = Path(output_root)
        self.log_root = self.output_root / "logs"
        self.status_root = self.output_root / "status"
        self.log_root.mkdir(parents=True, exist_ok=True)
        self.status_root.mkdir(parents=True, exist_ok=True)
        self.dry_run = bool(dry_run)
        self.resume = bool(resume)
        self.run_signature = str(run_signature)

    def completed(self, stage: str) -> bool:
        path = self.status_root / f"{stage}.json"
        if not self.resume or not path.is_file():
            return False
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return bool(
                payload.get("state") == "completed"
                and payload.get("run_signature", "") == self.run_signature
            )
        except (json.JSONDecodeError, OSError):
            return False

    def _record(self, stage: str, payload: Mapping[str, Any]) -> None:
        path = self.status_root / f"{stage}.json"
        record = dict(payload)
        record["run_signature"] = self.run_signature
        path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")

    def run(
        self,
        stage: str,
        command: Sequence[str],
        *,
        env: Mapping[str, str] | None = None,
    ) -> None:
        command_list = [str(value) for value in command]
        if self.completed(stage):
            print(f"[{stage}] already completed; resume skip", flush=True)
            return
        print(f"[{stage}] {' '.join(command_list)}", flush=True)
        if self.dry_run:
            self._record(stage, {"state": "dry-run", "command": command_list})
            return
        started = time.time()
        self._record(stage, {"state": "running", "command": command_list, "started": started})
        process_env = _process_environment()
        if env:
            process_env.update({key: str(value) for key, value in env.items()})
        log_path = self.log_root / f"{stage}.log"
        with log_path.open("a", encoding="utf-8") as log:
            log.write("\n$ " + " ".join(command_list) + "\n")
            log.flush()
            result = subprocess.run(
                command_list,
                cwd=str(REPO_ROOT),
                env=process_env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        payload = {
            "state": "completed" if result.returncode == 0 else "failed",
            "command": command_list,
            "returncode": int(result.returncode),
            "started": started,
            "finished": time.time(),
            "log": str(log_path.resolve()),
        }
        self._record(stage, payload)
        if result.returncode != 0:
            raise RuntimeError(f"Stage {stage} failed; see {log_path}")

    def run_parallel(
        self,
        stage: str,
        jobs: Sequence[Tuple[Sequence[str], Mapping[str, str]]],
    ) -> None:
        if self.completed(stage):
            print(f"[{stage}] already completed; resume skip", flush=True)
            return
        if self.dry_run:
            for index, (command, env) in enumerate(jobs):
                self.run(f"{stage}_worker{index}", command, env=env)
            return
        started = time.time()
        processes = []
        commands = []
        logs = []
        for index, (command, extra_env) in enumerate(jobs):
            command_list = [str(value) for value in command]
            log_path = self.log_root / f"{stage}_worker{index}.log"
            handle = log_path.open("a", encoding="utf-8")
            handle.write("\n$ " + " ".join(command_list) + "\n")
            handle.flush()
            process_env = _process_environment()
            process_env.update({key: str(value) for key, value in extra_env.items()})
            process = subprocess.Popen(
                command_list,
                cwd=str(REPO_ROOT),
                env=process_env,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
            processes.append((process, handle, log_path))
            commands.append(command_list)
            logs.append(str(log_path.resolve()))
        self._record(
            stage,
            {"state": "running", "commands": commands, "logs": logs, "started": started},
        )
        failures = []
        for index, (process, handle, log_path) in enumerate(processes):
            returncode = process.wait()
            handle.close()
            if returncode != 0:
                failures.append((index, returncode, log_path))
        self._record(
            stage,
            {
                "state": "failed" if failures else "completed",
                "commands": commands,
                "logs": logs,
                "started": started,
                "finished": time.time(),
                "failures": [
                    {"worker": index, "returncode": code, "log": str(path)}
                    for index, code, path in failures
                ],
            },
        )
        if failures:
            raise RuntimeError(f"Parallel stage {stage} failed: {failures}")

    def run_callable(self, stage: str, function: Any) -> Any:
        if self.completed(stage):
            print(f"[{stage}] already completed; resume skip", flush=True)
            return None
        if self.dry_run:
            print(f"[{stage}] callable dry-run", flush=True)
            self._record(stage, {"state": "dry-run", "kind": "callable"})
            return None
        started = time.time()
        self._record(stage, {"state": "running", "kind": "callable", "started": started})
        try:
            result = function()
        except Exception as error:
            self._record(
                stage,
                {
                    "state": "failed",
                    "kind": "callable",
                    "started": started,
                    "finished": time.time(),
                    "error": repr(error),
                },
            )
            raise
        self._record(
            stage,
            {
                "state": "completed",
                "kind": "callable",
                "started": started,
                "finished": time.time(),
            },
        )
        return result


def _python(python: str | None) -> str:
    return str(python or sys.executable)


def train_image_targets(
    config: Mapping[str, Any],
    runner: StageRunner,
    *,
    datasets: Sequence[str],
    devices: Sequence[str],
    python: str | None = None,
) -> None:
    paths = config["paths"]
    training = config["image"]["target_training"]
    dataset_names = config["image"]["dataset_names"]
    target_root = Path(paths["image_target_root"])
    for index, dataset in enumerate(datasets):
        device = devices[index % len(devices)] if devices else "0"
        command = [
            _python(python),
            str(WORKFLOW_ROOT / "train_image_target.py"),
            "--dataset",
            str(dataset_names[dataset]),
            "--augmentation",
            str(training["augmentation"]),
            "--output-dir",
            str(target_root / dataset),
            "--data-root",
            str(paths["image_data_root"]),
            "--seed",
            str(training["seed"]),
            "--epochs",
            str(training["epochs"]),
            "--batch-size",
            str(training["batch_size"]),
            "--lr",
            str(training["lr"]),
            "--weight-decay",
            str(training["weight_decay"]),
        ]
        runner.run(
            f"train_image_target_{dataset}",
            command,
            env={"CUDA_VISIBLE_DEVICES": str(device)},
        )


def train_image_references(
    config: Mapping[str, Any],
    runner: StageRunner,
    *,
    datasets: Sequence[str],
    devices: Sequence[str],
    python: str | None = None,
) -> None:
    paths = config["paths"]
    training = config["image"]["reference_training"]
    dataset_names = [config["image"]["dataset_names"][dataset] for dataset in datasets]
    common = [
        _python(python),
        str(WORKFLOW_ROOT / "train_image_references.py"),
        "--output-root",
        str(paths["image_reference_root"]),
        "--data-root",
        str(paths["image_data_root"]),
        "--datasets",
        ",".join(dataset_names),
        "--augmentation",
        str(training["augmentation"]),
        "--architecture",
        str(training["architecture"]),
        "--num-references",
        str(training["num_references"]),
        "--design-seed",
        str(training["design_seed"]),
        "--seed-base",
        str(training["model_seed_base"]),
        "--epochs",
        str(training["epochs"]),
        "--batch-size",
        str(training["batch_size"]),
        "--lr",
        str(training["lr"]),
        "--weight-decay",
        str(training["weight_decay"]),
        "--reference-ids",
        f"0-{int(training['num_references']) - 1}",
    ]
    runner.run("prepare_image_reference_splits", common + ["--prepare-only"])
    worker_devices = list(devices or [""])
    jobs = []
    for worker_id, device in enumerate(worker_devices):
        command = common + [
            "--skip-prepare",
            "--skip-existing",
            "--worker-id",
            str(worker_id),
            "--num-train-workers",
            str(len(worker_devices)),
        ]
        env = {"CUDA_VISIBLE_DEVICES": str(device)} if device != "" else {}
        jobs.append((command, env))
    runner.run_parallel("train_image_references", jobs)


def cache_image_outputs(
    config: Mapping[str, Any],
    runner: StageRunner,
    *,
    datasets: Sequence[str],
    devices: Sequence[str],
    python: str | None = None,
) -> None:
    paths = config["paths"]
    cache = config["image"]["cache"]
    common = [
        _python(python),
        str(WORKFLOW_ROOT / "cache_image_outputs.py"),
        "--output-root",
        str(paths["image_cache_root"]),
        "--data-root",
        str(paths["image_data_root"]),
        "--datasets",
        ",".join(datasets),
        "--shadow-root",
        str(paths["image_reference_root"]),
        "--model-root",
        str(paths["image_target_root"]),
        "--include-targets",
        "--max-shadows-per-dataset",
        str(config["image"]["reference_training"]["num_references"]),
        "--views",
        str(cache["views"]),
        "--seed",
        str(cache["seed"]),
        "--microbatch",
        "2048",
    ]
    augmentation = str(config["image"]["target_training"]["augmentation"])
    if augmentation == "stdaug":
        common.extend(["--unique-stdaug-views", "--sample-index-views"])
    worker_devices = list(devices or ["cpu"])
    jobs = []
    for worker_id, device in enumerate(worker_devices):
        use_cpu = str(device).lower() == "cpu"
        command = common + [
            "--worker-index",
            str(worker_id),
            "--num-workers-total",
            str(len(worker_devices)),
            "--device",
            "cpu" if use_cpu else "cuda:0",
        ]
        env = {} if use_cpu else {"CUDA_VISIBLE_DEVICES": str(device)}
        jobs.append((command, env))
    runner.run_parallel("cache_image_outputs", jobs)


def train_tabular_models(
    config: Mapping[str, Any],
    runner: StageRunner,
    *,
    datasets: Sequence[str],
    device: str,
    python: str | None = None,
) -> None:
    paths = config["paths"]
    training = config["tabular"]["model_training"]
    command = [_python(python), str(MODEL_ROOT / "tabular.py")]
    for dataset in datasets:
        command.extend(["--dataset", dataset])
    command.extend(
        [
            "--output-root",
            str(paths["tabular_task_root"]),
            "--data-root",
            str(paths["tabular_data_root"]),
            "--device",
            "cpu" if str(device).lower() == "cpu" else "cuda:0",
            "--seed",
            str(training["seed"]),
            "--reference-count",
            str(training["reference_count"]),
            "--target-train-fraction",
            str(training["target_train_fraction"]),
            "--epochs",
            str(training["epochs"]),
            "--batch-size",
            str(training["batch_size"]),
            "--score-batch-size",
            str(training["score_batch_size"]),
            "--lr",
            str(training["lr"]),
            "--weight-decay",
            str(training["weight_decay"]),
            "--optimizer",
            str(training["optimizer"]),
        ]
    )
    env = {
        "INF2GUARD_REPO_ROOT": str(REPO_ROOT),
        "CUDA_VISIBLE_DEVICES": "" if str(device).lower() == "cpu" else str(device),
    }
    runner.run("train_tabular_models", command, env=env)


def quality_control_tabular_models(
    config: Mapping[str, Any],
    runner: StageRunner,
    *,
    datasets: Sequence[str],
    devices: Sequence[str],
    python: str | None = None,
) -> None:
    paths = config["paths"]
    quality = config["tabular"]["quality_control"]
    for index, dataset in enumerate(datasets):
        device = devices[index % len(devices)] if devices else "cpu"
        command = [
            _python(python),
            str(WORKFLOW_ROOT / "quality_control_tabular.py"),
            "--task-dir",
            str(Path(paths["tabular_task_root"]) / dataset),
            "--task",
            dataset,
            "--data-dir",
            str(paths["tabular_data_root"]),
            "--device",
            "cpu" if str(device).lower() == "cpu" else "cuda",
            "--threshold",
            str(quality["minimum_train_accuracy"]),
            "--score-batch-size",
            "2048",
            "--early-stop-acc",
            str(quality["early_stop_accuracy"]),
            "--early-stop-min-epochs",
            str(quality["early_stop_min_epochs"]),
        ]
        runner.run(
            f"quality_control_tabular_{dataset}",
            command,
            env={
                "CUDA_VISIBLE_DEVICES": ""
                if str(device).lower() == "cpu"
                else str(device)
            },
        )


def cache_tabular_outputs(
    config: Mapping[str, Any],
    runner: StageRunner,
    *,
    datasets: Sequence[str],
    python: str | None = None,
) -> None:
    paths = config["paths"]
    task_root = Path(paths["tabular_task_root"])
    cache_root = Path(paths["tabular_cache_root"])
    for dataset in datasets:
        dataset_root = cache_root / dataset
        if not runner.dry_run:
            dataset_root.mkdir(parents=True, exist_ok=True)
            for name in ("roles", "shadows"):
                link = dataset_root / name
                target = task_root / dataset / name
                if link.exists() or link.is_symlink():
                    if link.is_symlink() and link.resolve() == target.resolve():
                        continue
                    raise FileExistsError(f"Refusing to replace {link}")
                link.symlink_to(target.resolve(), target_is_directory=True)
        command = [
            _python(python),
            str(WORKFLOW_ROOT / "cache_tabular_outputs.py"),
            "--task-dir",
            str(task_root / dataset),
            "--output-dir",
            str(dataset_root),
            "--reference-count",
            "256",
            "--expected-out",
            "128",
        ]
        runner.run(f"cache_tabular_{dataset}", command)


def doctor(
    config: Mapping[str, Any],
    *,
    image_enabled: bool = True,
    tabular_enabled: bool = True,
) -> Dict[str, Any]:
    required_scripts = [
        MODEL_ROOT / "image.py",
        WORKFLOW_ROOT / "train_image_target.py",
        WORKFLOW_ROOT / "train_image_references.py",
        WORKFLOW_ROOT / "cache_image_outputs.py",
        MODEL_ROOT / "tabular.py",
        WORKFLOW_ROOT / "quality_control_tabular.py",
        WORKFLOW_ROOT / "cache_tabular_outputs.py",
    ]
    checks: List[Dict[str, Any]] = []
    for path in required_scripts:
        checks.append({"kind": "code", "path": str(path), "ok": path.is_file()})
    paths = config["paths"]
    data_keys = []
    if image_enabled:
        data_keys.append("image_data_root")
    if tabular_enabled:
        data_keys.append("tabular_data_root")
    for key in data_keys:
        path = Path(paths[key])
        checks.append({"kind": "data", "path": str(path), "ok": path.exists()})
    report = {
        "mode": "source-to-results",
        "python": sys.version,
        "repo_root": str(REPO_ROOT),
        "checks": checks,
        "ok": all(bool(row["ok"]) for row in checks),
    }
    artifact = Path(config["paths"]["image_target_root"])
    existing = artifact
    while not existing.exists() and existing != existing.parent:
        existing = existing.parent
    usage = shutil.disk_usage(existing)
    report["storage"] = {
        "filesystem_probe": str(existing),
        "free_bytes": int(usage.free),
        "free_gib": float(usage.free / (1024**3)),
        "recommended_free_gib": 160,
        "enough_for_recommended_full_run": bool(usage.free >= 160 * 1024**3),
    }
    if not report["ok"]:
        missing = [row["path"] for row in checks if not row["ok"]]
        raise FileNotFoundError(f"Doctor found missing dependencies: {missing}")
    return report
