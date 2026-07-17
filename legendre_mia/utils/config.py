from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping


PACKAGE_ROOT = Path(__file__).resolve().parents[2]


def _detect_repo_root() -> Path:
    configured = os.environ.get("INF2GUARD_REPO_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    for candidate in (PACKAGE_ROOT, *PACKAGE_ROOT.parents):
        if (candidate / "decision-boundary-lab").is_dir() and (
            candidate / "external_mia_baselines"
        ).is_dir():
            return candidate
    # Default to the repository containing this package.
    return PACKAGE_ROOT.parents[1]


REPO_ROOT = _detect_repo_root()
DEFAULT_CONFIG = PACKAGE_ROOT / "config" / "default.json"


def _expand(value: Any, tokens: Mapping[str, str]) -> Any:
    if isinstance(value, dict):
        return {key: _expand(item, tokens) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand(item, tokens) for item in value]
    if isinstance(value, str):
        out = value
        for key, replacement in tokens.items():
            out = out.replace("${" + key + "}", replacement)
        return os.path.expandvars(os.path.expanduser(out))
    return value


def load_config(
    path: Path = DEFAULT_CONFIG,
    *,
    artifact_root: Path,
    output_root: Path,
) -> Dict[str, Any]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    tokens = {
        "REPO_ROOT": str(REPO_ROOT),
        "PACKAGE_ROOT": str(PACKAGE_ROOT),
        "ARTIFACT_ROOT": str(Path(artifact_root).expanduser().resolve()),
        "OUTPUT_ROOT": str(Path(output_root).expanduser().resolve()),
    }
    config = _expand(raw, tokens)
    config["_resolved"] = tokens
    config["_config_path"] = str(Path(path).resolve())
    return config


def write_resolved_config(config: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
