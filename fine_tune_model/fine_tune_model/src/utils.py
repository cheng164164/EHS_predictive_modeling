import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_path(path: str | Path, base: str | Path | None = None) -> Path:
    p = Path(path).expanduser()
    if p.is_absolute():
        return p
    candidates = []
    if base is not None:
        candidates.append(Path(base) / p)
    candidates.append(Path.cwd() / p)
    candidates.append(project_root() / p)
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


def load_config(path: str) -> Dict[str, Any]:
    cfg_path = resolve_path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(
            f'Config file not found: {path}. Tried: {cfg_path}. '
            'Run from the project root or pass --config /full/path/to/config.yaml.'
        )
    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    cfg['_config_path'] = str(cfg_path)
    cfg['_project_root'] = str(cfg_path.parents[1]) if cfg_path.parent.name == 'configs' else str(project_root())
    return cfg


def load_yaml(path: str | Path) -> Dict[str, Any]:
    p = resolve_path(path)
    with open(p, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def ensure_dir(path: str | Path) -> Path:
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = Path.cwd() / p
    p.mkdir(parents=True, exist_ok=True)
    return p


def cfg_path(cfg: Dict[str, Any], key_path: str, default: str | None = None) -> Path:
    cur: Any = cfg
    for k in key_path.split('.'):
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            if default is None:
                raise KeyError(key_path)
            cur = default
            break
    p = Path(str(cur)).expanduser()
    if p.is_absolute():
        return p
    return Path(cfg.get('_project_root', project_root())) / p


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def write_jsonl(records: List[Dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    if not path.is_absolute():
        path = Path.cwd() / path
    ensure_dir(path.parent)
    with open(path, 'w', encoding='utf-8') as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows = []
    path = Path(path)
    if not path.is_absolute():
        path = Path.cwd() / path
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def safe_str(x: Any) -> str:
    if x is None:
        return ''
    if isinstance(x, float) and np.isnan(x):
        return ''
    return str(x).strip()


def compact_space(text: str) -> str:
    return ' '.join(safe_str(text).replace('\r', ' ').replace('\n', ' ').split())
