import yaml
from types import SimpleNamespace

def _to_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in d.items()})
    return d

def load_config(path):
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return _to_namespace(raw)
