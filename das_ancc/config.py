from dataclasses import dataclass
from typing import Any
import yaml


@dataclass
class Config:
    data: dict[str, Any]
    preprocess: dict[str, Any]
    correlate: dict[str, Any]
    stack: dict[str, Any]
    dispersion: dict[str, Any]
    imaging: dict[str, Any]
    parallel: dict[str, Any]


def load_config(path: str) -> Config:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Config(**raw)
