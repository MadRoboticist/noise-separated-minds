"""Save / load the full system state with a version tag."""

import pickle
from pathlib import Path
from typing import Union

SCHEMA_VERSION = 1


def save(system, path: Union[str, Path]):
    """Serialize with a version header."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": SCHEMA_VERSION, "system": system}
    with path.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def load(path: Union[str, Path]):
    """Restore, checking the schema version."""
    path = Path(path)
    with path.open("rb") as f:
        payload = pickle.load(f)
    if isinstance(payload, dict) and "version" in payload:
        if payload["version"] != SCHEMA_VERSION:
            raise ValueError(
                f"Schema version {payload['version']} does not match {SCHEMA_VERSION}"
            )
        return payload["system"]
    # Legacy pickle of the raw system
    return payload
