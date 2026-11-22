"""Save / load the full system state."""

import pickle
from pathlib import Path
from typing import Union


def save(system, path: Union[str, Path]):
    """Serialize the entire NoiseSeparatedSystem."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(system, f, protocol=pickle.HIGHEST_PROTOCOL)


def load(path: Union[str, Path]):
    """Restore a NoiseSeparatedSystem."""
    path = Path(path)
    with path.open("rb") as f:
        return pickle.load(f)
