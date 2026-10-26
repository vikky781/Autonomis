"""Immutable JSON records; stage files are published only after complete writes."""

import json
import os
import tempfile
from pathlib import Path


def write_json(path, value):
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    return write_bytes(path, payload.encode("utf-8"))


def write_bytes(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"Refusing to overwrite immutable record: {path}")
        return path
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        # Link publishes a complete file and fails if another writer already claimed the name.
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return path


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))
