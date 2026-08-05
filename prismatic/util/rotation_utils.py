"""Stable cache keys for RLDS standardization transforms."""

import json
from typing import Any, Dict


def standardize_fn_hash_key(standardize_fn: Any) -> str:
    if standardize_fn is None:
        return ""

    fn_obj = getattr(standardize_fn, "func", standardize_fn)
    payload: Dict[str, Any] = {
        "module": getattr(fn_obj, "__module__", ""),
        "qualname": getattr(fn_obj, "__qualname__", repr(fn_obj)),
    }

    args = getattr(standardize_fn, "args", ())
    keywords = getattr(standardize_fn, "keywords", None) or {}
    if args:
        payload["args"] = [repr(arg) for arg in args]
    if keywords:
        payload["keywords"] = {key: repr(value) for key, value in sorted(keywords.items())}

    return json.dumps(payload, sort_keys=True)
