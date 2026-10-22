"""Scalar-only progress events; no scheduling, RNG draws, tensors or handlers."""
import json
import logging
import time


def progress(event, **fields):
    logging.getLogger("visual_rl.flash140").info(json.dumps(
        {"time_unix": time.time(), "event": event, **fields},
        sort_keys=True, allow_nan=False))
