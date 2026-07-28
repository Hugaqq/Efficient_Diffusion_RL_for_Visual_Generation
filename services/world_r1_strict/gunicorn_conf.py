"""Gunicorn config hook for the World-R1 strict companion service.

This module deliberately defines only the ``worker_exit`` hook.  Bind, worker
count/class, threads and timeouts live exclusively in the two frozen Gunicorn
commands documented in README.md; this file must not import the app, World-R1,
Torch or any heavy dependency.
"""

import os


def worker_exit(server, worker):  # noqa: ANN001 - signature fixed by Gunicorn
    """Close the registered manager exactly once inside the exiting worker."""

    del server, worker
    from services.world_r1_strict import reference_contract

    reference_contract.close_registered_manager(expected_pid=os.getpid())
