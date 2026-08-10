"""Gunicorn config hook for the World-R1 strict companion service.

Bind, worker count/class, threads and timeouts live exclusively in the two
frozen Gunicorn commands documented in README.md. These hooks import only
lightweight supervision/lifecycle modules, never the app, World-R1, Torch or
another heavy dependency.
"""

import os


def on_starting(server):  # noqa: ANN001 - signature fixed by Gunicorn
    """Make the master reap manager descendants if a worker dies abruptly."""

    del server
    from services.world_r1_strict.process_supervision import (
        configure_child_subreaper,
    )

    configure_child_subreaper()


def _close_worker_manager() -> None:
    from services.world_r1_strict import reference_contract

    reference_contract.close_registered_manager(expected_pid=os.getpid())


def worker_int(worker):  # noqa: ANN001 - signature fixed by Gunicorn
    """Close before the worker exits on SIGINT or SIGQUIT."""

    del worker
    _close_worker_manager()


def worker_abort(worker):  # noqa: ANN001 - signature fixed by Gunicorn
    """Close before the worker exits after a Gunicorn timeout."""

    del worker
    _close_worker_manager()


def worker_exit(server, worker):  # noqa: ANN001 - signature fixed by Gunicorn
    """Close the registered manager exactly once inside the exiting worker."""

    del server, worker
    _close_worker_manager()
