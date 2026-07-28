"""WSGI app for the strict World-R1 general (HPS) reward origin.

Exposes ``create_app(*, manager, server_revision)`` for contract tests with
fake managers and zero-argument ``build_app()`` for deployment.  No World-R1,
Torch or CUDA import exists at module top level.
"""

from __future__ import annotations

import os
from typing import Any

from visual_rl.world_r1_protocol import (
    ERROR_COMPUTE_FAILED,
    ERROR_INVALID_REQUEST,
    ERROR_MANAGER_NOT_READY,
    ERROR_REVISION_MISMATCH,
    HEALTH_ROUTE,
    REWARD_GENERAL,
    SCORE_ROUTE,
    WorldR1ProtocolError,
    WorldR1RevisionError,
    build_health_payload,
    validate_server_revision,
)
from services.world_r1_strict import protocol, reference_contract

REWARD = REWARD_GENERAL


def _state(app: Any) -> dict[str, Any]:
    return app.extensions["world_r1_strict"]


def _close_inherited_manager(state: dict[str, Any]) -> None:
    if state["inherited_closed"]:
        return
    state["inherited_closed"] = True
    try:
        state["manager"].shutdown()
    except Exception:  # noqa: BLE001 - closing an inherited manager is best-effort
        pass


def _pid_guard(app: Any):
    from flask import jsonify

    state = _state(app)
    if os.getpid() == state["created_pid"]:
        return None
    _close_inherited_manager(state)
    body, status = jsonify(protocol.error_body(ERROR_MANAGER_NOT_READY)), 503
    return body, status


def create_app(*, manager: Any, server_revision: str):
    """Build the Flask app around an injected (possibly fake) strict manager."""

    from flask import Flask, jsonify, request

    revision = validate_server_revision(server_revision)
    reference_contract.require_strict_manager(manager, reward=REWARD)

    app = Flask(__name__)
    app.extensions["world_r1_strict"] = {
        "manager": manager,
        "created_pid": os.getpid(),
        "inherited_closed": False,
    }
    reference_contract.register_manager(manager=manager, pid=os.getpid())

    @app.get(HEALTH_ROUTE)
    def healthz():  # noqa: ANN202
        guard = _pid_guard(app)
        if guard is not None:
            return guard
        if not manager.is_ready():
            return jsonify(protocol.error_body(ERROR_MANAGER_NOT_READY)), 503
        return jsonify(build_health_payload(reward=REWARD, server_revision=revision)), 200

    @app.post(SCORE_ROUTE)
    def score():  # noqa: ANN202
        guard = _pid_guard(app)
        if guard is not None:
            return guard
        if not manager.is_ready():
            return jsonify(protocol.error_body(ERROR_MANAGER_NOT_READY)), 503
        try:
            score_request = protocol.decode_score_request(
                request.get_data(),
                content_type=request.content_type,
                reward=REWARD,
                server_revision=revision,
            )
        except WorldR1RevisionError:
            return jsonify(protocol.error_body(ERROR_REVISION_MISMATCH)), 409
        except WorldR1ProtocolError:
            return jsonify(protocol.error_body(ERROR_INVALID_REQUEST)), 400
        try:
            outputs = manager.compute_batch_scores(
                list(score_request.images), list(score_request.prompts)
            )
            body = protocol.encode_score_response(
                server_revision=revision,
                sample_ids=score_request.sample_ids,
                outputs=outputs,
            )
        except Exception:  # noqa: BLE001 - any compute failure is fail-closed HTTP 500
            return jsonify(protocol.error_body(ERROR_COMPUTE_FAILED)), 500
        return app.response_class(
            response=protocol.encode_response_body(body),
            status=200,
            mimetype="application/json",
        )

    return app


def _load_manager_class() -> type:
    from reward_server.general_reward import MultiGPUGeneralRewardManager

    return MultiGPUGeneralRewardManager


def build_app():
    """Zero-argument deployment entry point used by the frozen Gunicorn command."""

    revision = validate_server_revision(os.environ["WORLD_R1_SERVER_REVISION"])
    reference_contract.require_service_runtime()
    manager_class = _load_manager_class()
    reference_contract.require_strict_manager(manager_class, reward=REWARD)
    reference_contract.run_native_fault_injection_gate(manager_class, reward=REWARD)
    manager = manager_class()
    try:
        manager.initialize()
    except BaseException:
        manager.shutdown()
        raise
    return create_app(manager=manager, server_revision=revision)


__all__ = ("REWARD", "build_app", "create_app")
