import argparse
import importlib
import json
from pathlib import Path
import sys
from wsgiref.simple_server import make_server

import yaml

from visual_rl.artifacts.hashing import file_sha256
import visual_rl.feedback.video_hpsv3 as hps


def build_application(
    *,
    checkpoint: str | Path,
    checkpoint_sha256: str,
    runtime_manifest_sha256: str,
    scorer_revision: str,
    config: str | Path,
    base_model_root: str | Path,
    hps_source_root: str | Path,
    device: str,
) -> hps.VideoHPSv3JSONApplication:
    checkpoint = Path(checkpoint).expanduser().resolve(strict=True)
    config = Path(config).expanduser().resolve(strict=True)
    base_model_root = Path(base_model_root).expanduser().resolve(strict=True)
    hps_source_root = Path(hps_source_root).expanduser().resolve(strict=True)
    if not checkpoint.is_file() or not config.is_file() or not base_model_root.is_dir():
        raise ValueError("Video HPSv3 checkpoint/config/base model paths are invalid.")
    package_root = hps_source_root / "hpsv3"
    if not package_root.is_dir():
        raise ValueError("HPS source root must contain the hpsv3 package.")

    declared_identity = hps.video_hpsv3_identity(
        scorer_revision=scorer_revision,
        checkpoint_sha256=checkpoint_sha256,
        runtime_manifest_sha256=runtime_manifest_sha256,
    )
    actual_checkpoint_sha256 = file_sha256(checkpoint)
    if actual_checkpoint_sha256 != declared_identity["checkpoint_sha256"]:
        raise ValueError("Video HPSv3 checkpoint SHA-256 mismatch.")
    actual_runtime_sha256 = hps.video_hpsv3_runtime_manifest_sha256(
        checkpoint_path=checkpoint,
        hps_source_root=hps_source_root,
        config_path=config,
        base_model_root=base_model_root,
    )
    if actual_runtime_sha256 != declared_identity["runtime_manifest_sha256"]:
        raise ValueError("Video HPSv3 runtime manifest SHA-256 mismatch.")

    values = yaml.safe_load(config.read_text(encoding="utf-8"))
    if not isinstance(values, dict) or not isinstance(
        values.get("model_name_or_path"), str
    ):
        raise ValueError("HPSv3 config must define model_name_or_path.")
    configured_base = Path(values["model_name_or_path"]).expanduser()
    if not configured_base.is_absolute():
        configured_base = config.parent / configured_base
    if configured_base.resolve(strict=True) != base_model_root:
        raise ValueError(
            "HPSv3 config model_name_or_path does not match base model root."
        )

    sys.path.insert(0, str(hps_source_root))
    package = importlib.import_module("hpsv3")
    package_file = Path(package.__file__).resolve(strict=True)
    if not package_file.is_relative_to(package_root.resolve(strict=True)):
        raise ValueError("Imported hpsv3 package is not from the declared source root.")
    inferencer = package.HPSv3RewardInferencer(
        device=device,
        config_path=str(config),
        checkpoint_path=str(checkpoint),
    )
    return hps.VideoHPSv3JSONApplication(
        scorer_identity=hps.video_hpsv3_identity(
            scorer_revision=scorer_revision,
            checkpoint_sha256=actual_checkpoint_sha256,
            runtime_manifest_sha256=actual_runtime_sha256,
        ),
        scorer=inferencer,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("checkpoint_sha256")
    parser.add_argument("runtime_manifest_sha256")
    parser.add_argument("scorer_revision")
    parser.add_argument("--config", required=True)
    parser.add_argument("--base-model-root", required=True)
    parser.add_argument("--hps-source-root", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    app = build_application(
        checkpoint=args.checkpoint,
        checkpoint_sha256=args.checkpoint_sha256,
        runtime_manifest_sha256=args.runtime_manifest_sha256,
        scorer_revision=args.scorer_revision,
        config=args.config,
        base_model_root=args.base_model_root,
        hps_source_root=args.hps_source_root,
        device=args.device,
    )

    def wsgi(environ, start_response):
        length = int(environ.get("CONTENT_LENGTH", "0"))
        payload = json.loads(environ["wsgi.input"].read(length))
        body = json.dumps(app.handle(payload), allow_nan=False).encode()
        start_response("200 OK", [("Content-Type", "application/json")])
        return [body]

    with make_server("127.0.0.1", args.port, wsgi) as server:
        server.serve_forever()


if __name__ == "__main__":
    main()
