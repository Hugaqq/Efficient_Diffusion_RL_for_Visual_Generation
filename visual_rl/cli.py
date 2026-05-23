"""VisualRL command line interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _register_builtin_plugins() -> None:
    import visual_rl.model_adapters.mock  # noqa: F401
    import visual_rl.model_adapters.sd15  # noqa: F401
    import visual_rl.model_adapters.tiny_diffusion  # noqa: F401
    import visual_rl.model_adapters.wan  # noqa: F401
    import visual_rl.model_adapters.sd3  # noqa: F401
    import visual_rl.model_adapters.flux  # noqa: F401
    import visual_rl.model_adapters.qwenimage  # noqa: F401
    import visual_rl.algorithms.flash_grpo  # noqa: F401
    import visual_rl.algorithms.grpo  # noqa: F401
    import visual_rl.algorithms.tempflow_grpo  # noqa: F401
    import visual_rl.rewards.clients  # noqa: F401
    import visual_rl.rewards.image_rewards  # noqa: F401


def smoke_imports(args: argparse.Namespace) -> int:
    _register_builtin_plugins()
    import visual_rl
    from visual_rl.core.registry import ALGORITHMS, MODEL_ADAPTERS, REWARD_CLIENTS

    payload = {
        "visual_rl_version": visual_rl.__version__,
        "algorithms": ALGORITHMS.keys(),
        "model_adapters": MODEL_ADAPTERS.keys(),
        "reward_clients": REWARD_CLIENTS.keys(),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def smoke_mock(args: argparse.Namespace) -> int:
    _register_builtin_plugins()
    from visual_rl.configs.schema import load_config
    from visual_rl.trainer.trainer import VisualRLTrainer

    default_config = Path(__file__).parent / "configs" / "presets" / "world_r1_wan_v02_mock.yaml"
    config = load_config(args.config or default_config)
    if args.output_dir:
        config.output_dir = args.output_dir
    trainer = VisualRLTrainer(config)
    metrics = trainer.train(max_steps=args.steps)
    print(json.dumps({"output_dir": config.output_dir, "metrics": metrics}, indent=2, sort_keys=True))
    return 0


def tempflow_smoke(args: argparse.Namespace) -> int:
    _register_builtin_plugins()
    from visual_rl.configs.schema import load_config
    from visual_rl.trainer.trainer import VisualRLTrainer

    default_config = Path(__file__).parent / "configs" / "presets" / "tempflow_tiny_branching.yaml"
    config = load_config(args.config or default_config)
    if args.output_dir:
        config.output_dir = args.output_dir
        config.paths.output_dir = args.output_dir
    trainer = VisualRLTrainer(config)
    metrics = trainer.train(max_steps=args.steps)
    print(json.dumps({"output_dir": config.output_dir, "metrics": metrics}, indent=2, sort_keys=True))
    return 0


def flash_smoke(args: argparse.Namespace) -> int:
    _register_builtin_plugins()
    from visual_rl.configs.schema import load_config
    from visual_rl.trainer.trainer import VisualRLTrainer

    default_config = Path(__file__).parent / "configs" / "presets" / "flash_tiny_single_step.yaml"
    config = load_config(args.config or default_config)
    if args.output_dir:
        config.output_dir = args.output_dir
        config.paths.output_dir = args.output_dir
    trainer = VisualRLTrainer(config)
    metrics = trainer.train(max_steps=args.steps)
    print(json.dumps({"output_dir": config.output_dir, "metrics": metrics}, indent=2, sort_keys=True))
    return 0


def image_train(args: argparse.Namespace) -> int:
    _register_builtin_plugins()
    from visual_rl.configs.schema import load_config
    from visual_rl.trainer.image_trainer import ImageRLTrainer

    default_config = Path(__file__).parent / "configs" / "presets" / "sd15_lora_rl.yaml"
    config = load_config(args.config or default_config)
    if args.model_path:
        config.model.model_path = args.model_path
        config.paths.pretrained_model = args.model_path
    if args.output_dir:
        config.output_dir = args.output_dir
        config.paths.output_dir = args.output_dir
    trainer = ImageRLTrainer(config)
    metrics = trainer.train(max_steps=args.steps)
    print(json.dumps({"output_dir": config.output_dir, "metrics": metrics}, indent=2, sort_keys=True))
    return 0


def adapter_probe(args: argparse.Namespace) -> int:
    _register_builtin_plugins()
    from visual_rl.configs.schema import load_config, section_to_dict
    from visual_rl.core.registry import MODEL_ADAPTERS

    if args.config:
        config = load_config(args.config)
        model_config = section_to_dict(config.model)
        model_config.setdefault("use_lora", config.use_lora)
        model_config.setdefault("lora_path", config.train.lora_path)
    else:
        model_config = {"name": args.adapter, "model_path": args.model_path or "", "model_family": "image", "extra": {}}
    model_config.setdefault("extra", {})
    if args.device:
        model_config["extra"]["device"] = args.device
    if not args.load:
        model_config["extra"]["defer_load"] = True

    adapter_cls = MODEL_ADAPTERS.get(model_config["name"])
    adapter = adapter_cls(model_config)
    payload = {"adapter": adapter.name, "loaded": bool(args.load), "model_path": model_config.get("model_path", "")}
    if args.load:
        params = list(adapter.parameters())
        payload["trainable_parameter_tensors"] = len(params)
        payload["trainable_parameters"] = int(sum(parameter.numel() for parameter in params))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def wan_plan(args: argparse.Namespace) -> int:
    _register_builtin_plugins()
    from visual_rl.configs.schema import load_config
    from visual_rl.trainer.wan_trainer import WanTrainer

    default_config = Path(__file__).parent / "configs" / "presets" / "wan_runtime_v02_plan.yaml"
    config = load_config(args.config or default_config)
    if args.model_path is not None:
        config.model.model_path = args.model_path
    if args.output_dir:
        config.output_dir = args.output_dir
        config.paths.output_dir = args.output_dir
    trainer = WanTrainer(config)
    print(json.dumps(trainer.build_runtime_plan().to_dict(), indent=2, sort_keys=True))
    return 0


def world_r1_plan(args: argparse.Namespace) -> int:
    from visual_rl.trainer.world_r1_launcher import build_world_r1_launch_plan

    plan = build_world_r1_launch_plan(
        model_path=args.model_path,
        repo_dir=args.repo_dir,
        train_visible_devices=args.gpus,
        output_root=args.output_root,
        smoke=not args.full,
    )
    print(json.dumps(plan.to_dict(), indent=2, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="visual-rl")
    subparsers = parser.add_subparsers(dest="command", required=True)

    imports_parser = subparsers.add_parser("smoke-imports")
    imports_parser.set_defaults(func=smoke_imports)

    mock_parser = subparsers.add_parser("smoke-mock")
    mock_parser.add_argument("--config", default=None)
    mock_parser.add_argument("--output-dir", default=None)
    mock_parser.add_argument("--steps", type=int, default=2)
    mock_parser.set_defaults(func=smoke_mock)

    tempflow_parser = subparsers.add_parser("tempflow-smoke")
    tempflow_parser.add_argument("--config", default=None)
    tempflow_parser.add_argument("--output-dir", default=None)
    tempflow_parser.add_argument("--steps", type=int, default=2)
    tempflow_parser.set_defaults(func=tempflow_smoke)

    flash_parser = subparsers.add_parser("flash-smoke")
    flash_parser.add_argument("--config", default=None)
    flash_parser.add_argument("--output-dir", default=None)
    flash_parser.add_argument("--steps", type=int, default=2)
    flash_parser.set_defaults(func=flash_smoke)

    image_parser = subparsers.add_parser("image-train")
    image_parser.add_argument("--config", default=None)
    image_parser.add_argument("--model-path", default=None)
    image_parser.add_argument("--output-dir", default=None)
    image_parser.add_argument("--steps", type=int, default=1)
    image_parser.set_defaults(func=image_train)

    adapter_parser = subparsers.add_parser("adapter-probe")
    adapter_parser.add_argument("--config", default=None)
    adapter_parser.add_argument("--adapter", default="sd15_lora")
    adapter_parser.add_argument("--model-path", default=None)
    adapter_parser.add_argument("--device", default=None)
    adapter_parser.add_argument("--load", action="store_true")
    adapter_parser.set_defaults(func=adapter_probe)

    plan_parser = subparsers.add_parser("world-r1-plan")
    plan_parser.add_argument("--model-path", required=True)
    plan_parser.add_argument("--repo-dir", default="reference_code/World-R1-main")
    plan_parser.add_argument("--gpus", default="6,7")
    plan_parser.add_argument("--output-root", default="runs/world_r1_v01")
    plan_parser.add_argument("--full", action="store_true")
    plan_parser.set_defaults(func=world_r1_plan)

    wan_parser = subparsers.add_parser("wan-plan")
    wan_parser.add_argument("--config", default=None)
    wan_parser.add_argument("--model-path", default=None)
    wan_parser.add_argument("--output-dir", default=None)
    wan_parser.set_defaults(func=wan_plan)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
