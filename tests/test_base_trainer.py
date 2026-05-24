def test_base_trainer_optimizer_accepts_string_numeric_train_config(tmp_path):
    import torch

    from visual_rl.configs.schema import VisualRLConfig
    from visual_rl.trainer.base import BaseTrainer

    class MinimalTrainer(BaseTrainer):
        def train(self, *args, **kwargs):
            return {}

    cfg = VisualRLConfig(run_name="optimizer_string_values", output_dir=str(tmp_path))
    cfg.paths.output_dir = str(tmp_path)
    cfg.train.learning_rate = "0.001"
    cfg.train.adam_beta1 = "0.8"
    cfg.train.adam_beta2 = "0.95"
    cfg.train.adam_weight_decay = "0.01"
    cfg.train.adam_epsilon = "1e-06"

    trainer = MinimalTrainer(cfg)
    parameter = torch.nn.Parameter(torch.ones(1))

    optimizer = trainer.setup_optimizer([parameter])

    assert isinstance(optimizer, torch.optim.AdamW)
    group = optimizer.param_groups[0]
    assert group["lr"] == 0.001
    assert group["betas"] == (0.8, 0.95)
    assert group["weight_decay"] == 0.01
    assert group["eps"] == 1e-06
    assert (tmp_path / "config.resolved.json").exists()
