"""Fixed MG1 continuous worker."""

from experiments.v0_7.mg1_nccl import run_mg1_role


if __name__ == "__main__":
    run_mg1_role("mg1_tiny_grpo_c20_continuous")
