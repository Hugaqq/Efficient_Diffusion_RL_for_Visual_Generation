"""07: Runner 只协调 rollout、feedback、optimizer 和 artifacts。"""


class FakeRollout:
    def sample(self, prompts):
        return {"prompts": prompts, "media": [1, 2]}


class FakeFeedback:
    def score(self, batch):
        return {"rewards": [0.2, 0.8], "batch": batch}


class FakeOptimizerPlugin:
    def step(self, batch, rewards, context):
        return {"loss": 0.5, "reward_mean": sum(rewards["rewards"]) / 2}


class FakeArtifacts:
    def __init__(self):
        self.rows = []

    def record(self, **row):
        self.rows.append(row)


class ExperimentRunner:
    def __init__(self):
        self.rollout = FakeRollout()
        self.feedback = FakeFeedback()
        self.optimizer = FakeOptimizerPlugin()
        self.artifacts = FakeArtifacts()

    def run_step(self, step, prompts):
        # FILL_ME 1: rollout.sample。
        # FILL_ME 2: feedback.score。
        # FILL_ME 3: optimizer.step，并由 Runner 加入 step metric。
        # FILL_ME 4: artifacts.record 后返回 metrics。
        raise NotImplementedError("FILL_ME")


def check() -> None:
    runner = ExperimentRunner()
    metrics = runner.run_step(4, ["red", "blue"])
    assert metrics == {"step": 4, "loss": 0.5, "reward_mean": 0.5}
    assert len(runner.artifacts.rows) == 1
    assert runner.artifacts.rows[0]["metrics"] == metrics
    assert runner.artifacts.rows[0]["batch"]["prompts"] == ["red", "blue"]
    print("07 PASS")


if __name__ == "__main__":
    check()
