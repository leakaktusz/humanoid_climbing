import csv
from pathlib import Path
from typing import Callable, Dict

from dm_control import mjcf

Reward = float
RewardFn = Callable[[mjcf.Physics], Reward]


class CompositeReward:

    def __init__(self, **kwargs) -> None:
        self._reward_fns: Dict[str, RewardFn] = {}
        for name, reward_fn in kwargs.items():
            self.add(name, reward_fn)
        self._reward_terms: Dict[str, Reward] = {}

        self._csv_file = Path("reward_logs") / "rewards.csv"
        self._csv_file.parent.mkdir(exist_ok=True)
        self._csv_initialized = False

    def add(self, name: str, reward_fn: RewardFn) -> None:
        self._reward_fns[name] = reward_fn

    def remove(self, name: str) -> None:
        del self._reward_fns[name]

    def compute(self, physics: mjcf.Physics) -> float:
        sum_of_rewards = 0.0
        for name, reward_fn in self._reward_fns.items():
            rew = reward_fn(physics)
            sum_of_rewards += rew
            self._reward_terms[name] = rew

        self._save_to_csv(sum_of_rewards)

        return sum_of_rewards

    def _save_to_csv(self, total_reward: float) -> None:
        if not self._csv_initialized:
            with open(self._csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                header = list(self._reward_terms.keys()) + ['total_reward']
                writer.writerow(header)
            self._csv_initialized = True

        with open(self._csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            row = list(self._reward_terms.values()) + [total_reward]
            writer.writerow(row)

    @property
    def reward_fns(self) -> Dict[str, RewardFn]:
        return self._reward_fns

    @property
    def reward_terms(self) -> Dict[str, Reward]:
        return self._reward_terms
