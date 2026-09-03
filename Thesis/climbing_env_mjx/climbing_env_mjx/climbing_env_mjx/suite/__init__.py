from pathlib import Path
from typing import Any, Mapping, Optional

from dm_control import composer
from mujoco_utils import composer_utils

from climbing_env_mjx.suite.tasks import climber_walking

CLIMBING = "climbing_env_mjx-climber-v0"


class Environment(composer_utils.Environment):
    def get_statistics(self):
        if hasattr(self.task, 'get_statistics'):
            return self.task.get_statistics()
        return {}


def load(
    environment_name: str,
    midi_file: Optional[Path] = None,
    seed: Optional[int] = None,
    stretch: float = 1.0,
    shift: int = 0,
    recompile_physics: bool = False,
    legacy_step: bool = True,
    task_kwargs: Optional[Mapping[str, Any]] = None,
) -> composer.Environment:
    task_kwargs = task_kwargs or {}

    if environment_name == CLIMBING:
        allowed_kwargs = {
            'max_episode_steps'
        }
        filtered_kwargs = {k: v for k, v in task_kwargs.items() if k in allowed_kwargs}

        return Environment(
            task=climber_walking.ClimberForWalking(**filtered_kwargs),
            random_state=seed,
            strip_singleton_obs_buffer_dim=True,
            recompile_physics=recompile_physics,
            legacy_step=legacy_step,
        )

__all__ = [
    "ALL",
    "CLIMBING",
    "load",
]
