from pathlib import Path
from typing import Any, Mapping, Optional

from climbing_env_mjx.mjx.tasks import climber_walking
from climbing_env_mjx.mjx.tasks.make_model import make_model

CLIMBING = "climbing_env_mjx-climber-v1"
CLIMB = "climbing_env_mjx-climb-v1"
CLIMB_FINGERS = "climbing_env_mjx-climb-fingers-v1"
HALFCHEETAH = "mjx-halfcheetah-v1"
_HALFCHEETAH_ALIASES = {HALFCHEETAH, "brax-halfcheetah-v1"}
HUMANOID = "mjx-humanoid-v1"

def load(
    environment_name: str,
    midi_file: Optional[Path] = None,
    seed: Optional[int] = None,
    stretch: float = 1.0,
    shift: int = 0,
    recompile_physics: bool = False,
    legacy_step: bool = True,
    task_kwargs: Optional[Mapping[str, Any]] = None,
) -> Any:
    task_kwargs = task_kwargs or {}

    if environment_name == CLIMBING:
        allowed_kwargs = {
            'max_episode_steps'
        }
        filtered_kwargs = {k: v for k, v in task_kwargs.items() if k in allowed_kwargs}

        mj_model = make_model()
        return climber_walking.ClimberWalkingMJX(mj_model, **filtered_kwargs)

    if environment_name == CLIMB:
        from climbing_env_mjx.mjx.tasks import climber_climbing
        wall_kwargs = {k: v for k, v in task_kwargs.items() if k == "wall"}
        return climber_climbing.ClimberClimbingMJX(**wall_kwargs)

    if environment_name == CLIMB_FINGERS:
        from climbing_env_mjx.mjx.tasks import climber_fingers_climbing
        wall_kwargs = {k: v for k, v in task_kwargs.items() if k == "wall"}
        return climber_fingers_climbing.ClimberFingersClimbingMJX(**wall_kwargs)

    if environment_name in _HALFCHEETAH_ALIASES:
        from climbing_env_mjx.mjx.tasks.halfcheetah import HalfCheetahMJX
        return HalfCheetahMJX()

    if environment_name == HUMANOID:
        from climbing_env_mjx.mjx.tasks.humanoid import HumanoidMJX
        return HumanoidMJX()

    raise ValueError(
        f"Unknown environment: {environment_name}. "
        f"Expected one of {CLIMBING}, {CLIMB}, {CLIMB_FINGERS}, {HALFCHEETAH}, {HUMANOID}"
    )

__all__ = [
    "CLIMBING",
    "CLIMB",
    "CLIMB_FINGERS",
    "HALFCHEETAH",
    "HUMANOID",
    "load",
]
