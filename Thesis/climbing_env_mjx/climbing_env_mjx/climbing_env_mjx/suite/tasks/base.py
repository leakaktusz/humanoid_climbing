from dm_control import composer
from climbing_env_mjx.models.humanoid.humanoid import Humanoid
from mujoco_utils import composer_utils
from climbing_env_mjx.models.floor.floor import Floor
from climbing_env_mjx.models.climbing_wall.climbing_wall_easy import ClimbingWall
_PHYSICS_TIMESTEP = 0.005
_CONTROL_TIMESTEP = 0.05

class WallAndBody(composer.Task):
    def __init__(
        self,
        arena: composer_utils.Arena,
        physics_timestep: float = _PHYSICS_TIMESTEP,
        control_timestep: float = _CONTROL_TIMESTEP,
    ) -> None:
        self._arena = arena
        self.climbing_wall= ClimbingWall()
        self.arena.attach(self.climbing_wall)
        self.body = Humanoid()
        self.arena.attach(self.body)
        self.set_timesteps(
        control_timestep=control_timestep, physics_timestep=physics_timestep
        )

    @property
    def root_entity(self):
        return self._arena

    @property
    def arena(self):
        return self._arena

    def get_reward(self, physics) -> float:
        del physics
        return 0.0

class FloorAndBody(composer.Task):
    def __init__(
        self,
        arena: composer_utils.Arena,
        physics_timestep: float = _PHYSICS_TIMESTEP,
        control_timestep: float = _CONTROL_TIMESTEP,
    ) -> None:
        self._arena = arena
        self.floor= Floor()
        self.arena.attach(self.floor)
        self.body = Humanoid()
        self.set_timesteps(
        control_timestep=control_timestep, physics_timestep=physics_timestep
        )
        wrapper_body=self.arena.attach(self.body)
        wrapper_body.add("freejoint", name="body_root_free")

    @property
    def root_entity(self):
        return self._arena

    @property
    def arena(self):
        return self._arena

    def get_reward(self, physics) -> float:
        del physics
        return 0.0
