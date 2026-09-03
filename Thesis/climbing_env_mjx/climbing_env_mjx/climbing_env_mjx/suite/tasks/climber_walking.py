from climbing_env_mjx.models.arenas import stage
from climbing_env_mjx.suite.tasks.base import FloorAndBody
from dm_control.composer.observation import observable
import numpy as np
from climbing_env_mjx.suite import composite_reward
from dm_control import mjcf
from dm_control.utils import rewards

_TRUNK_STAND_HEIGHT = 0.55
_WALK_SPEED = 1.0


class ClimberForWalking(FloorAndBody):
    def __init__(self, **kwargs) -> None:
        super().__init__(arena=stage.Stage(), **kwargs)
        self._trunk_body = self.body.mjcf_model.find('body', 'Trunk')
        self._add_observables()
        self._set_rewards()
        self._t_idx: int = 0
        self._max_episode_steps: int = 500

    def initialize_episode(self, physics, random_state) -> None:
        self.body.set_pose(physics, position=[0, 0, 0])
        self._t_idx = 0

    def should_terminate_episode(self, physics):
        return self._t_idx >= self._max_episode_steps or self._is_fallen(physics)

    def _set_rewards(self) -> None:
        self._reward_fn = composite_reward.CompositeReward(
            standing=self._standing_reward,
            upright=self._upright_reward,
            move=self._move_reward,
            small_control=self._small_control_reward,
        )

    def _add_observables(self) -> None:
        self.body.observables.joint_positions.enabled = True

        def _joint_velocities(physics):
            return physics.bind(self.body.joints).qvel.copy()

        def _trunk_position(physics):
            return physics.bind(self._trunk_body).xpos.copy()

        def _trunk_orientation(physics):
            return physics.bind(self._trunk_body).xmat.reshape(3, 3)[:, 2].copy()

        def _com_velocity(physics):
            return physics.bind(self.body.root_body).subtree_linvel.copy()

        jv_obs = observable.Generic(_joint_velocities)
        jv_obs.enabled = True
        tp_obs = observable.Generic(_trunk_position)
        tp_obs.enabled = True
        to_obs = observable.Generic(_trunk_orientation)
        to_obs.enabled = True
        cv_obs = observable.Generic(_com_velocity)
        cv_obs.enabled = True

        self._task_observables = {
            "joint_velocities": jv_obs,
            "trunk_position": tp_obs,
            "trunk_orientation": to_obs,
            "com_velocity": cv_obs,
        }

    @property
    def task_observables(self):
        return self._task_observables

    def before_step(self, physics, action, random_state) -> None:
        self.body.apply_action(physics, action, random_state)

    def after_step(self, physics, random_state) -> None:
        self._t_idx += 1

    def get_reward(self, physics: mjcf.Physics) -> float:
        standing = self._standing_reward(physics)
        upright = self._upright_reward(physics)
        stand_reward = standing * upright

        ctrl = physics.bind(self.body.actuators).ctrl
        small_control = rewards.tolerance(
            np.linalg.norm(ctrl), bounds=(0, 0), margin=10.0,
            value_at_margin=0, sigmoid='quadratic'
        )
        small_control = (4 + small_control) / 5

        forward_vel = physics.bind(self.body.root_body).subtree_linvel[0]
        move = rewards.tolerance(
            forward_vel,
            bounds=(_WALK_SPEED, float('inf')),
            margin=_WALK_SPEED,
            value_at_margin=0,
            sigmoid='linear',
        )
        move = (5 * move + 1) / 6

        self._reward_fn._reward_terms = {
            'standing': standing,
            'upright': upright,
            'stand_reward': stand_reward,
            'move': move,
            'small_control': small_control,
        }

        return small_control * stand_reward * move

    def _standing_reward(self, physics) -> float:
        trunk_z = physics.bind(self._trunk_body).xpos[2]
        return rewards.tolerance(
            trunk_z,
            bounds=(_TRUNK_STAND_HEIGHT, float('inf')),
            margin=_TRUNK_STAND_HEIGHT / 4,
        )

    def _upright_reward(self, physics) -> float:
        torso_upright = physics.bind(self._trunk_body).xmat[8]
        return rewards.tolerance(
            torso_upright,
            bounds=(0.9, float('inf')),
            sigmoid='linear',
            margin=1.9,
            value_at_margin=0,
        )

    def _move_reward(self, physics) -> float:
        forward_vel = physics.bind(self.body.root_body).subtree_linvel[0]
        return rewards.tolerance(
            forward_vel,
            bounds=(_WALK_SPEED, float('inf')),
            margin=_WALK_SPEED,
            value_at_margin=0,
            sigmoid='linear',
        )

    def _small_control_reward(self, physics) -> float:
        ctrl = physics.bind(self.body.actuators).ctrl
        return rewards.tolerance(
            np.linalg.norm(ctrl), bounds=(0, 0), margin=10.0,
            value_at_margin=0, sigmoid='quadratic'
        )

    def _is_fallen(self, physics) -> bool:
        trunk_z = physics.bind(self._trunk_body).xpos[2]
        return trunk_z < _TRUNK_STAND_HEIGHT * 0.5
