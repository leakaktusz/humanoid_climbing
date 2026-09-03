import os
from dm_control import composer
from dm_control import mjcf


from dm_control.composer.observation import observable
from dm_env import specs
from mujoco_utils import spec_utils


class HumanoidObservables(composer.Observables):
    @composer.observable
    def joint_positions(self):
        return observable.Generic(self._entity.joint_positions)

class Humanoid(composer.Entity):

    def _build(self):
        self._action_spec = None
        self.name= "body"
        self._mjcf_root = mjcf.from_path(
            os.path.join(os.path.dirname(__file__),'booster_t1', 't1.xml'))

    @property
    def mjcf_model(self):
        return self._mjcf_root

    @property
    def actuators(self):
        return self._mjcf_root.find_all('actuator')

    def apply_action(self, physics, action, random_state):
        physics.bind(self.actuators).ctrl = action

    def action_spec(self, physics: mjcf.Physics) -> specs.BoundedArray:
        if self._action_spec is None:
            self._action_spec = spec_utils.create_action_spec(
                physics=physics, actuators=self.body.actuators, prefix=self.body.name
            )

        return self._action_spec

    def _build_observables(self):
        return HumanoidObservables(self)

    @property
    def joints(self):
        return self._mjcf_root.find_all('joint')

    def joint_positions(self, physics):
        return physics.bind(self.joints).qpos
