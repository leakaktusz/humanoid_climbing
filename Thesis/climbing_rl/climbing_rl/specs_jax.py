from dataclasses import dataclass

import dm_env
import jax
import jax.numpy as jnp

from dm_env import specs


@dataclass(frozen=True)
class EnvironmentSpec:
    observation: specs.Array
    action: specs.Array

    @staticmethod
    def make(env: dm_env.Environment) -> "EnvironmentSpec":
        return EnvironmentSpec(
            observation=env.observation_spec(),
            action=env.action_spec(),
        )

    def sample_action(self, key: jax.Array) -> jnp.array:
        if not isinstance(self.action, specs.BoundedArray):
            raise ValueError("Only BoundedArray action specs are supported.")

        action = jax.random.uniform(
            key,
            shape=self.action.shape,
            minval=self.action.minimum,
            maxval=self.action.maximum,
            dtype=self.action.dtype
        )
        return action

    @property
    def observation_dim(self) -> int:
        return self.observation.shape[-1]

    @property
    def action_dim(self) -> int:
        return self.action.shape[-1]


def zeros_like(spec: specs.Array) -> jnp.array:
    return jax.tree_util.tree_map(lambda x: jnp.zeros(x.shape, x.dtype), spec)
