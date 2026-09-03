import jax
from mujoco import mjx
import mujoco
from typing import Any, Dict, NamedTuple

class State(NamedTuple):
    pipeline_state: mjx.Data
    obs: jax.Array
    reward: jax.Array
    done: jax.Array
    metrics: Dict[str, jax.Array] = {}
    info: Dict[str, Any] = {}

class MJXEnv:
    def __init__(self, mj_model: mujoco.MjModel):
        self.mj_model = mj_model
        self.mjx_model = mjx.put_model(mj_model)

    def reset(self, rng: jax.Array) -> State:
        raise NotImplementedError

    def step(self, state: State, action: jax.Array) -> State:
        raise NotImplementedError

    def _get_obs(self, data: mjx.Data) -> jax.Array:
        raise NotImplementedError
