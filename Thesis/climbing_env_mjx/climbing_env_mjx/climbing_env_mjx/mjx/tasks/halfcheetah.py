from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
from dm_env import specs

from climbing_env_mjx.mjx.tasks.base import MJXEnv, State

_XML_PATH = (
    Path(__file__).resolve().parents[2] / "models" / "halfcheetah" / "half_cheetah.xml"
)

_EPISODE_LEN = 1000
_FRAME_SKIP = 5
_FORWARD_REWARD_WEIGHT = 1.0
_CTRL_COST_WEIGHT = 0.1
_RESET_NOISE_SCALE = 0.1


def make_model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(str(_XML_PATH))


class HalfCheetahMJX(MJXEnv):
    def __init__(self, mj_model: mujoco.MjModel = None, **kwargs):
        if mj_model is None:
            mj_model = make_model()
        super().__init__(mj_model)

        self.root_body_id = mj_model.body("torso").id
        self.root_jnt_id = mj_model.joint("rootx").id

        self.n_substeps = _FRAME_SKIP
        self.dt = self.n_substeps * mj_model.opt.timestep

        self._action_shape = (mj_model.nu,)
        self._obs_shape = (mj_model.nq - 1 + mj_model.nv,)

        self._stagger_next_reset = True

    def observation_spec(self):
        return specs.Array(shape=self._obs_shape, dtype=jnp.float32, name="state")

    def action_spec(self):
        return specs.BoundedArray(
            shape=self._action_shape, dtype=jnp.float32,
            minimum=-1.0, maximum=1.0, name="action",
        )

    def _get_obs(self, data: mjx.Data) -> jax.Array:
        return jnp.concatenate([data.qpos[1:], data.qvel])

    def reset(self, rng: jax.Array) -> State:
        stagger = self._stagger_next_reset
        self._stagger_next_reset = False

        data = mjx.make_data(self.mjx_model)
        rng, rq, rv, rs = jax.random.split(rng, 4)
        qpos = self.mjx_model.qpos0 + jax.random.uniform(
            rq, (self.mjx_model.nq,), minval=-_RESET_NOISE_SCALE, maxval=_RESET_NOISE_SCALE
        )
        qvel = _RESET_NOISE_SCALE * jax.random.normal(rv, (self.mjx_model.nv,))
        data = data.replace(qpos=qpos, qvel=qvel)
        data = mjx.forward(self.mjx_model, data)

        if stagger:
            steps0 = jax.random.randint(rs, (), 0, _EPISODE_LEN).astype(jnp.int32)
        else:
            steps0 = jnp.zeros((), dtype=jnp.int32)

        return State(
            pipeline_state=data,
            obs=self._get_obs(data),
            reward=jnp.zeros(()),
            done=jnp.zeros((), dtype=jnp.bool_),
            metrics={"steps": steps0},
            info={"discount": jnp.ones((), dtype=jnp.float32)},
        )

    def step(self, state: State, action: jax.Array) -> State:
        action = jnp.clip(action, -1.0, 1.0).astype(jnp.float32)
        x_before = state.pipeline_state.qpos[0]

        def _phys(d, _):
            return mjx.step(self.mjx_model, d.replace(ctrl=action)), None

        data, _ = jax.lax.scan(_phys, state.pipeline_state, None, self.n_substeps)

        x_after = data.qpos[0]
        forward_vel = (x_after - x_before) / self.dt
        forward_reward = _FORWARD_REWARD_WEIGHT * forward_vel
        ctrl_cost = _CTRL_COST_WEIGHT * jnp.sum(jnp.square(action))
        reward = forward_reward - ctrl_cost

        steps = state.metrics["steps"] + 1
        done = steps >= _EPISODE_LEN

        return State(
            pipeline_state=data,
            obs=self._get_obs(data),
            reward=reward,
            done=done,
            metrics={"steps": steps},
            info={"discount": jnp.ones((), dtype=jnp.float32)},
        )
