from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
from dm_env import specs

from climbing_env_mjx.mjx.tasks.base import MJXEnv, State

_XML_PATH = (
    Path(__file__).resolve().parents[2] / "models" / "humanoid_gym" / "humanoid.xml"
)

_EPISODE_LEN = 1000
_FRAME_SKIP = 5
_FORWARD_REWARD_WEIGHT = 1.25
_CTRL_COST_WEIGHT = 0.1
_HEALTHY_REWARD = 5.0
_HEALTHY_Z = (1.0, 2.0)
_RESET_NOISE_SCALE = 1e-2


def make_model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(str(_XML_PATH))


class HumanoidMJX(MJXEnv):
    def __init__(self, mj_model: mujoco.MjModel = None, **kwargs):
        if mj_model is None:
            mj_model = make_model()
        super().__init__(mj_model)

        self.root_body_id = mj_model.body("torso").id
        self.root_jnt_id = next(
            i for i in range(mj_model.njnt)
            if mj_model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE
        )

        self.n_substeps = _FRAME_SKIP
        self.dt = self.n_substeps * mj_model.opt.timestep

        cr = jnp.asarray(mj_model.actuator_ctrlrange)
        self._ctrl_center = (cr[:, 0] + cr[:, 1]) * 0.5
        self._ctrl_scale = (cr[:, 1] - cr[:, 0]) * 0.5

        self._body_mass = jnp.asarray(mj_model.body_mass)
        self._mass_sum = float(mj_model.body_mass.sum())

        nq, nv, nbody = mj_model.nq, mj_model.nv, mj_model.nbody
        self._action_shape = (mj_model.nu,)
        self._obs_shape = ((nq - 2) + nv + nbody * 10 + nbody * 6 + nv + nbody * 6,)

    def observation_spec(self):
        return specs.Array(shape=self._obs_shape, dtype=jnp.float32, name="state")

    def action_spec(self):
        return specs.BoundedArray(
            shape=self._action_shape, dtype=jnp.float32,
            minimum=-1.0, maximum=1.0, name="action",
        )

    def _mass_center(self, data: mjx.Data) -> jax.Array:
        return (self._body_mass @ data.xipos) / self._mass_sum

    def _get_obs(self, data: mjx.Data) -> jax.Array:
        cfrc = getattr(data, "cfrc_ext", None)
        if cfrc is None:
            cfrc = jnp.zeros((self.mjx_model.nbody, 6))
        return jnp.concatenate([
            data.qpos[2:],
            data.qvel,
            data.cinert.ravel(),
            data.cvel.ravel(),
            data.qfrc_actuator,
            cfrc.ravel(),
        ])

    def reset(self, rng: jax.Array) -> State:
        data = mjx.make_data(self.mjx_model)
        rng, rq, rv = jax.random.split(rng, 3)
        qpos = self.mjx_model.qpos0 + jax.random.uniform(
            rq, (self.mjx_model.nq,), minval=-_RESET_NOISE_SCALE, maxval=_RESET_NOISE_SCALE
        )
        qvel = jax.random.uniform(
            rv, (self.mjx_model.nv,), minval=-_RESET_NOISE_SCALE, maxval=_RESET_NOISE_SCALE
        )
        data = data.replace(qpos=qpos, qvel=qvel)
        data = mjx.forward(self.mjx_model, data)

        return State(
            pipeline_state=data,
            obs=self._get_obs(data),
            reward=jnp.zeros(()),
            done=jnp.zeros((), dtype=jnp.bool_),
            metrics={"steps": jnp.zeros((), dtype=jnp.int32)},
            info={"discount": jnp.ones((), dtype=jnp.float32)},
        )

    def step(self, state: State, action: jax.Array) -> State:
        action = jnp.clip(action, -1.0, 1.0)
        ctrl = (self._ctrl_center + action * self._ctrl_scale).astype(jnp.float32)

        com_before = self._mass_center(state.pipeline_state)[0]

        def _phys(d, _):
            return mjx.step(self.mjx_model, d.replace(ctrl=ctrl)), None

        data, _ = jax.lax.scan(_phys, state.pipeline_state, None, self.n_substeps)

        com_after = self._mass_center(data)[0]
        x_velocity = (com_after - com_before) / self.dt

        forward_reward = _FORWARD_REWARD_WEIGHT * x_velocity
        ctrl_cost = _CTRL_COST_WEIGHT * jnp.sum(jnp.square(ctrl))
        reward = forward_reward + _HEALTHY_REWARD - ctrl_cost

        z = data.qpos[2]
        healthy = (z > _HEALTHY_Z[0]) & (z < _HEALTHY_Z[1])
        terminated = ~healthy
        steps = state.metrics["steps"] + 1
        truncated = steps >= _EPISODE_LEN
        done = terminated | truncated

        return State(
            pipeline_state=data,
            obs=self._get_obs(data),
            reward=reward,
            done=done,
            metrics={"steps": steps},
            info={"discount": (1.0 - terminated.astype(jnp.float32))},
        )
