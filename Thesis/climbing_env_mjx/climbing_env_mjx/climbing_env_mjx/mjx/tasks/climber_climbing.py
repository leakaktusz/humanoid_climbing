from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx
from dm_control import mjcf
from dm_env import specs

from climbing_env_mjx.mjx.tasks.base import MJXEnv, State
from climbing_env_mjx.models.climbing_wall.climbing_wall_kang import KangClimbingWall
from climbing_env_mjx.models.climbing_wall.climbing_wall_easy import ClimbingWall as EasyClimbingWall

_HUMANOID_XML = (
    Path(__file__).resolve().parents[2] / "models" / "humanoid_gym" / "humanoid.xml"
)

_EPISODE_LEN = 1000
_FRAME_SKIP = 5
_RESET_NOISE_SCALE = 1e-2

_SPAWN_X = 0.05

_SPAWN_CLEARANCE = 0.02

def _spawn_x_for(mj_model, margin: float = _SPAWN_CLEARANCE,
                 hold_margin: float = None) -> float:
    if hold_margin is None:
        hold_margin = 2.0 * _GRIP_RADIUS
    wall_gid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM,
                                 "climbing_name/wall")
    free_j = [j for j in range(mj_model.njnt)
              if mj_model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
    if wall_gid < 0 or not free_j:
        return _SPAWN_X
    adr = mj_model.jnt_qposadr[free_j[0]]
    free_b = mj_model.jnt_bodyid[free_j[0]]

    robot = []
    for b in range(mj_model.nbody):
        anc = b
        while anc > 0:
            if anc == free_b:
                robot.extend(range(mj_model.body_geomadr[b],
                                   mj_model.body_geomadr[b] + mj_model.body_geomnum[b]))
                break
            anc = mj_model.body_parentid[anc]
    robot = np.asarray(robot)
    holds = np.asarray([g for g in range(mj_model.ngeom)
                        if "hold_" in (mujoco.mj_id2name(
                            mj_model, mujoco.mjtObj.mjOBJ_GEOM, g) or "")])
    limbs = np.asarray([mj_model.geom(n).id for n in
                        ("left_hand", "right_hand", "left_foot", "right_foot")])
    if robot.size == 0 or holds.size == 0:
        return _SPAWN_X

    d = mujoco.MjData(mj_model)

    def _clearances(x):
        d.qpos[:] = mj_model.qpos0
        d.qpos[adr] += x
        mujoco.mj_forward(mj_model, d)
        R = d.geom_xmat[wall_gid].reshape(3, 3)
        local_x = R[:, 0]
        face_pt = d.geom_xpos[wall_gid] - local_x * float(mj_model.geom_size[wall_gid][0])
        wall_c = float(((d.geom_xpos[robot] - face_pt) @ (-local_x)
                        - mj_model.geom_rbound[robot]).min())
        diffs = d.geom_xpos[holds][None, :, :] - d.geom_xpos[limbs][:, None, :]
        hold_c = float((np.linalg.norm(diffs, axis=-1)
                        - mj_model.geom_size[limbs][:, 0][:, None]
                        - mj_model.geom_size[holds][:, 0][None, :]).min())
        return wall_c, hold_c

    wall_c, hold_c = _clearances(_SPAWN_X)
    if wall_c >= margin and hold_c >= hold_margin:
        return _SPAWN_X

    for k in range(1, 301):
        x = _SPAWN_X - 0.01 * k
        wall_c, hold_c = _clearances(x)
        if wall_c >= margin and hold_c >= hold_margin:
            return round(x, 3)

    return _SPAWN_X

_WORLD_GROUP = 1
_HOLD_GROUP = 2

_WALL_INCLINE_DEG = -15.0
_WALL_X = 0.5
_WALL_WIDTH = 2.0
_WALL_LENGTH = 5.2
_HOLD_ROWS = 12
_HOLD_COLS = 4
_HOLD_V0 = 0.3
_HOLD_ROW_DV = 0.4
_HOLD_COL_DY = 0.4
_HOLD_R = 0.05
_HOLD_PROTRUDE = 0.03
_HOLD_JITTER = 0.0

_GAMMA = 1.0
_GRIP_RADIUS = 0.04
_HOLD_REWARD_TIMEOUT = 4.0

_HOLD_HEIGHT_GAIN = 1.5
_REACH_LENGTHSCALE = 0.5

_STICKY_HOLDS = True
_STICKY_KP = 1200.0
_STICKY_KD = 50.0
_STICKY_FMAX = 400.0
_FALL_Z = 1.0
_FALL_PENALTY = -10.0

_SCALES = {
    "progress":  10.0,
    "hands":      4.0,
    "feet":       0.05,
    "reach":      1.5,
    "grip":       0.5,
    "torque":     0.02,
    "smooth":     0.01,
}


_KANG_BASE = dict(
    incline_deg=_WALL_INCLINE_DEG,
    wall_x=_WALL_X,
    wall_width=_WALL_WIDTH,
    wall_length=_WALL_LENGTH,
    hold_rows=_HOLD_ROWS,
    hold_cols=_HOLD_COLS,
    hold_v0=_HOLD_V0,
    hold_row_dv=_HOLD_ROW_DV,
    hold_col_dy=_HOLD_COL_DY,
    hold_radius=_HOLD_R,
    hold_protrude=_HOLD_PROTRUDE,
    hold_jitter=_HOLD_JITTER,
    hold_contype=_HOLD_GROUP,
    hold_conaffinity=_HOLD_GROUP,
)

def _grid_route(rows=_HOLD_ROWS, cols=_HOLD_COLS, v0=_HOLD_V0, dv=_HOLD_ROW_DV,
                dy=_HOLD_COL_DY, skip_rows=()):
    y0 = -dy * (cols - 1) / 2.0
    return [(v0 + r * dv, y0 + c * dy)
            for r in range(rows) if r not in skip_rows for c in range(cols)]

WALL_PRESETS = {
    "kang": {},
    "easy": {},

    "kang_mirror":  dict(mirror_y=True),
    "kang_phase20": dict(hold_v0=_HOLD_V0 + 0.2),
    "kang_shift20": dict(hold_y_offset=0.2),

    "kang_dv30": dict(hold_row_dv=0.3),
    "kang_dv50": dict(hold_row_dv=0.5),
    "kang_dv60": dict(hold_row_dv=0.6),
    "kang_dy30": dict(hold_col_dy=0.3),
    "kang_dy50": dict(hold_col_dy=0.5),
    "kang_cols2": dict(hold_cols=2),
    "kang_cols6": dict(hold_cols=6),
    "kang_drop20": dict(hold_dropout=0.2),
    "kang_drop40": dict(hold_dropout=0.4),

    "kang_ladder":   dict(holds=_grid_route(cols=2)),
    "kang_single":   dict(holds=[(_HOLD_V0 + r * _HOLD_ROW_DV, 0.0)
                                 for r in range(_HOLD_ROWS)]),
    "kang_zigzag":   dict(holds=[(_HOLD_V0 + r * _HOLD_ROW_DV,
                                  s + (0.3 if r % 2 else -0.3))
                                 for r in range(_HOLD_ROWS)
                                 for s in (-0.2, 0.2)]),
    "kang_traverse": dict(holds=[(_HOLD_V0 + r * _HOLD_ROW_DV,
                                  -0.6 + 1.2 * r / (_HOLD_ROWS - 1) + s)
                                 for r in range(_HOLD_ROWS)
                                 for s in (-0.2, 0.2)]),
    "kang_crux":     dict(holds=_grid_route(skip_rows=(5, 6))),

    "kang_x45": dict(wall_x=0.45),
    "kang_x55": dict(wall_x=0.55),
    "kang_r03": dict(hold_radius=0.03),
    "kang_r08": dict(hold_radius=0.08),
    "kang_prot01": dict(hold_protrude=0.01),
    "kang_prot05": dict(hold_protrude=0.05),
}

for _j, _tag in ((0.05, "05"), (0.10, "10"), (0.15, "15")):
    for _s in range(5):
        WALL_PRESETS[f"kang_jitter{_tag}_s{_s}"] = dict(hold_jitter=_j, hold_seed=_s)
        WALL_PRESETS[f"kang_jitter{_tag}_s{_s}_mirror"] = dict(
            hold_jitter=_j, hold_seed=_s, mirror_y=True)

for _inc in (-30, -20, -10, -5, 0, 10, 20):
    WALL_PRESETS[f"kang_incline{_inc}"] = dict(incline_deg=float(_inc))

def _build_wall_mjcf(wall: str = "kang"):
    if wall in WALL_PRESETS and wall != "easy":
        return KangClimbingWall(**{**_KANG_BASE, **WALL_PRESETS[wall]}).mjcf_model

    if wall == "easy":
        root = EasyClimbingWall().mjcf_model
        for col, body_name in enumerate(("jug_body", "jug_body_2")):
            body = root.worldbody.find("body", body_name)
            if body is None:
                continue
            for geom in body.find_all("geom"):
                geom.contype = 0
                geom.conaffinity = 0
            body.add(
                "geom",
                name=f"hold_0_{col}",
                type="sphere",
                size=[_HOLD_R],
                rgba=[0.58, 0.32, 0.76, 0.35],
                friction=[2.0, 0.5, 0.5],
                condim=3,
                contype=_HOLD_GROUP,
                conaffinity=_HOLD_GROUP,
            )
        return root

    raise ValueError(
        f"unknown wall '{wall}'; expected one of {sorted(WALL_PRESETS)}")

def make_model(wall: str = "kang") -> mujoco.MjModel:
    humanoid = mjcf.from_path(str(_HUMANOID_XML))

    floor = humanoid.worldbody.find("geom", "floor")
    if floor is not None:
        floor.remove()

    humanoid.attach(_build_wall_mjcf(wall))

    physics = mjcf.Physics.from_mjcf_model(humanoid)
    mj_model = physics.model._model

    mj_model.geom_margin[:] = 0.0
    mj_model.geom_gap[:] = 0.0

    mj_model.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER
    mj_model.opt.solver = mujoco.mjtSolver.mjSOL_CG
    mj_model.opt.iterations = 10
    mj_model.opt.ls_iterations = 6

    for hname in ("left_hand", "right_hand", "left_foot", "right_foot"):
        gid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, hname)
        mj_model.geom_condim[gid] = 3
        mj_model.geom_friction[gid] = [2.0, 0.5, 0.5]
        mj_model.geom_contype[gid] = _WORLD_GROUP | _HOLD_GROUP
        mj_model.geom_conaffinity[gid] = _WORLD_GROUP | _HOLD_GROUP
    return mj_model

class ClimberClimbingMJX(MJXEnv):
    def __init__(self, mj_model: mujoco.MjModel = None, wall: str = "kang", **kwargs):
        if mj_model is None:
            mj_model = make_model(wall)
        super().__init__(mj_model)

        self.torso_body_id = mj_model.body("torso").id
        self.root_body_id = self.torso_body_id
        self.root_jnt_id = next(
            i for i in range(mj_model.njnt)
            if mj_model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE
        )
        self.root_qposadr = int(mj_model.jnt_qposadr[self.root_jnt_id])

        self.hand_geom_ids = jnp.array([
            mj_model.geom("left_hand").id,
            mj_model.geom("right_hand").id,
        ])
        self.hand_body_ids = jnp.array([
            mj_model.body("left_lower_arm").id,
            mj_model.body("right_lower_arm").id,
        ])
        self.foot_geom_ids = jnp.array([
            mj_model.geom("left_foot").id,
            mj_model.geom("right_foot").id,
        ])
        self.foot_body_ids = jnp.array([
            mj_model.body("left_foot").id,
            mj_model.body("right_foot").id,
        ])
        self.limb_geom_ids = jnp.concatenate([self.hand_geom_ids, self.foot_geom_ids])
        self.limb_body_ids = jnp.concatenate([self.hand_body_ids, self.foot_body_ids])
        self.n_limbs = int(self.limb_geom_ids.shape[0])
        self.nbody = mj_model.nbody
        hum = [b for b in range(mj_model.nbody)
               if "climbing_name/" not in
               (mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, b) or "")]
        self._hum_body_ids = jnp.array(hum)
        self._hum_nbody = len(hum)

        def _is_hold(g):
            nm = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
            return nm.rsplit("/", 1)[-1].startswith("hold_")
        hold_ids = [g for g in range(mj_model.ngeom) if _is_hold(g)]
        assert len(hold_ids) >= 1, "no hold geoms ('hold_*') found in the model"
        self.hold_geom_ids = jnp.array(hold_ids)
        self.n_holds = len(hold_ids)
        self._hold_radii = jnp.array([
            float(mj_model.geom_size[g, 0]) for g in hold_ids
        ])
        self._limb_radii = jnp.array([
            float(mj_model.geom_size[int(g), 0]) for g in self.limb_geom_ids
        ])
        _d = mujoco.MjData(mj_model)
        mujoco.mj_forward(mj_model, _d)
        hold_z = _d.geom_xpos[hold_ids, 2]
        z_span = max(float(hold_z.max() - hold_z.min()), 1e-6)
        self._hold_height_w = jnp.asarray(
            1.0 + _HOLD_HEIGHT_GAIN * (hold_z - hold_z.min()) / z_span
        )

        self._spawn_x = _spawn_x_for(mj_model)

        self.n_substeps = _FRAME_SKIP
        self.dt = self.n_substeps * mj_model.opt.timestep
        self.max_episode_steps = _EPISODE_LEN

        cr = jnp.asarray(mj_model.actuator_ctrlrange)
        self._ctrl_center = (cr[:, 0] + cr[:, 1]) * 0.5
        self._ctrl_scale = (cr[:, 1] - cr[:, 0]) * 0.5

        nq, nv = mj_model.nq, mj_model.nv
        self._action_shape = (mj_model.nu,)
        nb = self._hum_nbody
        n_hold_feat = self.n_limbs * 3 + self.n_limbs
        self._obs_shape = ((nq - 2) + nv + nb * 10 + nb * 6 + nv + nb * 6 + n_hold_feat,)

    def observation_spec(self):
        return specs.Array(shape=self._obs_shape, dtype=jnp.float32, name="state")

    def action_spec(self):
        return specs.BoundedArray(
            shape=self._action_shape, dtype=jnp.float32,
            minimum=-1.0, maximum=1.0, name="action",
        )

    def _torso_z(self, data: mjx.Data) -> jax.Array:
        return data.xpos[self.torso_body_id][2]

    def _hands_z(self, data: mjx.Data) -> jax.Array:
        return data.geom_xpos[self.hand_geom_ids, 2].mean()

    def _feet_z(self, data: mjx.Data) -> jax.Array:
        return data.geom_xpos[self.foot_geom_ids, 2].mean()

    def _limb_hold_dists(self, data: mjx.Data):
        limbs = data.geom_xpos[self.limb_geom_ids]
        holds = data.geom_xpos[self.hold_geom_ids]
        diffs = holds[None, :, :] - limbs[:, None, :]
        d = (jnp.linalg.norm(diffs, axis=-1)
             - self._limb_radii[:, None] - self._hold_radii[None, :])
        return diffs, d

    def _limb_hold_top_dists(self, data: mjx.Data):
        limbs = data.geom_xpos[self.limb_geom_ids]
        tops = data.geom_xpos[self.hold_geom_ids].at[:, 2].add(self._hold_radii)
        diffs = tops[None, :, :] - limbs[:, None, :]
        return jnp.linalg.norm(diffs, axis=-1) - self._limb_radii[:, None]

    def _reach_phi(self, data: mjx.Data, active: jax.Array) -> jax.Array:
        d_top = self._limb_hold_top_dists(data)
        kern = jnp.exp(-jnp.maximum(d_top, 0.0) / _REACH_LENGTHSCALE)
        return jnp.mean(jnp.sum(
            self._hold_height_w[None, :] * kern * active[None, :], axis=1))

    def _hand_hold_features(self, data: mjx.Data, active: jax.Array = None):
        diffs, d = self._limb_hold_dists(data)
        if active is not None:
            d = jnp.where(active[None, :], d, jnp.inf)
        nearest = jnp.argmin(d, axis=1)
        rel = jnp.take_along_axis(diffs, nearest[:, None, None], axis=1)[:, 0, :]
        dist = jnp.min(d, axis=1)
        grip = (dist < _GRIP_RADIUS).astype(jnp.float32)
        return rel, dist, grip

    def _get_obs(self, data: mjx.Data, active: jax.Array) -> jax.Array:
        cfrc = getattr(data, "cfrc_ext", None)
        if cfrc is None:
            cfrc = jnp.zeros((self.nbody, 6))
        bid = self._hum_body_ids
        rel, _dist, _grip_paying = self._hand_hold_features(data, active)
        rel = jnp.where(active.any(), rel, 0.0)
        _, _, grip = self._hand_hold_features(data)
        return jnp.concatenate([
            data.qpos[2:],
            data.qvel,
            data.cinert[bid].ravel(),
            data.cvel[bid].ravel(),
            data.qfrc_actuator,
            cfrc[bid].ravel(),
            rel.ravel(),
            grip,
        ])

    def reset(self, rng: jax.Array) -> State:
        data = mjx.make_data(self.mjx_model)
        rng, rq, rv = jax.random.split(rng, 3)
        qpos = self.mjx_model.qpos0 + jax.random.uniform(
            rq, (self.mjx_model.nq,), minval=-_RESET_NOISE_SCALE, maxval=_RESET_NOISE_SCALE
        )
        qpos = qpos.at[self.root_qposadr].add(self._spawn_x)
        qvel = jax.random.uniform(
            rv, (self.mjx_model.nv,), minval=-_RESET_NOISE_SCALE, maxval=_RESET_NOISE_SCALE
        )
        data = data.replace(qpos=qpos, qvel=qvel)
        data = mjx.forward(self.mjx_model, data)

        return State(
            pipeline_state=data,
            obs=self._get_obs(data, jnp.ones(self.n_holds, dtype=bool)),
            reward=jnp.zeros(()),
            done=jnp.zeros((), dtype=jnp.bool_),
            metrics={
                "steps": jnp.zeros((), dtype=jnp.int32),
                "last_act": jnp.zeros(self._action_shape),
                "prev_torso_z": self._torso_z(data),
                "prev_hands_z": self._hands_z(data),
                "prev_feet_z": self._feet_z(data),
                "hold_touch_time": jnp.zeros(self.n_holds),
                **{f"rew_{k}": jnp.zeros(()) for k in _SCALES},
                "rew_fall": jnp.zeros(()),
            },
            info={"discount": jnp.ones((), dtype=jnp.float32)},
        )

    def step(self, state: State, action: jax.Array) -> State:
        action = jnp.clip(action, -1.0, 1.0)
        ctrl = (self._ctrl_center + action * self._ctrl_scale).astype(jnp.float32)

        z_before = state.metrics["prev_torso_z"]

        ps = state.pipeline_state
        if _STICKY_HOLDS:
            rel, _dist, grip = self._hand_hold_features(ps)
            limb_vel = ps.cvel[self.limb_body_ids][:, 3:6]
            force = grip[:, None] * (_STICKY_KP * rel - _STICKY_KD * limb_vel)
            mag = jnp.linalg.norm(force, axis=1, keepdims=True)
            force = jnp.where(mag > _STICKY_FMAX,
                              force * (_STICKY_FMAX / (mag + 1e-8)), force)
            xfrc = jnp.zeros((self.nbody, 6)).at[self.limb_body_ids, 0:3].set(force)
            ps = ps.replace(xfrc_applied=xfrc)

        def _phys(d, _):
            return mjx.step(self.mjx_model, d.replace(ctrl=ctrl)), None

        data, _ = jax.lax.scan(_phys, ps, None, self.n_substeps)

        z_after = self._torso_z(data)

        progress = (_GAMMA * z_after - z_before) / self.dt

        hands_z_after = self._hands_z(data)
        hands = (_GAMMA * hands_z_after - state.metrics["prev_hands_z"]) / self.dt
        feet_z_after = self._feet_z(data)
        feet = (_GAMMA * feet_z_after - state.metrics["prev_feet_z"]) / self.dt

        _, d_all = self._limb_hold_dists(data)
        touched = (d_all < _GRIP_RADIUS).any(axis=0)
        t_prev = state.metrics["hold_touch_time"]
        hold_t = jnp.where((t_prev > 0.0) | touched, t_prev + self.dt, 0.0)
        active = hold_t < _HOLD_REWARD_TIMEOUT

        _rel, _dist, grip = self._hand_hold_features(data, active=active)
        reach = (_GAMMA * self._reach_phi(data, active)
                 - self._reach_phi(state.pipeline_state, active)) / self.dt
        d_act = jnp.where(active[None, :], d_all, jnp.inf)
        w_limb = self._hold_height_w[jnp.argmin(d_act, axis=1)]
        grip_r = jnp.sum(w_limb * grip)

        torque = -jnp.sum(jnp.square(ctrl))
        smooth = -jnp.sum(jnp.square(action - state.metrics["last_act"]))

        terms = {
            "progress": progress,
            "hands":    hands,
            "feet":     feet,
            "reach":    reach,
            "grip":     grip_r,
            "torque":   torque,
            "smooth":   smooth,
        }
        reward = sum(_SCALES[k] * v for k, v in terms.items())

        fell = z_after < _FALL_Z
        nan = jnp.isnan(data.qpos).any() | jnp.isnan(data.qvel).any()
        terminated = fell | nan
        steps = state.metrics["steps"] + 1
        truncated = steps >= _EPISODE_LEN
        done = terminated | truncated

        reward = reward + _FALL_PENALTY * terminated.astype(jnp.float32)

        return State(
            pipeline_state=data,
            obs=self._get_obs(data, active),
            reward=reward,
            done=done,
            metrics={
                "steps": steps,
                "last_act": action,
                "prev_torso_z": z_after,
                "prev_hands_z": hands_z_after,
                "prev_feet_z": feet_z_after,
                "hold_touch_time": hold_t,
                **{f"rew_{k}": _SCALES[k] * v for k, v in terms.items()},
                "rew_fall": _FALL_PENALTY * terminated.astype(jnp.float32),
            },
            info={"discount": (1.0 - terminated.astype(jnp.float32))},
        )
