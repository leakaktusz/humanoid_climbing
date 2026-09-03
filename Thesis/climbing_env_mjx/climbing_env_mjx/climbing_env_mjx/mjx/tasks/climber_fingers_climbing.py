from pathlib import Path

import numpy as np
import mujoco
from dm_control import mjcf

from climbing_env_mjx.mjx.tasks import climber_climbing as cc
from climbing_env_mjx.mjx.tasks.climber_climbing import ClimberClimbingMJX

_HUMANOID_FINGERS_XML = (
    Path(__file__).resolve().parents[2] / "models" / "humanoid_gym"
    / "humanoid_fingers.xml"
)

_THUMB_GROUP = 4


def _apply_hand_groups(mj_model: mujoco.MjModel) -> None:
    for g in range(mj_model.ngeom):
        base = (mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_GEOM, g) or "").rsplit("/", 1)[-1]
        is_finger = any(k in base for k in ("_f1_", "_f2_", "_f3_", "_f4_"))
        is_thumb = "_th_" in base
        if not (is_finger or is_thumb):
            continue
        mj_model.geom_condim[g] = 3
        mj_model.geom_friction[g] = [2.0, 0.5, 0.5]
        if is_thumb:
            mj_model.geom_contype[g] = _THUMB_GROUP
            mj_model.geom_conaffinity[g] = cc._WORLD_GROUP | cc._HOLD_GROUP
        else:
            mj_model.geom_contype[g] = cc._HOLD_GROUP
            mj_model.geom_conaffinity[g] = cc._WORLD_GROUP


def make_model(wall: str = "kang") -> mujoco.MjModel:
    humanoid = mjcf.from_path(str(_HUMANOID_FINGERS_XML))

    floor = humanoid.worldbody.find("geom", "floor")
    if floor is not None:
        floor.remove()

    humanoid.attach(cc._build_wall_mjcf(wall))

    physics = mjcf.Physics.from_mjcf_model(humanoid)
    mj_model = physics.model._model

    mj_model.geom_margin[:] = 0.0
    mj_model.geom_gap[:] = 0.0

    mj_model.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER
    mj_model.opt.solver = mujoco.mjtSolver.mjSOL_CG
    mj_model.opt.iterations = 20
    mj_model.opt.ls_iterations = 6

    for hname in ("left_hand", "right_hand", "left_foot", "right_foot"):
        gid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, hname)
        mj_model.geom_condim[gid] = 3
        mj_model.geom_friction[gid] = [2.0, 0.5, 0.5]
        mj_model.geom_contype[gid] = cc._WORLD_GROUP | cc._HOLD_GROUP
        mj_model.geom_conaffinity[gid] = cc._WORLD_GROUP | cc._HOLD_GROUP

    _apply_hand_groups(mj_model)
    return mj_model

_LIMB_SPHERE_RADII = (0.04, 0.04, 0.075, 0.075)

_FINGERTIP_CLEARANCE = 0.02
_FINGERTIP_NAMES = [
    "left_f1_dist", "left_f2_dist", "left_f3_dist", "left_f4_dist", "left_th_dist",
    "right_f1_dist", "right_f2_dist", "right_f3_dist", "right_f4_dist", "right_th_dist",
]


class ClimberFingersClimbingMJX(ClimberClimbingMJX):

    def __init__(self, mj_model: mujoco.MjModel = None, wall: str = "kang", **kwargs):
        super().__init__(mj_model if mj_model is not None else make_model(wall), **kwargs)
        import jax.numpy as jnp
        assert len(_LIMB_SPHERE_RADII) == self.n_limbs, (
            f"expected {self.n_limbs} limb radii, got {len(_LIMB_SPHERE_RADII)}"
        )
        self._limb_radii = jnp.asarray(_LIMB_SPHERE_RADII)

        self.hand_body_ids = jnp.array([
            self.mj_model.body("left_hand").id,
            self.mj_model.body("right_hand").id,
        ])
        self.limb_body_ids = jnp.concatenate([self.hand_body_ids, self.foot_body_ids])

        self.fingertip_geom_ids = jnp.array(
            [self.mj_model.geom(n).id for n in _FINGERTIP_NAMES])
        self._fingertip_radii = jnp.asarray(
            [float(self.mj_model.geom_size[self.mj_model.geom(n).id, 0])
             for n in _FINGERTIP_NAMES])

    def _fingertip_reward(self, data, active, grip):
        import jax.numpy as jnp
        tips = data.geom_xpos[self.fingertip_geom_ids]
        holds = data.geom_xpos[self.hold_geom_ids]
        d = (jnp.linalg.norm(holds[None, :, :] - tips[:, None, :], axis=-1)
             - self._hold_radii[None, :] - self._fingertip_radii[:, None])
        d = jnp.where(active[None, :], d, jnp.inf)
        touch = (jnp.min(d, axis=1) < _FINGERTIP_CLEARANCE).astype(jnp.float32)
        return grip[0] * jnp.mean(touch[:5]) + grip[1] * jnp.mean(touch[5:])

_N_LIMBS = 4


def obs_semantic_keys(m: mujoco.MjModel):
    jname = lambda j: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
    bname = lambda b: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b)
    nq, nv, nbody = m.nq, m.nv, m.nbody

    qp_j = [-1] * nq; qp_c = [0] * nq
    qv_j = [-1] * nv; qv_c = [0] * nv
    for j in range(m.njnt):
        jt = m.jnt_type[j]
        if jt == mujoco.mjtJoint.mjJNT_FREE:   nqj, nvj = 7, 6
        elif jt == mujoco.mjtJoint.mjJNT_BALL: nqj, nvj = 4, 3
        else:                                  nqj, nvj = 1, 1
        qa, va = int(m.jnt_qposadr[j]), int(m.jnt_dofadr[j])
        for c in range(nqj): qp_j[qa + c] = j; qp_c[qa + c] = c
        for c in range(nvj): qv_j[va + c] = j; qv_c[va + c] = c

    def qp_key(q):
        j = qp_j[q]
        return ("qpos_free", qp_c[q]) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE \
            else ("qpos_joint", jname(j))

    def dof_key(prefix, v):
        j = qv_j[v]
        return (prefix + "_free", qv_c[v]) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE \
            else (prefix + "_joint", jname(j))

    bid = [b for b in range(nbody) if "climbing_name/" not in (bname(b) or "")]
    keys = []
    keys += [qp_key(q) for q in range(2, nq)]
    keys += [dof_key("qvel", v) for v in range(nv)]
    keys += [("cinert", bname(b), e) for b in bid for e in range(10)]
    keys += [("cvel",   bname(b), e) for b in bid for e in range(6)]
    keys += [dof_key("qfrc", v) for v in range(nv)]
    keys += [("cfrc",   bname(b), e) for b in bid for e in range(6)]
    keys += [("rel", l, e) for l in range(_N_LIMBS) for e in range(3)]
    keys += [("grip", l) for l in range(_N_LIMBS)]
    return keys


def actuator_names(m: mujoco.MjModel):
    return [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a) for a in range(m.nu)]


def remap_plain_actor(actor: dict) -> dict:
    actor = actor["actor"] if "actor" in actor else actor
    mp, mf = cc.make_model(), make_model()
    old_keys, new_keys = obs_semantic_keys(mp), obs_semantic_keys(mf)
    old_acts, new_acts = actuator_names(mp), actuator_names(mf)

    old_obs_idx = {k: i for i, k in enumerate(old_keys)}
    old_act_idx = {a: i for i, a in enumerate(old_acts)}
    obs_src = np.array([old_obs_idx.get(k, -1) for k in new_keys], dtype=np.int64)
    act_src = np.array([old_act_idx.get(a, -1) for a in new_acts], dtype=np.int64)

    def _remap(d, path=""):
        out = {}
        for k, v in d.items():
            p = f"{path}/{k}"
            if isinstance(v, dict):
                out[k] = _remap(v, p); continue
            a = np.asarray(v)
            if p.endswith("Dense_0/kernel"):
                assert a.shape[0] == len(old_keys), (
                    f"{p} rows {a.shape[0]} != plain-climb obs {len(old_keys)}; "
                    "not a plain-climb actor")
                new = np.zeros((len(new_keys), a.shape[1]), a.dtype)
                m = obs_src >= 0; new[m] = a[obs_src[m]]
                out[k] = new
            elif "Output" in p and p.endswith("kernel"):
                assert a.shape[1] == len(old_acts), (
                    f"{p} cols {a.shape[1]} != plain-climb action {len(old_acts)}")
                new = np.zeros((a.shape[0], len(new_acts)), a.dtype)
                m = act_src >= 0; new[:, m] = a[:, act_src[m]]
                out[k] = new
            elif "Output" in p and p.endswith("bias"):
                assert a.shape[0] == len(old_acts)
                new = np.zeros((len(new_acts),), a.dtype)
                m = act_src >= 0; new[m] = a[act_src[m]]
                out[k] = new
            else:
                out[k] = a
        return out

    return _remap(actor)
