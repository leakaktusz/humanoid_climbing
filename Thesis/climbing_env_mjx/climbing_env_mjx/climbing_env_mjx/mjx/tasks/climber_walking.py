import jax
import jax.numpy as jnp
from mujoco import mjx
import mujoco
from dm_env import specs
from climbing_env_mjx.mjx.tasks.base import MJXEnv, State

_HOME = jnp.array([
    0.0, 0.0,
    0.0, -1.4, 0.0, -0.4,
    0.0,  1.4, 0.0,  0.4,
    0.0,
    -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,
    -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,
])

_POSE_WEIGHTS = jnp.array([
    1.0, 1.0,
    0.1, 1.0, 1.0, 1.0,
    0.1, 1.0, 1.0, 1.0,
    1.0,
    0.01, 1.0, 1.0, 0.01, 1.0, 1.0,
    0.01, 1.0, 1.0, 0.01, 1.0, 1.0,
])
_HIP_INDICES  = jnp.array([12, 13, 18, 19])
_KNEE_INDICES = jnp.array([14, 20])

_ACTION_SCALE = 1.0
_FOOT_Z       = 0.03
_CONTACT_DIST = 1e-3
_EPISODE_LEN  = 1000

_PUSH_ENABLE   = True
_PUSH_INTERVAL = (5.0, 10.0)
_PUSH_MAG      = (0.1, 1.0)

_CMD_X   = (-1.0, 1.0)
_CMD_Y   = (-0.8, 0.8)
_CMD_YAW = (-1.0, 1.0)
_GAIT_FREQ = (1.25, 1.75)

_TRACK_SIGMA = 0.25
_MAX_FOOT_H  = 0.12
_REWARD_SCALE = 30.0
_SCALES = {
    "tracking_lin_vel":     1.0,
    "tracking_ang_vel":     0.5,
    "ang_vel_xy":          -0.15,
    "orientation":         -1.0,
    "feet_slip":           -0.5,
    "feet_air_time":        2.0,
    "feet_phase":           1.0,
    "alive":                0.25,
    "joint_deviation_hip": -0.1,
    "joint_deviation_knee":-0.1,
    "dof_pos_limits":      -1.0,
    "pose":                -1.0,
    "feet_distance":       -1.0,
}


def _get_rz(phi: jax.Array, swing_height: float) -> jax.Array:
    def bezier(y0, y1, x):
        return y0 + (y1 - y0) * (x ** 3 + 3.0 * (x ** 2 * (1.0 - x)))
    x = (phi + jnp.pi) / (2.0 * jnp.pi)
    stance = bezier(0.0, swing_height, 2.0 * x)
    swing  = bezier(swing_height, 0.0, 2.0 * x - 1.0)
    return jnp.where(x <= 0.5, stance, swing)


class ClimberWalkingMJX(MJXEnv):
    def __init__(self, mj_model: mujoco.MjModel, **kwargs):
        super().__init__(mj_model)

        self.root_jnt_id = next(
            i for i in range(mj_model.njnt)
            if mj_model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE
        )
        self.root_body_id  = mj_model.body('T1/Trunk').id
        self.left_foot_id  = mj_model.body('T1/left_foot_link').id
        self.right_foot_id = mj_model.body('T1/right_foot_link').id

        self.imu_site_id   = mj_model.site('T1/imu').id
        self.lfoot_site_id = mj_model.site('T1/left_foot').id
        self.rfoot_site_id = mj_model.site('T1/right_foot').id

        plane_geoms = [g for g in range(mj_model.ngeom)
                       if mj_model.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE]
        assert len(plane_geoms) == 1, f"expected 1 floor plane, found {plane_geoms}"
        self.floor_geom_id = int(plane_geoms[0])
        def _collision_geom(body_id):
            gs = [g for g in range(mj_model.ngeom)
                  if mj_model.geom_bodyid[g] == body_id and mj_model.geom_contype[g] != 0]
            assert len(gs) == 1, f"expected 1 collision geom on body {body_id}, found {gs}"
            return int(gs[0])
        self.lfoot_geom_id = _collision_geom(self.left_foot_id)
        self.rfoot_geom_id = _collision_geom(self.right_foot_id)

        def _sensor(name):
            sid = mj_model.sensor(name).id
            return int(mj_model.sensor_adr[sid]), int(mj_model.sensor_dim[sid])
        self._local_linvel  = _sensor('T1/local_linvel')
        self._global_linvel = _sensor('T1/global_linvel')
        self._gyro          = _sensor('T1/angular-velocity')
        self._lfoot_linvel  = _sensor('T1/left_foot_global_linvel')
        self._rfoot_linvel  = _sensor('T1/right_foot_global_linvel')

        self.home_joints = _HOME

        jnt_range = jnp.array(mj_model.jnt_range[1:])
        lo, hi = jnt_range[:, 0], jnt_range[:, 1]
        c, r = (lo + hi) * 0.5, (hi - lo)
        self._soft_lo = c - 0.5 * r * 0.95
        self._soft_hi = c + 0.5 * r * 0.95

        mj_model.opt.timestep = 0.002
        self.mjx_model = mjx.put_model(mj_model)
        self.n_substeps = 10
        self.dt = self.n_substeps * 0.002

        self._obs_shape    = (85,)
        self._action_shape = (mj_model.nu,)

    def observation_spec(self):
        return specs.Array(shape=self._obs_shape, dtype=jnp.float32, name="state")

    def action_spec(self):
        return specs.BoundedArray(
            shape=self._action_shape, dtype=jnp.float32,
            minimum=-1.0, maximum=1.0, name="action",
        )

    def _sensor_read(self, data, adr_dim):
        a, d = adr_dim
        return data.sensordata[a:a + d]

    def _gravity(self, data):
        R = data.site_xmat[self.imu_site_id].reshape(3, 3)
        return R.T @ jnp.array([0.0, 0.0, -1.0])

    def _foot_z(self, data):
        return jnp.array([
            data.site_xpos[self.lfoot_site_id][2],
            data.site_xpos[self.rfoot_site_id][2],
        ])

    def _feet_contact(self, data):
        g1, g2 = data.contact.geom1, data.contact.geom2
        dist   = data.contact.dist
        floor  = self.floor_geom_id

        def touching(foot_geom):
            pair = (((g1 == foot_geom) & (g2 == floor)) |
                    ((g1 == floor) & (g2 == foot_geom)))
            return jnp.any(pair & (dist < _CONTACT_DIST))

        return jnp.array([touching(self.lfoot_geom_id),
                          touching(self.rfoot_geom_id)])

    def sample_command(self, rng):
        r1, r2, r3, r4 = jax.random.split(rng, 4)
        vx  = jax.random.uniform(r1, (), minval=_CMD_X[0],   maxval=_CMD_X[1])
        vy  = jax.random.uniform(r2, (), minval=_CMD_Y[0],   maxval=_CMD_Y[1])
        yaw = jax.random.uniform(r3, (), minval=_CMD_YAW[0], maxval=_CMD_YAW[1])
        cmd = jnp.array([vx, vy, yaw])
        return jnp.where(jax.random.bernoulli(r4, 0.1), jnp.zeros(3), cmd)

    def reset(self, rng: jax.Array) -> State:
        data = mjx.make_data(self.mjx_model)
        nq, nv = self.mjx_model.nq, self.mjx_model.nv

        init_q = jnp.concatenate(
            [jnp.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]), self.home_joints])

        rng, r_dxy, r_yaw, r_j, r_v, r_f, r_c, r_p = jax.random.split(rng, 8)
        qpos = init_q.at[0:2].add(jax.random.uniform(r_dxy, (2,), minval=-0.5, maxval=0.5))
        half = 0.5 * jax.random.uniform(r_yaw, (), minval=-3.14, maxval=3.14)
        qpos = qpos.at[3:7].set(jnp.array([jnp.cos(half), 0.0, 0.0, jnp.sin(half)]))
        qpos = qpos.at[7:].multiply(jax.random.uniform(r_j, (nq - 7,), minval=0.5, maxval=1.5))
        qvel = jnp.zeros(nv).at[0:6].set(jax.random.uniform(r_v, (6,), minval=-0.5, maxval=0.5))

        data = data.replace(qpos=qpos, qvel=qvel, ctrl=self.home_joints)
        data = mjx.forward(self.mjx_model, data)

        gait_freq = jax.random.uniform(r_f, (), minval=_GAIT_FREQ[0], maxval=_GAIT_FREQ[1])
        push_interval = jax.random.uniform(r_p, (), minval=_PUSH_INTERVAL[0], maxval=_PUSH_INTERVAL[1])
        metrics = {
            'steps':        jnp.zeros((), dtype=jnp.int32),
            'cmd_step':     jnp.zeros((), dtype=jnp.int32),
            'push_step':    jnp.zeros((), dtype=jnp.int32),
            'push_interval_steps': jnp.round(push_interval / self.dt).astype(jnp.int32),
            'rng':          rng,
            'command':      self.sample_command(r_c),
            'last_act':     jnp.zeros(self._action_shape),
            'feet_air_time':jnp.zeros(2),
            'last_contact': jnp.zeros(2, dtype=jnp.bool_),
            'swing_peak':   jnp.zeros(2),
            'phase':        jnp.array([0.0, jnp.pi]),
            'phase_dt':     2.0 * jnp.pi * self.dt * gait_freq,
        }
        return State(
            pipeline_state=data,
            obs=self._get_obs(data, metrics),
            reward=jnp.zeros(()),
            done=jnp.zeros((), dtype=jnp.bool_),
            metrics=metrics,
            info={'truncation': jnp.zeros((), dtype=jnp.bool_),
                  'termination': jnp.zeros((), dtype=jnp.bool_),
                  'discount': jnp.ones((), dtype=jnp.float32)},
        )

    def step(self, state: State, action: jax.Array) -> State:
        m = state.metrics

        rng, push1, push2 = jax.random.split(m['rng'], 3)
        push_theta = jax.random.uniform(push1, (), maxval=2.0 * jnp.pi)
        push_mag   = jax.random.uniform(push2, (), minval=_PUSH_MAG[0], maxval=_PUSH_MAG[1])
        fire = (jnp.mod(m['push_step'] + 1, m['push_interval_steps']) == 0) & _PUSH_ENABLE
        push = jnp.array([jnp.cos(push_theta), jnp.sin(push_theta)]) * push_mag * fire
        qvel = state.pipeline_state.qvel.at[:2].add(push)
        ps = state.pipeline_state.replace(qvel=qvel)

        motor_targets = (self.home_joints + action * _ACTION_SCALE).astype(jnp.float32)

        def _phys(d, _):
            return mjx.step(self.mjx_model, d.replace(ctrl=motor_targets)), None
        data, _ = jax.lax.scan(_phys, ps, None, self.n_substeps)

        foot_z   = self._foot_z(data)
        contact  = self._feet_contact(data)
        contact_filt = contact | m['last_contact']
        first_contact = (m['feet_air_time'] > 0.0) & contact_filt
        air_time = m['feet_air_time'] + self.dt
        swing_peak = jnp.maximum(m['swing_peak'], foot_z)

        reward, _ = self._get_reward(data, action, m, contact, first_contact, air_time)

        steps     = m['steps'] + 1
        cmd_step  = m['cmd_step'] + 1
        push_step = m['push_step'] + 1
        rng, cmd_rng = jax.random.split(rng)
        command = jnp.where(cmd_step > 500, self.sample_command(cmd_rng), m['command'])
        cmd_step = jnp.where(cmd_step > 500, 0, cmd_step)

        phase = m['phase'] + m['phase_dt']
        phase = jnp.fmod(phase + jnp.pi, 2.0 * jnp.pi) - jnp.pi
        phase = jnp.where(jnp.linalg.norm(command) > 0.01, phase, jnp.ones(2) * jnp.pi)

        new_metrics = {
            'steps': steps, 'cmd_step': cmd_step, 'push_step': push_step,
            'push_interval_steps': m['push_interval_steps'], 'rng': rng,
            'command': command, 'last_act': action,
            'feet_air_time': air_time * (~contact),
            'last_contact': contact,
            'swing_peak': swing_peak * (~contact),
            'phase': phase, 'phase_dt': m['phase_dt'],
        }

        gravity = self._gravity(data)
        trunk_z = data.xpos[self.root_body_id][2]
        termination = ((gravity[2] > 0.0) | (trunk_z < 0.4)
                       | jnp.isnan(data.qpos).any() | jnp.isnan(data.qvel).any())
        truncation  = steps >= _EPISODE_LEN
        done = termination | truncation
        discount = 1.0 - termination.astype(jnp.float32)

        return State(
            pipeline_state=data,
            obs=self._get_obs(data, new_metrics),
            reward=reward,
            done=done,
            metrics=new_metrics,
            info={'truncation': truncation, 'termination': termination, 'discount': discount},
        )

    def _get_obs(self, data: mjx.Data, m: dict) -> jax.Array:
        local_linvel = self._sensor_read(data, self._local_linvel)
        gyro         = self._sensor_read(data, self._gyro)
        gravity      = self._gravity(data)
        phase = jnp.concatenate([jnp.cos(m['phase']), jnp.sin(m['phase'])])
        return jnp.concatenate([
            local_linvel,
            gyro,
            gravity,
            m['command'],
            data.qpos[7:] - self.home_joints,
            data.qvel[6:],
            m['last_act'],
            phase,
        ])

    def _get_reward(self, data, action, m, contact, first_contact, air_time):
        command = m['command']
        cmd_norm = jnp.linalg.norm(command)
        local_linvel = self._sensor_read(data, self._local_linvel)
        gyro         = self._sensor_read(data, self._gyro)
        gravity      = self._gravity(data)
        qpos_j       = data.qpos[7:]
        foot_z       = self._foot_z(data)

        lin_err = jnp.sum(jnp.square(command[:2] - local_linvel[:2]))
        r_track_lin = jnp.exp(-lin_err / _TRACK_SIGMA)
        ang_err = jnp.square(command[2] - gyro[2])
        r_track_ang = jnp.exp(-ang_err / _TRACK_SIGMA)

        c_ang_vel_xy  = jnp.sum(jnp.square(gyro[:2]))
        c_orientation = jnp.sum(jnp.square(gravity[:2]))

        lfoot_vel = self._sensor_read(data, self._lfoot_linvel)[:2]
        rfoot_vel = self._sensor_read(data, self._rfoot_linvel)[:2]
        contact_f = contact.astype(jnp.float32)
        c_feet_slip = (jnp.linalg.norm(lfoot_vel) * contact_f[0]
                       + jnp.linalg.norm(rfoot_vel) * contact_f[1])
        at = jnp.clip((air_time - 0.2) * first_contact.astype(jnp.float32), None, 0.3)
        r_feet_air = jnp.sum(at) * (cmd_norm > 0.1)
        rz = _get_rz(m['phase'], _MAX_FOOT_H)
        r_feet_phase = jnp.exp(-jnp.sum(jnp.square(foot_z - rz)) / 0.01)

        c_dev_hip = jnp.sum(jnp.abs(qpos_j[_HIP_INDICES] - self.home_joints[_HIP_INDICES])) * (jnp.abs(command[1]) > 0.1)
        c_dev_knee = jnp.sum(jnp.abs(qpos_j[_KNEE_INDICES] - self.home_joints[_KNEE_INDICES]))
        oob = -jnp.clip(qpos_j - self._soft_lo, None, 0.0) + jnp.clip(qpos_j - self._soft_hi, 0.0, None)
        c_dof_lim = jnp.sum(oob)
        c_pose = jnp.sum(jnp.square(qpos_j - self.home_joints) * _POSE_WEIGHTS)

        lp = data.site_xpos[self.lfoot_site_id]
        rp = data.site_xpos[self.rfoot_site_id]
        R  = data.site_xmat[self.imu_site_id].reshape(3, 3)
        yaw = jnp.arctan2(R[1, 0], R[0, 0])
        feet_dist = jnp.abs(jnp.cos(yaw) * (lp[1] - rp[1]) - jnp.sin(yaw) * (lp[0] - rp[0]))
        c_feet_dist = jnp.clip(0.2 - feet_dist, 0.0, 0.1)

        terms = {
            "tracking_lin_vel": r_track_lin,
            "tracking_ang_vel": r_track_ang,
            "ang_vel_xy": c_ang_vel_xy,
            "orientation": c_orientation,
            "feet_slip": c_feet_slip,
            "feet_air_time": r_feet_air,
            "feet_phase": r_feet_phase,
            "alive": jnp.array(1.0),
            "joint_deviation_hip": c_dev_hip,
            "joint_deviation_knee": c_dev_knee,
            "dof_pos_limits": c_dof_lim,
            "pose": c_pose,
            "feet_distance": c_feet_dist,
        }
        total = sum(_SCALES[k] * v for k, v in terms.items())
        reward = jnp.clip(total * self.dt * _REWARD_SCALE, 0.0, 10000.0)
        return reward, terms
