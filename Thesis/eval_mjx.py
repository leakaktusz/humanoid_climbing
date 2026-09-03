import os, sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "climbing_rl", "climbing_rl"))
sys.path.insert(0, os.path.join(_HERE, "climbing_env_mjx", "climbing_env_mjx"))

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Tuple

import time
import jax
import jax.numpy as jnp
import numpy as np
import flax.serialization
import mujoco
from mujoco import mjx
from dm_env import specs as dm_specs
import tyro

import sac_jax as sac
import specs_jax as specs_util
from climbing_env_mjx import mjx as climbing_mjx


@dataclass
class Args:
    checkpoint: Path = Path("outputs/t1_walk/checkpoints/best_actor.msgpack")
    env: str = "climbing_env_mjx-climber-v1"
    episodes: int = 5
    seed: int = 0
    record_dir: Optional[Path] = None
    fps: Optional[int] = None
    slowmo: float = 2.0
    max_steps: int = 500
    stochastic: bool = False
    viewer: bool = False
    random: bool = False
    wall: str = "kang"
    list_walls: bool = False
    camera: Literal["default", "back"] = "default"


def _sync_mjdata(model: mujoco.MjModel, mj_data: mujoco.MjData, pipeline_state) -> None:
    cpu = mjx.get_data(model, pipeline_state)
    mj_data.qpos[:] = np.array(cpu.qpos)
    mj_data.qvel[:] = np.array(cpu.qvel)
    mujoco.mj_forward(model, mj_data)


def _camera_preset(camera: str, is_climbing: bool):
    if camera == "back":
        lookat = [0.3, 0.0, 1.4] if is_climbing else [0.0, 0.0, 0.8]
        return 3.0, 0, -10, lookat
    if is_climbing:
        return 3.5, 90, -5, [0.3, 0.0, 1.4]
    return 3.0, 135, -20, [0.0, 0.0, 0.5]


def _load_agent(checkpoint: Path, env, seed: int) -> Tuple[sac.SAC, int, int]:
    with open(checkpoint, "rb") as f:
        raw = flax.serialization.msgpack_restore(f.read())

    if "MLP_0" not in raw and "actor" in raw:
        print("[INIT] Full-state checkpoint — using its 'actor' params")
        raw = raw["actor"]

    ckpt_obs_dim = raw["MLP_0"]["Dense_0"]["kernel"].shape[0]
    ckpt_action_dim = raw["OutputDenseMean"]["kernel"].shape[1]
    print(f"[INIT] Checkpoint  obs={ckpt_obs_dim}  action={ckpt_action_dim}")

    env_obs_dim = env._obs_shape[0]
    env_act_dim = env._action_shape[0]
    if ckpt_obs_dim != env_obs_dim or ckpt_action_dim != env_act_dim:
        print(f"[WARN] Env dims: obs={env_obs_dim}  action={env_act_dim} — will pad/trim")

    obs_spec = dm_specs.Array(shape=(ckpt_obs_dim,), dtype=jnp.float32, name="obs")
    act_spec = dm_specs.BoundedArray(
        shape=(ckpt_action_dim,), dtype=jnp.float32,
        minimum=-1.0, maximum=1.0, name="action",
    )
    env_spec = specs_util.EnvironmentSpec(observation=obs_spec, action=act_spec)
    agent = sac.SAC.initialize(spec=env_spec, config=sac.SACConfig(), seed=seed, discount=0.99)
    params = flax.serialization.from_state_dict(agent.actor.params, raw)
    agent = agent.replace(actor=agent.actor.replace(params=params))

    ckpt_sum = float(sum(np.sum(np.array(p)) for p in jax.tree_util.tree_leaves(params)))
    print(f"[INIT] Actor loaded  checksum={ckpt_sum:.4f}")
    return agent, ckpt_obs_dim, ckpt_action_dim


def main(args: Args) -> None:
    from climbing_env_mjx.mjx.tasks import climber_climbing as _cc

    if args.list_walls:
        print("Available routes (--wall <name>):")
        for _nm in sorted(_cc.WALL_PRESETS):
            _ov = _cc.WALL_PRESETS[_nm]
            _desc = "the training wall" if _nm == "kang" else (
                ", ".join(f"{k}={v}" for k, v in _ov.items() if k != "holds")
                or "explicit hold list")
            print(f"  {_nm:28s} {_desc}")
        return

    if args.wall not in _cc.WALL_PRESETS:
        raise SystemExit(
            f"[ERR] unknown wall '{args.wall}'. Run with --list_walls to see them.")

    print(f"[INIT] Loading '{args.env}' …  (wall='{args.wall}')")
    env = climbing_mjx.load(environment_name=args.env, task_kwargs={"wall": args.wall})
    print(f"[INIT] Env  obs={env._obs_shape[0]}  action={env._action_shape[0]}")

    if hasattr(env, "n_holds"):
        import mujoco as _mj
        _d = _mj.MjData(env.mj_model)
        _mj.mj_forward(env.mj_model, _d)
        _hz = _d.geom_xpos[np.asarray(env.hold_geom_ids), 2]
        _delta = {k: v for k, v in _cc.WALL_PRESETS[args.wall].items() if k != "holds"}
        print(f"[INIT] Route '{args.wall}': {env.n_holds} holds, "
              f"z {_hz.min():.2f}–{_hz.max():.2f} m"
              + (f"  (vs training: {_delta})" if _delta else "  (= training wall)"))

    is_climbing = args.env.endswith("-climb-v1")

    if args.random:
        agent = None
        ckpt_obs_dim = env._obs_shape[0]
        ckpt_action_dim = env._action_shape[0]
        print("[EVAL] Mode: random actions (no checkpoint) — visualizing the env\n")
    else:
        agent, ckpt_obs_dim, ckpt_action_dim = _load_agent(args.checkpoint, env, args.seed)
        mode_str = "stochastic — samples from π (mirrors training)" if args.stochastic \
                   else "deterministic — mode of π"
        print(f"[EVAL] Mode: {mode_str}\n")

    print("[INIT] Compiling step/reset …")
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)
    _s0 = jit_reset(jax.random.PRNGKey(999))
    jit_step(_s0, jnp.zeros(env._action_shape)).obs.block_until_ready()
    print("[INIT] Compilation done.\n")

    renderer = None
    if args.record_dir is not None:
        args.record_dir.mkdir(parents=True, exist_ok=True)
        renderer = mujoco.Renderer(env.mj_model, height=480, width=640)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.distance, cam.azimuth, cam.elevation, lookat = _camera_preset(args.camera, is_climbing)
        cam.lookat[:] = lookat
        CAM_BOX = 0.2

    rng = jax.random.PRNGKey(args.seed)

    def get_action(obs_jax: jnp.ndarray) -> np.ndarray:
        nonlocal rng
        if args.random:
            rng, key = jax.random.split(rng)
            return np.asarray(jax.random.uniform(
                key, (ckpt_action_dim,), minval=-1.0, maxval=1.0))
        if args.stochastic:
            rng, key = jax.random.split(rng)
            dist = agent.actor.apply_fn({"params": agent.actor.params}, obs_jax)
            return np.asarray(dist.sample(seed=key))
        return np.asarray(agent.eval_actions(obs_jax))

    def run_episode(ep: int, viewer=None):
        nonlocal rng
        rng, reset_key = jax.random.split(rng)
        state = jit_reset(reset_key)
        total_reward = 0.0
        frames = []
        step = 0

        while (viewer is None or viewer.is_running()) \
              and not bool(state.done) and step < args.max_steps:

            obs = state.obs
            if obs.shape[0] < ckpt_obs_dim:
                obs = jnp.concatenate([obs, jnp.zeros(ckpt_obs_dim - obs.shape[0])])
            else:
                obs = obs[:ckpt_obs_dim]

            action = get_action(obs)
            env_act = env._action_shape[0]
            if action.shape[0] < env_act:
                action = np.concatenate([np.zeros(env_act - action.shape[0]), action])
            else:
                action = action[:env_act]

            state = jit_step(state, jnp.array(action))
            total_reward += float(state.reward)
            step += 1

            trunk_z = float(state.pipeline_state.xpos[env.root_body_id, 2])
            leg_q = np.array(state.pipeline_state.qpos[18:22])
            print(f"  step={step:3d}  trunk_z={trunk_z:.3f}  leg_q={leg_q.round(3)}")

            if viewer is not None:
                _sync_mjdata(env.mj_model, mj_data, state.pipeline_state)
                viewer.sync()
                time.sleep(0.05)

            if renderer is not None:
                rpos = np.asarray(state.pipeline_state.xpos[env.root_body_id][:3])
                follow_axes = (0, 1, 2) if is_climbing else (0, 1)
                for i in follow_axes:
                    d = rpos[i] - cam.lookat[i]
                    if d > CAM_BOX:
                        cam.lookat[i] = rpos[i] - CAM_BOX
                    elif d < -CAM_BOX:
                        cam.lookat[i] = rpos[i] + CAM_BOX
                cpu = mjx.get_data(env.mj_model, state.pipeline_state)
                renderer.update_scene(cpu, camera=cam)
                frames.append(renderer.render().copy())

        trunk_z = float(state.pipeline_state.xpos[env.root_body_id, 2])
        x_vel = float(state.pipeline_state.qvel[0])
        outcome = "FELL" if bool(state.done) and step < args.max_steps else "timeout"
        print(f"Episode {ep+1}/{args.episodes}:  "
              f"return={total_reward:.2f}  steps={step}  "
              f"trunk_z={trunk_z:.3f} m  x_vel={x_vel:+.3f} m/s  [{outcome}]")

        if renderer is not None and frames:
            import imageio
            path = args.record_dir / f"eval_ep{ep+1:02d}.mp4"
            fps = args.fps if args.fps is not None else max(1, round(1.0 / (env.dt * args.slowmo)))
            imageio.mimwrite(str(path), frames, fps=fps)
            print(f"  → {path}")

    if args.viewer:
        import mujoco.viewer as mjv

        mj_data = mujoco.MjData(env.mj_model)
        _sync_mjdata(env.mj_model, mj_data, jit_reset(jax.random.PRNGKey(0)).pipeline_state)

        with mjv.launch_passive(env.mj_model, mj_data) as viewer:
            dist, az, elev, lookat = _camera_preset(args.camera, is_climbing)
            viewer.cam.distance = dist
            viewer.cam.azimuth = az
            viewer.cam.elevation = elev
            if is_climbing or args.camera == "back":
                viewer.cam.lookat[:] = lookat

            for ep in range(args.episodes):
                if not viewer.is_running():
                    break
                run_episode(ep, viewer=viewer)
    else:
        for ep in range(args.episodes):
            run_episode(ep, viewer=None)

    if renderer is not None:
        renderer.close()

if __name__ == "__main__":
    main(tyro.cli(Args))
