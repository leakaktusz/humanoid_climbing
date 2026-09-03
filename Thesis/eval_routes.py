import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "climbing_rl", "climbing_rl"))
sys.path.insert(0, os.path.join(_HERE, "climbing_env_mjx", "climbing_env_mjx"))
sys.path.insert(0, _HERE)

import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import tyro
from mujoco import mjx

from climbing_env_mjx import mjx as climbing_mjx
from climbing_env_mjx.mjx.tasks import climber_climbing as cc

import eval_mjx


def _jitter(tag):
    return [f"kang_jitter{tag}_s{s}" for s in range(5)]

WALL_GROUPS = {
    "anchor":    ["kang"],
    "matched":   ["kang", "kang_phase20", "kang_shift20"] + _jitter("05"),
    "jitter05":  ["kang"] + _jitter("05"),
    "jitter10":  ["kang"] + _jitter("10"),
    "jitter15":  ["kang"] + _jitter("15"),
    "mirror":    ["kang"] + [w for s in range(5)
                             for w in (f"kang_jitter10_s{s}",
                                       f"kang_jitter10_s{s}_mirror")],
    "graded":    ["kang", "kang_dv30", "kang_dv50", "kang_dv60",
                  "kang_dy30", "kang_dy50", "kang_cols2", "kang_cols6",
                  "kang_drop20", "kang_drop40"],
    "routes":    ["kang", "kang_ladder", "kang_single", "kang_zigzag",
                  "kang_traverse", "kang_crux"],
    "incline":   ["kang"] + [f"kang_incline{i}"
                             for i in (-30, -20, -10, -5, 0, 10, 20)],
    "latent":    ["kang", "kang_x45", "kang_x55", "kang_r03", "kang_r08",
                  "kang_prot01", "kang_prot05"],
}


@dataclass
class Args:
    checkpoint: Path = Path("climbing_rl/climbing_rl/best_actor_full.msgpack")
    env:        str  = "climbing_env_mjx-climb-v1"
    walls:      str  = "matched"
    episodes:   int  = 10
    max_steps:  Optional[int] = None
    seed:       int  = 0
    stochastic: bool = False
    frame_skip: Optional[int] = None
    out:        Optional[Path] = None
    list_walls: bool = False

    record_dir: Optional[Path] = None
    record_eps: int = 1
    camera: Literal["default", "back"] = "default"
    fps:    Optional[int] = None
    slowmo: float = 2.0


def _resolve_walls(spec: str) -> list:
    if spec == "all":
        names = ["kang"] + [w for w in sorted(cc.WALL_PRESETS)
                            if w not in ("kang", "easy")]
        return names
    out = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok in WALL_GROUPS:
            out.extend(WALL_GROUPS[tok])
        elif tok in cc.WALL_PRESETS:
            out.append(tok)
        else:
            raise SystemExit(
                f"[ERR] '{tok}' is neither a wall preset nor a group.\n"
                f"      groups: {sorted(WALL_GROUPS)}\n"
                f"      run --list_walls for every preset")
    seen, uniq = set(), []
    for w in out:
        if w not in seen:
            seen.add(w)
            uniq.append(w)
    return uniq

_REACH_WARN = 0.15
_CLIP_TOL = 0.001

_CAM_BOX = 0.2


def _robot_geoms(m):
    free_b = next(m.jnt_bodyid[j] for j in range(m.njnt)
                  if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE)
    out = set()
    for b in range(m.nbody):
        anc = b
        while anc > 0:
            if anc == free_b:
                out.update(range(m.body_geomadr[b],
                                 m.body_geomadr[b] + m.body_geomnum[b]))
                break
            anc = m.body_parentid[anc]
    return out


def _spawn_clip(env, pipeline_state):
    m = env.mj_model
    d = mujoco.MjData(m)
    cpu = mjx.get_data(m, pipeline_state)
    d.qpos[:] = np.array(cpu.qpos)
    d.qvel[:] = np.array(cpu.qvel)
    mujoco.mj_forward(m, d)

    rgeoms = _robot_geoms(m)
    worst, pair = 0.0, ""
    for c in range(d.ncon):
        con = d.contact[c]
        g1, g2 = int(con.geom1), int(con.geom2)
        if (g1 in rgeoms) == (g2 in rgeoms):
            continue
        nm = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or f"geom{g}"
              for g in (g1, g2)]
        if any("hold" in n for n in nm):
            continue
        if con.dist < worst:
            worst, pair = float(con.dist), f"{nm[0]} <-> {nm[1]}"
    return worst, pair


def _reset_geometry(env):
    state = jax.jit(env.reset)(jax.random.PRNGKey(0))
    _diffs, d = env._limb_hold_dists(state.pipeline_state)
    d = np.asarray(d)
    pen, pair = _spawn_clip(env, state.pipeline_state)
    return float(d.min()), int((d < 0).sum()), pen, pair


def _status(gap0: float, overlap: int, pen: float = 0.0) -> str:
    if pen < -_CLIP_TOL:
        return "IN WALL"
    if overlap:
        return "OVERLAP"
    if gap0 > _REACH_WARN:
        return "UNREACHABLE"
    return ""


def _run_wall(wall: str, args: Args) -> list:
    if args.frame_skip is not None:
        cc._FRAME_SKIP = args.frame_skip
    env = climbing_mjx.load(environment_name=args.env, task_kwargs={"wall": wall})
    agent, ckpt_obs, ckpt_act = eval_mjx._load_agent(args.checkpoint, env, args.seed)
    env_act = env._action_shape[0]

    jit_reset, jit_step = jax.jit(env.reset), jax.jit(env.step)
    print("    compiling", end="", flush=True)
    t0 = time.time()
    _s = jit_reset(jax.random.PRNGKey(0))
    jit_step(_s, jnp.zeros(env._action_shape)).obs.block_until_ready()
    compile_s = time.time() - t0
    print(f" {compile_s:.0f}s", flush=True)

    gap0, overlap, pen, pen_pair = _reset_geometry(env)
    max_steps = (args.max_steps if args.max_steps is not None
                 else env.max_episode_steps)

    renderer, cam, cam_home = None, None, None
    is_climbing = args.env.endswith("-climb-v1")
    if args.record_dir is not None and args.record_eps > 0:
        args.record_dir.mkdir(parents=True, exist_ok=True)
        renderer = mujoco.Renderer(env.mj_model, height=480, width=640)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        _cam = eval_mjx._camera_preset(args.camera, is_climbing)
        cam.distance, cam.azimuth, cam.elevation, cam_home = _cam

    rng = jax.random.PRNGKey(args.seed)
    rows = []
    for ep in range(args.episodes):
        state = jit_reset(jax.random.PRNGKey(args.seed + ep))
        z0 = float(state.pipeline_state.xpos[env.root_body_id, 2])
        max_z, grip_steps, step = z0, 0, 0
        total_r = 0.0

        print(f"    ep {ep + 1:2d}/{args.episodes} ", end="", flush=True)
        rec, frames = (renderer is not None and ep < args.record_eps), []
        if rec:
            cam.lookat[:] = cam_home

        while not bool(state.done) and step < max_steps:
            obs = state.obs
            obs = (jnp.concatenate([obs, jnp.zeros(ckpt_obs - obs.shape[0])])
                   if obs.shape[0] < ckpt_obs else obs[:ckpt_obs])
            if args.stochastic:
                rng, ak = jax.random.split(rng)
                dist = agent.actor.apply_fn({"params": agent.actor.params}, obs)
                action = np.asarray(dist.sample(seed=ak))
            else:
                action = np.asarray(agent.eval_actions(obs))
            action = (np.concatenate([np.zeros(env_act - action.shape[0]), action])
                      if action.shape[0] < env_act else action[:env_act])

            state = jit_step(state, jnp.array(action))
            total_r += float(state.reward)
            step += 1
            max_z = max(max_z, float(state.pipeline_state.xpos[env.root_body_id, 2]))
            grip_steps += int(np.asarray(state.obs[-4:]).sum() > 0)
            if step % 100 == 0:
                print(".", end="", flush=True)

            if rec:
                rpos = np.asarray(state.pipeline_state.xpos[env.root_body_id][:3])
                for i in ((0, 1, 2) if is_climbing else (0, 1)):
                    d = rpos[i] - cam.lookat[i]
                    if d > _CAM_BOX:
                        cam.lookat[i] = rpos[i] - _CAM_BOX
                    elif d < -_CAM_BOX:
                        cam.lookat[i] = rpos[i] + _CAM_BOX
                renderer.update_scene(mjx.get_data(env.mj_model,
                                                   state.pipeline_state), camera=cam)
                frames.append(renderer.render().copy())

        video = ""
        if rec and frames:
            import imageio
            path = args.record_dir / f"{wall}_ep{ep + 1:02d}.mp4"
            fps = args.fps
            if fps is None:
                fps = max(1, round(1.0 / (env.dt * args.slowmo)))
            imageio.mimwrite(str(path), frames, fps=fps)
            video = str(path)

        fell = bool(state.done) and step < max_steps
        holds = int((np.asarray(state.metrics["hold_touch_time"]) > 0).sum())
        print(f" gain {max_z - z0:+.3f} m  steps {step:4d}  holds {holds}  "
              f"{'FELL' if fell else 'alive'}"
              + (f"  -> {Path(video).name}" if video else ""), flush=True)
        rows.append(dict(
            wall=wall, episode=ep, steps=step, fell=int(fell),
            z0=z0, max_z=max_z, gain=max_z - z0,
            holds=holds,
            grip=grip_steps / max(step, 1),
            ret=total_r, n_holds=int(env.n_holds), compile_s=compile_s,
            frame_skip=env.n_substeps, dt=env.dt,
            gap0=gap0, overlap=overlap, pen=pen, pen_pair=pen_pair,
            status=_status(gap0, overlap, pen),
            video=video,
        ))

    if renderer is not None:
        renderer.close()
    return rows


def _agg(rows, key):
    v = np.array([r[key] for r in rows], dtype=float)
    ci = 1.96 * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
    return v.mean(), ci


def main(args: Args) -> None:
    if args.list_walls:
        print("groups:", ", ".join(sorted(WALL_GROUPS)))
        print("\npresets:")
        for nm in sorted(cc.WALL_PRESETS):
            ov = cc.WALL_PRESETS[nm]
            desc = "the training wall" if nm == "kang" else (
                ", ".join(f"{k}={v}" for k, v in ov.items() if k != "holds")
                or "explicit hold list")
            print(f"  {nm:28s} {desc}")
        return

    walls = _resolve_walls(args.walls)
    print(f"[INIT] checkpoint : {args.checkpoint}")
    print(f"[INIT] walls      : {len(walls)} ({', '.join(walls)})")
    print(f"[INIT] episodes   : {args.episodes} per wall, "
          f"{'stochastic' if args.stochastic else 'deterministic'}, "
          f"max {args.max_steps if args.max_steps is not None else 'env'} steps")
    _fs = args.frame_skip if args.frame_skip is not None else cc._FRAME_SKIP
    print(f"[INIT] frame_skip : {_fs} (dt = {_fs * 0.003:.3f} s"
          f"{'' if args.frame_skip is not None else ', task default'})"
          f" -- must match what the checkpoint was TRAINED with\n")

    all_rows, summary = [], []
    for i, wall in enumerate(walls, 1):
        print(f"[{i}/{len(walls)}] {wall}", flush=True)
        rows = _run_wall(wall, args)
        all_rows.extend(rows)
        gain, gain_ci = _agg(rows, "gain")
        summary.append((wall, rows, gain, gain_ci))
        print(f"    = mean gain {gain:+.3f} +/-{gain_ci:.3f} m "
              f"over {len(rows)} episodes", flush=True)

    anchor = next((s for s in summary if s[0] == "kang"), None)
    print(f"\n{'route':28s} {'gain (m)':>16s} {'vs kang':>9s} {'max_z':>7s}"
          f" {'holds':>6s} {'grip':>6s} {'steps':>7s} {'fell':>6s}"
          f" {'gap0':>6s} {'pen':>6s}  {'status':<11s}")
    print("-" * 120)
    flagged = []
    for wall, rows, gain, gain_ci in summary:
        delta = "  --   " if (anchor is None or wall == "kang") else \
                f"{100 * (gain - anchor[2]) / abs(anchor[2]):+6.0f}%" \
                if abs(anchor[2]) > 1e-6 else "   n/a "
        st = rows[0]["status"]
        if st:
            flagged.append((wall, st, rows[0]["gap0"], rows[0]["overlap"],
                            rows[0]["pen"], rows[0]["pen_pair"]))
        print(f"{wall:28s} {gain:+7.3f} +/-{gain_ci:5.3f} {delta:>9s}"
              f" {_agg(rows,'max_z')[0]:7.2f} {_agg(rows,'holds')[0]:6.1f}"
              f" {_agg(rows,'grip')[0]:6.2f} {_agg(rows,'steps')[0]:7.0f}"
              f" {_agg(rows,'fell')[0]:6.2f} {rows[0]['gap0']:+6.3f} {rows[0]['pen']:+6.3f}  {st:<11s}")

    if flagged:
        print()
        for wall, st, gap0, ov, pen, pen_pair in flagged:
            if st == "IN WALL":
                print(f"[INVALID] {wall}: the robot spawns {abs(pen):.3f} m "
                      f"INSIDE world geometry ({pen_pair}). An overhanging "
                      f"face leans into a spawn pose that _SPAWN_X "
                      f"({cc._SPAWN_X} m) never adjusts for, so MuJoCo opens "
                      f"the episode by pushing the robot out -- this row is "
                      f"the shove, not the policy.")
            elif st == "OVERLAP":
                print(f"[INVALID] {wall}: {ov} limb/hold pair(s) interpenetrate at "
                      f"reset (gap {gap0:+.3f} m). MuJoCo starts the episode by "
                      f"pushing them apart, so this row measures the spawn, not "
                      f"the policy.")
            else:
                print(f"[INVALID] {wall}: nearest hold is {gap0:.3f} m from the "
                      f"closest limb at reset, vs a grip radius of "
                      f"{cc._GRIP_RADIUS} m. Nothing is in reach from the spawn "
                      f"pose, so this row cannot show climbing.")

    if anchor is not None and anchor[2] < 0.10:
        print(f"\n[WARN] On its OWN training wall this checkpoint gains only "
              f"{anchor[2]:+.3f} m over {_agg(anchor[1], 'steps')[0]:.0f} steps "
              f"({_agg(anchor[1], 'fell')[0]:.0%} falls).")
        print("[WARN] There is no baseline to generalize FROM, so the "
              "differences between routes below are not interpretable.")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            w.writerows(all_rows)
        print(f"\n-> {args.out}  ({len(all_rows)} episodes)")

if __name__ == "__main__":
    main(tyro.cli(Args))
