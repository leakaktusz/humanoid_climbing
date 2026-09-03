import sys
import os
from pathlib import Path
from typing import Optional
import tyro
from dataclasses import dataclass, asdict, replace as dataclass_replace
import os
os.environ["WANDB_SILENT"] = "true"
import wandb
import time
import random
import gc
from tqdm import tqdm
import subprocess
import flax.serialization
import jax
import jax.numpy as jnp
from dm_env import specs as dm_specs
from functools import partial

import sac_jax as sac
import specs_jax as specs
import replay_pure_jax as replay
from climbing_env_mjx import mjx


@dataclass(frozen=True)
class Args:
    root_dir: str = "../outputs"
    seed: int = 53
    name: str = ""
    max_steps: int = 1_000_000
    warmstart_steps: int = 50_000
    log_interval: int = 1_000
    eval_interval: int = 10_000
    num_envs: int = 128
    batch_size: int = 128
    rollout_length: int = 16
    updates_per_step: int = 16
    auto_tau: bool = True
    discount: float = 0.99
    tqdm_bar: bool = False
    replay_capacity: int = 200_000
    project: str = "climbing_env_mjx"
    mode: str = "online"
    environment_name: str = "climbing_env_mjx-climber-v1"
    n_steps_lookahead: int = 10
    trim_silence: bool = False
    gravity_compensation: bool = False
    reduced_action_space: bool = False
    control_timestep: float = 0.05
    stretch_factor: float = 1.0
    shift_factor: int = 0
    clip: bool = True
    record_dir: Optional[Path] = None
    resume: Optional[str] = None
    critic_warmup_steps: int = 100_000
    checkpoint_interval: int = 8_000_000
    agent_config: sac.SACConfig = sac.SACConfig()


def get_env_mjx(args: Args, record_dir: Optional[Path] = None):
    camera_id = -1
    env = mjx.load(
        environment_name=args.environment_name,
        seed=args.seed,
        stretch=args.stretch_factor,
        shift=args.shift_factor,
        task_kwargs=dict(
            n_steps_lookahead=args.n_steps_lookahead,
            trim_silence=args.trim_silence,
            gravity_compensation=args.gravity_compensation,
            reduced_action_space=args.reduced_action_space,
            control_timestep=args.control_timestep,
        ),
    )
    return env

class AsyncSaver:
    def __init__(self, save_fn):
        self.prev_params = None
        self.save_fn = save_fn

    def __call__(self, params, name):
        self.prev_params, to_save = params, self.prev_params

        if to_save is None:
            return

        self.save_fn(to_save, name)

class RecordWriter:
    def __init__(self, window: int = 16, save_fn=None):
        self.window   = window
        self.save_fn  = save_fn
        self.best_ep_reward = -float("inf")
        self.reward_buf = []
        self.metrics_buf = []
        self.ep_len_buf = []
        self.terms_buf = []

    def __call__(self, reward_batch, done_batch, steps_batch, cur_metrics: dict, agent=None, env_step=None, reward_terms=None):
        self.reward_buf.append(reward_batch)
        self.metrics_buf.append(cur_metrics)
        self.ep_len_buf.append((done_batch, steps_batch))
        if reward_terms:
            self.terms_buf.append(reward_terms)

        if len(self.reward_buf) >= self.window:
            reward = float(jnp.mean(jnp.stack(self.reward_buf)))

            all_done  = jnp.ravel(jnp.stack([d for d, _ in self.ep_len_buf]))
            all_steps = jnp.ravel(jnp.stack([s for _, s in self.ep_len_buf]))
            n_done    = float(jnp.sum(all_done))
            mean_ep_len = (
                float(jnp.sum(all_steps * all_done.astype(jnp.float32))) / max(n_done, 1.0)
            )

            mean_ep_reward = reward * mean_ep_len

            log_dict = {
                "train/mean_ep_reward": mean_ep_reward,
                "train/mean_ep_len":    mean_ep_len,
                "train/mean_reward":    reward,
                "train/n_episodes":     n_done,
                "train/critic_loss":  float(jnp.mean(jnp.stack([m["critic_loss"]  for m in self.metrics_buf]))),
                "train/actor_loss":   float(jnp.mean(jnp.stack([m["actor_loss"]   for m in self.metrics_buf]))),
                "train/q_values":     float(jnp.mean(jnp.stack([m["q"]            for m in self.metrics_buf]))),
                "train/entropy":      float(jnp.mean(jnp.stack([m["entropy"]      for m in self.metrics_buf]))),
                "train/temperature":  float(jnp.mean(jnp.stack([m["temperature"]  for m in self.metrics_buf]))),
            }
            if self.terms_buf:
                for k in self.terms_buf[0]:
                    log_dict[f"train/reward_terms/{k[len('rew_'):]}"] = float(
                        jnp.mean(jnp.stack([t[k] for t in self.terms_buf]))
                    )
            if env_step is not None:
                log_dict["env_step"] = env_step
            wandb.log(log_dict)
            print(f"[TRAIN]: ep_reward={mean_ep_reward:.1f}  ep_len={mean_ep_len:.1f}  reward/step={reward:.3f}  n_ep={int(n_done)}")

            if self.save_fn is not None and agent is not None and n_done > 0:
                if mean_ep_reward > self.best_ep_reward:
                    self.best_ep_reward = mean_ep_reward
                    self.save_fn(agent.actor.params, "best_actor")

            self.reward_buf.clear()
            self.metrics_buf.clear()
            self.ep_len_buf.clear()
            self.terms_buf.clear()


def main(args: Args) -> None:
    if args.name:
        run_name = args.name
    else:
        run_name = f"SAC-{args.environment_name}-{args.seed}-{time.time()}"

    def step_cond_reset(state, action, reset_key):
            next_state = env.step(state, action)
            def do_reset(_):
                return env.reset(reset_key)

            def keep(_):
                return next_state

            new_state = jax.lax.cond(
                next_state.done > 0.5,
                do_reset,
                keep,
                operand=None
            )
            return new_state, next_state

    vmap_step_cond = jax.vmap(step_cond_reset)

    def scan_step_random(carry, inputs):
        state, agent, replay_buffer, rng = carry
        rng, key_action, key_env = jax.random.split(rng, 3)

        action = jax.random.uniform(key_action, shape=(args.num_envs, action_dim), minval=-1.0, maxval=1.0)
        env_keys = jax.random.split(key_env, args.num_envs)

        new_state, real_next_state = vmap_step_cond(state, action, env_keys)
        reward = real_next_state.reward
        discount = real_next_state.info['discount']

        replay_buffer = replay_buffer.insert_batch(
            state.obs, action, reward, discount, real_next_state.obs
        )

        done  = real_next_state.done
        steps = real_next_state.metrics['steps']
        return (new_state, agent, replay_buffer, rng), (reward, done, steps)

    def scan_step_policy_collect(carry, inputs):
        state, agent, replay_buffer, rng = carry
        rng, key_env = jax.random.split(rng)

        agent, action = agent.sample_actions(state.obs)
        env_keys = jax.random.split(key_env, args.num_envs)

        new_state, real_next_state = vmap_step_cond(state, action, env_keys)
        reward = real_next_state.reward
        discount = real_next_state.info['discount']

        replay_buffer = replay_buffer.insert_batch(
            state.obs, action, reward, discount, real_next_state.obs
        )

        done  = real_next_state.done
        steps = real_next_state.metrics['steps']
        return (new_state, agent, replay_buffer, rng), (reward, done, steps)

    def scan_step(carry, inputs, critic_only: bool = False):
        state, agent, replay_buffer, rng = carry
        rng, key_env, key_update = jax.random.split(rng, 3)

        agent, action = agent.sample_actions(state.obs)
        env_keys = jax.random.split(key_env, args.num_envs)

        new_state, real_next_state = vmap_step_cond(state, action, env_keys)
        reward = real_next_state.reward
        discount = real_next_state.info['discount']

        replay_buffer = replay_buffer.insert_batch(
            state.obs, action, reward, discount, real_next_state.obs
        )

        def do_update(carry, _):
            agent, rng = carry
            rng, key = jax.random.split(rng)
            transitions = replay_buffer.sample(key)
            if critic_only:
                agent, metrics = agent.update_critic_only(transitions)
            else:
                agent, metrics = agent.update(transitions)
            return (agent, rng), metrics

        rng, key_update = jax.random.split(rng)
        (agent, _), metrics = jax.lax.scan(do_update, (agent, key_update), None, length=args.updates_per_step)

        done  = real_next_state.done
        steps = real_next_state.metrics['steps']
        rew_terms = {k: jnp.mean(v) for k, v in real_next_state.metrics.items()
                     if k.startswith("rew_")}
        return (new_state, agent, replay_buffer, rng), (reward, done, steps, metrics, rew_terms)

    def save_actor(params,name):
        try:
            actor_bytes = flax.serialization.to_bytes(params)
            with open(checkpoint_dir / f"{name}.msgpack", 'wb') as f:
                f.write(actor_bytes)

            safe_run_name = "".join(c if (c.isalnum() or c in "_-.") else "-" for c in run_name)
            artifact = wandb.Artifact(f"{name}_{safe_run_name}", type="model")

            initial_path = checkpoint_dir / f"{name}.msgpack"
            artifact.add_file(str(initial_path))
            wandb.log_artifact(artifact)
        except Exception as e:
            print(f"[ERROR] Failed to save actor: {e}")

    def save_full_state(agent, name):
        try:
            payload = {
                "actor": agent.actor.params,
                "critic": agent.critic.params,
                "target_critic": agent.target_critic.params,
                "temp": agent.temp.params,
            }
            full_path = checkpoint_dir / f"{name}.msgpack"
            with open(full_path, "wb") as f:
                f.write(flax.serialization.to_bytes(payload))

            safe_run_name = "".join(c if (c.isalnum() or c in "_-.") else "-" for c in run_name)
            artifact = wandb.Artifact(f"{name}_{safe_run_name}", type="model")
            artifact.add_file(str(full_path))
            wandb.log_artifact(artifact)
        except Exception as e:
            print(f"[ERROR] Failed to save full state: {e}")

    def record_best_actor_video(args: Args, checkpoint_dir: Path, experiment_dir: Path) -> None:
        best_ckpt = checkpoint_dir / "best_actor.msgpack"
        if not best_ckpt.exists():
            print("[EVAL] No best_actor checkpoint found — skipping video.")
            return

        video_dir = experiment_dir / "eval_video"
        eval_script = Path(__file__).resolve().parent / "eval_mjx.py"

        cmd = [
            sys.executable, str(eval_script),
            "--checkpoint", str(best_ckpt),
            "--env", args.environment_name,
            "--episodes", "1",
            "--max_steps", "2000",
            "--record_dir", str(video_dir),
            "--seed", str(args.seed),
            "--slowmo", "1",
        ]
        print(f"[EVAL] Recording best actor: {' '.join(cmd)}")
        child_env = dict(os.environ)
        child_env.setdefault("MUJOCO_GL", "egl")
        child_env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        try:
            subprocess.run(cmd, check=True, env=child_env,
                           capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print(f"[EVAL] Failed to record video: {e}")
            tail = "\n".join((e.stderr or "").strip().splitlines()[-30:])
            if tail:
                print(f"[EVAL] eval_mjx.py stderr (last 30 lines):\n{tail}")
            return
        except Exception as e:
            print(f"[EVAL] Failed to record video: {e}")
            return

        videos = sorted(video_dir.glob("*.mp4"))
        if not videos:
            print("[EVAL] eval_mjx.py produced no video — skipping wandb log.")
            return
        try:
            wandb.log({"eval/best_actor_video": wandb.Video(str(videos[0]), fps=20, format="mp4")})
            print(f"[EVAL] Logged {videos[0]} to wandb.")
        except Exception as e:
            print(f"[EVAL] Failed to log video to wandb: {e}")

    experiment_dir = Path(args.root_dir) / run_name
    experiment_dir.mkdir(parents=True)
    checkpoint_dir = experiment_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    random.seed(args.seed)
    writer = RecordWriter()
    saver = AsyncSaver(save_actor)

    wandb.init(
        project=args.project,
        tags=[],
        notes= None,
        config=asdict(args),
        mode=args.mode,
        name=run_name,
        settings=wandb.Settings(start_method="thread"),
    )

    wandb.define_metric("env_step")
    wandb.define_metric("train/*", step_metric="env_step")
    wandb.define_metric("diag/*", step_metric="env_step")

    if wandb.run:
        print(f"[WANDB] Run URL: {wandb.run.get_url()}")
    else:
        print("[WANDB_ERROR] Wandb run not initialized!")

    env = get_env_mjx(args)

    def _git_rev(path: Path) -> str:
        try:
            rev = subprocess.run(["git", "-C", str(path), "rev-parse", "--short", "HEAD"],
                                 capture_output=True, text=True, check=True).stdout.strip()
            dirty = subprocess.run(["git", "-C", str(path), "status", "--porcelain"],
                                   capture_output=True, text=True, check=True).stdout.strip()
            return rev + ("-dirty" if dirty else "")
        except Exception:
            return "unknown"

    env_mod = sys.modules[type(env).__module__]
    reward_consts = {
        n: v for n, v in vars(env_mod).items()
        if n.startswith("_") and n.isupper() and isinstance(v, (int, float, bool, dict))
    }
    wandb.config.update({
        "env_reward_consts": reward_consts,
        "env_task_module": type(env).__module__,
        "env_git": _git_rev(Path(env_mod.__file__).parent),
        "train_git": _git_rev(Path(__file__).parent),
    }, allow_val_change=True)

    print(f"[DIAG] root_body_id: {env.root_body_id}")
    print(f"[DIAG] root_joint_id: {env.root_jnt_id}")
    print(f"[DIAG] body name: {env.mj_model.body(env.root_body_id).name}")

    rng = jax.random.PRNGKey(args.seed)
    rng, key = jax.random.split(rng)

    vmap_reset = jax.vmap(env.reset)
    env_keys = jax.random.split(key, args.num_envs)
    initial_state = vmap_reset(env_keys)
    print(f"[DIAG] initial trunk z-axis z: {float(initial_state.obs[0, -1]):.4f}")
    print(f"[DIAG] qpos at reset: {initial_state.obs[0, :7]}")

    obs_shape = initial_state.obs.shape[1:]
    action_dim = env.action_spec().shape[0]

    observation_spec = dm_specs.Array(shape=obs_shape, dtype=jnp.float32, name='observation')
    action_spec = dm_specs.BoundedArray(shape=(action_dim,), dtype=jnp.float32, minimum=-1.0, maximum=1.0, name='action')

    spec = specs.EnvironmentSpec(observation=observation_spec, action=action_spec)
    replay_buffer = replay.PureJaxBuffer.create(
        max_size=args.replay_capacity,
        state_dim=spec.observation_dim,
        action_dim=spec.action_dim,
        batch_size=args.batch_size,
    )
    agent_config = args.agent_config
    if args.auto_tau:
        scaled_tau = args.agent_config.tau * (16.0 / args.updates_per_step)
        agent_config = dataclass_replace(args.agent_config, tau=scaled_tau)
    print(f"[CFG] updates_per_step={args.updates_per_step}  "
          f"UTD={args.updates_per_step / args.num_envs:.3f}  "
          f"tau={agent_config.tau:.6g} (auto_tau={args.auto_tau})")
    agent = sac.SAC.initialize(
        spec=spec,
        config=agent_config,
        seed=args.seed,
        discount=args.discount,
    )

    critic_warmup_steps = 0
    if args.resume:
        raw = Path(args.resume).read_bytes()
        restored = flax.serialization.msgpack_restore(raw)
        if isinstance(restored, dict) and "critic" in restored:
            target = {
                "actor": agent.actor.params,
                "critic": agent.critic.params,
                "target_critic": agent.target_critic.params,
                "temp": agent.temp.params,
            }
            new = flax.serialization.from_bytes(target, raw)
            agent = agent.replace(
                actor=agent.actor.replace(params=new["actor"]),
                critic=agent.critic.replace(params=new["critic"]),
                target_critic=agent.target_critic.replace(params=new["target_critic"]),
                temp=agent.temp.replace(params=new["temp"]),
            )
            print(f"[RESUME] Full state (actor+critic+target+temperature) from {args.resume} — no critic warmup.")
        else:
            ckpt_in = restored["MLP_0"]["Dense_0"]["kernel"].shape[0]
            env_in = int(obs_shape[0])
            if ckpt_in != env_in:
                from climbing_env_mjx.mjx.tasks.climber_fingers_climbing import remap_plain_actor
                print(f"[RESUME] Actor input {ckpt_in} != env obs {env_in} — "
                      f"auto-remapping plain-climb actor to this env by joint/body name.")
                restored = remap_plain_actor(restored)
                raw = flax.serialization.to_bytes(restored)
            new_params = flax.serialization.from_bytes(agent.actor.params, raw)
            agent = agent.replace(actor=agent.actor.replace(params=new_params))
            critic_warmup_steps = args.critic_warmup_steps
            print(f"[RESUME] Actor-only checkpoint from {args.resume} — critic warmup {critic_warmup_steps:,} steps.")
        print("[RESUME] Warmstart phase collects with the resumed policy (not random actions).")

    warmup_scan = scan_step_policy_collect if args.resume else scan_step_random
    jit_rollout_random = jax.jit(
        lambda carry: jax.lax.scan(warmup_scan, carry, None, length=args.rollout_length)
    )

    jit_rollout_policy = jax.jit(
        lambda carry: jax.lax.scan(scan_step, carry, None, length=args.rollout_length)
    )

    jit_rollout_critic_only = jax.jit(
        lambda carry: jax.lax.scan(
            partial(scan_step, critic_only=True), carry, None, length=args.rollout_length
        )
    )
    save_actor(agent.actor.params, "initial_actor")

    state = initial_state

    episode_return = 0.0
    episode_length = 0

    start_time = time.time()
    best_eval_reward = -float('inf')
    bar = tqdm(total=args.max_steps, disable=not args.tqdm_bar)

    last_perf_time = start_time
    steps_since_perf = 0

    params_before = jax.tree_util.tree_leaves(agent.actor.params)[0]
    critic_before = jax.tree_util.tree_leaves(agent.critic.params)[0]
    buffer_before = jax.tree_util.tree_leaves(replay_buffer)[0]
    step_idx = 0
    last_log_step = 0
    loop_idx = 0
    total_steps_in_batch = args.rollout_length * args.num_envs
    printed_diag = False
    printed_critic_warmup_done = False
    if critic_warmup_steps > 0:
        print(f"[RESUME] Steps {args.warmstart_steps:,}–{args.warmstart_steps + critic_warmup_steps:,}: "
              f"critic-only updates (actor frozen).")
    while step_idx < args.max_steps:
        loop_idx += 1

        if step_idx < args.warmstart_steps:
            (state, agent, replay_buffer, rng), (reward_batch, done_batch, steps_batch) = jit_rollout_random((state, agent, replay_buffer, rng))
        else:
            if not printed_diag:
                printed_diag = True
                rng, key = jax.random.split(rng)
                sample = replay_buffer.sample(key)
                print(f"[DIAG] buffer size: {int(replay_buffer._size)}")
                print(f"[DIAG] reward mean: {float(jnp.mean(sample.reward)):.4f}")
                print(f"[DIAG] reward std: {float(jnp.std(sample.reward)):.4f}")
                print(f"[DIAG] reward min: {float(jnp.min(sample.reward)):.4f}")
                print(f"[DIAG] reward max: {float(jnp.max(sample.reward)):.4f}")
                print(f"[DIAG] obs std: {float(jnp.std(sample.state)):.4f}")
                print(f"[DIAG] discount mean: {float(jnp.mean(sample.discount)):.4f}")

            in_critic_warmup = critic_warmup_steps > 0 and step_idx < args.warmstart_steps + critic_warmup_steps
            if in_critic_warmup:
                (state, agent, replay_buffer, rng), (reward_batch, done_batch, steps_batch, metrics, rew_terms_batch) = jit_rollout_critic_only((state, agent, replay_buffer, rng))
            else:
                if critic_warmup_steps > 0 and not printed_critic_warmup_done:
                    printed_critic_warmup_done = True
                    print(f"[RESUME] Critic warmup done at step {step_idx:,} — actor/temperature updates now active.")
                (state, agent, replay_buffer, rng), (reward_batch, done_batch, steps_batch, metrics, rew_terms_batch) = jit_rollout_policy((state, agent, replay_buffer, rng))

        if step_idx >= args.warmstart_steps and step_idx < args.warmstart_steps + 2 * total_steps_in_batch:
            params_after = jax.tree_util.tree_leaves(agent.actor.params)[0]
            critic_after = jax.tree_util.tree_leaves(agent.critic.params)[0]
            actor_changed  = not jnp.allclose(params_before, params_after)
            critic_changed = not jnp.allclose(critic_before, critic_after)
            changed = critic_changed if critic_warmup_steps > 0 else actor_changed
            param_delta = float(jnp.max(jnp.abs(params_after - params_before)))
            print(f"[DIAG] Actor params changed after update: {actor_changed} (max delta: {param_delta:.6f})")
            print(f"[DIAG] Critic params changed after update: {critic_changed}")
            rng, diag_key = jax.random.split(rng)
            diag_obs = state.obs[0]
            diag_action = agent.eval_actions(diag_obs)
            print(f"[DIAG] Eval action (first 5 joints): {diag_action[:5]}")
            print(f"[DIAG] Action mean: {float(jnp.mean(diag_action)):.4f}, std: {float(jnp.std(diag_action)):.4f}")
            if not changed:
                print("[DIAG] ❌ Agent is NOT updating — bug is in sac_jax.py or JIT tracing")
            else:
                print("[DIAG] ✅ Agent is updating — bug is in data pipeline or reward logging")

            buffer_after = jax.tree_util.tree_leaves(replay_buffer)[0]
            buffer_changed = not jnp.allclose(buffer_before, buffer_after)
            print(f"[DIAG] Replay buffer changed after update: {buffer_changed}")
            if not buffer_changed:
                print("[DIAG] ❌ Replay buffer is NOT updating — bug is in rollout or buffer insert")
            else:
                print("[DIAG] ✅ Replay buffer is updating")

        if step_idx >= args.warmstart_steps and loop_idx % 10 == 0:
            root_vx  = state.pipeline_state.qvel[:, 0]
            trunk_z  = state.pipeline_state.xpos[:, env.root_body_id, 2]
            root_x   = state.pipeline_state.qpos[:, 0]
            print(
                f"[CHECK8 step={step_idx:,d}] "
                f"root_vx={float(jnp.mean(root_vx)):+.3f} m/s (target 1.0)  "
                f"trunk_z={float(jnp.mean(trunk_z)):.3f} m (stand 0.70)  "
                f"root_x={float(jnp.mean(root_x)):.3f} m"
            )
            diag_log = {
                "diag/mean_root_vx": float(jnp.mean(root_vx)),
                "diag/max_root_vx":  float(jnp.max(root_vx)),
                "diag/mean_trunk_z": float(jnp.mean(trunk_z)),
                "diag/mean_root_x":  float(jnp.mean(root_x)),
                "env_step":          step_idx,
            }
            if "best_z" in state.metrics and "spawn_z" in state.metrics:
                climb_gain = state.metrics["best_z"] - state.metrics["spawn_z"]
                diag_log["diag/mean_climb_gain"] = float(jnp.mean(climb_gain))
                diag_log["diag/max_climb_gain"]  = float(jnp.max(climb_gain))
                print(f"[CHECK8] climb_gain mean={diag_log['diag/mean_climb_gain']:.3f} "
                      f"max={diag_log['diag/max_climb_gain']:.3f} m")
            wandb.log(diag_log)

        if step_idx >= args.warmstart_steps:
            done_flat  = jnp.ravel(done_batch)
            steps_flat = jnp.ravel(steps_batch)
            n_done = float(jnp.sum(done_flat))
            if n_done > 0:
                mean_ep_len       = float(jnp.sum(steps_flat * done_flat.astype(jnp.float32))) / n_done
                mean_reward_step  = float(jnp.mean(reward_batch))
                current_ep_reward = mean_ep_len * mean_reward_step
                if current_ep_reward > best_eval_reward:
                    best_eval_reward = current_ep_reward
                    save_actor(agent.actor.params, "best_actor")
                    save_full_state(agent, "best_actor_full")
            writer(reward_batch, done_batch, steps_batch, metrics, env_step=step_idx,
                   reward_terms=rew_terms_batch)

            checkpoint_every = max(1, args.checkpoint_interval // total_steps_in_batch)
            if loop_idx % checkpoint_every == 0:
                saver(agent.actor.params, name=f"actor_{step_idx}")

        step_idx += total_steps_in_batch

    bar.close()

    del state, agent, replay_buffer, jit_rollout_random, jit_rollout_policy
    gc.collect()
    jax.clear_caches()

    record_best_actor_video(args, checkpoint_dir, experiment_dir)

    wandb.finish()
    print("[INFO] Training complete!")

if __name__ == "__main__":
    main(tyro.cli(Args, description=__doc__))
