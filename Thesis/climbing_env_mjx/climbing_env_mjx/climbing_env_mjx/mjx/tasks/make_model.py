import mujoco
from dm_control import composer
from climbing_env_mjx.suite.tasks.climber_walking import ClimberForWalking

def make_model() -> mujoco.MjModel:
    task = ClimberForWalking()
    env = composer.Environment(task, strip_singleton_obs_buffer_dim=True)

    return env.physics.model._model

if __name__ == "__main__":
    model = make_model()
    print(f"Successfully created model with {model.nq} qpos and {model.nv} qvel")
