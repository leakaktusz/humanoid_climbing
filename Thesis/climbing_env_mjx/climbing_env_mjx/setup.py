from setuptools import setup, find_packages

setup(
    name="climbing_env_mjx",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        "dm_control",
        "dm_env",
        "mujoco",
        "numpy",
        "absl-py",
        "jax",
        "jaxlib",
    ],
)
