from setuptools import find_packages, setup

setup(
    name="active_adaptation",
    author="btx0424@SUSTech",
    keywords=["robotics", "rl"],
    packages=find_packages("."),
    install_requires=[
        "hydra-core",
        "omegaconf",
        "wandb",
        "moviepy",
        "imageio",
        "einops",
        "av", # for moviepy
        "pandas",
        "termcolor",
        "pygame", # for game controller
        "mujoco",
        "onnxscript==0.3.0",
        "onnxruntime==1.22.0",
        "torch==2.7.1",
        "torchrl==0.7.0",
        "tensordict==0.7.0",
    ],
)
