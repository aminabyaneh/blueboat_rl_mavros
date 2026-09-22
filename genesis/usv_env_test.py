"""
usv_env_test.py

Smoke test for the USV environment: builds it from the default configuration and
drives it with random actions. Use this to confirm Genesis is installed correctly
before starting a training run.
"""

import torch

from usv_env_genesis import USVEnv
from usv_env_cfg import env_cfg, obs_cfg, command_cfg, reward_cfg


def main():
    """ Test the USV environment.
    """
    # initialize the environment with the given configurations
    env = USVEnv(env_cfg=env_cfg,
                 obs_cfg=obs_cfg,
                 reward_cfg=reward_cfg,
                 command_cfg=command_cfg)

    # set the device to GPU if available, otherwise use CPU
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # reset the environment
    env.reset()

    while True:

        # create random actions for the environments
        action = torch.rand((env_cfg["num_envs"], 2), device=device)

        # perform a step in the environment with the action
        obs, _, reward, reset, extras = env.step(action)


if __name__ == "__main__":
    main()
