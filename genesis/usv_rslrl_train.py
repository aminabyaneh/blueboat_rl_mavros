import argparse
import os
import pickle
import shutil

import torch

# NOTE: Only compatible with:
# git clone https://github.com/leggedrobotics/rsl_rl
# cd rsl_rl && git checkout v1.0.2 && pip install -e .

from rsl_rl.runners import OnPolicyRunner

from usv_env_genesis import USVEnv
from usv_env_cfg import env_cfg, obs_cfg, command_cfg, reward_cfg


def get_train_cfg(exp_name, max_iterations):
    train_cfg_dict = {
        "algorithm": {
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.002,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.0003,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "init_member_classes": {},
        "policy": {
            "activation": "tanh",
            "actor_hidden_dims": [256, 256, 256],
            "critic_hidden_dims": [128, 128, 128],
            "init_noise_std": 1.0,
        },
        "runner": {
            "algorithm_class_name": "PPO",
            "checkpoint": -1,
            "experiment_name": exp_name,
            "load_run": -1,
            "log_interval": 1,
            "max_iterations": max_iterations,
            "num_steps_per_env": 24,
            "policy_class_name": "ActorCritic",
            "record_interval": -1,
            "resume": False,
            "resume_path": None,
            "run_name": "",
            "runner_class_name": "runner_class_name",
            "save_interval": 100,
        },
        "runner_class_name": "OnPolicyRunner",
        "seed": 1,
    }

    return train_cfg_dict

def main():
    # parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="usv_env")
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--render", action="store_true", default=False)
    parser.add_argument("--max_iterations", type=int, default=500)
    parser.add_argument("--resume_path", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true", default=False,
                        help="Delete an existing logs/<exp_name> directory instead of refusing to start.")
    args = parser.parse_args()

    log_dir = f"logs/{args.exp_name}"
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations)

    # never silently discard a previous run's checkpoints
    if os.path.exists(log_dir):
        if not args.overwrite:
            raise FileExistsError(
                f"{log_dir} already exists. Pass --overwrite to delete it, "
                f"or pick a different --exp_name.")
        print(f"Overwriting existing run at {log_dir}")
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    # customize env_cfg
    env_cfg["num_envs"] = args.num_envs
    env_cfg["render"] = args.render

    # save the configurations before the env mutates them (reward scales are multiplied by dt)
    with open(f"{log_dir}/cfgs.pkl", "wb") as cfg_file:
        pickle.dump([env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg], cfg_file)

    # define the environment
    env = USVEnv(num_envs=args.num_envs, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg,
                 command_cfg=command_cfg)

    # policy runner, on the same device the environment picked
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=device)

    if args.resume_path is not None:
        print(f"Loading the model from {args.resume_path}")
        runner.load(args.resume_path)

    # learning loop
    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()
