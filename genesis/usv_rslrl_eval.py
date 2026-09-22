import argparse
import os
import pickle

import torch
from usv_env_genesis import USVEnv
from usv_env_cfg import env_cfg as default_env_cfg, command_cfg as default_command_cfg
from rsl_rl.runners import OnPolicyRunner


def fill_missing(saved, defaults, label):
    """ Backfill configuration keys that did not exist when a run was saved.

    Older runs were pickled before some keys were introduced, and the environment would
    raise a KeyError on them. Saved values always win, defaults only fill the gaps.
    """

    missing = sorted(set(defaults) - set(saved))
    if missing:
        print(f"Backfilling {label} keys absent from this run: {missing}")
    return {**defaults, **saved}


def main():
    # parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="usv_env")
    parser.add_argument("--ckpt", type=int, default=500)
    parser.add_argument("--record", action="store_true", default=False)
    parser.add_argument("--headless", action="store_true", default=False,
                        help="Disable the interactive viewer (useful over SSH or when only recording).")
    args = parser.parse_args()

    # the environment picks its own device; keep the policy on the same one
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # load the configurations
    log_dir = f"logs/{args.exp_name}"
    if not os.path.exists(log_dir):
        raise FileNotFoundError(
            f"No run named '{args.exp_name}' under logs/. Available runs: {sorted(os.listdir('logs'))}")
    with open(f"{log_dir}/cfgs.pkl", "rb") as cfg_file:
        env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(cfg_file)
    reward_cfg["reward_scales"] = {}

    # a run saved before a config key existed must still be replayable
    env_cfg = fill_missing(env_cfg, default_env_cfg, "env_cfg")
    command_cfg = fill_missing(command_cfg, default_command_cfg, "command_cfg")

    # visualize the target
    env_cfg["visualize_target"] = True
    # for video recording
    env_cfg["visualize_camera"] = args.record
    # set the max FPS for visualization
    env_cfg["max_visualize_fps"] = 60
    # open the interactive viewer (training runs save render=False, so re-enable it here)
    env_cfg["render"] = not args.headless
    env_cfg["num_visualize_envs"] = 1

    # create the environment
    env = USVEnv(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
    )

    # policy runner
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=device)
    resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
    if not os.path.exists(resume_path):
        available = sorted(f for f in os.listdir(log_dir) if f.startswith("model_"))
        raise FileNotFoundError(
            f"No checkpoint at {resume_path}. Available in {log_dir}: {available}")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=device)

    obs, _ = env.reset()

    # rollout the policy
    max_sim_step = int(5 * env_cfg["episode_length_seconds"] * env_cfg["max_visualize_fps"])
    with torch.no_grad():
        if args.record:
            env.cam.start_recording()
            for _ in range(max_sim_step):
                actions = policy(obs)
                obs, _, rews, dones, infos = env.step(actions)
                env.cam.render()
            env.cam.stop_recording(save_to_filename=f"{args.exp_name}_ckpt{args.ckpt}.mp4",
                                   fps=env_cfg["max_visualize_fps"])
        else:
            for _ in range(max_sim_step):
                actions = policy(obs)
                obs, _, rews, dones, infos = env.step(actions)


if __name__ == "__main__":
    main()
