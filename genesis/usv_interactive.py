import torch

from pynput import keyboard

from usv_env_genesis import USVEnv
from usv_env_cfg import env_cfg, obs_cfg, command_cfg, reward_cfg

device = "cuda:0" if torch.cuda.is_available() else "cpu"

env_cfg["num_envs"] = 1
env_cfg["render"] = True
env_cfg["num_visualize_envs"] = 1

# simulation time
# NOTE: this must be set before the env is built, the env derives its episode
# length from it at construction time
total_sim_time = 60 # seconds
env_cfg["episode_length_seconds"] = total_sim_time
total_sim_steps = int(total_sim_time / env_cfg["dt"])

# create the USV environment
env = USVEnv(
    env_cfg=env_cfg,
    obs_cfg=obs_cfg,
    reward_cfg=reward_cfg,
    command_cfg=command_cfg,
)

# define initial velocities
linear_velocity = 0
angular_velocity = 0

linear_increment = env_cfg["boat_max_lin_speed"] / 10
angular_increment = env_cfg["boat_max_ang_speed"] / 10

# keyboard handle
def on_press(key):
    global linear_velocity, angular_velocity

    if key == keyboard.Key.up:
        linear_velocity += linear_increment
    if key == keyboard.Key.down:
        linear_velocity -= linear_increment

    if key == keyboard.Key.left:
        angular_velocity += angular_increment
    if key == keyboard.Key.right:
        angular_velocity -= angular_increment

    # clamp the velocities
    linear_velocity = max(-env_cfg["boat_max_lin_speed"], min(env_cfg["boat_max_lin_speed"], linear_velocity))
    angular_velocity = max(-env_cfg["boat_max_ang_speed"], min(env_cfg["boat_max_ang_speed"], angular_velocity))

def on_release(key):
    global linear_velocity, angular_velocity

    if key == keyboard.Key.up:
        linear_velocity = 0
    if key == keyboard.Key.down:
        linear_velocity = 0

    if key == keyboard.Key.left:
        angular_velocity = 0
    if key == keyboard.Key.right:
        angular_velocity = 0

# keyboard initiate
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

print("Controls: arrow up/down = forward/reverse, arrow left/right = turn. Ctrl+C to quit.")

# simulation main loop
for simulation_step in range(total_sim_steps):
    # create the action tensor
    action = torch.tensor([[linear_velocity, angular_velocity]], device=device)

    # replicate the action for all environments (not necessary for single environment)
    actions = action.repeat(env_cfg["num_envs"], 1)

    # step the simulation
    obs, _, reward, reset, extras = env.step(actions)

    # show the reward
    print(f"Step: {simulation_step}, Reward: {reward.item():.4f}")