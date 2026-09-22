# Blue Boat RL and MavROS Guide

Part A is focused on `ROS networking` setups and access to MavROS Topics. In part B, you can directly set up a `UDP connection` to the boat, and use MavLink Python package to communicate (still testing).

Part C introduces the `Blue Boat Genesis Environment`, efficient and parallelized simulation for reinforcement learning!

Code partly adopted from [this repository](https://github.com/ImStian/blueboat_globalpathplanner/tree/main).

## Repository Layout

```
genesis/     Parallelized Genesis simulation of the BlueBoat, plus RSL-RL training and evaluation
  assets/    Boat STL mesh used for visualization
  logs/      Reference training runs and checkpoints (see "Reference runs" below)
  videos/    Recorded policy rollouts
mavlink/     Direct MAVLink (UDP) communication with the boat: link test, helper library, data logger
```

The two halves are independent. `genesis/` needs no boat; `mavlink/` needs no simulator.

## Installation

Developed and tested on Python 3.12 / Linux, which is what the reference runs were produced
on. The `mavlink/` scripts have no version-specific syntax and should work on Python 3.8+.

```bash
# option 1: exact Linux environment the reference runs were produced with
conda env create -f boat_conda_env.yaml && conda activate boat

# option 2: direct dependencies only, works on other platforms
pip install -r requirements.txt
```

Training and evaluation additionally need RSL-RL, which is pinned to a tag that is not on PyPI:

```bash
git clone https://github.com/leggedrobotics/rsl_rl
cd rsl_rl && git checkout v1.0.2 && pip install -e .
```

A CUDA GPU is strongly recommended for training. Everything falls back to CPU, but 8192
parallel environments will not be practical there.

## A. ROS Networking Setup

### **Step 0: Connect to the Blue Boat** 🌐

Before setting up the ROS network, ensure you are connected to the Blue Boat's WiFi router.

1. Connect to the WiFi Router:
    - Follow the instructions provided [on the software guide](https://bluerobotics.com/learn/blueboat-software-setup/) to connect your laptop to the Blue Boat's WiFi network.

2. Access BlueOS:
    - Make sure the Blue Boat is up and running based on the [setup guide](https://bluerobotics.com/learn/blueboat-general-integration-guide/), and ensure you can see it connected to the router based on the [software guide](https://bluerobotics.com/learn/blueboat-software-setup/).
    - Open a web browser and navigate to the BlueOS interface using the IP address provided [here](https://bluerobotics.com/learn/blueboat-software-setup/#verifying-the-network-connection).

3. Open the BlueOS ROS Interface:
    - Once in the BlueOS interface, navigate to the ROS section to access the ROS settings and tools.

> **⚠️ Important:** Ensure MavROS is up and running! You should not see any errors in the ROS panel of the BlueOS. If you do, it's most likely because the TCP port for MavROS is set wrong. Follow these steps to fix this problem:
>
> 1. Activate the pirate mode in BlueOS on the top right of the homepage.
> 2. Set the MavROS port to the same port as the TCP_USER port.
> 3. Run the MavROS command in the BlueOS ROS terminals again, using the correct TCP port.

---

### **Step 1: Find IP Addresses** 🔍

You need to determine the **IP addresses** of both the **robot** and the **laptop**.

#### **On the Robot Computer** 🤖

1. Run the following to get the IP:

    ```bash
    ip a | grep inet
    ```

    or

    ```bash
    hostname -I
    ```

2. Identify the local IP (e.g., `BLUE_OS_IP`). Avoid `127.0.0.1`, as it is a loopback address.
3. You might see two IPs for the BlueOS, pick the one you used to access BlueOS.

#### **On the Laptop** 💻

1. Run the same command:

    ```bash
    ip a | grep inet  # Linux/macOS
    ```

2. Identify the laptop's IP (e.g., `COMPUTER_IP`).

---

### **Step 2: Set Environment Variables** 🌍

Now, set the **ROS_MASTER_URI** and **ROS_IP** on both devices.

#### **On the Robot Computer** 🤖

1. Add the following commands (replace `ROBOT_IP` with the actual IP of the robot, e.g., `BLUE_OS_IP`):

    ```bash
    export ROS_MASTER_URI=http://192.168.2.2:11311
    export ROS_IP=192.168.2.2
    ```

2. To make these settings permanent, add the lines to the `~/.bashrc` file:

    ```bash
    echo "export ROS_MASTER_URI=http://BLUE_OS_IP:11311" >> ~/.bashrc
    echo "export ROS_IP=BLUE_OS_IP" >> ~/.bashrc
    source ~/.bashrc
    ```

> **⚠️ Critical:** This must be done for all terminals in the BlueBoat!! We are working on a solution to run this at boot-up on BlueOS.

#### **On the Laptop** 💻

1. Open a terminal.
2. Set the variables (replace `ROBOT_IP` and `LAPTOP_IP` with the actual values):

    ```bash
    export ROS_MASTER_URI=http://192.168.2.2:11311
    export ROS_IP=192.168.2.1
    ```

3. To make these permanent, add them to `~/.bashrc`:

    ```bash
    echo "export ROS_MASTER_URI=http://BLUE_OS_IP:11311" >> ~/.bashrc
    echo "export ROS_IP=COMPUTER_IP" >> ~/.bashrc
    source ~/.bashrc
    ```

---

### **Step 3: Test the Connection** 🔗

#### **On the Robot Computer** 🤖

* Check that `roscore` is active in BlueOS ROS terminals.

#### **On the Laptop** 💻

* Get a list of topics:

    ```bash
    rostopic list
    ```

    If everything is set up correctly, you should see the list of available `/mavros/*` topics.

---

### **Step 4: Troubleshooting (hopefully not)** 🛠️

If the connection is not working:

1. **Check IP addresses** on both computers:

    ```bash
    echo $ROS_IP
    echo $ROS_MASTER_URI
    ```

2. **Ping the robot from the laptop**:

    ```bash
    ping BLUE_OS_IP
    ```

    If there’s no response, ensure both are on the same network.
3. **Disable firewalls** temporarily to test:

    ```bash
    sudo ufw disable  # Linux
    ```

---

### MavROS Documentation 📚

For detailed information and advanced usage of MavROS, refer to the official [MavROS documentation](https://wiki.ros.org/mavros).

MavROS is a ROS package that provides communication between ROS and MAVLink-based autopilots. It includes various plugins that allow you to interact with different MAVLink messages and services.

> 📝 **Note**: By following the MavROS documentation, you can leverage the full capabilities of your MAVLink-compatible Blue Boat within the ROS ecosystem.

### Designated Topics (Not needed when using MavLink Python package!)

These are the topics we found useful during the initial testing phase. Feel free to add to the list.

| Topic                       | Message Type             | Description                                      |
|-----------------------------|--------------------------|--------------------------------------------------|
| /mavros/state               | mavros_msgs/State        | Current state of the MAVLink device              |
| /mavros/imu/data            | sensor_msgs/Imu          | IMU data including orientation and angular velocity|
| /mavros/local_position/pose | geometry_msgs/PoseStamped| Local position of the MAVLink device             |
| /mavros/rc/out              | mavros_msgs/RCOut        | RC output values                                 |
| /mavros/battery             | sensor_msgs/BatteryState | Battery status of the MAVLink device             |

### Designated Services

These are the services we found useful during the initial testing phase. Feel free to add to the list.

| Service                  | Service Type               | Description                                      |
|--------------------------|----------------------------|--------------------------------------------------|
| /mavros/cmd/arming       | mavros_msgs/CommandBool    | Arm or disarm the MAVLink device                 |
| /mavros/set_mode         | mavros_msgs/SetMode        | Set the flight mode of the MAVLink device        |
| /mavros/param/get        | mavros_msgs/ParamGet       | Retrieve a parameter from the MAVLink device     |
| /mavros/param/set        | mavros_msgs/ParamSet       | Set a parameter on the MAVLink device            |
| /mavros/command/takeoff  | mavros_msgs/CommandTOL     | Command the MAVLink device to take off           |
| /mavros/command/land     | mavros_msgs/CommandTOL     | Command the MAVLink device to land               |


## B. MAVLink Communication Module

The Python MAVLink package creates a reliable UDP link with the boat for planning missions,
and basically all functionalities of QGroundControl. Run everything below from the
`mavlink/` folder, with your host machine on the same base-station network as the boat.

Use the BlueOS webpage, typically at `192.168.2.2` on the local network, to check the
MAVLink server address and ports.

> **⚠️ Note:** We successfully tested this using the **GCS_Client_Link** endpoint, which was
> `192.168.2.2:14550` at the time. It can be found in the **MavLink Endpoints** menu, in
> **pirate mode**. The scripts listen for it by binding locally to `udpin:192.168.2.1:14550`.
> If your addresses differ, pass `--connection` instead of editing the scripts.

### Link Test

Confirm the link before anything else:

```bash
python mavlink_udp_heartbeat.py
python mavlink_udp_heartbeat.py --connection udpin:192.168.2.1:14550   # explicit endpoint
```

You should see a heartbeat line followed by a stream of `GLOBAL_POSITION_INT` messages.

### Data Logging

[`mavlink_data_logger.py`](mavlink/mavlink_data_logger.py) records GPS and IMU messages and
writes them to `mavlink/logs/<folder_name>/` as timestamped CSV files on exit:

```bash
python mavlink_data_logger.py my_first_run
```

Press `Ctrl+C` to stop. The CSV files are written even if the script exits with an error.

### MAVLink Python Library

[`mavlink_library.py`](mavlink/mavlink_library.py) holds the reusable helpers: arming,
mode setting, home position, and full mission upload/start/monitor. Import from it to build
your own scripts, the way `mavlink_data_logger.py` does.

## C. Genesis Environment

The `genesis/` folder contains the Blue Boat Genesis Environment, designed for efficient and
parallelized simulation for reinforcement learning.

1. **Environment Setup**: The environment is built using Genesis components and the blue boat STL file.
2. **Parallelization**: The simulation supports parallel execution, allowing thousands of instances to run simultaneously for faster training.
3. **Customization**: All knobs live in [`usv_env_cfg.py`](genesis/usv_env_cfg.py), making the environment versatile for various training needs.

The task: drive the boat from its start pose to a fixed target at `(5, 0)` without hitting
any of the randomly placed cylindrical obstacles. Actions are two-dimensional, a forward
speed and a turn rate, both in `[-1, 1]` and scaled by the limits in the config.

Run all commands below from the `genesis/` folder.

### Simulation in Action 🎥

Here's a quick preview of the Blue Boat Genesis Environment in action:

![Blue Boat Simulation](genesis/videos/obstacle_avoidance_rollout.gif)

### Try It Out First

```bash
python usv_env_test.py      # smoke test, random actions, confirms Genesis works
python usv_interactive.py   # drive the boat yourself with the arrow keys
```

### Training with RSL-RL

```bash
python usv_rslrl_train.py --exp_name my_run --num_envs 8192 --max_iterations 500
```

Checkpoints and TensorBoard events are written to `genesis/logs/<exp_name>/`. The script
refuses to start if that directory already exists, so a previous run is never silently
deleted. Pass `--overwrite` when you do want to replace it, or use a new `--exp_name`.

| Flag | Default | Meaning |
|------|---------|---------|
| `--exp_name` | `usv_env` | Name of the run directory under `logs/` |
| `--num_envs` | `64` | Parallel environments. Use several thousand on a GPU |
| `--max_iterations` | `500` | PPO iterations |
| `--render` | off | Open the viewer during training (slow, for debugging only) |
| `--resume_path` | none | Checkpoint `.pt` to warm-start from |
| `--overwrite` | off | Delete an existing `logs/<exp_name>` instead of refusing |

Monitor progress with TensorBoard, from the repository root:

```bash
tensorboard --logdir genesis/logs/
```

### Evaluating a Policy

```bash
python usv_rslrl_eval.py --exp_name ama_1 --ckpt 1900            # watch it in the viewer
python usv_rslrl_eval.py --exp_name ama_1 --ckpt 1900 --record   # also save an mp4
```

`--record` writes `<exp_name>_ckpt<N>.mp4` into `genesis/`. Add `--headless` to skip the
interactive viewer, which is what you want over SSH.

### Reference Runs

One trained run is committed under `genesis/logs/ama_1/`, with the final checkpoint and one
mid-training checkpoint so you can compare them:

| Checkpoint | Iterations | Notes |
|------------|-----------|-------|
| `model_900` | 900 | partially trained, still clips obstacles |
| `model_1900` | 1900 | final, the one the demo gif was recorded from |

```bash
python usv_rslrl_eval.py --exp_name ama_1 --ckpt 1900
```

The full TensorBoard event file for the run is kept alongside them, so the complete training
curve is still available even though the intermediate checkpoints were dropped.

> **Note:** `ama_1` predates two reward fixes. Rewards used to be computed *after* the
> environments were reset, so the final transition of each episode was scored from the
> already-reset state, and `_reward_smooth` evaluated to exactly zero on every step because
> the previous action was overwritten before it was read. Both are fixed now. The checkpoint
> still loads and runs, since neither change affects the observation layout, but a fresh run
> will train differently.
>
> Its `cfgs.pkl` also predates several configuration keys. `usv_rslrl_eval.py` backfills any
> key a run was saved without, taking the value from `usv_env_cfg.py` and printing which keys
> it filled, so older runs stay replayable as the config grows.

### Goal-Conditioned Training

By default the target sits at a fixed `(5, 0)` and the policy learns that one destination.
To train a policy that generalizes to arbitrary targets, set the following in
[`usv_env_cfg.py`](genesis/usv_env_cfg.py):

```python
command_cfg["randomize_target"] = True   # resample the target on every reset
obs_cfg["num_obs"] = 25                  # 23 + the boat-to-target vector
```

The relevant keys in `command_cfg` are:

| Key | Default | Meaning |
|-----|---------|---------|
| `randomize_target` | `False` | Resample the target on every reset |
| `target_pos` | `(5.0, 0.0, 0.0)` | Where the target sits when randomization is off |
| `pos_x_range` | `(3.0, 5.0)` | Sampling range along x |
| `pos_y_range` | `(-1.5, 1.5)` | Sampling range along y |
| `pos_z_range` | `(0.0, 0.0)` | Sampling range along z, the boat is a surface vehicle |

With this enabled the environment resamples the target from `pos_x_range` / `pos_y_range`
on every reset, moves the visual marker to match, and appends the boat-to-target vector
`(dx, dy)` to the observation. That last part is what makes the task learnable: with a
moving target and no target in the observation, the policy has no way to know where to go.

The committed `ama_1` checkpoint expects 23 observations and will not load in this mode.
The environment reports the mismatch instead of failing silently, so train a fresh run:

```bash
python usv_rslrl_train.py --exp_name goal_conditioned --num_envs 8192 --max_iterations 2000
```

### Configuration Notes

`obs_cfg["num_obs"]` in [`usv_env_cfg.py`](genesis/usv_env_cfg.py) must match the
observation layout, which is:

```
6                      boat pose (x, y, yaw) and velocity (vx, vy, yaw_rate)
+ num_obstacles        distance to each obstacle
+ 2 * num_obstacles    each obstacle's (x, y)
+ num_actions          previous action
+ 2                    boat-to-target vector, only when randomize_target is True
```

With the defaults that is `6 + 5 + 10 + 2 = 23`, or 25 in goal-conditioned mode. The
environment checks this at construction and raises a descriptive error on a mismatch, so if
you change `num_obstacles` you must update `num_obs` to match.

Only reward terms listed in `reward_cfg["reward_scales"]` are active. `_reward_collision`,
`_reward_grid`, and `_reward_angular` are implemented in the environment but switched off by
default. Add them to `reward_scales` to enable them.

## Known Limitations

Worth knowing before you build on this:

- **The boat start pose is fixed.** The target can be randomized, but the boat always starts
  from the same pose. Randomizing it too would make the task considerably harder.
- **Dynamic obstacles are untested.** Set `obstacle_static: False` to enable them, but the
  observation vector does not include obstacle velocities, so a policy cannot anticipate them.
- **Reaching the target does not end the episode.** `_at_target()` is implemented but unused,
  so episodes always run to the time limit or to a collision. The reward is displacement
  based, which works, but there is no terminal bonus for arriving.
- **Physics are kinematic.** Velocities are set directly on the boat body. There is no
  hydrodynamic drag, no thruster model, and the "sea" is a static non-colliding box, so the
  sim-to-real gap to the actual BlueBoat is large.
- **No automated tests.** A smoke test that builds the environment and checks the observation
  shape would catch configuration mismatches before a training run starts.
