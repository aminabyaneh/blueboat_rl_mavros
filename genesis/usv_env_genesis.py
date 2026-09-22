"""
usv_env_genesis.py

BoatEnv is an instanciable class for simulating an environment with boats and obstacles using the Genesis simulation framework.

TODO:
- Randomize the boat and target positions.
- Add more complex reward functions: speed, fuel consumption, etc.
- Merge boat visualization and physics.
"""

import torch
import math
import random

import genesis as gs

from typing import Dict
from scipy.spatial.transform import Rotation as R


class USVEnv:
    """ USVEnv is a simulation environment for Unmanned Surface Vehicles (USVs) using reinforcement learning.

    Example:

    >>>    env = USVEnv(env_cfg=env_cfg,
    >>>                 obs_cfg=obs_cfg,
    >>>                 reward_cfg=reward_cfg,
    >>>                 command_cfg=command_cfg)
    >>>
    >>>    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    >>>    action = torch.tensor([1.0, 0.5], device=device)
    >>>    action = action.repeat(env_cfg["num_envs"], 1)
    >>>    obs, _, reward, reset, extras = env.step(action)

    1. Simple test script can be found in the usv_env_test.py file.
    2. Training pipeline using Nvidia's RSL_RL can be found in the usv_rslrl_train.py file.
    3. Evaluation pipeline using Nvidia's RSL_RL can be found in the usv_rslrl_eval.py file.
    """

    def __init__(self,
                 env_cfg: Dict,
                 obs_cfg: Dict,
                 command_cfg: Dict,
                 reward_cfg: Dict,
                 num_envs=None):
        """ Initialize the USV environment.

        Configuration dicts are used to configure the environment, observations, commands, and rewards.
        They can be found in the usv_env_cfg.py file.

        Args:
            env_cfg (dict): Configuration dictionary for the environment.
            obs_cfg (dict): Configuration dictionary for observations.
            command_cfg (dict): Configuration dictionary for commands.
            reward_cfg (dict): Configuration dictionary for rewards.
            num_envs (int, optional): Number of environments. Defaults to None.
        """

        # set the device to GPU
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # torch random seed
        torch.manual_seed(random.randint(0, 10000))

        # harvest environment configurations
        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg # TODO: to be used for goal conditioned setup

        self.num_envs = num_envs if num_envs is not None else self.env_cfg["num_envs"]
        self.num_obs = obs_cfg["num_obs"]
        self.num_privileged_obs = None
        self.num_actions = self.env_cfg["num_actions"]
        self.num_commands = command_cfg["num_commands"]

        # goal-conditioned mode: resample the target on every reset and expose the
        # boat-to-target vector in the observation
        # NOTE: every target key is resolved with a default, so configs pickled before
        # these keys existed (such as logs/ama_1/cfgs.pkl) still load
        self.randomize_target = command_cfg.get("randomize_target", False)
        self.default_target = command_cfg.get("target_pos", (5.0, 0.0, 0.0))
        self.target_ranges = (command_cfg.get("pos_x_range", (3.0, 5.0)),
                              command_cfg.get("pos_y_range", (-1.5, 1.5)),
                              command_cfg.get("pos_z_range", (0.0, 0.0)))

        self.dt = self.env_cfg["dt"]
        # copy: scales are multiplied by dt below, and mutating the caller's dict
        # would double-scale them if the env is rebuilt from the same config
        self.reward_scales = dict(reward_cfg["reward_scales"])
        self.max_episode_length = math.ceil(self.env_cfg["episode_length_seconds"] / self.dt)

        # initialize the physics engine
        gs.init(backend=gs.cuda if self.device == "cuda:0" else gs.cpu)

        # create scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=self.env_cfg["max_visualize_fps"],
                camera_pos=self.env_cfg["camera_pos"],
                camera_lookat=self.env_cfg["camera_lookat"],
                camera_fov=self.env_cfg["camera_fov"],
            ),
            vis_options=gs.options.VisOptions(n_rendered_envs=self.env_cfg["num_visualize_envs"]),
            show_viewer=self.env_cfg["render"],
            show_FPS=self.env_cfg["show_fps"],
        )

        # add camera
        if self.env_cfg["visualize_camera"]:
            self.cam = self.scene.add_camera(
                res=(640, 480),
                pos=(0.0, -4.0, 6),
                lookat=(0, 0, 0),
                fov=70,
                GUI=True,
            )

        # add plane
        self.scene.add_entity(gs.morphs.Plane())

        # add random obstacles, initial placement
        self.dynamic_obstacles = []

        self.num_obstacles = self.env_cfg["num_obstacles"]
        for _ in range(self.num_obstacles):
            while True:
                # random radius
                radius = self.env_cfg["obstacle_max_radius"] # random.uniform(self.env_cfg["obstacle_min_radius"], self.env_cfg["obstacle_max_radius"])

                # random positions
                pose = (random.uniform(self.env_cfg["obstacle_min_pos"][0], self.env_cfg["obstacle_max_pos"][0]),
                            random.uniform(self.env_cfg["obstacle_min_pos"][1], self.env_cfg["obstacle_max_pos"][1]),
                            self.env_cfg["obstacle_z_pos"], 0.0, 0.0, 0.0)

                # check for initial collision
                if not self._is_collision(pose[:2], radius, self.dynamic_obstacles):
                    break

            obstacle = self.scene.add_entity(gs.morphs.Cylinder(pos=pose[:3], radius=radius,
                                             height=self.env_cfg["obstacle_height"], collision=False))
            self.dynamic_obstacles.append({"obstacle": obstacle, "pose": pose, "radius": radius})

        # sea as a non-colliding static object, change to liquid in future
        self.sea = self.scene.add_entity(
            morph=gs.morphs.Box(
                pos=(0.0, 0.0, self.env_cfg["obstacle_height"] / 8),
                size=(self.env_cfg["grid_size"][0] * 2 + 4.0,
                      self.env_cfg["grid_size"][1] * 2 + 1.0,
                      self.env_cfg["obstacle_height"] / 4),
                fixed=True,
                collision=False,
            ),
            surface=gs.surfaces.Rough(
                diffuse_texture=gs.textures.ColorTexture(
                    color=(0.0, 0.5, 1.0),
                ),
            ),
        )

        # add target marker
        self.target = None
        if self.env_cfg["visualize_target"]:
            self.target = self.scene.add_entity(
                morph=gs.morphs.Mesh(
                    file="meshes/sphere.obj",
                    scale=0.2,
                    pos=self.default_target,
                    # a fixed entity has no DOFs and cannot be repositioned per environment,
                    # so the marker is only pinned when the target never moves
                    fixed=not self.randomize_target,
                    collision=False,
                ),
                surface=gs.surfaces.Rough(
                    diffuse_texture=gs.textures.ColorTexture(
                        color=self.env_cfg["target_color"],
                    ),
                ),
            )

        # add boat cover for visualization
        self.boat_cover = self.scene.add_entity(
            morph=gs.morphs.Mesh(
                file="assets/boat.stl",
                scale=self.env_cfg["boat_shell_scale"],
                collision=False,
            ),
            surface=gs.surfaces.Rough(
                diffuse_texture=gs.textures.ColorTexture(
                    color=self.env_cfg["boat_shell_color"],
                ),
            ),
        )

        # add boat core for physics
        self.boat_core_pose_init = torch.tensor(self.env_cfg["boat_core_start_pose"],
                                                device=self.device, dtype=gs.tc_float)

        self.boat_core = self.scene.add_entity(
            gs.morphs.Cylinder(pos=self.boat_core_pose_init[:3].cpu().numpy(),
                               radius=self.env_cfg["boat_core_radius"],
                               height=self.env_cfg["boat_core_height"],
                               collision=True,
                               visualization=False),
        )

        # build scene
        self.scene.build(n_envs=self.num_envs, env_spacing=self.env_cfg["env_spacing"])

        # prepare reward functions and multiply reward scales by dt
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)

        # make obstacles tensor (position, velocity, radius)
        self.obstacle_position = torch.zeros((self.num_envs, self.num_obstacles, 2),
                                             device=self.device, dtype=gs.tc_float)
        self.obstacle_pose = torch.zeros((self.num_envs, self.num_obstacles, 6),
                                             device=self.device, dtype=gs.tc_float)
        # velocity needs to be 6 to be applied
        self.obstacle_velocity = torch.zeros((self.num_envs, self.num_obstacles, 6),
                                             device=self.device, dtype=gs.tc_float)

        # fill the init position and radius
        for i, obstacle_dict in enumerate(self.dynamic_obstacles):
            self.obstacle_position[:, i] = torch.tensor(obstacle_dict["pose"][:2],
                                                        device=self.device, dtype=gs.tc_float)
            self.obstacle_pose[:, i] = torch.tensor(obstacle_dict["pose"],
                                                    device=self.device, dtype=gs.tc_float)

        # boat cover tensors
        self.boat_cover_pose = torch.tensor(self.env_cfg["boat_shell_start_pose"],
                                            device=self.device, dtype=gs.tc_float)

        self.boat_cover_pose = self.boat_cover_pose.repeat(self.num_envs, 1)
        self.boat_cover.set_dofs_position(self.boat_cover_pose)

        # rl buffers
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device, dtype=gs.tc_float)
        self.rew_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        self.reset_buf = torch.ones((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.obstacle_radius = torch.tensor(self.env_cfg["obstacle_max_radius"], device=self.device, dtype=gs.tc_float)

        # targets
        self.commands = torch.tensor(self.default_target,
                                     device=self.device, dtype=gs.tc_float).repeat(self.num_envs, 1)

        # 6-dof pose of the visual target marker, only used when the target is resampled
        self.target_pose = torch.zeros((self.num_envs, 6), device=self.device, dtype=gs.tc_float)
        self.target_pose[:, :3] = self.commands
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device, dtype=gs.tc_float)
        self.last_actions = torch.zeros_like(self.actions)

        # boat core tensors
        self.boat_core_pose = self.boat_core_pose_init.repeat(self.num_envs, 1)
        self.boat_core_velocity = torch.zeros((self.num_envs, 6), device=self.device, dtype=gs.tc_float)
        self.last_core_pose = torch.zeros_like(self.boat_core_pose)
        self.rel_boat_pos = self.commands - self.boat_core_pose[:, :3]
        self.last_rel_boat_pos = torch.zeros_like(self.rel_boat_pos)

        # leaving grid buffer
        self.leaving_grid_buf = torch.zeros_like(self.rew_buf)

        # distance to obstacles
        self.dist_to_obstacles = torch.norm(self.obstacle_position - self.boat_core_pose.unsqueeze(1).repeat(1, self.num_obstacles, 1)[:, :, :2], dim=2)

        # obs layout: boat pose (x, y, yaw) + boat velocity (vx, vy, yaw_rate)
        #             + distance to each obstacle + obstacle (x, y) + last actions
        expected_num_obs = 6 + self.num_obstacles + 2 * self.num_obstacles + self.num_actions
        if self.randomize_target:
            expected_num_obs += 2 # boat-to-target vector
        if self.num_obs != expected_num_obs:
            raise ValueError(
                f'obs_cfg["num_obs"] is {self.num_obs} but the observation layout produces '
                f'{expected_num_obs} values for num_obstacles={self.num_obstacles}, '
                f'num_actions={self.num_actions} and randomize_target={self.randomize_target}. '
                f'Update obs_cfg in usv_env_cfg.py, or load a checkpoint that was trained '
                f'with a matching layout.')

        # extra information for logging
        self.extras = dict()

        # step counter TODO: merge with episode length?
        self.step_counter = 0

    def step(self, actions):
        """ Step the simulation with the given actions.

        Args:
            actions (torch.Tensor): Actions to apply to the environment.

        Returns:
            Tuple(torch.Tensor): Observations, Previllaged observations, Rewards, Resets, Extras.
        """

        # apply actions
        # action size is: (self.env_cfg.num_envs, 1 (velocity magnitude), 1 (velocity angle))
        # action variable range is: [-1, 1]
        self.actions[:] = torch.clamp(actions, -self.env_cfg["action_clamp"], +self.env_cfg["action_clamp"])

        # decouple actions
        linear_velocity = self.actions[:, 0].unsqueeze(1) * self.env_cfg["boat_max_lin_speed"]
        angular_velocity = self.actions[:, 1].unsqueeze(1) * self.env_cfg["boat_max_ang_speed"]

        self.boat_core_pose = self.boat_core.get_dofs_position()
        current_angle = self.boat_core_pose[:, 5].unsqueeze(1)

        # form a single velocity vector
        linear_velocity_x = linear_velocity * torch.cos(current_angle)
        linear_velocity_y = linear_velocity * torch.sin(current_angle)

        self.boat_core_velocity = torch.cat((linear_velocity_x,
                                             linear_velocity_y,
                                             torch.zeros_like(linear_velocity_y),
                                             torch.zeros_like(angular_velocity),
                                             torch.zeros_like(angular_velocity),
                                             angular_velocity), dim=1)

        # check if the boat is about to leave the grid on the next step
        next_x_pos = self.boat_core_pose[:, 0] + self.boat_core_velocity[:, 0] * self.env_cfg["dt"]
        mask_x_boat = (next_x_pos > (self.env_cfg["grid_size"][0] + 1.5)) |  \
                      (next_x_pos < -(self.env_cfg["grid_size"][0] + 1.5))

        next_y_pos = self.boat_core_pose[:, 1] + self.boat_core_velocity[:, 1] * self.env_cfg["dt"]
        mask_y_boat = (next_y_pos > self.env_cfg["grid_size"][1]) | \
                      (next_y_pos < -(self.env_cfg["grid_size"][1]))

        # leaving the grid penalty
        if mask_x_boat.any() or mask_y_boat.any():
            self.leaving_grid_buf[mask_x_boat | mask_y_boat] = 1
        else:
            self.leaving_grid_buf[:] = 0

        # zero vel if hitting the boundries
        self.boat_core_velocity[mask_x_boat | mask_y_boat] = 0.0

        # apply the velocity
        self.boat_core.set_dofs_velocity(self.boat_core_velocity)

        # update dynamic objects
        if not self.env_cfg["obstacle_static"]:
            self._update_obstacle_velocities()

        # step the simulation
        self.scene.step()
        self.step_counter += 1

        # place the cover nicely
        self.boat_cover_pose[:, 0] = self.boat_core.get_dofs_position()[:, 0]
        self.boat_cover_pose[:, 1] = self.boat_core.get_dofs_position()[:, 1]
        self.boat_cover_pose[:, 5] = self.boat_core.get_dofs_position()[:, 5] - 0.08
        self.boat_cover.set_dofs_position(self.boat_cover_pose)

        # place the obstacles
        for i, obstacle_dict in enumerate(self.dynamic_obstacles):
            obstacle_dict["obstacle"].set_dofs_position(self.obstacle_pose[:, i], zero_velocity=True)

        # a resampled target is not a fixed entity, so hold it against gravity
        if self.randomize_target and self.target is not None:
            self.target.set_dofs_position(self.target_pose, zero_velocity=True)

        # update buffers
        self.episode_length_buf += 1
        self.last_core_pose[:] = self.boat_core_pose[:]
        self.boat_core_pose[:] = self.boat_core.get_dofs_position()

        self.rel_boat_pos = self.commands - self.boat_core_pose[:, :3] # only xyz matters
        self.last_rel_boat_pos = self.commands - self.last_core_pose[:, :3]

        # check if boat has collided with any obstacle
        boat_positions_repeated = self.boat_core_pose[:, :2].unsqueeze(1).repeat(1, self.num_obstacles, 1)
        self.dist_to_obstacles = torch.norm(boat_positions_repeated - self.obstacle_position, dim=2)

        # check termination and reset
        self.reset_buf = (self.episode_length_buf > self.max_episode_length)

        # build a collision mask, reset if collision!
        collision_threshold = (self.obstacle_radius + self.env_cfg["boat_core_radius"]
                               + self.env_cfg["collision_margin"])
        collision_mask = self.dist_to_obstacles < collision_threshold
        if collision_mask.any():
            self.reset_buf[collision_mask.any(dim=1)] = True

        # compute reward, according to reward_cfg
        # NOTE: this runs before reset_idx so the terminal transition of an episode is
        # scored from the state the boat actually reached, not from the reset state
        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        # last action, recorded after the reward so that _reward_smooth compares the
        # current action against the previous step's action instead of against itself
        self.last_actions[:] = self.actions[:]

        # reset the envs, this also zeroes last_actions for the environments it touches
        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).flatten())

        # collect observations, NOTE: add obstacle velocity if planning to use dynamic obstacles
        obs_terms = [
            # boat
            self.boat_core_pose[:, :2],
            self.boat_core_pose[:, 5].unsqueeze(1),
            self.boat_core_velocity[:, :2],
            self.boat_core_velocity[:, 5].unsqueeze(1),

            # obstacles
            self.dist_to_obstacles.view(self.num_envs, -1),
            self.obstacle_position.view(self.num_envs, -1),
            # self.obstacle_velocity[:, :, :2].reshape(self.num_envs, -1),

            # history
            self.last_actions,
        ]

        # the target only belongs in the observation when it moves, a fixed target is
        # a constant the policy can absorb into its weights
        if self.randomize_target:
            obs_terms.append(self.rel_boat_pos[:, :2])

        self.obs_buf = torch.cat(obs_terms, axis=-1)

        return self.obs_buf, None, self.rew_buf, self.reset_buf, self.extras # extras are empty for now

    def get_observations(self):
        """ Get the observations.
        """

        return self.obs_buf

    def get_privileged_observations(self):
        """ Get the privileged observations.

        NOTE: Not implemented, and not yet necessary.
        """

        return None

    def reset_idx(self, envs_idx):
        """ Reset the given environments.

        Args:
            envs_idx (ArrayLike): Indices of the environments to reset.
        """

        # check if there are any environments to reset
        if len(envs_idx) == 0:
            return

        # reset boat core
        self.boat_core_pose[envs_idx] = self.boat_core_pose_init
        self.last_core_pose[envs_idx] = torch.zeros_like(self.boat_core_pose_init)

        # draw a new target before the relative position below is derived from it
        self._resample_commands(envs_idx)

        self.rel_boat_pos[envs_idx] = self.commands[envs_idx] - self.boat_core_pose[envs_idx, :3]
        self.last_rel_boat_pos[envs_idx] = self.rel_boat_pos[envs_idx]
        self.boat_core.set_dofs_position(self.boat_core_pose[envs_idx], zero_velocity=True, envs_idx=envs_idx)
        self.boat_core_velocity[envs_idx] = torch.zeros_like(self.boat_core_velocity[envs_idx])

        # reset buffers
        self.last_actions[envs_idx] = torch.zeros_like(self.last_actions[envs_idx])
        self.episode_length_buf[envs_idx] = 0.0
        self.reset_buf[envs_idx] = True
        self.leaving_grid_buf[envs_idx] = 0
        self.dist_to_obstacles[envs_idx, :] = 0.0

        # randomly place obstacles
        for idx in range(self.num_obstacles):
            # random position
            pose_tensor = torch.zeros_like(self.obstacle_pose[envs_idx, idx, :])
            pose_tensor[:, 0] = self._gs_rand_float(self.env_cfg["obstacle_min_pos"][0], self.env_cfg["obstacle_max_pos"][0], (len(envs_idx),), self.device)
            pose_tensor[:, 1] = self._gs_rand_float(self.env_cfg["obstacle_min_pos"][1], self.env_cfg["obstacle_max_pos"][1], (len(envs_idx),), self.device)
            pose_tensor[:, 2] = self.env_cfg["obstacle_z_pos"]

            # obstacle pose and position
            self.obstacle_position[envs_idx, idx] = pose_tensor[:, :2]
            self.obstacle_pose[envs_idx, idx] = pose_tensor

            # apply the pose
            obstacle = self.dynamic_obstacles[idx]["obstacle"]
            obstacle.set_dofs_position(self.obstacle_pose[envs_idx, idx], zero_velocity=True, envs_idx=envs_idx)

        # fill extras (logging and tensorboard)
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item() / self.env_cfg["episode_length_seconds"]
            )
            self.episode_sums[key][envs_idx] = 0.0

    def reset(self):
        """ Reset the environment.
        """

        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        return self.obs_buf, None

    # ---------------------- resampling functions -------------------------#
    def _resample_commands(self, envs_idx):
        """ Resample the target positions for the given environments.

        Only has an effect when command_cfg["randomize_target"] is True. The sampling
        ranges come from command_cfg, and the visual marker is moved to match.

        Args:
            envs_idx (ArrayLike): Indices of the environments to resample the target.
        """

        if not self.randomize_target or len(envs_idx) == 0:
            return

        for axis, value_range in enumerate(self.target_ranges):
            self.commands[envs_idx, axis] = self._gs_rand_float(*value_range, (len(envs_idx),), self.device)

        # move the marker, it carries free DOFs in this mode so it is driven like the obstacles
        if self.target is not None:
            self.target_pose[envs_idx, :3] = self.commands[envs_idx]
            self.target.set_dofs_position(self.target_pose[envs_idx], zero_velocity=True, envs_idx=envs_idx)

    # -------------------------- reward functions ------------------------- #
    def _reward_target(self):
        target_rew = torch.sum(torch.square(self.last_rel_boat_pos), dim=1) - torch.sum(torch.square(self.rel_boat_pos), dim=1)

        return target_rew

    def _reward_smooth(self):
        smooth_rew = torch.sum(torch.square(self.actions - self.last_actions), dim=1)
        return smooth_rew

    def _reward_collision(self):
        collision_rew = (self.dist_to_obstacles.min(dim=1).values) - \
                        (self.env_cfg["boat_core_radius"] + self.obstacle_radius)

        collision_rew = torch.where(collision_rew > 0.2, -1 * torch.ones_like(collision_rew),
                                    1.0 / (collision_rew.abs() + 0.0001))
        return collision_rew

    def _reward_grid(self):
        grid_rew = self.leaving_grid_buf
        return grid_rew

    def _reward_angular(self):
        angular_rew = torch.norm(self.boat_core_velocity[:, 5].unsqueeze(1) / torch.pi, dim=1)
        return angular_rew

    # ---------------------- misc functions ---------------------------- #
    def _at_target(self):
        at_target = (
            (torch.norm(self.rel_boat_pos, dim=1) < self.env_cfg["at_target_threshold"]).nonzero(as_tuple=False).flatten()
        )
        return at_target

    def _is_collision(self, pos, radius, existing_obstacles):
        """ Check if a new obstacle collides with existing obstacles.
        """

        for obstacle_dict in existing_obstacles:
            obs_pos = obstacle_dict["pose"][:2]
            obs_radius = obstacle_dict["radius"]

            if (abs(pos[0] - obs_pos[0]) < (radius + obs_radius) and
                abs(pos[1] - obs_pos[1]) < (radius + obs_radius)):
                return True

        return False

    def _convert_rotation(self, input_rotation, input_type='quat', output_type='euler'):
        """ Convert rotation representation.

        Args:
            input_rotation (ArrayLike): Input rotation.
            input_type (str, optional): Representation type. Defaults to 'quat'.
            output_type (str, optional): Ourput rotation type. Defaults to 'euler'.

        Returns:
            ArrayLike: Output rotation.
        """
        if input_type == output_type:
            return input_rotation

        if input_type == 'quat':
            r = R.from_quat(input_rotation)
        elif input_type == 'euler':
            r = R.from_euler('zyx', input_rotation, degrees=True)

        if output_type == 'euler':
            return r.as_euler('xyz', degrees=False)
        elif output_type == 'quat':
            return r.as_quat()

        return None

    def _gs_rand_float(self, lower, upper, shape, device):
        """ Generate a random float tensor.

        Args:
            lower (float): Lower bound.
            upper (float): Upper bound.
            shape (ArrayLike): Shape of the tensor.
            device (str): Device to use.
        """

        return (upper - lower) * torch.rand(size=shape, device=device) + lower

    def _update_obstacle_velocities(self):
        """ Update the velocity of dynamic obstacles, and check for collisions with the grid.

        TODO: Make this more efficient.
        """

        # resample obstacle velocities
        # NOTE: dynamic obstacles are not used in the current setup
        resample = self.step_counter % self.env_cfg["obstacle_resample_velocity_steps"] == 0
        if resample:
            random_velocity = torch.rand(self.num_envs, self.num_obstacles, 6, device=self.device)

            random_velocity = (random_velocity - 0.5) * 2 * self.env_cfg["obstacle_base_velocity"]
            random_velocity[:, :, 2:] = 0.0

            self.obstacle_velocity = random_velocity

            for i, obstacle_dict in enumerate(self.dynamic_obstacles):
                obstacle_dict["obstacle"].set_dofs_velocity(self.obstacle_velocity[:, i, :])

            # update obstacle positions
            self.obstacle_position += self.obstacle_velocity[:, :, :2] * self.env_cfg["dt"]

        # make sure obstacles stay in the area
        mask_x_obstacles = (self.obstacle_position[:, :, 0] + self.obstacle_velocity[:, :, 0] * self.env_cfg["dt"] > self.env_cfg["obstacle_max_pos"][0]) | \
            (self.obstacle_position[:, :, 0] + self.obstacle_velocity[:, :, 0] * self.env_cfg["dt"] < self.env_cfg["obstacle_min_pos"][0])

        mask_y_obstacles = (self.obstacle_position[:, :, 1] + self.obstacle_velocity[:, :, 1] * self.env_cfg["dt"] > self.env_cfg["obstacle_max_pos"][1]) | \
            (self.obstacle_position[:, :, 1] + self.obstacle_velocity[:, :, 1] * self.env_cfg["dt"] < self.env_cfg["obstacle_min_pos"][1])

        self.obstacle_velocity[:, :, 0][mask_x_obstacles] = -self.obstacle_velocity[:, :, 0][mask_x_obstacles]
        self.obstacle_velocity[:, :, 1][mask_y_obstacles] = -self.obstacle_velocity[:, :, 1][mask_y_obstacles]

        for i, obstacle_dict in enumerate(self.dynamic_obstacles):
            obstacle_dict["obstacle"].set_dofs_velocity(self.obstacle_velocity[:, i, :])
