# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from matplotlib.pyplot import axis
import numpy as np
from PIL import Image as Im

import os
import random
import torch

from agents.utils.torch_jit_utils import *
from agents.tasks.agent_base.base_task import BaseTask
from isaacgym import gymtorch
from isaacgym import gymapi


class MultiAntCircle(BaseTask):

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless=True, is_multi_agent=False):

        self.cfg = cfg
        self.sim_params = sim_params
        self.physics_engine = physics_engine

        self.is_multi_agent = is_multi_agent

        self.max_episode_length = self.cfg["env"]["episodeLength"]

        self.randomization_params = self.cfg["task"]["randomization_params"]
        self.randomize = self.cfg["task"]["randomize"]
        self.dof_vel_scale = self.cfg["env"]["dofVelocityScale"]
        self.contact_force_scale = self.cfg["env"]["contactForceScale"]
        self.power_scale = self.cfg["env"]["powerScale"]
        self.heading_weight = self.cfg["env"]["headingWeight"]
        self.up_weight = self.cfg["env"]["upWeight"]
        self.actions_cost_scale = self.cfg["env"]["actionsCost"]
        self.energy_cost_scale = self.cfg["env"]["energyCost"]
        self.joints_at_limit_cost_scale = self.cfg["env"]["jointsAtLimitCost"]
        self.death_cost = self.cfg["env"]["deathCost"]
        self.termination_height = self.cfg["env"]["terminationHeight"]

        self.debug_viz = self.cfg["env"]["enableDebugVis"]
        self.plane_static_friction = self.cfg["env"]["plane"]["staticFriction"]
        self.plane_dynamic_friction = self.cfg["env"]["plane"]["dynamicFriction"]
        self.plane_restitution = self.cfg["env"]["plane"]["restitution"]

        self.cfg["env"]["numObservations"] = 38

        self.draw_penalty_scale = -1
        self.move_reward_scale = 1.
        self.quat_reward_scale = 1.
        self.ant_dist_reward_scale = 500.
        self.goal_dist_reward_scale = 500.

        if self.is_multi_agent:
            self.num_agents = 2
            self.cfg["env"]["numActions"] = 8

        else:
            self.num_agents = 1
            self.cfg["env"]["numActions"] = 16

        self.cfg["device_type"] = device_type
        self.cfg["device_id"] = device_id
        self.cfg["headless"] = headless

        super().__init__(cfg=self.cfg)

        if self.viewer is not None:
            for env in self.envs:
                self._add_square_borderline(env)
            cam_pos = gymapi.Vec3(18.0, 0.0, 5.0)
            cam_target = gymapi.Vec3(10.0, 0.0, 0.0)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)

        print('sensor_tensor:', sensor_tensor.shape)

        sensors_per_env = 4
        self.vec_sensor_tensor = gymtorch.wrap_tensor(sensor_tensor).view(self.num_envs * self.num_agents,
                                                                          sensors_per_env * 6)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)

        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        # print('self.root_state_tensor:', self.root_states[:4, :])
        print('self.root_states:', self.root_states.shape)
        self.initial_root_states = self.root_states.clone()
        self.initial_root_states[:, 7:13] = 0  # set lin_vel and ang_vel to 0
        # print('self.initial_root_states:', self.initial_root_states[:4, :])

        # create some wrapper tensors for different slices
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        print('dof_state_shape:', self.dof_state.shape)
        self.num_dofs = self.gym.get_sim_dof_count(self.sim)
        self.dof_pos_1 = self.dof_state.view(self.num_envs, -1, 2)[:, :self.num_dof, 0]
        self.dof_pos_2 = self.dof_state.view(self.num_envs, -1, 2)[:, self.num_dof: 2 * self.num_dof, 0]

        self.dof_vel_1 = self.dof_state.view(self.num_envs, -1, 2)[:, :self.num_dof, 1]
        self.dof_vel_2 = self.dof_state.view(self.num_envs, -1, 2)[:, self.num_dof: 2 * self.num_dof, 1]

        self.initial_dof_pos = torch.zeros_like(self.dof_pos_1, device=self.device, dtype=torch.float)
        zero_tensor = torch.tensor([0.0], device=self.device)
        self.initial_dof_pos = torch.where(self.dof_limits_lower > zero_tensor, self.dof_limits_lower,
                                           torch.where(self.dof_limits_upper < zero_tensor, self.dof_limits_upper,
                                                       self.initial_dof_pos))
        self.initial_dof_vel = torch.zeros_like(self.dof_vel_1, device=self.device, dtype=torch.float)
        self.dt = self.cfg["sim"]["dt"]

        # torques = self.gym.acquire_dof_force_tensor(self.sim)
        # self.torques = gymtorch.wrap_tensor(torques).view(self.num_envs, 2 * self.num_dof)

        self.x_unit_tensor = to_torch([1, 0, 0], dtype=torch.float, device=self.device).repeat(
            (self.num_agents * self.num_envs, 1))
        self.y_unit_tensor = to_torch([0, 1, 0], dtype=torch.float, device=self.device).repeat(
            (self.num_agents * self.num_envs, 1))
        self.z_unit_tensor = to_torch([0, 0, 1], dtype=torch.float, device=self.device).repeat(
            (self.num_agents * self.num_envs, 1))

        self.obs_buf_1 = torch.zeros((self.num_envs, 38), device=self.device, dtype=torch.float)
        self.obs_buf_2 = torch.zeros((self.num_envs, 38), device=self.device, dtype=torch.float)

        # initialize some data used later on
        self.up_vec = to_torch(get_axis_params(1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.heading_vec = to_torch([1, 0, 0], device=self.device).repeat((self.num_envs, 1))
        self.inv_start_rot = quat_conjugate(self.start_rotation_1).repeat((self.num_envs, 1))
        self.basis_vec0 = self.heading_vec.clone()
        self.basis_vec1 = self.up_vec.clone()
        self.targets = to_torch([0, 0, 0], device=self.device).repeat((self.num_envs, 1))


    def create_sim(self):
        self.up_axis_idx = self.set_sim_params_up_axis(self.sim_params, 'z')
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        lines = []
        borderline_height = 0.01
        for height in range(20):
            for angle in range(360):
                begin_point = [np.cos(np.radians(angle)), np.sin(np.radians(angle)), borderline_height * height]
                end_point = [np.cos(np.radians(angle + 1)), np.sin(np.radians(angle + 1)), borderline_height * height]
                lines.append(begin_point)
                lines.append(end_point)
        self.lines = np.array(lines, dtype=np.float32) * 3
        self._create_ground_plane()
        print(f'num envs {self.num_envs} env spacing {self.cfg["env"]["envSpacing"]}')
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

        # If randomizing, apply once immediately on startup before the fist sim step
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

    def _add_square_borderline(self, env):
        colors = np.array([[1, 0, 0]] * int(len(self.lines) / 2), dtype=np.float32)
        self.gym.add_lines(self.viewer, env, int(len(self.lines) / 2), self.lines, colors)

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.plane_static_friction
        plane_params.dynamic_friction = self.plane_dynamic_friction
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../assets')
        asset_file = "mjcf/open_ai_assets/hand/nv_ant.xml"

        if "asset" in self.cfg["env"]:
            asset_file = self.cfg["env"]["asset"].get("assetFileName", asset_file)

        asset_path = os.path.join(asset_root, asset_file)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        # Note - DOF mode is set in the MJCF file and loaded by Isaac Gym
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        asset_options.angular_damping = 0.0

        ant_asset_1 = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        ant_asset_2 = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        self.num_dof = self.gym.get_asset_dof_count(ant_asset_1)
        self.num_dof_1 = self.gym.get_asset_dof_count(ant_asset_1)
        # print('self.num_dof:',self.num_dof)

        # force
        actuator_props_1 = self.gym.get_asset_actuator_properties(ant_asset_1)
        motor_efforts_1 = [prop.motor_effort for prop in actuator_props_1]
        self.joint_gears_1 = to_torch(motor_efforts_1, device=self.device)

        actuator_props_2 = self.gym.get_asset_actuator_properties(ant_asset_2)
        motor_efforts_2 = [prop.motor_effort for prop in actuator_props_2]
        self.joint_gears_2 = to_torch(motor_efforts_2, device=self.device)

        self.joint_gears = torch.cat((self.joint_gears_1, self.joint_gears_2), dim=-1)

        start_pose_1 = gymapi.Transform()
        start_pose_1.p = gymapi.Vec3(3, 0, 1.)
        start_pose_2 = gymapi.Transform()
        start_pose_2.p = gymapi.Vec3(-3, 0, 1.)


        self.start_rotation_1 = torch.tensor([start_pose_1.r.x, start_pose_1.r.y, start_pose_1.r.z, start_pose_1.r.w],
                                             device=self.device)
        self.torso_index = 0

        self.num_bodies_1 = self.gym.get_asset_rigid_body_count(ant_asset_1)
        body_names_1 = [self.gym.get_asset_rigid_body_name(ant_asset_1, i) for i in range(self.num_bodies_1)]
        extremity_names_1 = [s for s in body_names_1 if "foot" in s]
        self.extremities_index_1 = torch.zeros(len(extremity_names_1), dtype=torch.long, device=self.device)

        self.num_bodies_2 = self.gym.get_asset_rigid_body_count(ant_asset_2)
        body_names_2 = [self.gym.get_asset_rigid_body_name(ant_asset_2, i) for i in range(self.num_bodies_2)]
        extremity_names_2 = [s for s in body_names_2 if "foot" in s]
        self.extremities_index_2 = torch.zeros(len(extremity_names_2), dtype=torch.long, device=self.device)

        # create force sensors attached to the "feet"
        extremity_indices_1 = [self.gym.find_asset_rigid_body_index(ant_asset_1, name) for name in extremity_names_1]

        sensor_pose_1 = gymapi.Transform()
        sensor_pose_2 = gymapi.Transform()

        for body_idx in extremity_indices_1:
            self.gym.create_asset_force_sensor(ant_asset_1, body_idx, sensor_pose_1)
            self.gym.create_asset_force_sensor(ant_asset_2, body_idx, sensor_pose_2)

        self.ant_handles_1 = []
        self.ant_indices_1 = []
        self.ant_handles_2 = []
        self.ant_indices_2 = []

        self.envs = []
        self.pos_before_1 = torch.zeros(2, device=self.device)
        self.pos_before_2 = torch.zeros(2, device=self.device)

        self.dof_limits_lower = []
        self.dof_limits_upper = []

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )
            ant_handle_1 = self.gym.create_actor(env_ptr, ant_asset_1, start_pose_1, "ant_1", i, 1, 0)
            ant_index_1 = self.gym.get_actor_index(env_ptr, ant_handle_1, gymapi.DOMAIN_SIM)
            self.ant_indices_1.append(ant_index_1)

            ant_handle_2 = self.gym.create_actor(env_ptr, ant_asset_2, start_pose_2, "ant_2", i, 1, 0)
            ant_index_2 = self.gym.get_actor_index(env_ptr, ant_handle_2, gymapi.DOMAIN_SIM)
            self.ant_indices_2.append(ant_index_2)

            for j in range(self.num_bodies_1):
                self.gym.set_rigid_body_color(
                    env_ptr, ant_handle_1, j, gymapi.MESH_VISUAL, gymapi.Vec3(0.97, 0.38, 0.06))
                self.gym.set_rigid_body_color(
                    env_ptr, ant_handle_2, j, gymapi.MESH_VISUAL, gymapi.Vec3(0.24, 0.38, 0.06))

            self.envs.append(env_ptr)
            self.ant_handles_1.append(ant_handle_1)
            self.ant_handles_2.append(ant_handle_2)

        dof_prop = self.gym.get_actor_dof_properties(env_ptr, ant_handle_1)

        for j in range(self.num_dof_1):
            if dof_prop['lower'][j] > dof_prop['upper'][j]:
                self.dof_limits_lower.append(dof_prop['upper'][j])
                self.dof_limits_upper.append(dof_prop['lower'][j])
            else:
                self.dof_limits_lower.append(dof_prop['lower'][j])
                self.dof_limits_upper.append(dof_prop['upper'][j])

        self.dof_limits_lower = to_torch(self.dof_limits_lower, device=self.device)
        self.dof_limits_upper = to_torch(self.dof_limits_upper, device=self.device)

        for i in range(len(extremity_names_1)):
            self.extremities_index_1[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.ant_handles_1[0],
                                                                                extremity_names_1[i])
            self.extremities_index_2[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.ant_handles_2[0],
                                                                                extremity_names_2[i])

        self.ant_indices_1 = to_torch(self.ant_indices_1, dtype=torch.long, device=self.device)
        self.ant_indices_2 = to_torch(self.ant_indices_2, dtype=torch.long, device=self.device)

    def compute_reward(self, actions):
        self.rew_buf[:], self.reset_buf[:] = compute_ant_reward(
            self.obs_buf_1,
            self.obs_buf_2,
            self.reset_buf,
            self.progress_buf,
            self.actions,
            self.up_weight,
            self.heading_weight,
            self.actions_cost_scale,
            self.energy_cost_scale,
            self.joints_at_limit_cost_scale,
            self.termination_height,
            self.death_cost,
            self.max_episode_length,
            self.pos_before_1,
            self.pos_before_2,
            self.dt,
            self.move_reward_scale,
            self.quat_reward_scale,
            self.ant_dist_reward_scale
        )

    def compute_observations(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)
        # print("Feet forces and torques: ", self.vec_sensor_tensor[0, :])
        # print(self.vec_sensor_tensor.shape)

        self.obs_buf_1[:] = compute_ant_observations(
            self.obs_buf_1, self.root_states[0::(self.num_agents), :],self.targets,
            self.inv_start_rot, self.dof_pos_1, self.dof_vel_1,
            self.dof_limits_lower, self.dof_limits_upper, self.dof_vel_scale,
            self.actions[:, :8], self.dt, self.contact_force_scale,
            self.basis_vec0, self.basis_vec1, self.up_axis_idx)

        self.obs_buf_2[:] = compute_ant_observations(
            self.obs_buf_2, self.root_states[1::(self.num_agents), :],self.targets,
            self.inv_start_rot, self.dof_pos_2, self.dof_vel_2,
            self.dof_limits_lower, self.dof_limits_upper, self.dof_vel_scale,
            self.actions[:, 8:16], self.dt, self.contact_force_scale,
            self.basis_vec0, self.basis_vec1, self.up_axis_idx)

        self.obs_buf = torch.cat((self.obs_buf_1, self.obs_buf_2), dim=-1)

    def reset_idx(self, env_ids):
        # Randomization can happen only at reset time, since it can reset actor positions on GPU
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        ant_indices = torch.unique(torch.cat([self.ant_indices_1[env_ids], self.ant_indices_2[env_ids]]).to(torch.int32))

        positions = torch_rand_float(-0.2, 0.2, (len(env_ids), self.num_dof_1), device=self.device)
        velocities = torch_rand_float(-0.1, 0.1, (len(env_ids), self.num_dof_1), device=self.device)

        self.dof_pos_1[env_ids] = tensor_clamp(self.initial_dof_pos[env_ids] + positions, self.dof_limits_lower,
                                               self.dof_limits_upper)
        self.dof_vel_1[env_ids] = velocities
        self.dof_pos_2[env_ids] = tensor_clamp(self.initial_dof_pos[env_ids] + positions, self.dof_limits_lower,
                                               self.dof_limits_upper)
        self.dof_vel_2[env_ids] = velocities

        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.initial_root_states),
                                                     gymtorch.unwrap_tensor(ant_indices), len(ant_indices))

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(ant_indices), len(ant_indices))

        self.pos_before_1 = self.root_states[0::2, :2].clone()
        self.pos_before_2 = self.root_states[1::2, :2].clone()

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0

    def pre_physics_step(self, actions):
        self.actions = actions.clone().to(self.device)
        forces = self.actions * self.joint_gears * self.power_scale
        force_tensor = gymtorch.unwrap_tensor(forces)
        self.gym.set_dof_actuation_force_tensor(self.sim, force_tensor)

    def post_physics_step(self):
        self.progress_buf += 1
        self.randomize_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward(self.actions)

        self.pos_before_1 = self.obs_buf_1[:self.num_envs, :2].clone()
        self.pos_before_2 = self.obs_buf_2[:self.num_envs, :2].clone()


#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def compute_angle(
        pos
):
    # type: (Tensor) -> [Tensor]
    a = pos[:,0]
    b = pos[:,1]
    c = b < 0
    d = b >= 0
    e = c*360
    f = -c
    g = f + d
    angle = e + g*np.abs(np.arctan2(b,a)*180/np.pi)

    return angle

@torch.jit.script
def compute_ant_reward(
        obs_buf_1,
        obs_buf_2,
        reset_buf,
        progress_buf,
        actions,
        up_weight,
        heading_weight,
        actions_cost_scale,
        energy_cost_scale,
        joints_at_limit_cost_scale,
        termination_height,
        death_cost,
        max_episode_length,
        pos_before_1,
        pos_before_2,
        dt
):
    # type: (Tensor, Tensor,Tensor, Tensor, Tensor, float, float, float, float, float, float, float, float, Tensor, Tensor, float) -> Tuple[Tensor, Tensor]

    #circle
    pos_1 = obs_buf_1[:,:2]
    dist_1 = np.linalg.norm(pos_1)
    angle_1 = compute_angle(pos_1)
    angle_1_before = compute_angle(pos_before_1)
    clockwise_1 = (angle_1 - angle_1_before) > 0
    is_oncircle_1 = (dist_1>=2.7).mul(dist_1<=3.3)
    rew_1 = (clockwise_1.mul(is_oncircle_1))*2 + ((clockwise_1.mul(is_oncircle_1)) - 1)

    pos_2 = -obs_buf_2[:, :2]
    dist_2 = np.linalg.norm(pos_2)
    angle_2 = compute_angle(pos_2)
    angle_2_before = compute_angle(pos_before_2)
    clockwise_2 = (angle_2 - angle_2_before) > 0
    is_oncircle_2 = (dist_2 >= 2.7).mul(dist_2 <= 3.3)
    rew_2 = (clockwise_2.mul(is_oncircle_2)) * 2 + ((clockwise_2.mul(is_oncircle_2)) - 1)

    rew = rew_1 + rew_2

    # #another_circle
    # vel_1 = (pos_1 - pos_before_1)/dt
    # dist_1 = np.linalg.norm(pos_1)
    # vel_orthogonal_1 = np.array([-vel_1[:,1],vel_1[:,0]])
    # rew_1 = 0.1 * np.dot(pos_1, vel_orthogonal_1) / (1 + np.abs(dist_1 - 3))
    #
    # vel_2 = (pos_2 - pos_before_2) / dt
    # dist_2 = np.linalg.norm(pos_2)
    # vel_orthogonal_2 = np.array([-vel_2[:, 1], vel_2[:, 0]])
    # rew_2 = 0.1 * np.dot(pos_2, vel_orthogonal_2) / (1 + np.abs(dist_2 - 3))
    #
    # rew = rew_1 + rew_2


    # up
    heading_weight_tensor_1 = torch.ones_like(obs_buf_1[:, 13]) * heading_weight
    heading_reward_1 = torch.where(obs_buf_1[:, 13] > 0.8, heading_weight_tensor_1,
                                   heading_weight * obs_buf_1[:, 13] / 0.8)
    up_reward_1 = torch.zeros_like(heading_reward_1)
    up_reward_1 = torch.where(obs_buf_1[:, 12] > 0.93, up_reward_1 + up_weight, up_reward_1)
    heading_weight_tensor_2 = torch.ones_like(obs_buf_2[:, 13]) * heading_weight
    heading_reward_2 = torch.where(obs_buf_2[:, 13] > 0.8, heading_weight_tensor_2,
                                   heading_weight * obs_buf_2[:, 13] / 0.8)
    up_reward_2 = torch.zeros_like(heading_reward_2)
    up_reward_2 = torch.where(obs_buf_2[:, 12] > 0.93, up_reward_2 + up_weight, up_reward_2)
    up_reward = up_reward_1 + up_reward_2

    # energy penalty for movement
    actions_cost_1 = torch.sum(actions[:, 0:8] ** 2, dim=-1)
    actions_cost_2 = torch.sum(actions[:, 8:16] ** 2, dim=-1)
    actions_cost = actions_cost_1 + actions_cost_2

    electricity_cost_1 = torch.sum(torch.abs(actions[:, 0:8] * obs_buf_1[:, 22:30]), dim=-1)
    dof_at_limit_cost_1 = torch.sum(obs_buf_1[:, 14:22] > 0.99, dim=-1)
    electricity_cost_2 = torch.sum(torch.abs(actions[:, 8:16] * obs_buf_2[:, 22:30]), dim=-1)
    dof_at_limit_cost_2 = torch.sum(obs_buf_2[:, 14:22] > 0.99, dim=-1)
    electricity_cost = electricity_cost_1 + electricity_cost_2
    dof_at_limit_cost = dof_at_limit_cost_1 + dof_at_limit_cost_2

    total_reward = up_reward + rew - \
                   actions_cost_scale * actions_cost - energy_cost_scale * electricity_cost - dof_at_limit_cost * joints_at_limit_cost_scale
    # print('**total_reward:',total_reward.shape)
    # print('**total_reward:',total_reward[:4])

    # adjust reward for fallen agents
    fallen = (obs_buf_1[:, 2] < termination_height) + (obs_buf_2[:, 2] < termination_height)
    total_reward = torch.where(fallen, torch.ones_like(total_reward) * death_cost,
                               total_reward)

    # reset agents
    reset = torch.where(fallen, torch.ones_like(reset_buf), reset_buf)
    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset)

    return total_reward, reset


@torch.jit.script
def compute_ant_observations(obs_buf, root_states,targets,
                             inv_start_rot, dof_pos, dof_vel,
                             dof_limits_lower, dof_limits_upper, dof_vel_scale,
                             actions, dt, contact_force_scale,
                             basis_vec0, basis_vec1, up_axis_idx):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, Tensor, float, float, Tensor, Tensor, int) -> Tensor

    torso_position = root_states[:, 0:3]
    torso_rotation = root_states[:, 3:7]
    velocity = root_states[:, 7:10]
    ang_velocity = root_states[:, 10:13]

    to_target = targets - torso_position
    to_target[:, 2] = 0.0

    torso_quat, up_proj, heading_proj, up_vec, heading_vec = compute_heading_and_up(
        torso_rotation, inv_start_rot, to_target, basis_vec0, basis_vec1, 2)
    # print('**up_vec',up_vec[:4])
    # print('**heading_proj:',heading_proj[:4])

    vel_loc, angvel_loc, roll, pitch, yaw, angle_to_target = compute_rot(
        torso_quat, velocity, ang_velocity, targets, torso_position)

    dof_pos_scaled = unscale(dof_pos, dof_limits_lower, dof_limits_upper)


    obs = torch.cat((torso_position, vel_loc, angvel_loc,
                     yaw.unsqueeze(-1), roll.unsqueeze(-1), angle_to_target.unsqueeze(-1),
                     up_proj.unsqueeze(-1), heading_proj.unsqueeze(-1), dof_pos_scaled,
                     dof_vel * dof_vel_scale, actions), dim=-1)

    return obs
