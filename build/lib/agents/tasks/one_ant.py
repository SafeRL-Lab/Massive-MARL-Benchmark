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


class OneAnt(BaseTask):

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

        self.cfg["env"]["numObservations"] = 60
        self.cfg["env"]["numActions"] = 8

        self.draw_penalty_scale = -1
        self.move_reward_scale = 1.
        self.quat_reward_scale = 1.
        self.ant_dist_reward_scale = 500.
        self.goal_dist_reward_scale = 500.


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
        self.vec_sensor_tensor = gymtorch.wrap_tensor(sensor_tensor).view(self.num_envs, sensors_per_env * 6)

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
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]

        # print('dof_pos_shape:', self.dof_pos.shape)
        # print('dof_vel:', self.dof_vel.shape)
        # print('num_dofs:', self.num_dofs)

        self.initial_dof_pos = torch.zeros_like(self.dof_pos, device=self.device, dtype=torch.float)
        zero_tensor = torch.tensor([0.0], device=self.device)
        self.initial_dof_pos = torch.where(self.dof_limits_lower > zero_tensor, self.dof_limits_lower,
                                           torch.where(self.dof_limits_upper < zero_tensor, self.dof_limits_upper,
                                                       self.initial_dof_pos))
        self.initial_dof_vel = torch.zeros_like(self.dof_vel, device=self.device, dtype=torch.float)
        self.dt = self.cfg["sim"]["dt"]

        # torques = self.gym.acquire_dof_force_tensor(self.sim)
        # self.torques = gymtorch.wrap_tensor(torques).view(self.num_envs, 2 * self.num_dof)

        self.x_unit_tensor = to_torch([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = to_torch([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = to_torch([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

        self.ant_pos = torch.zeros((self.num_envs, 2), device=self.device,
                                   dtype=torch.float)
        self.box_pos = torch.zeros((self.num_envs, 2), device=self.device,
                                   dtype=torch.float)
        self.box_quat = torch.zeros((self.num_envs, 4), device=self.device,
                                    dtype=torch.float)
        self.hp = torch.ones((self.num_envs,), device=self.device, dtype=torch.float32) * 100

        # initialize some data used later on
        self.up_vec = to_torch(get_axis_params(1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        # print('***up_vec:',self.up_vec[:4])
        self.heading_vec = to_torch([1, 0, 0], device=self.device).repeat((self.num_envs, 1))
        # print('**heading_vec:',self.heading_vec[:4])
        self.inv_start_rot = quat_conjugate(self.start_rotation).repeat((self.num_envs, 1))

        self.basis_vec0 = self.heading_vec.clone()
        self.basis_vec1 = self.up_vec.clone()

        self.targets = to_torch([0, 0, 0], device=self.device).repeat((self.num_envs, 1))
        self.box_targets = to_torch([0, 0], device=self.device).repeat((self.num_envs, 1))
        self.target_dirs = to_torch([1, 0, 0], device=self.device).repeat((self.num_envs, 1))
        self.potentials = to_torch([-4/ self.dt], device=self.device).repeat(self.num_envs)
        self.prev_potentials = self.potentials.clone()
        self.x_goal = 0.
        self.y_goal = 1.
        self.z_goal = 0.



    def create_sim(self):
        self.up_axis_idx = self.set_sim_params_up_axis(self.sim_params, 'z')
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)

        lines = []
        borderline_height = 0.05
        for height in range(20):
            for angle in range(360):
                x1, y1 = self.compute_angle(angle)
                x2, y2 = self.compute_angle(angle + 1)
                begin_point = [x1, y1, borderline_height * height]
                end_point = [x2, y2, borderline_height * height]
                lines.append(begin_point)
                lines.append(end_point)
        self.lines = np.array(lines, dtype=np.float32) * 1

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

    def compute_angle(self, angle):
        if angle >= 0 and angle <= 45:
            x = 0.5
            y = angle / 90
        if angle > 45 and angle <= 135:
            x = (90 - angle) / 90
            y = 0.5
        if angle > 135 and angle <= 225:
            x = -0.5
            y = (180 - angle) / 90
        if angle > 225 and angle <= 315:
            x = (angle - 270) / 90
            y = -0.5
        if angle > 315 and angle <= 360:
            x = 0.5
            y = (angle - 360) / 90
        return x, y

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../assets')
        asset_file = "mjcf/nv_ant.xml"

        if "asset" in self.cfg["env"]:
            asset_file = self.cfg["env"]["asset"].get("assetFileName", asset_file)

        asset_path = os.path.join(asset_root, asset_file)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        # Note - DOF mode is set in the MJCF file and loaded by Isaac Gym
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        asset_options.angular_damping = 0.0

        ant_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(ant_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(ant_asset)

        # Note - for this asset we are loading the actuator info from the MJCF
        actuator_props = self.gym.get_asset_actuator_properties(ant_asset)
        motor_efforts = [prop.motor_effort for prop in actuator_props]
        self.joint_gears = to_torch(motor_efforts, device=self.device)

        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(-6, 0, 1.)

        self.start_rotation = torch.tensor([start_pose.r.x, start_pose.r.y, start_pose.r.z, start_pose.r.w],
                                           device=self.device)

        self.torso_index = 0
        self.num_bodies = self.gym.get_asset_rigid_body_count(ant_asset)
        body_names = [self.gym.get_asset_rigid_body_name(ant_asset, i) for i in range(self.num_bodies)]
        extremity_names = [s for s in body_names if "foot" in s]
        self.extremities_index = torch.zeros(len(extremity_names), dtype=torch.long, device=self.device)

        # create force sensors attached to the "feet"
        extremity_indices = [self.gym.find_asset_rigid_body_index(ant_asset, name) for name in extremity_names]
        sensor_pose = gymapi.Transform()
        for body_idx in extremity_indices:
            self.gym.create_asset_force_sensor(ant_asset, body_idx, sensor_pose)

        self.ant_handles = []
        self.ant_indices = []
        self.box_handles = []
        self.box_indices = []
        self.envs = []
        self.pos_before = torch.zeros(2, device=self.device)
        self.box_before = torch.zeros(2, device=self.device)
        self.dof_limits_lower = []
        self.dof_limits_upper = []

        # box
        asset_options = gymapi.AssetOptions()
        asset_options.density = 1.0
        asset_box = self.gym.create_box(self.sim, 1, 1, 1, asset_options)
        box_pose = gymapi.Transform()
        box_pose.p = gymapi.Vec3(-4, 0, 1)

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )
            ant_handle = self.gym.create_actor(env_ptr, ant_asset, start_pose, "ant", i, 1, 0)
            ant_index = self.gym.get_actor_index(env_ptr, ant_handle, gymapi.DOMAIN_SIM)
            self.ant_indices.append(ant_index)

            box_handle = self.gym.create_actor(env_ptr, asset_box, box_pose, 'box', i, 0)
            self.box_handles.append(box_handle)
            box_index = self.gym.get_actor_index(env_ptr, box_handle, gymapi.DOMAIN_SIM)
            self.box_indices.append(box_index)
            shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, self.box_handles[-1])
            shape_props[0].friction = 0.
            shape_props[0].rolling_friction = 0.
            shape_props[0].torsion_friction = 0.
            self.gym.set_actor_rigid_shape_properties(env_ptr, self.box_handles[-1], shape_props)
            self.gym.set_rigid_body_color(env_ptr, box_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(2, 0.1, 2.))

            for j in range(self.num_bodies):
                self.gym.set_rigid_body_color(
                    env_ptr, ant_handle, j, gymapi.MESH_VISUAL, gymapi.Vec3(0.97, 0.38, 0.06))

            self.envs.append(env_ptr)
            self.ant_handles.append(ant_handle)

        dof_prop = self.gym.get_actor_dof_properties(env_ptr, ant_handle)
        for j in range(self.num_dof):
            if dof_prop['lower'][j] > dof_prop['upper'][j]:
                self.dof_limits_lower.append(dof_prop['upper'][j])
                self.dof_limits_upper.append(dof_prop['lower'][j])
            else:
                self.dof_limits_lower.append(dof_prop['lower'][j])
                self.dof_limits_upper.append(dof_prop['upper'][j])

        self.dof_limits_lower = to_torch(self.dof_limits_lower, device=self.device)
        self.dof_limits_upper = to_torch(self.dof_limits_upper, device=self.device)

        for i in range(len(extremity_names)):
            self.extremities_index[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.ant_handles[0],
                                                                              extremity_names[i])

        self.box_indices = to_torch(self.box_indices, dtype=torch.long, device=self.device)
        self.ant_indices = to_torch(self.ant_indices, dtype=torch.long, device=self.device)

    def compute_reward(self, actions):
        self.rew_buf[:], self.reset_buf[:] = compute_ant_reward(
            self.obs_buf,
            self.reset_buf,
            self.progress_buf,
            self.actions,
            self.up_weight,
            self.heading_weight,
            self.potentials,
            self.prev_potentials,
            self.actions_cost_scale,
            self.energy_cost_scale,
            self.joints_at_limit_cost_scale,
            self.termination_height,
            self.death_cost,
            self.max_episode_length,
            self.pos_before,
            self.box_before,
            self.ant_pos,
            self.box_pos,
            self.dt,
            self.move_reward_scale,
            self.box_quat,
            self.x_goal,
            self.y_goal,
            self.z_goal,
            self.quat_reward_scale,
            self.ant_dist_reward_scale,
            self.box_targets,
            self.goal_dist_reward_scale
        )

    def compute_observations(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)
        # print("Feet forces and torques: ", self.vec_sensor_tensor[0, :])
        # print(self.vec_sensor_tensor.shape)

        self.obs_buf[:], self.potentials[:], self.prev_potentials[:], self.up_vec[:], self.heading_vec[:], self.ant_pos[
                                                                                                           :] = compute_ant_observations(
            self.obs_buf, self.root_states[0::2, :], self.root_states[1::2,:], self.targets, self.potentials,
            self.inv_start_rot, self.dof_pos, self.dof_vel,
            self.dof_limits_lower, self.dof_limits_upper, self.dof_vel_scale,
            self.vec_sensor_tensor, self.actions, self.dt, self.contact_force_scale,
            self.basis_vec0, self.basis_vec1, self.up_axis_idx)

        self.box_pos[:], self.box_quat[:] = compute_box_pos(self.root_states[1::2, :])

    def reset_idx(self, env_ids):
        # Randomization can happen only at reset time, since it can reset actor positions on GPU
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        ant_box_indices = torch.unique(torch.cat([self.ant_indices[env_ids],
                                                  self.box_indices[env_ids]]).to(torch.int32))

        positions = torch_rand_float(-0.2, 0.2, (len(env_ids), self.num_dof), device=self.device)
        velocities = torch_rand_float(-0.1, 0.1, (len(env_ids), self.num_dof), device=self.device)

        self.dof_pos[env_ids] = tensor_clamp(self.initial_dof_pos[env_ids] + positions, self.dof_limits_lower,
                                             self.dof_limits_upper)
        self.dof_vel[env_ids] = velocities

        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.initial_root_states),
                                                     gymtorch.unwrap_tensor(ant_box_indices), len(ant_box_indices))

        ant_indices = self.ant_indices[env_ids].to(torch.int32)

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(ant_indices), len(ant_indices))

        self.pos_before = self.root_states[0::2, :2].clone()
        self.box_before = self.root_states[1::2, :2].clone()
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0

        # print('**11self.pos_before:',self.pos_before[:4,:])
        # print('**11self.box_before:',self.box_before[:4,:])

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

        self.pos_before = self.ant_pos[:self.num_envs, :2].clone()
        self.box_before = self.box_pos[:self.num_envs, :2].clone()
        # print('**22self.pos_before:',self.pos_before[:4,:])
        # print('**22self.box_before:',self.box_before[:4,:])


#####################################################################
###=========================jit functions=========================###
#####################################################################


#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def compute_box_quat(box_quat):
    # type: (Tensor) -> Tuple[Tensor, Tensor, Tensor]
    qw = box_quat[:, 3].clone()
    qx = box_quat[:, 0].clone()
    qy = box_quat[:, 1].clone()
    qz = box_quat[:, 2].clone()

    x = 2 * (qx * qy + qw * qz)
    y = 1 - 2 * (qx * qx + qz * qz)
    z = 2 * (qy * qz - qw * qx)
    return x, y, z

@torch.jit.script
def compute_box_quat_dist(x_goal, y_goal, z_goal, x, y, z):
    # type: (float, float, float, Tensor, Tensor, Tensor) -> Tensor
    x_1 = x * x_goal
    y_1 = y * y_goal
    z_1 = z * z_goal
    quat_dist = (x_1 + y_1 + z_1) / (torch.sqrt(x ** 2 + y ** 2 + z ** 2)) / (
        torch.sqrt(x_goal ** 2 + y_goal ** 2 + z_goal ** 2))

    return quat_dist

@torch.jit.script
def l2_dist(
        a,
        b
):
    # type: (Tensor,Tensor) -> Tensor
    c = a - b
    c1 = c[:, 0].clone()
    c2 = c[:, 1].clone()
    c = c1 ** 2 + c2 ** 2
    return torch.sqrt(c)

@torch.jit.script
def compute_ant_reward(
        obs_buf,
        reset_buf,
        progress_buf,
        actions,
        up_weight,
        heading_weight,
        potentials,
        prev_potentials,
        actions_cost_scale,
        energy_cost_scale,
        joints_at_limit_cost_scale,
        termination_height,
        death_cost,
        max_episode_length,
        pos_before,
        box_before,
        ant_pos,
        box_pos,
        dt,
        move_reward_scale,
        box_quat,
        x_goal,
        y_goal,
        z_goal,
        quat_reward_scale,
        ant_dist_reward_scale,
        box_targets,
        goal_dist_reward_scale
):
     # type: (Tensor, Tensor, Tensor, Tensor, float, float, Tensor, Tensor, float, float, float, float, float, float, Tensor, Tensor, Tensor, Tensor, float, float, Tensor, float, float, float, float, float, Tensor, float) -> Tuple[Tensor, Tensor]

    # quat reward
    x, y, z = compute_box_quat(box_quat)
    quat_dist = compute_box_quat_dist(x_goal, y_goal, z_goal, x, y, z)
    quat_reward = quat_reward_scale * quat_dist

    # ant and box reward
    ant_push = l2_dist(ant_pos, box_pos) < 1.5
    ant_push = abs(ant_push - 1)
    # print('**ant_push:', ant_push.shape, ant_push[:4])
    ant_dist = l2_dist(pos_before, box_before) - l2_dist(ant_pos, box_pos)
    ant_dist_reward = ant_dist_reward_scale * ant_dist * ant_push
    # print("ant_dist_reward:",ant_dist_reward.shape, ant_dist_reward[:4])

    goal_dist_before = l2_dist(box_targets, box_before)
    goal_dist = l2_dist(box_targets, box_pos)
    goal_arrive = goal_dist < 0.5
    goal_dist_reward = goal_dist_reward_scale * (goal_dist_before - goal_dist)
    goal_arrive_reward = 2 * goal_arrive

    # success
    quat_arrive = quat_dist > 0.9
    success_reward = quat_arrive * goal_arrive * 10


    # reward from direction headed
    heading_weight_tensor = torch.ones_like(obs_buf[:, 11]) * heading_weight
    heading_reward = torch.where(obs_buf[:, 11] > 0.8, heading_weight_tensor, heading_weight * obs_buf[:, 11] / 0.8)
    # print('**heading_reward:',heading_reward[:4])

    # aligning up axis of ant and environment
    up_reward = torch.zeros_like(heading_reward)
    up_reward = torch.where(obs_buf[:, 10] > 0.93, up_reward + up_weight, up_reward)
    # print('**up_reward:',up_reward[:4])

    # energy penalty for movement
    actions_cost = torch.sum(actions ** 2, dim=-1)
    electricity_cost = torch.sum(torch.abs(actions * obs_buf[:, 20:28]), dim=-1)
    dof_at_limit_cost = torch.sum(obs_buf[:, 12:20] > 0.99, dim=-1)
    # print('*****actions:', actions[:4])
    # print(obs_buf[:4, 20:28])
    # print((torch.abs(actions * obs_buf[:, 20:28]))[0:4])

    # reward for duration of staying alive
    alive_reward = torch.ones_like(potentials) * 0.5
    # print('***alive_reward:',alive_reward[:4])
    progress_reward = potentials - prev_potentials
    # print('***progress_reward',progress_reward[:4])


    total_reward = alive_reward + up_reward + quat_reward + ant_dist_reward + goal_dist_reward + goal_arrive_reward + success_reward- \
                   actions_cost_scale * actions_cost - energy_cost_scale * electricity_cost - dof_at_limit_cost * joints_at_limit_cost_scale
    # print('**total_reward:',total_reward.shape)
    # print('**total_reward:',total_reward[:4])

    # adjust reward for fallen agents
    total_reward = torch.where(obs_buf[:, 0] < termination_height, torch.ones_like(total_reward) * death_cost,
                               total_reward)

    # reset agents
    reset = torch.where(obs_buf[:, 0] < termination_height, torch.ones_like(reset_buf), reset_buf)
    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset)

    return total_reward, reset


@torch.jit.script
def compute_ant_observations(obs_buf, root_states, root_states_box, targets, potentials,
                             inv_start_rot, dof_pos, dof_vel,
                             dof_limits_lower, dof_limits_upper, dof_vel_scale,
                             sensor_force_torques, actions, dt, contact_force_scale,
                             basis_vec0, basis_vec1, up_axis_idx):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, Tensor, Tensor, float, float, Tensor, Tensor, int) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]

    torso_position = root_states[:, 0:3]
    # print('***:',torso_position[0:4])
    # print(torso_position[:, up_axis_idx].view(-1, 1)[0:4])
    torso_rotation = root_states[:, 3:7]
    velocity = root_states[:, 7:10]
    ang_velocity = root_states[:, 10:13]
    ant_pos = root_states[:, 0:2]

    to_target = targets - torso_position
    to_target[:, 2] = 0.0

    box_torso_position = root_states_box[:, 0:3]
    to_target_box = targets - box_torso_position
    to_target_box[:,2]=0.0

    prev_potentials_new = potentials.clone()
    # print('**prev_potentials_new:',prev_potentials_new[:4])
    potentials = -torch.norm(to_target_box, p=2, dim=-1) / dt
    # print('**potentials:',potentials[:4])

    torso_quat, up_proj, heading_proj, up_vec, heading_vec = compute_heading_and_up(
        torso_rotation, inv_start_rot, to_target, basis_vec0, basis_vec1, 2)
    # print('**up_vec',up_vec[:4])
    # print('**heading_proj:',heading_proj[:4])

    vel_loc, angvel_loc, roll, pitch, yaw, angle_to_target = compute_rot(
        torso_quat, velocity, ang_velocity, targets, torso_position)

    dof_pos_scaled = unscale(dof_pos, dof_limits_lower, dof_limits_upper)

    # print('***torso_position:', torso_position[:, up_axis_idx].view(-1, 1).shape)
    # print('***vel_loc:', vel_loc.shape)
    # print('***angvel_loc', angvel_loc.shape)
    # print('***yaw.unsqueeze(-1), roll.unsqueeze(-1), angle_to_target.unsqueeze(-1):',yaw.unsqueeze(-1).shape, roll.unsqueeze(-1).shape, angle_to_target.unsqueeze(-1).shape)
    # print('***up_proj.unsqueeze(-1), heading_proj.unsqueeze(-1), dof_pos_scaled:',up_proj.unsqueeze(-1).shape, heading_proj.unsqueeze(-1).shape, dof_pos_scaled.shape)
    # print('***dof_vel * dof_vel_scale, sensor_force_torques.view(-1, 24) * contact_force_scale:', dof_vel.shape, sensor_force_torques.view(-1, 24).shape)
    # print('***actions',actions.shape)

    # obs_buf shapes: 1, 3, 3, 1, 1, 1, 1, 1, num_dofs(8), num_dofs(8), 24, num_dofs(8)
    obs = torch.cat((torso_position[:, up_axis_idx].view(-1, 1), vel_loc, angvel_loc,
                     yaw.unsqueeze(-1), roll.unsqueeze(-1), angle_to_target.unsqueeze(-1),
                     up_proj.unsqueeze(-1), heading_proj.unsqueeze(-1), dof_pos_scaled,
                     dof_vel * dof_vel_scale, sensor_force_torques.view(-1, 24) * contact_force_scale,
                     actions), dim=-1)
    # print('**obs:', obs.shape)
    # print('**obs', obs[:4])

    return obs, potentials, prev_potentials_new, up_vec, heading_vec, ant_pos


@torch.jit.script
def compute_box_pos(root_states):
    # type: (Tensor) -> Tuple[Tensor, Tensor]
    box_pos = root_states[:, :2]
    box_quat = root_states[:, 3:7]

    return box_pos, box_quat