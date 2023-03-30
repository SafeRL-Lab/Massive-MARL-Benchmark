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
from agents.tasks.hand_base.base_task import BaseTask
from isaacgym import gymtorch
from isaacgym import gymapi


class TenAnt(BaseTask):

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless, is_multi_agent=False):

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
            self.num_agents = 10
            self.cfg["env"]["numActions"] = 8
            
        else:
            self.num_agents = 1
            self.cfg["env"]["numActions"] = 80

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
        self.vec_sensor_tensor = gymtorch.wrap_tensor(sensor_tensor).view(self.num_envs * self.num_agents, sensors_per_env * 6)

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
        self.dof_pos_3 = self.dof_state.view(self.num_envs, -1, 2)[:, 2 * self.num_dof: 3 * self.num_dof, 0]
        self.dof_pos_4 = self.dof_state.view(self.num_envs, -1, 2)[:, 3 * self.num_dof: 4 * self.num_dof, 0]
        self.dof_pos_5 = self.dof_state.view(self.num_envs, -1, 2)[:, 4 * self.num_dof: 5 * self.num_dof, 0]
        self.dof_pos_6 = self.dof_state.view(self.num_envs, -1, 2)[:, 5 * self.num_dof: 6 * self.num_dof, 0]
        self.dof_pos_7 = self.dof_state.view(self.num_envs, -1, 2)[:, 6 * self.num_dof: 7 * self.num_dof, 0]
        self.dof_pos_8 = self.dof_state.view(self.num_envs, -1, 2)[:, 7 * self.num_dof: 8 * self.num_dof, 0]
        self.dof_pos_9 = self.dof_state.view(self.num_envs, -1, 2)[:, 8 * self.num_dof: 9 * self.num_dof, 0]
        self.dof_pos_10 = self.dof_state.view(self.num_envs, -1, 2)[:, 9 * self.num_dof: 10 * self.num_dof, 0]

        self.dof_vel_1 = self.dof_state.view(self.num_envs, -1, 2)[:, :self.num_dof, 1]
        self.dof_vel_2 = self.dof_state.view(self.num_envs, -1, 2)[:, self.num_dof: 2 * self.num_dof, 1]
        self.dof_vel_3 = self.dof_state.view(self.num_envs, -1, 2)[:, 2 * self.num_dof: 3 * self.num_dof, 1]
        self.dof_vel_4 = self.dof_state.view(self.num_envs, -1, 2)[:, 3 * self.num_dof: 4 * self.num_dof, 1]
        self.dof_vel_5 = self.dof_state.view(self.num_envs, -1, 2)[:, 4 * self.num_dof: 5 * self.num_dof, 1]
        self.dof_vel_6 = self.dof_state.view(self.num_envs, -1, 2)[:, 5 * self.num_dof: 6 * self.num_dof, 1]
        self.dof_vel_7 = self.dof_state.view(self.num_envs, -1, 2)[:, 6 * self.num_dof: 7 * self.num_dof, 1]
        self.dof_vel_8 = self.dof_state.view(self.num_envs, -1, 2)[:, 7 * self.num_dof: 8 * self.num_dof, 1]
        self.dof_vel_9 = self.dof_state.view(self.num_envs, -1, 2)[:, 8 * self.num_dof: 9 * self.num_dof, 1]
        self.dof_vel_10 = self.dof_state.view(self.num_envs, -1, 2)[:, 9 * self.num_dof: 10 * self.num_dof, 1]

        # print('dof_pos_shape:', self.dof_pos.shape)
        # print('dof_vel:', self.dof_vel.shape)
        # print('num_dofs:', self.num_dofs)

        self.initial_dof_pos = torch.zeros_like(self.dof_pos_1, device=self.device, dtype=torch.float)
        zero_tensor = torch.tensor([0.0], device=self.device)
        self.initial_dof_pos = torch.where(self.dof_limits_lower > zero_tensor, self.dof_limits_lower,
                                           torch.where(self.dof_limits_upper < zero_tensor, self.dof_limits_upper,
                                                       self.initial_dof_pos))
        self.initial_dof_vel = torch.zeros_like(self.dof_vel_1, device=self.device, dtype=torch.float)
        self.dt = self.cfg["sim"]["dt"]

        # torques = self.gym.acquire_dof_force_tensor(self.sim)
        # self.torques = gymtorch.wrap_tensor(torques).view(self.num_envs, 2 * self.num_dof)

        self.x_unit_tensor = to_torch([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_agents * self.num_envs, 1))
        self.y_unit_tensor = to_torch([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_agents * self.num_envs, 1))
        self.z_unit_tensor = to_torch([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_agents * self.num_envs, 1))

        self.obs_buf_1 = torch.zeros((self.num_envs, 38), device=self.device,dtype=torch.float)
        self.obs_buf_2 = torch.zeros((self.num_envs, 38), device=self.device,dtype=torch.float)
        self.obs_buf_3 = torch.zeros((self.num_envs, 38), device=self.device,dtype=torch.float)
        self.obs_buf_4 = torch.zeros((self.num_envs, 38), device=self.device,dtype=torch.float)
        self.obs_buf_5 = torch.zeros((self.num_envs, 38), device=self.device,dtype=torch.float)
        self.obs_buf_6 = torch.zeros((self.num_envs, 38), device=self.device,dtype=torch.float)
        self.obs_buf_7 = torch.zeros((self.num_envs, 38), device=self.device,dtype=torch.float)
        self.obs_buf_8 = torch.zeros((self.num_envs, 38), device=self.device,dtype=torch.float)
        self.obs_buf_9 = torch.zeros((self.num_envs, 38), device=self.device,dtype=torch.float)
        self.obs_buf_10 = torch.zeros((self.num_envs, 38), device=self.device,dtype=torch.float)

        self.box_pos = torch.zeros((self.num_envs, 2), device=self.device, dtype=torch.float)
        self.box_quat = torch.zeros((self.num_envs, 4), device=self.device, dtype=torch.float)

        # initialize some data used later on
        self.up_vec_1 = to_torch(get_axis_params(1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.up_vec_2 = to_torch(get_axis_params(1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.up_vec_3 = to_torch(get_axis_params(1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.up_vec_4 = to_torch(get_axis_params(1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.up_vec_5 = to_torch(get_axis_params(1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.up_vec_6 = to_torch(get_axis_params(1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.up_vec_7 = to_torch(get_axis_params(1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.up_vec_8 = to_torch(get_axis_params(1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.up_vec_9 = to_torch(get_axis_params(1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.up_vec_10 = to_torch(get_axis_params(1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))

        self.heading_vec_1 = to_torch([1, 0, 0], device=self.device).repeat((self.num_envs, 1))
        self.heading_vec_2 = to_torch([1, 0, 0], device=self.device).repeat((self.num_envs, 1))
        self.heading_vec_3 = to_torch([1, 0, 0], device=self.device).repeat((self.num_envs, 1))
        self.heading_vec_4 = to_torch([1, 0, 0], device=self.device).repeat((self.num_envs, 1))
        self.heading_vec_5 = to_torch([1, 0, 0], device=self.device).repeat((self.num_envs, 1))
        self.heading_vec_6 = to_torch([1, 0, 0], device=self.device).repeat((self.num_envs, 1))
        self.heading_vec_7 = to_torch([1, 0, 0], device=self.device).repeat((self.num_envs, 1))
        self.heading_vec_8 = to_torch([1, 0, 0], device=self.device).repeat((self.num_envs, 1))
        self.heading_vec_9 = to_torch([1, 0, 0], device=self.device).repeat((self.num_envs, 1))
        self.heading_vec_10 = to_torch([1, 0, 0], device=self.device).repeat((self.num_envs, 1))

        self.inv_start_rot = quat_conjugate(self.start_rotation_1).repeat((self.num_envs, 1))

        self.basis_vec0 = self.heading_vec_1.clone()
        self.basis_vec1 = self.up_vec_1.clone()

        self.targets = to_torch([0, 0, 0], device=self.device).repeat((self.num_envs, 1))
        self.box_targets = to_torch([0, 0], device=self.device).repeat((self.num_envs, 1))
        self.target_dirs = to_torch([1, 0, 0], device=self.device).repeat((self.num_envs, 1))

        self.potentials = to_torch([-6 / self.dt], device=self.device).repeat(self.num_envs)
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
        return x, 28 * y

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
        ant_asset_3 = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        ant_asset_4 = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        ant_asset_5 = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        ant_asset_6 = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        ant_asset_7 = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        ant_asset_8 = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        ant_asset_9 = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        ant_asset_10 = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        self.num_dof = self.gym.get_asset_dof_count(ant_asset_1)
        self.num_dof_1 = self.gym.get_asset_dof_count(ant_asset_1)
        # print('self.num_dof:',self.num_dof)

        # Note - for this asset we are loading the actuator info from the MJCF
        actuator_props_1 = self.gym.get_asset_actuator_properties(ant_asset_1)
        actuator_props_2 = self.gym.get_asset_actuator_properties(ant_asset_2)
        motor_efforts = [prop.motor_effort for prop in actuator_props_1]
        self.joint_gears = to_torch(motor_efforts, device=self.device)

        start_pose_1 = gymapi.Transform()
        start_pose_1.p = gymapi.Vec3(6, -1.5, 1.)
        start_pose_2 = gymapi.Transform()
        start_pose_2.p = gymapi.Vec3(6, 1.5, 1.)
        start_pose_3 = gymapi.Transform()
        start_pose_3.p = gymapi.Vec3(6, -4.5, 1.)
        start_pose_4 = gymapi.Transform()
        start_pose_4.p = gymapi.Vec3(6, 4.5, 1.)
        start_pose_5 = gymapi.Transform()
        start_pose_5.p = gymapi.Vec3(6, -7.5, 1.)
        start_pose_6 = gymapi.Transform()
        start_pose_6.p = gymapi.Vec3(6, 7.5, 1.)
        start_pose_7 = gymapi.Transform()
        start_pose_7.p = gymapi.Vec3(6, -10.5, 1.)
        start_pose_8 = gymapi.Transform()
        start_pose_8.p = gymapi.Vec3(6, 10.5, 1.)
        start_pose_9 = gymapi.Transform()
        start_pose_9.p = gymapi.Vec3(6, -13.5, 1.)
        start_pose_10 = gymapi.Transform()
        start_pose_10.p = gymapi.Vec3(6, 13.5, 1.)

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

        self.num_bodies_3 = self.gym.get_asset_rigid_body_count(ant_asset_3)
        body_names_3 = [self.gym.get_asset_rigid_body_name(ant_asset_3, i) for i in range(self.num_bodies_3)]
        extremity_names_3 = [s for s in body_names_3 if "foot" in s]
        self.extremities_index_3 = torch.zeros(len(extremity_names_3), dtype=torch.long, device=self.device)

        self.num_bodies_4 = self.gym.get_asset_rigid_body_count(ant_asset_4)
        body_names_4 = [self.gym.get_asset_rigid_body_name(ant_asset_4, i) for i in range(self.num_bodies_4)]
        extremity_names_4 = [s for s in body_names_4 if "foot" in s]
        self.extremities_index_4 = torch.zeros(len(extremity_names_4), dtype=torch.long, device=self.device)

        self.num_bodies_5 = self.gym.get_asset_rigid_body_count(ant_asset_5)
        body_names_5 = [self.gym.get_asset_rigid_body_name(ant_asset_5, i) for i in range(self.num_bodies_5)]
        extremity_names_5 = [s for s in body_names_5 if "foot" in s]
        self.extremities_index_5 = torch.zeros(len(extremity_names_5), dtype=torch.long, device=self.device)

        self.num_bodies_6 = self.gym.get_asset_rigid_body_count(ant_asset_6)
        body_names_6 = [self.gym.get_asset_rigid_body_name(ant_asset_6, i) for i in range(self.num_bodies_6)]
        extremity_names_6 = [s for s in body_names_6 if "foot" in s]
        self.extremities_index_6 = torch.zeros(len(extremity_names_6), dtype=torch.long, device=self.device)

        self.num_bodies_7 = self.gym.get_asset_rigid_body_count(ant_asset_7)
        body_names_7 = [self.gym.get_asset_rigid_body_name(ant_asset_7, i) for i in range(self.num_bodies_7)]
        extremity_names_7 = [s for s in body_names_7 if "foot" in s]
        self.extremities_index_7 = torch.zeros(len(extremity_names_7), dtype=torch.long, device=self.device)

        self.num_bodies_8 = self.gym.get_asset_rigid_body_count(ant_asset_8)
        body_names_8 = [self.gym.get_asset_rigid_body_name(ant_asset_8, i) for i in range(self.num_bodies_8)]
        extremity_names_8 = [s for s in body_names_8 if "foot" in s]
        self.extremities_index_8 = torch.zeros(len(extremity_names_8), dtype=torch.long, device=self.device)

        self.num_bodies_9 = self.gym.get_asset_rigid_body_count(ant_asset_9)
        body_names_9 = [self.gym.get_asset_rigid_body_name(ant_asset_9, i) for i in range(self.num_bodies_9)]
        extremity_names_9 = [s for s in body_names_9 if "foot" in s]
        self.extremities_index_9 = torch.zeros(len(extremity_names_9), dtype=torch.long, device=self.device)

        self.num_bodies_10 = self.gym.get_asset_rigid_body_count(ant_asset_10)
        body_names_10 = [self.gym.get_asset_rigid_body_name(ant_asset_10, i) for i in range(self.num_bodies_10)]
        extremity_names_10 = [s for s in body_names_10 if "foot" in s]
        self.extremities_index_10 = torch.zeros(len(extremity_names_10), dtype=torch.long, device=self.device)

        # create force sensors attached to the "feet"
        extremity_indices_1 = [self.gym.find_asset_rigid_body_index(ant_asset_1, name) for name in extremity_names_1]
        
        sensor_pose_1 = gymapi.Transform()
        sensor_pose_2 = gymapi.Transform()
        sensor_pose_3 = gymapi.Transform()
        sensor_pose_4 = gymapi.Transform()
        sensor_pose_5 = gymapi.Transform()
        sensor_pose_6 = gymapi.Transform()
        sensor_pose_7 = gymapi.Transform()
        sensor_pose_8 = gymapi.Transform()
        sensor_pose_9 = gymapi.Transform()
        sensor_pose_10 = gymapi.Transform()

        for body_idx in extremity_indices_1:
            self.gym.create_asset_force_sensor(ant_asset_1, body_idx, sensor_pose_1)
            self.gym.create_asset_force_sensor(ant_asset_2, body_idx, sensor_pose_2)
            self.gym.create_asset_force_sensor(ant_asset_3, body_idx, sensor_pose_3)
            self.gym.create_asset_force_sensor(ant_asset_4, body_idx, sensor_pose_4)
            self.gym.create_asset_force_sensor(ant_asset_5, body_idx, sensor_pose_5)
            self.gym.create_asset_force_sensor(ant_asset_6, body_idx, sensor_pose_6)
            self.gym.create_asset_force_sensor(ant_asset_7, body_idx, sensor_pose_7)
            self.gym.create_asset_force_sensor(ant_asset_8, body_idx, sensor_pose_8)
            self.gym.create_asset_force_sensor(ant_asset_9, body_idx, sensor_pose_9)
            self.gym.create_asset_force_sensor(ant_asset_10, body_idx, sensor_pose_10)

        self.ant_handles_1 = []
        self.ant_indices_1 = []
        self.ant_handles_2 = []
        self.ant_indices_2 = []
        self.ant_handles_3 = []
        self.ant_indices_3 = []
        self.ant_handles_4 = []
        self.ant_indices_4 = []
        self.ant_handles_5 = []
        self.ant_indices_5 = []
        self.ant_handles_6 = []
        self.ant_indices_6 = []
        self.ant_handles_7 = []
        self.ant_indices_7 = []
        self.ant_handles_8 = []
        self.ant_indices_8 = []
        self.ant_handles_9 = []
        self.ant_indices_9 = []
        self.ant_handles_10 = []
        self.ant_indices_10 = []

        self.box_handles = []
        self.box_indices = []
        self.envs = []
        self.pos_before_1 = torch.zeros(2, device=self.device)
        self.pos_before_2 = torch.zeros(2, device=self.device)
        self.pos_before_3 = torch.zeros(2, device=self.device)
        self.pos_before_4 = torch.zeros(2, device=self.device)
        self.pos_before_5 = torch.zeros(2, device=self.device)
        self.pos_before_6 = torch.zeros(2, device=self.device)
        self.pos_before_7 = torch.zeros(2, device=self.device)
        self.pos_before_8 = torch.zeros(2, device=self.device)
        self.pos_before_9 = torch.zeros(2, device=self.device)
        self.pos_before_10 = torch.zeros(2, device=self.device)

        self.box_before = torch.zeros(2, device=self.device)
        self.dof_limits_lower = []
        self.dof_limits_upper = []

        # box
        asset_options = gymapi.AssetOptions()
        asset_options.density = 1.
        asset_box = self.gym.create_box(self.sim, 1, 28, 1, asset_options)
        box_pose = gymapi.Transform()
        box_pose.p = gymapi.Vec3(4, 0, 1)

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

            ant_handle_3 = self.gym.create_actor(env_ptr, ant_asset_3, start_pose_3, "ant_3", i, 1, 0)
            ant_index_3 = self.gym.get_actor_index(env_ptr, ant_handle_3, gymapi.DOMAIN_SIM)
            self.ant_indices_3.append(ant_index_3)

            ant_handle_4 = self.gym.create_actor(env_ptr, ant_asset_4, start_pose_4, "ant_4", i, 1, 0)
            ant_index_4 = self.gym.get_actor_index(env_ptr, ant_handle_4, gymapi.DOMAIN_SIM)
            self.ant_indices_4.append(ant_index_4)

            ant_handle_5 = self.gym.create_actor(env_ptr, ant_asset_5, start_pose_5, "ant_5", i, 1, 0)
            ant_index_5 = self.gym.get_actor_index(env_ptr, ant_handle_5, gymapi.DOMAIN_SIM)
            self.ant_indices_5.append(ant_index_5)

            ant_handle_6 = self.gym.create_actor(env_ptr, ant_asset_6, start_pose_6, "ant_6", i, 1, 0)
            ant_index_6 = self.gym.get_actor_index(env_ptr, ant_handle_6, gymapi.DOMAIN_SIM)
            self.ant_indices_6.append(ant_index_6)

            ant_handle_7 = self.gym.create_actor(env_ptr, ant_asset_7, start_pose_7, "ant_7", i, 1, 0)
            ant_index_7 = self.gym.get_actor_index(env_ptr, ant_handle_7, gymapi.DOMAIN_SIM)
            self.ant_indices_7.append(ant_index_7)

            ant_handle_8 = self.gym.create_actor(env_ptr, ant_asset_8, start_pose_8, "ant_8", i, 1, 0)
            ant_index_8 = self.gym.get_actor_index(env_ptr, ant_handle_8, gymapi.DOMAIN_SIM)
            self.ant_indices_8.append(ant_index_8)

            ant_handle_9 = self.gym.create_actor(env_ptr, ant_asset_9, start_pose_9, "ant_9", i, 1, 0)
            ant_index_9 = self.gym.get_actor_index(env_ptr, ant_handle_9, gymapi.DOMAIN_SIM)
            self.ant_indices_9.append(ant_index_9)

            ant_handle_10 = self.gym.create_actor(env_ptr, ant_asset_10, start_pose_10, "ant_10", i, 1, 0)
            ant_index_10 = self.gym.get_actor_index(env_ptr, ant_handle_10, gymapi.DOMAIN_SIM)
            self.ant_indices_10.append(ant_index_10)

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

            for j in range(self.num_bodies_1):
                self.gym.set_rigid_body_color(
                    env_ptr, ant_handle_1, j, gymapi.MESH_VISUAL, gymapi.Vec3(0.97, 0.38, 0.06))
                self.gym.set_rigid_body_color(
                    env_ptr, ant_handle_2, j, gymapi.MESH_VISUAL, gymapi.Vec3(0.24, 0.38, 0.06))
                self.gym.set_rigid_body_color(
                    env_ptr, ant_handle_3, j, gymapi.MESH_VISUAL, gymapi.Vec3(0.01, 0.38, 0.06))
                self.gym.set_rigid_body_color(
                    env_ptr, ant_handle_4, j, gymapi.MESH_VISUAL, gymapi.Vec3(0.11, 0.38, 0.06))
                self.gym.set_rigid_body_color(
                    env_ptr, ant_handle_5, j, gymapi.MESH_VISUAL, gymapi.Vec3(0.33, 0.38, 0.06))
                self.gym.set_rigid_body_color(
                    env_ptr, ant_handle_6, j, gymapi.MESH_VISUAL, gymapi.Vec3(0.44, 0.38, 0.06))
                self.gym.set_rigid_body_color(
                    env_ptr, ant_handle_7, j, gymapi.MESH_VISUAL, gymapi.Vec3(0.55, 0.38, 0.06))
                self.gym.set_rigid_body_color(
                    env_ptr, ant_handle_8, j, gymapi.MESH_VISUAL, gymapi.Vec3(0.66, 0.38, 0.06))
                self.gym.set_rigid_body_color(
                    env_ptr, ant_handle_9, j, gymapi.MESH_VISUAL, gymapi.Vec3(0.77, 0.38, 0.06))
                self.gym.set_rigid_body_color(
                    env_ptr, ant_handle_10, j, gymapi.MESH_VISUAL, gymapi.Vec3(0.88, 0.38, 0.06))

            self.envs.append(env_ptr)
            self.ant_handles_1.append(ant_handle_1)
            self.ant_handles_2.append(ant_handle_2)
            self.ant_handles_3.append(ant_handle_3)
            self.ant_handles_4.append(ant_handle_4)
            self.ant_handles_5.append(ant_handle_5)
            self.ant_handles_6.append(ant_handle_6)
            self.ant_handles_7.append(ant_handle_7)
            self.ant_handles_8.append(ant_handle_8)
            self.ant_handles_9.append(ant_handle_9)
            self.ant_handles_10.append(ant_handle_10)

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
            self.extremities_index_3[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.ant_handles_3[0],
                                                                              extremity_names_3[i])
            self.extremities_index_4[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.ant_handles_4[0],
                                                                              extremity_names_4[i])
            self.extremities_index_5[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.ant_handles_5[0],
                                                                              extremity_names_5[i])
            self.extremities_index_6[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.ant_handles_6[0],
                                                                              extremity_names_6[i])
            self.extremities_index_7[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.ant_handles_7[0],
                                                                              extremity_names_7[i])
            self.extremities_index_8[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.ant_handles_8[0],
                                                                              extremity_names_8[i])
            self.extremities_index_9[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.ant_handles_9[0],
                                                                              extremity_names_9[i])
            self.extremities_index_10[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.ant_handles_10[0],
                                                                              extremity_names_10[i])

        self.box_indices = to_torch(self.box_indices, dtype=torch.long, device=self.device)
        self.ant_indices_1 = to_torch(self.ant_indices_1, dtype=torch.long, device=self.device)
        self.ant_indices_2 = to_torch(self.ant_indices_2, dtype=torch.long, device=self.device)
        self.ant_indices_3 = to_torch(self.ant_indices_3, dtype=torch.long, device=self.device)
        self.ant_indices_4 = to_torch(self.ant_indices_4, dtype=torch.long, device=self.device)
        self.ant_indices_5 = to_torch(self.ant_indices_5, dtype=torch.long, device=self.device)
        self.ant_indices_6 = to_torch(self.ant_indices_6, dtype=torch.long, device=self.device)
        self.ant_indices_7 = to_torch(self.ant_indices_7, dtype=torch.long, device=self.device)
        self.ant_indices_8 = to_torch(self.ant_indices_8, dtype=torch.long, device=self.device)
        self.ant_indices_9 = to_torch(self.ant_indices_9, dtype=torch.long, device=self.device)
        self.ant_indices_10 = to_torch(self.ant_indices_10, dtype=torch.long, device=self.device)

    def compute_reward(self, actions):
        self.rew_buf[:], self.reset_buf[:] = compute_ant_reward(
            self.obs_buf_1,
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
            self.box_before,
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

        self.obs_buf_1[:], self.up_vec_1[:], self.heading_vec_1[:]= compute_ant_observations(
            self.obs_buf_1, self.root_states[0::(self.num_agents + 1), :], self.targets, 
            self.inv_start_rot, self.dof_pos_1, self.dof_vel_1,
            self.dof_limits_lower, self.dof_limits_upper, self.dof_vel_scale,
            self.actions[:,:8], self.dt, self.contact_force_scale,
            self.basis_vec0, self.basis_vec1, self.up_axis_idx)
        
        self.obs_buf_2[:], self.up_vec_2[:], self.heading_vec_2[:]= compute_ant_observations(
            self.obs_buf_2, self.root_states[1::(self.num_agents + 1), :], self.targets, 
            self.inv_start_rot, self.dof_pos_2, self.dof_vel_2,
            self.dof_limits_lower, self.dof_limits_upper, self.dof_vel_scale,
            self.actions[:,8:16], self.dt, self.contact_force_scale,
            self.basis_vec0, self.basis_vec1, self.up_axis_idx)

        self.obs_buf_3[:], self.up_vec_3[:], self.heading_vec_3[:]= compute_ant_observations(
            self.obs_buf_3, self.root_states[2::(self.num_agents + 1), :], self.targets, 
            self.inv_start_rot, self.dof_pos_3, self.dof_vel_3,
            self.dof_limits_lower, self.dof_limits_upper, self.dof_vel_scale,
            self.actions[:,16:24], self.dt, self.contact_force_scale,
            self.basis_vec0, self.basis_vec1, self.up_axis_idx)
        
        self.obs_buf_4[:], self.up_vec_4[:], self.heading_vec_4[:]= compute_ant_observations(
            self.obs_buf_4, self.root_states[3::(self.num_agents + 1), :], self.targets, 
            self.inv_start_rot, self.dof_pos_4, self.dof_vel_4,
            self.dof_limits_lower, self.dof_limits_upper, self.dof_vel_scale,
            self.actions[:,24:32], self.dt, self.contact_force_scale,
            self.basis_vec0, self.basis_vec1, self.up_axis_idx)
        
        self.obs_buf_5[:], self.up_vec_5[:], self.heading_vec_5[:]= compute_ant_observations(
            self.obs_buf_5, self.root_states[4::(self.num_agents + 1), :], self.targets, 
            self.inv_start_rot, self.dof_pos_5, self.dof_vel_5,
            self.dof_limits_lower, self.dof_limits_upper, self.dof_vel_scale,
            self.actions[:,32:40], self.dt, self.contact_force_scale,
            self.basis_vec0, self.basis_vec1, self.up_axis_idx)
        
        self.obs_buf_6[:], self.up_vec_6[:], self.heading_vec_6[:]= compute_ant_observations(
            self.obs_buf_6, self.root_states[5::(self.num_agents + 1), :], self.targets, 
            self.inv_start_rot, self.dof_pos_6, self.dof_vel_6,
            self.dof_limits_lower, self.dof_limits_upper, self.dof_vel_scale,
            self.actions[:,40:48], self.dt, self.contact_force_scale,
            self.basis_vec0, self.basis_vec1, self.up_axis_idx)
        
        self.obs_buf_7[:], self.up_vec_7[:], self.heading_vec_7[:]= compute_ant_observations(
            self.obs_buf_7, self.root_states[6::(self.num_agents + 1), :], self.targets, 
            self.inv_start_rot, self.dof_pos_7, self.dof_vel_7,
            self.dof_limits_lower, self.dof_limits_upper, self.dof_vel_scale,
            self.actions[:,48:56], self.dt, self.contact_force_scale,
            self.basis_vec0, self.basis_vec1, self.up_axis_idx)
        
        self.obs_buf_8[:], self.up_vec_8[:], self.heading_vec_8[:]= compute_ant_observations(
            self.obs_buf_8, self.root_states[7::(self.num_agents + 1), :], self.targets, 
            self.inv_start_rot, self.dof_pos_8, self.dof_vel_8,
            self.dof_limits_lower, self.dof_limits_upper, self.dof_vel_scale,
            self.actions[:,56:64], self.dt, self.contact_force_scale,
            self.basis_vec0, self.basis_vec1, self.up_axis_idx)
        
        self.obs_buf_9[:], self.up_vec_9[:], self.heading_vec_9[:]= compute_ant_observations(
            self.obs_buf_9, self.root_states[8::(self.num_agents + 1), :], self.targets, 
            self.inv_start_rot, self.dof_pos_9, self.dof_vel_9,
            self.dof_limits_lower, self.dof_limits_upper, self.dof_vel_scale,
            self.actions[:,64:72], self.dt, self.contact_force_scale,
            self.basis_vec0, self.basis_vec1, self.up_axis_idx)
        
        self.obs_buf_10[:], self.up_vec_10[:], self.heading_vec_10[:]= compute_ant_observations(
            self.obs_buf_10, self.root_states[9::(self.num_agents + 1), :], self.targets, 
            self.inv_start_rot, self.dof_pos_10, self.dof_vel_10,
            self.dof_limits_lower, self.dof_limits_upper, self.dof_vel_scale,
            self.actions[:,72:80], self.dt, self.contact_force_scale,
            self.basis_vec0, self.basis_vec1, self.up_axis_idx)
        
        self.box_pos[:], self.box_quat[:] = compute_box_pos(self.root_states[10::(self.num_agents + 1), :])
    

        self.obs_buf = torch.cat((self.obs_buf_1,self.obs_buf_2,self.obs_buf_3,self.obs_buf_4,self.obs_buf_5,
                                  self.obs_buf_6,self.obs_buf_7,self.obs_buf_8,self.obs_buf_9,self.obs_buf_10
                                  ,self.box_pos,self.box_quat,self.box_targets),dim=-1)

    def reset_idx(self, env_ids):
        # Randomization can happen only at reset time, since it can reset actor positions on GPU
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        ant_box_indices = torch.unique(torch.cat([self.ant_indices_1[env_ids], self.ant_indices_2[env_ids],
                                                  self.ant_indices_3[env_ids], self.ant_indices_4[env_ids],
                                                  self.ant_indices_5[env_ids], self.ant_indices_6[env_ids],
                                                  self.ant_indices_7[env_ids], self.ant_indices_8[env_ids],
                                                  self.ant_indices_9[env_ids], self.ant_indices_10[env_ids]
                                                     , self.box_indices[env_ids]]).to(torch.int32))

        positions = torch_rand_float(-0.2, 0.2, (len(env_ids), self.num_dof_1), device=self.device)
        velocities = torch_rand_float(-0.1, 0.1, (len(env_ids), self.num_dof_1), device=self.device)

        self.dof_pos_1[env_ids] = tensor_clamp(self.initial_dof_pos[env_ids] + positions, self.dof_limits_lower,
                                             self.dof_limits_upper)
        self.dof_vel_1[env_ids] = velocities
        self.dof_pos_2[env_ids] = tensor_clamp(self.initial_dof_pos[env_ids] + positions, self.dof_limits_lower,
                                               self.dof_limits_upper)
        self.dof_vel_2[env_ids] = velocities
        self.dof_pos_3[env_ids] = tensor_clamp(self.initial_dof_pos[env_ids] + positions, self.dof_limits_lower,
                                             self.dof_limits_upper)
        self.dof_vel_3[env_ids] = velocities
        self.dof_pos_4[env_ids] = tensor_clamp(self.initial_dof_pos[env_ids] + positions, self.dof_limits_lower,
                                               self.dof_limits_upper)
        self.dof_vel_4[env_ids] = velocities
        self.dof_pos_5[env_ids] = tensor_clamp(self.initial_dof_pos[env_ids] + positions, self.dof_limits_lower,
                                             self.dof_limits_upper)
        self.dof_vel_5[env_ids] = velocities
        self.dof_pos_6[env_ids] = tensor_clamp(self.initial_dof_pos[env_ids] + positions, self.dof_limits_lower,
                                               self.dof_limits_upper)
        self.dof_vel_6[env_ids] = velocities
        self.dof_pos_7[env_ids] = tensor_clamp(self.initial_dof_pos[env_ids] + positions, self.dof_limits_lower,
                                             self.dof_limits_upper)
        self.dof_vel_7[env_ids] = velocities
        self.dof_pos_8[env_ids] = tensor_clamp(self.initial_dof_pos[env_ids] + positions, self.dof_limits_lower,
                                               self.dof_limits_upper)
        self.dof_vel_8[env_ids] = velocities
        self.dof_pos_9[env_ids] = tensor_clamp(self.initial_dof_pos[env_ids] + positions, self.dof_limits_lower,
                                               self.dof_limits_upper)
        self.dof_vel_9[env_ids] = velocities
        self.dof_pos_10[env_ids] = tensor_clamp(self.initial_dof_pos[env_ids] + positions, self.dof_limits_lower,
                                               self.dof_limits_upper)
        self.dof_vel_10[env_ids] = velocities

        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.initial_root_states),
                                                     gymtorch.unwrap_tensor(ant_box_indices), len(ant_box_indices))

        ant_indices = torch.unique(torch.cat([self.ant_indices_1[env_ids], self.ant_indices_2[env_ids],
                                                  self.ant_indices_3[env_ids], self.ant_indices_4[env_ids],
                                                  self.ant_indices_5[env_ids], self.ant_indices_6[env_ids],
                                                  self.ant_indices_7[env_ids], self.ant_indices_8[env_ids],
                                                  self.ant_indices_9[env_ids], self.ant_indices_10[env_ids]]).to(torch.int32))

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(ant_indices), len(ant_indices))

        self.pos_before_1 = self.root_states[0::11, :2].clone()
        self.pos_before_2 = self.root_states[1::11, :2].clone()
        self.pos_before_3 = self.root_states[2::11, :2].clone()
        self.pos_before_4 = self.root_states[3::11, :2].clone()
        self.pos_before_5 = self.root_states[4::11, :2].clone()
        self.pos_before_6 = self.root_states[5::11, :2].clone()
        self.pos_before_7 = self.root_states[6::11, :2].clone()
        self.pos_before_8 = self.root_states[7::11, :2].clone()
        self.pos_before_9 = self.root_states[8::11, :2].clone()
        self.pos_before_10 = self.root_states[9::11, :2].clone()

        self.box_before = self.root_states[10::11, :2].clone()
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0

    def pre_physics_step(self, actions):
        self.actions = actions.clone().to(self.device)
        # self.actions = torch.cat((self.actions, self.actions), dim=-1)
        # forces = self.actions * self.joint_gears * self.power_scale
        # force_tensor = gymtorch.unwrap_tensor(forces)
        # self.gym.set_dof_actuation_force_tensor(self.sim, force_tensor)
        targets = self.actions
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(targets))

    def post_physics_step(self):
        # print('**44')
        self.progress_buf += 1
        self.randomize_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward(self.actions)

        self.pos_before_1 = self.obs_buf_1[:self.num_envs, :2].clone()
        self.pos_before_2 = self.obs_buf_2[:self.num_envs, :2].clone()
        self.pos_before_3 = self.obs_buf_3[:self.num_envs, :2].clone()
        self.pos_before_4 = self.obs_buf_4[:self.num_envs, :2].clone()
        self.pos_before_5 = self.obs_buf_5[:self.num_envs, :2].clone()
        self.pos_before_6 = self.obs_buf_6[:self.num_envs, :2].clone()
        self.pos_before_7 = self.obs_buf_7[:self.num_envs, :2].clone()
        self.pos_before_8 = self.obs_buf_8[:self.num_envs, :2].clone()
        self.pos_before_9 = self.obs_buf_9[:self.num_envs, :2].clone()
        self.pos_before_10 = self.obs_buf_10[:self.num_envs, :2].clone()
        self.box_before = self.box_pos[:self.num_envs, :2].clone()
        # print('**33:',self.pos_before.shape)


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
        actions_cost_scale,
        energy_cost_scale,
        joints_at_limit_cost_scale,
        termination_height,
        death_cost,
        max_episode_length,
        pos_before,
        box_before,
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
     # type: (Tensor, Tensor, Tensor, Tensor, float, float, float, float, float, float, float, float, Tensor, Tensor, Tensor, float, float, Tensor, float, float, float, float, float, Tensor, float) -> Tuple[Tensor, Tensor]

    # quat reward
    x, y, z = compute_box_quat(box_quat)
    quat_dist = compute_box_quat_dist(x_goal, y_goal, z_goal, x, y, z)
    quat_reward = quat_reward_scale * quat_dist

    # ant and box reward
    # ant_push = l2_dist(ant_pos, box_pos) < 1.5
    # ant_push = abs(ant_push - 1)
    # print('**ant_push:', ant_push.shape, ant_push[:4])
    # ant_dist = l2_dist(pos_before, box_before) - l2_dist(ant_pos, box_pos)
    # ant_dist_reward = ant_dist_reward_scale * ant_dist * ant_push
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
    heading_weight_tensor = torch.ones_like(obs_buf[:, 13]) * heading_weight
    heading_reward = torch.where(obs_buf[:, 13] > 0.8, heading_weight_tensor, heading_weight * obs_buf[:, 13] / 0.8)
    # print('**heading_reward:',heading_reward[:4])

    # aligning up axis of ant and environment
    up_reward = torch.zeros_like(heading_reward)
    up_reward = torch.where(obs_buf[:, 12] > 0.93, up_reward + up_weight, up_reward)
    # print('**up_reward:',up_reward[:4])

    # energy penalty for movement
    actions_cost = torch.sum(actions ** 2, dim=-1)
    electricity_cost = torch.sum(torch.abs(actions[:,0:8] * obs_buf[:, 22:30]), dim=-1)
    dof_at_limit_cost = torch.sum(obs_buf[:, 14:22] > 0.99, dim=-1)
    # print('*****actions:', actions[:4])
    # print(obs_buf[:4, 20:28])
    # print((torch.abs(actions * obs_buf[:, 20:28]))[0:4])

    # reward for duration of staying alive
    # alive_reward = torch.ones_like(potentials) * 0.5
    # print('***alive_reward:',alive_reward[:4])
    # progress_reward = potentials - prev_potentials
    # print('***progress_reward',progress_reward[:4])


    total_reward = up_reward + quat_reward +  goal_dist_reward + goal_arrive_reward + success_reward- \
                   actions_cost_scale * actions_cost - energy_cost_scale * electricity_cost - dof_at_limit_cost * joints_at_limit_cost_scale
    # print('**total_reward:',total_reward.shape)
    # print('**total_reward:',total_reward[:4])

    # adjust reward for fallen agents
    total_reward = torch.where(obs_buf[:, 2] < termination_height, torch.ones_like(total_reward) * death_cost,
                               total_reward)

    # reset agents
    reset = torch.where(obs_buf[:, 2] < termination_height, torch.ones_like(reset_buf), reset_buf)
    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset)

    return total_reward, reset


@torch.jit.script
def compute_ant_observations(obs_buf, root_states, targets, 
                             inv_start_rot, dof_pos, dof_vel,
                             dof_limits_lower, dof_limits_upper, dof_vel_scale,
                             actions, dt, contact_force_scale,
                             basis_vec0, basis_vec1, up_axis_idx):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, Tensor, float, float, Tensor, Tensor, int) -> Tuple[Tensor, Tensor, Tensor]

    torso_position = root_states[:, 0:3]
    # print('***:',torso_position[0:4])
    # print(torso_position[:, up_axis_idx].view(-1, 1)[0:4])
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

    # print('***torso_position:', torso_position[:, up_axis_idx].view(-1, 1).shape)
    # print('***vel_loc:', vel_loc.shape)
    # print('***angvel_loc', angvel_loc.shape)
    # print('***yaw.unsqueeze(-1), roll.unsqueeze(-1), angle_to_target.unsqueeze(-1):',yaw.unsqueeze(-1).shape, roll.unsqueeze(-1).shape, angle_to_target.unsqueeze(-1).shape)
    # print('***up_proj.unsqueeze(-1), heading_proj.unsqueeze(-1), dof_pos_scaled:',up_proj.unsqueeze(-1).shape, heading_proj.unsqueeze(-1).shape, dof_pos_scaled.shape)
    # print('***dof_vel * dof_vel_scale, sensor_force_torques.view(-1, 24) * contact_force_scale:', dof_vel.shape, sensor_force_torques.view(-1, 24).shape)

    # obs_buf shapes: 1, 3, 3, 1, 1, 1, 1, 1, num_dofs(8), num_dofs(8), 24, num_dofs(8)
    obs = torch.cat((torso_position, vel_loc, angvel_loc,
                     yaw.unsqueeze(-1), roll.unsqueeze(-1), angle_to_target.unsqueeze(-1),
                     up_proj.unsqueeze(-1), heading_proj.unsqueeze(-1), dof_pos_scaled,
                     dof_vel * dof_vel_scale, actions), dim=-1)
    # print('**obs:', obs.shape)
    # print('**obs', obs[:4])

    return obs, up_vec, heading_vec


@torch.jit.script
def compute_box_pos(root_states):
    # type: (Tensor) -> Tuple[Tensor, Tensor]
    box_pos = root_states[:, :2]
    box_quat = root_states[:, 3:7]

    return box_pos, box_quat