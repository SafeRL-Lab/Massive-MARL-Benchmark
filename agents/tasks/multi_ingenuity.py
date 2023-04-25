# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from matplotlib.pyplot import axis
import numpy as np
import math
from PIL import Image as Im

import os
import random
import torch

from agents.utils.torch_jit_utils import *
from agents.tasks.agent_base.base_task import BaseTask
from isaacgym import gymtorch
from isaacgym import gymapi


class MultiIngenuity(BaseTask):

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


        self.cfg["env"]["numObservations"] = 13
        if self.is_multi_agent:
            self.num_agents = 4
            self.cfg["env"]["numActions"] = 6

        else:
            self.num_agents = 1
            self.cfg["env"]["numActions"] = 24

        self.draw_penalty_scale = -1
        self.move_reward_scale = 1.
        self.quat_reward_scale = 1.
        self.ant_dist_reward_scale = 500.
        self.goal_dist_reward_scale = 500.

        self.cfg["device_type"] = device_type
        self.cfg["device_id"] = device_id
        self.cfg["headless"] = headless

        super().__init__(cfg=self.cfg)

        dofs_per_env = 4
        bodies_per_env = 6

        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)

        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.initial_root_states = self.root_states.clone()
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos_1 = self.dof_state.view(self.num_envs, -1, 2)[:, :dofs_per_env, 0]
        self.dof_pos_2 = self.dof_state.view(self.num_envs, -1, 2)[:, dofs_per_env: 2 * dofs_per_env, 0]
        self.dof_pos_3 = self.dof_state.view(self.num_envs, -1, 2)[:, 2 * dofs_per_env: 3 * dofs_per_env, 0]
        self.dof_pos_4 = self.dof_state.view(self.num_envs, -1, 2)[:, 3 * dofs_per_env: 4 * dofs_per_env, 0]

        self.dof_vel_1 = self.dof_state.view(self.num_envs, -1, 2)[:, :dofs_per_env, 1]
        self.dof_vel_2 = self.dof_state.view(self.num_envs, -1, 2)[:, dofs_per_env: 2 * dofs_per_env, 1]
        self.dof_vel_3 = self.dof_state.view(self.num_envs, -1, 2)[:, 2 * dofs_per_env: 3 * dofs_per_env, 1]
        self.dof_vel_4 = self.dof_state.view(self.num_envs, -1, 2)[:, 3 * dofs_per_env: 4 * dofs_per_env, 1]
        self.initial_dof_states = self.dof_state.clone()

        self.obs_buf_1 = torch.zeros((self.num_envs, 13), device=self.device, dtype=torch.float)
        self.obs_buf_2 = torch.zeros((self.num_envs, 13), device=self.device, dtype=torch.float)
        self.obs_buf_3 = torch.zeros((self.num_envs, 13), device=self.device, dtype=torch.float)
        self.obs_buf_4 = torch.zeros((self.num_envs, 13), device=self.device, dtype=torch.float)

        self.goal_1 = to_torch([4, 2, 1], device=self.device).repeat((self.num_envs, 1))
        self.goal_2 = to_torch([4, -2, 1], device=self.device).repeat((self.num_envs, 1))
        self.goal_3 = to_torch([4, 6, 1], device=self.device).repeat((self.num_envs, 1))
        self.goal_4 = to_torch([4, -6, 1], device=self.device).repeat((self.num_envs, 1))


        self.thrust_lower_limit = 0
        self.thrust_upper_limit = 2000
        self.thrust_lateral_component = 0.2

        # control tensors
        self.thrusts = torch.zeros((self.num_envs, 4 * 2, 3), dtype=torch.float32, device=self.device,
                                   requires_grad=False)
        self.forces = torch.zeros((self.num_envs, bodies_per_env * 4, 3), dtype=torch.float32, device=self.device,
                                  requires_grad=False)


    def create_sim(self):
        self.sim_params.up_axis = gymapi.UP_AXIS_Z

        # Mars gravity
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -3.721

        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self.dt = self.sim_params.dt
        # self._create_ingenuity_asset()
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../assets')
        asset_file = "mjcf/open_ai_assets/ingenuity/ingenuity.xml"

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False
        asset_options.angular_damping = 0.0
        asset_options.max_angular_velocity = 4 * math.pi
        asset_options.slices_per_cylinder = 40

        asset_1 = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        asset_2 = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        asset_3 = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        asset_4 = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        default_pose_1 = gymapi.Transform()
        default_pose_1.p = gymapi.Vec3(0, 2, 1)
        default_pose_2 = gymapi.Transform()
        default_pose_2.p = gymapi.Vec3(0, -2, 1)
        default_pose_3 = gymapi.Transform()
        default_pose_3.p = gymapi.Vec3(0, 6, 1)
        default_pose_4 = gymapi.Transform()
        default_pose_4.p = gymapi.Vec3(0, -6, 1)

        self.envs = []
        self.actor_handles_1 = []
        self.actor_handles_2 = []
        self.actor_handles_3 = []
        self.actor_handles_4 = []
        self.actor_indices_1 = []
        self.actor_indices_2 = []
        self.actor_indices_3 = []
        self.actor_indices_4 = []

        self.pos_before_1 = torch.zeros(3, device=self.device)
        self.pos_before_2 = torch.zeros(3, device=self.device)
        self.pos_before_3 = torch.zeros(3, device=self.device)
        self.pos_before_4 = torch.zeros(3, device=self.device)

        for i in range(self.num_envs):
            # create env instance
            env = self.gym.create_env(self.sim, lower, upper, num_per_row)
            actor_handle_1 = self.gym.create_actor(env, asset_1, default_pose_1, "ingenuity_1", i, 1, 1)
            actor_index_1 = self.gym.get_actor_index(env, actor_handle_1, gymapi.DOMAIN_SIM)
            self.actor_indices_1.append(actor_index_1)

            actor_handle_2 = self.gym.create_actor(env, asset_2, default_pose_2, "ingenuity_2", i, 1, 1)
            actor_index_2 = self.gym.get_actor_index(env, actor_handle_2, gymapi.DOMAIN_SIM)
            self.actor_indices_2.append(actor_index_2)

            actor_handle_3 = self.gym.create_actor(env, asset_3, default_pose_3, "ingenuity_3", i, 1, 1)
            actor_index_3 = self.gym.get_actor_index(env, actor_handle_3, gymapi.DOMAIN_SIM)
            self.actor_indices_3.append(actor_index_3)

            actor_handle_4 = self.gym.create_actor(env, asset_4, default_pose_4, "ingenuity_4", i, 1, 1)
            actor_index_4 = self.gym.get_actor_index(env, actor_handle_4, gymapi.DOMAIN_SIM)
            self.actor_indices_4.append(actor_index_4)

            dof_props_1 = self.gym.get_actor_dof_properties(env, actor_handle_1)
            dof_props_1['stiffness'].fill(0)
            dof_props_1['damping'].fill(0)
            self.gym.set_actor_dof_properties(env, actor_handle_1, dof_props_1)

            dof_props_2 = self.gym.get_actor_dof_properties(env, actor_handle_2)
            dof_props_2['stiffness'].fill(0)
            dof_props_2['damping'].fill(0)
            self.gym.set_actor_dof_properties(env, actor_handle_2, dof_props_2)

            dof_props_3 = self.gym.get_actor_dof_properties(env, actor_handle_3)
            dof_props_3['stiffness'].fill(0)
            dof_props_3['damping'].fill(0)
            self.gym.set_actor_dof_properties(env, actor_handle_3, dof_props_3)

            dof_props_4 = self.gym.get_actor_dof_properties(env, actor_handle_4)
            dof_props_4['stiffness'].fill(0)
            dof_props_4['damping'].fill(0)
            self.gym.set_actor_dof_properties(env, actor_handle_4, dof_props_4)

            self.actor_handles_1.append(actor_handle_1)
            self.actor_handles_2.append(actor_handle_2)
            self.actor_handles_3.append(actor_handle_3)
            self.actor_handles_4.append(actor_handle_4)
            self.envs.append(env)

        self.actor_indices_1 = to_torch(self.actor_indices_1, dtype=torch.long, device=self.device)
        self.actor_indices_2 = to_torch(self.actor_indices_2, dtype=torch.long, device=self.device)
        self.actor_indices_3 = to_torch(self.actor_indices_3, dtype=torch.long, device=self.device)
        self.actor_indices_4 = to_torch(self.actor_indices_4, dtype=torch.long, device=self.device)

    def reset_idx(self, env_ids):

        # set rotor speeds
        self.dof_vel_1[:, 1] = -50
        self.dof_vel_1[:, 3] = 50
        self.dof_vel_2[:, 1] = -50
        self.dof_vel_2[:, 3] = 50
        self.dof_vel_3[:, 1] = -50
        self.dof_vel_3[:, 3] = 50
        self.dof_vel_4[:, 1] = -50
        self.dof_vel_4[:, 3] = 50
        # clear actions for reset envs
        self.thrusts[env_ids] = 0.0
        self.forces[env_ids] = 0.0

        num_resets = len(env_ids)

        actor_indices = torch.unique(torch.cat([self.actor_indices_1[env_ids], self.actor_indices_2[env_ids],
                                                  self.actor_indices_3[env_ids], self.actor_indices_4[env_ids]
                                                  ]).to(torch.int32))

        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.initial_root_states),
                                                     gymtorch.unwrap_tensor(actor_indices), len(actor_indices))

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(actor_indices), len(actor_indices))

        self.pos_before_1 = self.root_states[0::4, :3].clone()
        self.pos_before_2 = self.root_states[1::4, :3].clone()
        self.pos_before_3 = self.root_states[2::4, :3].clone()
        self.pos_before_4 = self.root_states[3::4, :3].clone()

        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

    def pre_physics_step(self, _actions):

        actions = _actions.to(self.device)

        thrust_action_speed_scale = 2000
        vertical_thrust_prop_a_1 = torch.clamp(actions[:, 2] * thrust_action_speed_scale, -self.thrust_upper_limit,
                                             self.thrust_upper_limit)
        vertical_thrust_prop_b_1 = torch.clamp(actions[:, 5] * thrust_action_speed_scale, -self.thrust_upper_limit,
                                             self.thrust_upper_limit)
        lateral_fraction_prop_a_1 = torch.clamp(actions[:, 0:2], -self.thrust_lateral_component,
                                              self.thrust_lateral_component)
        lateral_fraction_prop_b_1 = torch.clamp(actions[:, 3:5], -self.thrust_lateral_component,
                                              self.thrust_lateral_component)

        vertical_thrust_prop_a_2 = torch.clamp(actions[:, 8] * thrust_action_speed_scale, -self.thrust_upper_limit,
                                               self.thrust_upper_limit)
        vertical_thrust_prop_b_2 = torch.clamp(actions[:, 11] * thrust_action_speed_scale, -self.thrust_upper_limit,
                                               self.thrust_upper_limit)
        lateral_fraction_prop_a_2 = torch.clamp(actions[:, 6:8], -self.thrust_lateral_component,
                                                self.thrust_lateral_component)
        lateral_fraction_prop_b_2 = torch.clamp(actions[:, 9:11], -self.thrust_lateral_component,
                                                self.thrust_lateral_component)

        vertical_thrust_prop_a_3 = torch.clamp(actions[:, 14] * thrust_action_speed_scale, -self.thrust_upper_limit,
                                               self.thrust_upper_limit)
        vertical_thrust_prop_b_3 = torch.clamp(actions[:, 17] * thrust_action_speed_scale, -self.thrust_upper_limit,
                                               self.thrust_upper_limit)
        lateral_fraction_prop_a_3 = torch.clamp(actions[:, 12:14], -self.thrust_lateral_component,
                                                self.thrust_lateral_component)
        lateral_fraction_prop_b_3 = torch.clamp(actions[:, 15:17], -self.thrust_lateral_component,
                                                self.thrust_lateral_component)

        vertical_thrust_prop_a_4 = torch.clamp(actions[:, 20] * thrust_action_speed_scale, -self.thrust_upper_limit,
                                               self.thrust_upper_limit)
        vertical_thrust_prop_b_4 = torch.clamp(actions[:, 23] * thrust_action_speed_scale, -self.thrust_upper_limit,
                                               self.thrust_upper_limit)
        lateral_fraction_prop_a_4 = torch.clamp(actions[:, 18:20], -self.thrust_lateral_component,
                                                self.thrust_lateral_component)
        lateral_fraction_prop_b_4 = torch.clamp(actions[:, 21:23], -self.thrust_lateral_component,
                                                self.thrust_lateral_component)

        self.thrusts[:, 0, 2] = self.dt * vertical_thrust_prop_a_1
        self.thrusts[:, 0, 0:2] = self.thrusts[:, 0, 2, None] * lateral_fraction_prop_a_1
        self.thrusts[:, 1, 2] = self.dt * vertical_thrust_prop_b_1
        self.thrusts[:, 1, 0:2] = self.thrusts[:, 1, 2, None] * lateral_fraction_prop_b_1

        self.thrusts[:, 2, 2] = self.dt * vertical_thrust_prop_a_2
        self.thrusts[:, 2, 0:2] = self.thrusts[:, 2, 2, None] * lateral_fraction_prop_a_2
        self.thrusts[:, 3, 2] = self.dt * vertical_thrust_prop_b_2
        self.thrusts[:, 3, 0:2] = self.thrusts[:, 3, 2, None] * lateral_fraction_prop_b_2

        self.thrusts[:, 4, 2] = self.dt * vertical_thrust_prop_a_3
        self.thrusts[:, 4, 0:2] = self.thrusts[:, 4, 2, None] * lateral_fraction_prop_a_3
        self.thrusts[:, 5, 2] = self.dt * vertical_thrust_prop_b_3
        self.thrusts[:, 5, 0:2] = self.thrusts[:, 5, 2, None] * lateral_fraction_prop_b_3

        self.thrusts[:, 6, 2] = self.dt * vertical_thrust_prop_a_4
        self.thrusts[:, 6, 0:2] = self.thrusts[:, 6, 2, None] * lateral_fraction_prop_a_4
        self.thrusts[:, 7, 2] = self.dt * vertical_thrust_prop_b_4
        self.thrusts[:, 7, 0:2] = self.thrusts[:, 7, 2, None] * lateral_fraction_prop_b_4

        self.forces[:, 1] = self.thrusts[:, 0]
        self.forces[:, 3] = self.thrusts[:, 1]
        self.forces[:, 7] = self.thrusts[:, 2]
        self.forces[:, 9] = self.thrusts[:, 3]
        self.forces[:, 13] = self.thrusts[:, 4]
        self.forces[:, 15] = self.thrusts[:, 5]
        self.forces[:, 19] = self.thrusts[:, 6]
        self.forces[:, 21] = self.thrusts[:, 7]

        # apply actions
        self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(self.forces), None, gymapi.LOCAL_SPACE)

    def post_physics_step(self):

        self.progress_buf += 1
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward()

    def compute_observations(self):
        self.obs_buf_1[:] = self.root_states[0::4, :]
        self.obs_buf_2[:] = self.root_states[1::4, :]
        self.obs_buf_3[:] = self.root_states[2::4, :]
        self.obs_buf_4[:] = self.root_states[3::4, :]
        self.obs_buf = torch.cat((self.obs_buf_1, self.obs_buf_2, self.obs_buf_3, self.obs_buf_4), dim=-1)
        return self.obs_buf

    def compute_reward(self):
        self.rew_buf[:], self.reset_buf[:] = compute_ingenuity_reward(
            self.obs_buf_1,
            self.obs_buf_2,
            self.obs_buf_3,
            self.obs_buf_4,
            self.pos_before_1,
            self.pos_before_2,
            self.pos_before_3,
            self.pos_before_4,
            self.goal_1,
            self.goal_2,
            self.goal_3,
            self.goal_4,
            self.reset_buf, self.progress_buf, self.max_episode_length
        )


#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def compute_ingenuity_reward(obs_buf_1,obs_buf_2,obs_buf_3,obs_buf_4,pos_before_1,pos_before_2,pos_before_3,pos_before_4,
                             goal_1,goal_2,goal_3,goal_4,reset_buf,
                             progress_buf, max_episode_length):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor,Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor,Tensor, Tensor, float) -> Tuple[Tensor, Tensor]

    # distance to target
    root_positions_1 = obs_buf_1[:, :3]
    root_positions_2 = obs_buf_2[:, :3]
    root_positions_3 = obs_buf_3[:, :3]
    root_positions_4 = obs_buf_4[:, :3]

    target_dist_1 = torch.sqrt(torch.square(goal_1 - root_positions_1).sum(-1))
    target_dist_2 = torch.sqrt(torch.square(goal_2 - root_positions_2).sum(-1))
    target_dist_3 = torch.sqrt(torch.square(goal_3 - root_positions_3).sum(-1))
    target_dist_4 = torch.sqrt(torch.square(goal_4 - root_positions_4).sum(-1))

    pos_reward_1 = 1.0 / (1.0 + target_dist_1 * target_dist_1)
    pos_reward_2 = 1.0 / (1.0 + target_dist_2 * target_dist_2)
    pos_reward_3 = 1.0 / (1.0 + target_dist_3 * target_dist_3)
    pos_reward_4 = 1.0 / (1.0 + target_dist_4 * target_dist_4)

    pos_reward = pos_reward_1 + pos_reward_2 + pos_reward_3 + pos_reward_4

    # uprightness
    ups_1 = quat_axis(obs_buf_1[:,3:7], 2)
    tiltage_1 = torch.abs(1 - ups_1[..., 2])
    up_reward_1 = 5.0 / (1.0 + tiltage_1 * tiltage_1)

    ups_2 = quat_axis(obs_buf_2[:, 3:7], 2)
    tiltage_2 = torch.abs(1 - ups_2[..., 2])
    up_reward_2 = 5.0 / (1.0 + tiltage_2 * tiltage_2)

    ups_3 = quat_axis(obs_buf_3[:, 3:7], 2)
    tiltage_3 = torch.abs(1 - ups_3[..., 2])
    up_reward_3 = 5.0 / (1.0 + tiltage_3 * tiltage_3)

    ups_4 = quat_axis(obs_buf_4[:, 3:7], 2)
    tiltage_4 = torch.abs(1 - ups_4[..., 2])
    up_reward_4 = 5.0 / (1.0 + tiltage_4 * tiltage_4)

    up_reward = up_reward_1 + up_reward_2 + up_reward_3 + up_reward_4

    # spinning
    spinnage_1 = torch.abs(obs_buf_1[:, 12])
    spinnage_reward_1 = 1.0 / (1.0 + spinnage_1 * spinnage_1)

    spinnage_2 = torch.abs(obs_buf_2[:, 12])
    spinnage_reward_2 = 1.0 / (1.0 + spinnage_2 * spinnage_2)

    spinnage_3 = torch.abs(obs_buf_3[:, 12])
    spinnage_reward_3 = 1.0 / (1.0 + spinnage_3 * spinnage_3)

    spinnage_4 = torch.abs(obs_buf_4[:, 12])
    spinnage_reward_4 = 1.0 / (1.0 + spinnage_4 * spinnage_4)

    spinnage_reward = spinnage_reward_1 + spinnage_reward_2 + spinnage_reward_3 + spinnage_reward_4

    # combined reward
    # uprigness and spinning only matter when close to the target
    reward = pos_reward + pos_reward * (up_reward + spinnage_reward)

    # resets due to misbehavior
    ones = torch.ones_like(reset_buf)
    die = torch.zeros_like(reset_buf)
    die = torch.where(target_dist_1 > 8.0, ones, die) or torch.where(target_dist_2 > 8.0, ones, die) or torch.where(target_dist_3 > 8.0, ones, die) or torch.where(target_dist_4 > 8.0, ones, die)
    die = torch.where(root_positions_1[:, 2] < 0.5, ones, die) or torch.where(root_positions_2[:, 2] < 0.5, ones, die) or torch.where(root_positions_3[:, 2] < 0.5, ones, die) or torch.where(root_positions_4[:, 2] < 0.5, ones, die)

    # resets due to episode length
    reset = torch.where(progress_buf >= max_episode_length - 1, ones, die)

    return reward, reset
