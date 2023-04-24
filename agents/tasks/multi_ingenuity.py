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
        default_pose_3.p = gymapi.Vec3(0, -4, 1)
        default_pose_4 = gymapi.Transform()
        default_pose_4.p = gymapi.Vec3(0, 4, 1)

        self.envs = []
        self.actor_handles_1 = []
        self.actor_handles_2 = []
        self.actor_handles_3 = []
        self.actor_handles_4 = []
        self.actor_indices_1 = []
        self.actor_indices_2 = []
        self.actor_indices_3 = []
        self.actor_indices_4 = []

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

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)

        self.compute_observations()
        self.compute_reward()

    def compute_observations(self):
        self.obs_buf[:] = self.root_states
        return self.obs_buf

    def compute_reward(self):
        self.rew_buf[:], self.reset_buf[:] = compute_ingenuity_reward(
            self.root_states[0::4, 0:3],
            self.root_states[0::4, 0:3],
            self.root_states[0::4, 3:7],
            self.root_states[0::4, 7:10],
            self.root_states[0::4, 10:13],
            self.reset_buf, self.progress_buf, self.max_episode_length
        )


#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def compute_ingenuity_reward(root_positions, target_root_positions, root_quats, root_linvels, root_angvels, reset_buf,
                             progress_buf, max_episode_length):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float) -> Tuple[Tensor, Tensor]

    # distance to target
    target_dist = torch.sqrt(torch.square(target_root_positions - root_positions).sum(-1))
    pos_reward = 1.0 / (1.0 + target_dist * target_dist)

    # uprightness
    ups = quat_axis(root_quats, 2)
    tiltage = torch.abs(1 - ups[..., 2])
    up_reward = 5.0 / (1.0 + tiltage * tiltage)

    # spinning
    spinnage = torch.abs(root_angvels[..., 2])
    spinnage_reward = 1.0 / (1.0 + spinnage * spinnage)

    # combined reward
    # uprigness and spinning only matter when close to the target
    reward = pos_reward + pos_reward * (up_reward + spinnage_reward)

    # resets due to misbehavior
    ones = torch.ones_like(reset_buf)
    die = torch.zeros_like(reset_buf)
    die = torch.where(target_dist > 8.0, ones, die)
    die = torch.where(root_positions[..., 2] < 0.5, ones, die)

    # resets due to episode length
    reset = torch.where(progress_buf >= max_episode_length - 1, ones, die)

    return reward, reset
