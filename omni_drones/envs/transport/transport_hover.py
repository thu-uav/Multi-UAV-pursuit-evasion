from functorch import vmap

import omni.isaac.core.utils.torch as torch_utils
from omni_drones.utils.torch import euler_to_quaternion
import torch
import torch.distributions as D
from omni.isaac.core.objects import DynamicCuboid
from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import CompositeSpec, UnboundedContinuousTensorSpec

import omni_drones.utils.kit as kit_utils
import omni_drones.utils.scene as scene_utils

from omni_drones.envs.isaac_env import AgentSpec, IsaacEnv
from omni_drones.views import RigidPrimView
from omni_drones.utils.torch import cpos, off_diag, others
from omni_drones.robots.assembly.transportation_group import TransportationGroup
from omni_drones.robots.config import RobotCfg
from omni_drones.robots.drone import MultirotorBase


class TransportHover(IsaacEnv):
    def __init__(self, cfg, headless):
        super().__init__(cfg, headless)
        self.reward_effort_weight = self.cfg.task.reward_effort_weight
        self.reward_spin_weight = self.cfg.task.reward_spin_weight
        self.reward_swing_weight = self.cfg.task.reward_swing_weight

        self.reward_distance_scale = self.cfg.task.reward_distance_scale
        self.safe_distance = self.cfg.task.safe_distance

        self.group.initialize()
        self.payload = self.group.payload_view
        
        self.payload_target_visual = RigidPrimView(
            "/World/envs/.*/payloadTargetVis",
            reset_xform_properties=False
        )
        self.payload_target_visual.initialize()
        
        self.init_poses = self.group.get_world_poses(clone=True)
        self.init_velocities = torch.zeros_like(self.group.get_velocities())
        self.init_joint_pos = self.group.get_joint_positions(clone=True)
        self.init_joint_vel = torch.zeros_like(self.group.get_joint_velocities())

        self.init_drone_poses = self.drone.get_world_poses(clone=True)
        self.init_drone_vels = torch.zeros_like(self.drone.get_velocities())

        drone_state_dim = self.drone.state_spec.shape[0]
        observation_spec = CompositeSpec({
            "self": UnboundedContinuousTensorSpec((1, drone_state_dim)).to(self.device),
            "others": UnboundedContinuousTensorSpec((self.drone.n-1, drone_state_dim+1)).to(self.device),
            "payload": UnboundedContinuousTensorSpec((1, 22)).to(self.device)
        })

        state_spec = CompositeSpec(
            drones=UnboundedContinuousTensorSpec((self.drone.n, drone_state_dim)).to(self.device),
            payload=UnboundedContinuousTensorSpec((1, 22)).to(self.device)
        )

        self.agent_spec["drone"] = AgentSpec(
            "drone",
            4,
            observation_spec,
            self.drone.action_spec.to(self.device),
            UnboundedContinuousTensorSpec(1).to(self.device),
            state_spec=state_spec
        )

        self.payload_target_pos = torch.tensor([0., 0., 2.], device=self.device)
        self.payload_target_heading = torch.zeros(self.num_envs, 3, device=self.device)
        
        self.payload_target_rpy_dist = D.Uniform(
            torch.tensor([0., 0., 0.], device=self.device) * torch.pi,
            torch.tensor([0., 0., 2.], device=self.device) * torch.pi
        )
        self.payload_mass_dist = D.Uniform(
            torch.as_tensor(self.cfg.task.payload_mass_min, device=self.device),
            torch.as_tensor(self.cfg.task.payload_mass_max, device=self.device)
        )
        self.init_pos_dist = D.Uniform(
            torch.tensor([-3, -3, 1.], device=self.device),
            torch.tensor([3., 3., 2.5], device=self.device)
        )
        self.init_rpy_dist = D.Uniform(
            torch.tensor([0., 0., 0.], device=self.device) * torch.pi,
            torch.tensor([0., 0., 2.], device=self.device) * torch.pi
        )

        self.alpha = 0.7
        
        info_spec = CompositeSpec({
            "payload_mass": UnboundedContinuousTensorSpec((1,)),
            "payload_pos_error": UnboundedContinuousTensorSpec((1,)),
            "heading_alignment": UnboundedContinuousTensorSpec((1,)),
            "uprightness": UnboundedContinuousTensorSpec((1,)),
        }).expand(self.num_envs).to(self.device)
        self.observation_spec["info"] = info_spec
        self.info = info_spec.zero()

    def _design_scene(self):
        cfg = RobotCfg()
        drone_model = MultirotorBase.REGISTRY[self.cfg.task.drone_model]
        self.drone: MultirotorBase = drone_model(cfg=cfg)
        self.group = TransportationGroup(drone=self.drone)

        scene_utils.design_scene()

        DynamicCuboid(
            "/World/envs/env_0/payloadTargetVis",
            translation=torch.tensor([0., 0., 2.]),
            scale=torch.tensor([0.75, 0.5, 0.2]),
            color=torch.tensor([0.8, 0.1, 0.1]),
            size=2.01,
        )
        kit_utils.set_collision_properties(
            "/World/envs/env_0/payloadTargetVis",
            collision_enabled=False
        )
        kit_utils.set_rigid_body_properties(
            "/World/envs/env_0/payloadTargetVis",
            disable_gravity=True
        )

        self.group.spawn(translations=[(0, 0, 2.0)], enable_collision=False)
        return ["/World/defaultGroundPlane"]

    def _reset_idx(self, env_ids: torch.Tensor):
        pos = self.init_pos_dist.sample(env_ids.shape)
        rpy = self.init_rpy_dist.sample(env_ids.shape)
        rot = euler_to_quaternion(rpy)
        heading = torch_utils.quat_axis(rot, 0)
        up = torch_utils.quat_axis(rot, 2)

        self.group._reset_idx(env_ids)
        self.group.set_world_poses(pos + self.envs_positions[env_ids], rot, env_ids)
        self.group.set_velocities(self.init_velocities[env_ids], env_ids)

        self.group.set_joint_positions(self.init_joint_pos[env_ids], env_ids)
        self.group.set_joint_velocities(self.init_joint_vel[env_ids], env_ids)

        payload_target_rpy = self.payload_target_rpy_dist.sample(env_ids.shape)
        payload_target_rot = euler_to_quaternion(payload_target_rpy)
        payload_masses = self.payload_mass_dist.sample(env_ids.shape)

        self.payload_target_heading[env_ids] = torch_utils.quat_axis(payload_target_rot, 0)

        self.payload.set_masses(payload_masses, env_ids)
        self.payload_target_visual.set_world_poses(
            orientations=payload_target_rot,
            env_indices=env_ids
        )

        self.info["payload_mass"][env_ids] = payload_masses.unsqueeze(-1).clone()
        self.info["payload_pos_error"][env_ids] = torch.norm(
            self.payload_target_pos - pos, dim=-1, keepdim=True
        )
        self.info["heading_alignment"][env_ids] = torch.sum(
            self.payload_target_heading[env_ids] * heading, dim=-1, keepdim=True
        )
        self.info["uprightness"][env_ids] = up[:, 2].unsqueeze(-1)

    def _pre_sim_step(self, tensordict: TensorDictBase):
        actions = tensordict[("action", "drone.action")]
        self.effort = self.drone.apply_action(actions)

    def _compute_state_and_obs(self):
        self.drone_states = self.drone.get_state()
        drone_pos = self.drone_states[..., :3]

        self.payload_pos, self.payload_rot = self.get_env_poses(self.payload.get_world_poses())
        payload_vels = self.payload.get_velocities()
        self.payload_heading: torch.Tensor = torch_utils.quat_axis(self.payload_rot, axis=0)
        self.payload_up: torch.Tensor = torch_utils.quat_axis(self.payload_rot, axis=2)
        
        self.drone_rpos = vmap(cpos)(drone_pos, drone_pos)
        self.drone_rpos = vmap(off_diag)(self.drone_rpos)
        self.drone_pdist = torch.norm(self.drone_rpos, dim=-1, keepdim=True)
        payload_drone_rpos = self.payload_pos.unsqueeze(1) - drone_pos

        self.target_payload_rpos = self.payload_target_pos - self.payload_pos
        self.target_payload_rheading = self.payload_target_heading - self.payload_heading

        payload_state = torch.cat(
            [
                self.target_payload_rpos,  # 3
                self.target_payload_rheading,  # 3
                self.payload_rot,  # 4
                payload_vels,  # 6
                self.payload_heading,  # 3
                self.payload_up, # 3
            ],
            dim=-1,
        ).unsqueeze(1)

        obs = TensorDict({}, [self.num_envs, self.drone.n])
        obs["self"] = torch.cat(
            [-payload_drone_rpos, self.drone_states[..., 3:]], dim=-1
        ).unsqueeze(2) # [..., 1, state_dim]
        obs["others"] = torch.cat(
            [self.drone_rpos, self.drone_pdist, vmap(others)(self.drone_states[..., 3:])], dim=-1
        ) # [..., n-1, state_dim + 1]
        obs["payload"] = payload_state.expand(-1, self.drone.n, -1).unsqueeze(2) # [..., 1, 22]

        state = TensorDict({}, self.num_envs)
        state["payload"] = payload_state # [..., 1, 22]
        state["drones"] = obs["self"].squeeze(2) # [..., n, state_dim]

        pos_error = torch.norm(self.target_payload_rpos, dim=-1, keepdim=True)
        heading_alignment = torch.sum(
            self.payload_heading * self.payload_target_heading, dim=-1, keepdim=True
        )
        self.info["payload_pos_error"].mul_(self.alpha).add_((1-self.alpha) * pos_error)
        self.info["heading_alignment"].mul_(self.alpha).add_((1-self.alpha) * heading_alignment)
        self.info["uprightness"].mul_(self.alpha).add_((1-self.alpha) * self.payload_up[:, 2].unsqueeze(-1))

        return TensorDict({
            "drone.obs": obs, 
            "drone.state": state,
            "info": self.info
        }, self.num_envs)

    def _compute_reward_and_done(self):
        vels = self.payload.get_velocities()

        distance = torch.norm(
            torch.cat([self.target_payload_rpos, self.target_payload_rheading], dim=-1)
        , dim=-1, keepdim=True)
        separation = self.drone_pdist.min(dim=-2).values.min(dim=-2).values

        reward = torch.zeros(self.num_envs, self.drone.n, 1, device=self.device)
        reward_pose = 1 / (1 + torch.square(distance * self.reward_distance_scale))
        # reward_pose = torch.exp(-distance * self.reward_distance_scale)

        up = self.payload_up[:, 2]
        reward_up = torch.square((up + 1) / 2).unsqueeze(-1)

        spinnage = vels[:, -3:].abs().sum(-1, keepdim=True)
        reward_spin = self.reward_spin_weight * torch.exp(-torch.square(spinnage))

        swing = vels[:, :3].abs().sum(-1, keepdim=True)
        reward_swing = self.reward_swing_weight * torch.exp(-torch.square(swing))

        reward_effort = self.reward_effort_weight * torch.exp(-self.effort).mean(-1, keepdim=True)
        reward_separation = torch.square(separation / self.safe_distance).clamp(0, 1)

        reward[:] = (
            reward_separation * (
                reward_pose 
                + reward_pose * (reward_up + reward_spin + reward_swing) 
                + reward_effort
            )
        ).unsqueeze(-1)

        done_hasnan = torch.isnan(self.drone_states).any(-1)
        done_fall = self.drone_states[..., 2] < 0.2

        done = (
            (self.progress_buf >= self.max_episode_length).unsqueeze(-1) 
            | done_fall.any(-1, keepdim=True)
            | done_hasnan.any(-1, keepdim=True)
        )

        self._tensordict["return"] += reward
        return TensorDict(
            {
                "reward": {"drone.reward": reward},
                "return": self._tensordict["return"],
                "done": done,
            },
            self.batch_size,
        )

