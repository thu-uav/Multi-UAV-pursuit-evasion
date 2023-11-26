import torch
import numpy as np
import functorch
from torchrl.data import UnboundedContinuousTensorSpec, CompositeSpec
from tensordict.tensordict import TensorDict, TensorDictBase
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import wandb
import time
from functorch import vmap
from omni_drones.utils.torch import cpos, off_diag, others
import torch.distributions as D

import omni.isaac.core.objects as objects
# from omni.isaac.core.objects import VisualSphere, DynamicSphere, FixedCuboid, VisualCylinder, FixedCylinder, DynamicCylinder
# from omni.isaac.core.prims import RigidPrimView, GeometryPrimView
import omni.isaac.core.prims as prims
from omni_drones.views import RigidPrimView
from omni_drones.envs.isaac_env import IsaacEnv, AgentSpec
from omni_drones.robots.config import RobotCfg
from omni_drones.robots.drone import MultirotorBase
import omni_drones.utils.kit as kit_utils
# import omni_drones.utils.restart_sampling as rsp
from pxr import UsdGeom, Usd, UsdPhysics
import omni.isaac.core.utils.prims as prim_utils
import omni.physx.scripts.utils as script_utils
from omni_drones.utils.scene import design_scene
from .utils import create_obstacle
import pdb

# drones on land by default
# only cubes are available as walls

class PredatorPrey_debug(IsaacEnv): 
    def __init__(self, cfg, headless):
        super().__init__(cfg, headless)
        self.drone.initialize()

        self.target = RigidPrimView(
            "/World/envs/env_*/target", 
            reset_xform_properties=False
        )
        self.target.initialize()
        
        self.obstacles = RigidPrimView(
            "/World/envs/env_*/obstacle_*",
            reset_xform_properties=False,
            shape=[self.num_envs, -1],
            # track_contact_forces=True
        )
        self.obstacles.initialize()
        
        self.time_encoding = self.cfg.task.time_encoding

        self.target_init_vel = self.target.get_velocities(clone=True)
        self.env_ids = torch.from_numpy(np.arange(0, cfg.env.num_envs))
        self.size_min = self.cfg.size_min
        self.size_max = self.cfg.size_max
        self.size_dist = D.Uniform(
            torch.tensor([self.size_min], device=self.device),
            torch.tensor([self.size_max], device=self.device)
        )
        self.caught = self.progress_buf * 0
        self.returns = self.progress_buf * 0
        self.catch_radius = self.cfg.catch_radius
        self.collision_radius = self.cfg.collision_radius
        self.init_poses = self.drone.get_world_poses(clone=True)
        self.v_low = self.cfg.v_drone * self.cfg.v_low
        self.v_high = self.cfg.v_drone * self.cfg.v_high
        self.v_obstacle_min = self.cfg.v_drone * self.cfg.v_obstacle_min
        self.v_obstacle_max = self.cfg.v_drone * self.cfg.v_obstacle_max
        self.obstacle_control_fre = self.cfg.obstacle_control_fre

        # CL
        # self.goals = self.create_goalproposal_mix()
        
        drone_state_dim = self.drone.state_spec.shape.numel()
        frame_state_dim = 9 # target_pos_dim + target_vel
        if self.time_encoding:
            self.time_encoding_dim = 4
            frame_state_dim += self.time_encoding_dim        

        observation_spec = CompositeSpec({
            "state_self": UnboundedContinuousTensorSpec((1, 3 + 6 + drone_state_dim + self.drone.n)),
            "state_others": UnboundedContinuousTensorSpec((self.drone.n-1, 3)), # pos
            "state_frame": UnboundedContinuousTensorSpec((1, frame_state_dim)),
            "obstacles": UnboundedContinuousTensorSpec((self.num_obstacles, 6)), # pos + vel
        }).to(self.device)
        state_spec = CompositeSpec({
            "state_drones": UnboundedContinuousTensorSpec((self.drone.n, 3 + 6 + drone_state_dim + self.drone.n)),
            "state_frame": UnboundedContinuousTensorSpec((1, frame_state_dim)),
            "obstacles": UnboundedContinuousTensorSpec((self.num_obstacles, 6)),
        }).to(self.device)
        
        self.agent_spec["drone"] = AgentSpec(
            "drone",
            self.drone.n,
            observation_spec,
            self.drone.action_spec.to(self.device),
            UnboundedContinuousTensorSpec(1).to(self.device),
            state_spec
        )  
        
        self.task_config = TensorDict({
            "max_linvel": torch.ones(self.num_envs, 1, device=self.device)
        }, self.num_envs)

        # infos
        info_spec = CompositeSpec({
            "capture": UnboundedContinuousTensorSpec(1),
            "capture_episode": UnboundedContinuousTensorSpec(1),
            "capture_per_step": UnboundedContinuousTensorSpec(1),
            "cover_rate": UnboundedContinuousTensorSpec(1),
            "return": UnboundedContinuousTensorSpec(1),
            "drone1_speed_per_step": UnboundedContinuousTensorSpec(1),
            "drone2_speed_per_step": UnboundedContinuousTensorSpec(1),
            "drone3_speed_per_step": UnboundedContinuousTensorSpec(1),
            "drone1_max_speed": UnboundedContinuousTensorSpec(1),
            "drone2_max_speed": UnboundedContinuousTensorSpec(1),
            "drone3_max_speed": UnboundedContinuousTensorSpec(1),
            "prey_speed": UnboundedContinuousTensorSpec(1),
        }).expand(self.num_envs).to(self.device)
        self.observation_spec["info"] = info_spec
        self.info = info_spec.zero()
        
    def _design_scene(self):
        self.num_agents = self.cfg.num_agents
        self.num_obstacles = self.cfg.num_obstacles
        self.obstacle_size = self.cfg.obstacle_size
        self.size_min = self.cfg.size_min
        self.size_max = self.cfg.size_max

        # init drone
        drone_model = MultirotorBase.REGISTRY[self.cfg.task.drone_model]
        cfg = drone_model.cfg_cls(force_sensor=self.cfg.task.force_sensor)
        cfg.rigid_props.max_linear_velocity = self.cfg.v_drone
        self.drone: MultirotorBase = drone_model(cfg=cfg)

        translation = torch.zeros(self.num_agents, 3)
        translation[:, 0] = torch.arange(self.num_agents)
        translation[:, 1] = torch.arange(self.num_agents)
        translation[:, 2] = 0.5
        self.drone.spawn(translation)

        # init prey
        self.target_pos = torch.tensor([[0., 0.05, 0.5]], device=self.device)
        objects.DynamicSphere(
            prim_path="/World/envs/env_0/target",
            name="target",
            translation=self.target_pos,
            radius=0.05,
            # height=0.1,
            color=torch.tensor([1., 0., 0.]),
            mass=1.0
        )

        # init obstacle
        obstacle_pos = torch.zeros(self.num_obstacles, 3)
        size_dist = D.Uniform(
            torch.tensor([self.size_min], device=self.device),
            torch.tensor([self.size_max], device=self.device)
        )
        size = size_dist.sample().item()
        random_pos_dist = D.Uniform(
            torch.tensor([-size, -size, 0.0], device=self.device),
            torch.tensor([size, size, 0.0], device=self.device)
        )
        obstacle_pos = random_pos_dist.sample(obstacle_pos.shape[:-1])
        for idx in range(self.num_obstacles):
            objects.DynamicSphere(
                prim_path="/World/envs/env_0/obstacle_{}".format(idx),
                name="obstacle_{}".format(idx),
                translation=obstacle_pos[idx],
                radius=0.05,
                color=torch.tensor([1., 1., 1.]),
                mass=1.0
            )
        
        objects.VisualCuboid(
            prim_path="/World/envs/env_0/ground",
            name="ground",
            translation= torch.tensor([0., 0., 0.], device=self.device),
            scale=torch.tensor([size * 2, size * 2, 0.001], device=self.device),
            color=torch.tensor([0., 0., 0.]),
        )
    
        kit_utils.set_rigid_body_properties(
            prim_path="/World/envs/env_0/target",
            disable_gravity=True
        )        

        kit_utils.create_ground_plane(
            "/World/defaultGroundPlane",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        )

        return ["/World/defaultGroundPlane"]
    
    def _reset_idx(self, env_ids: torch.Tensor):
        n = self.num_agents
        init_pos, rot = self.init_poses
        self.drone._reset_idx(env_ids)

        n_envs = len(env_ids)
        drone_pos = []
        obstacle_pos = []
        target_pos = []
        self.size_list = []
        # reset size
        for idx in range(n_envs):
            size = self.size_dist.sample().item()
            
            self.size_list.append(size)
        
            drone_pos_dist = D.Uniform(
                torch.tensor([-size, -size, 0.0], device=self.device),
                torch.tensor([size, size, 2 * size], device=self.device)
            )
            drone_pos.append(drone_pos_dist.sample((1,n)))

            target_pos_dist = D.Uniform(
                torch.tensor([-size, -size, 0.0], device=self.device),
                torch.tensor([size, size, 2 * size], device=self.device)
            )
            target_pos.append(target_pos_dist.sample())

            obstacles_pos_dist = D.Uniform(
                torch.tensor([-size, -size, 0.0], device=self.device),
                torch.tensor([size, size, 2 * size], device=self.device)
            )
            obstacle_pos.append(obstacles_pos_dist.sample((1, self.num_obstacles)))

        drone_pos = torch.concat(drone_pos, dim=0)
        target_pos = torch.stack(target_pos, dim=0)
        obstacle_pos = torch.concat(obstacle_pos, dim=0)
        self.size_list = torch.Tensor(np.array(self.size_list)).to(self.device)
        
        # set position and velocity
        self.drone.set_world_poses(
            drone_pos + self.envs_positions[env_ids].unsqueeze(1), rot[env_ids], env_ids
        )
        drone_init_velocities = torch.zeros_like(self.drone.get_velocities())
        self.drone.set_velocities(torch.zeros_like(drone_init_velocities), env_ids)
        self.drone_sum_speed = drone_init_velocities[...,0].squeeze(-1)
        self.drone_max_speed = drone_init_velocities[...,0].squeeze(-1)

        # set target
        self.target.set_world_poses((self.envs_positions + target_pos)[env_ids], env_indices=env_ids)
        target_vel = self.target.get_velocities()
        self.target.set_velocities(2 * torch.rand_like(target_vel) - 1, self.env_ids)

        # obstalces
        self.obstacles.set_world_poses(
            (obstacle_pos + self.envs_positions[env_ids].unsqueeze(1))[env_ids], env_indices=env_ids
        )
        
        # obstacles begin to move
        # self.obstacles_start_move = torch.randint(0, self.max_episode_length // 2, size = (n_envs, self.num_obstacles)).to(self.device)
        
        # reset velocity of prey
        self.v_prey = torch.from_numpy(np.random.uniform(self.v_low, self.v_high, [self.num_envs, 1])).to(self.device)
        # reset velocity of obstacles
        self.v_obstacle = torch.from_numpy(np.random.uniform(self.v_obstacle_min, self.v_obstacle_max, [self.num_envs, 1])).to(self.device)
        
        # reset info
        info_spec = CompositeSpec({
            "capture": UnboundedContinuousTensorSpec(1),
            "capture_episode": UnboundedContinuousTensorSpec(1),
            "capture_per_step": UnboundedContinuousTensorSpec(1),
            "cover_rate": UnboundedContinuousTensorSpec(1),
            "return": UnboundedContinuousTensorSpec(1),
            "drone1_speed_per_step": UnboundedContinuousTensorSpec(1),
            "drone2_speed_per_step": UnboundedContinuousTensorSpec(1),
            "drone3_speed_per_step": UnboundedContinuousTensorSpec(1),
            "drone1_max_speed": UnboundedContinuousTensorSpec(1),
            "drone2_max_speed": UnboundedContinuousTensorSpec(1),
            "drone3_max_speed": UnboundedContinuousTensorSpec(1),
            "prey_speed": UnboundedContinuousTensorSpec(1),
        }).expand(self.num_envs).to(self.device)
        self.info = info_spec.zero()
        self.step_spec = 0

    def _pre_sim_step(self, tensordict: TensorDictBase):
        self.step_spec += 1
        actions = tensordict[("action", "drone.action")]
        self.effort = self.drone.apply_action(actions)
        
        target_vel = self.target.get_velocities()
        forces_target = self._get_dummy_policy_prey()
        
        # fixed velocity
        target_vel[:,:3] = self.v_prey * forces_target / (torch.norm(forces_target, dim=1).unsqueeze(1) + 1e-5)
        
        self.target.set_velocities(target_vel.type(torch.float32), self.env_ids)
        
        # set obstacles vel with fre = self.obstacle_control_fre
        obstacles_vel = self.obstacles.get_velocities()
    
        new_obstacle_flag = (self.progress_buf % self.obstacle_control_fre == 0).unsqueeze(1).expand(-1, self.num_obstacles).unsqueeze(-1)
        new_obstacles_vel = obstacles_vel.clone()
        direction_vel = 2 * torch.rand(self.num_envs, self.num_obstacles, 3) - 1
        direction_vel = (direction_vel / torch.norm(direction_vel, dim=-1).unsqueeze(-1)).to(self.device)
        new_obstacles_vel[:,:,:3] = torch.mul(self.v_obstacle.unsqueeze(1).expand(-1, self.num_obstacles, -1), direction_vel)
        
        obstacles_vel = new_obstacles_vel * new_obstacle_flag + obstacles_vel * ~(new_obstacle_flag)
        self.obstacles.set_velocities(obstacles_vel.type(torch.float32), self.env_ids)
        
        # clip, if out of area
        obstacle_pos, _ = self.get_env_poses(self.obstacles.get_world_poses())
        min_values = torch.stack([-self.size_list, -self.size_list, torch.zeros_like(self.size_list)], dim=-1).unsqueeze(1).expand(-1, self.num_obstacles, -1)  # min for each dim, shape=(envs, num_obstacles, 3)
        max_values = torch.stack([self.size_list, self.size_list, 2 * self.size_list], dim=-1).unsqueeze(1).expand(-1, self.num_obstacles, -1)  # max for each dim
        obstacle_pos = torch.clamp(obstacle_pos, min_values, max_values)
        self.obstacles.set_world_poses(
            (obstacle_pos + self.envs_positions[self.env_ids].unsqueeze(1))[self.env_ids], env_indices=self.env_ids
        )

    def _compute_state_and_obs(self):
        self.drone_states = self.drone.get_state()
        drone_pos = self.drone_states[..., :3]
        self.drone_rpos = vmap(cpos)(drone_pos, drone_pos)
        self.drone_rpos = vmap(off_diag)(self.drone_rpos)
        drone_vel = self.drone.get_velocities()
        
        drone_speed_norm = torch.norm(drone_vel[..., :3], dim=-1)
        self.drone_sum_speed += drone_speed_norm
        self.drone_max_speed = torch.max(torch.stack([self.drone_max_speed, drone_speed_norm], dim=-1), dim=-1).values
        self.info['drone1_speed_per_step'].set_(self.drone_sum_speed[:,0].unsqueeze(-1) / self.step_spec)
        self.info['drone2_speed_per_step'].set_(self.drone_sum_speed[:,1].unsqueeze(-1) / self.step_spec)
        self.info['drone3_speed_per_step'].set_(self.drone_sum_speed[:,2].unsqueeze(-1) / self.step_spec)
        self.info['drone1_max_speed'].set_(self.drone_max_speed[:,0].unsqueeze(-1))
        self.info['drone2_max_speed'].set_(self.drone_max_speed[:,1].unsqueeze(-1))
        self.info['drone3_max_speed'].set_(self.drone_max_speed[:,2].unsqueeze(-1))
        
        target_pos, _ = self.get_env_poses(self.target.get_world_poses())
        target_pos = target_pos.unsqueeze(1)
        target_vel = self.target.get_velocities()
        self.info["prey_speed"].set_(torch.norm(target_vel[:, :3], dim=-1).unsqueeze(-1))
        target_rpos = target_pos - self.drone_states[..., :3]
        if self.time_encoding:
            t = (self.progress_buf / self.max_episode_length).unsqueeze(-1)
            target_state = torch.cat([
                target_pos,
                target_vel.unsqueeze(1),
                t.expand(-1, self.time_encoding_dim).unsqueeze(1)
            ], dim=-1) # [num_envs, 1, 25+time_encoding_dim]
        else:
            target_state = torch.cat([
                target_pos,
                target_vel.unsqueeze(1)
            ], dim=-1) # [num_envs, 1, 25]

        identity = torch.eye(self.drone.n, device=self.device).expand(self.num_envs, -1, -1)

        obs = TensorDict({}, [self.num_envs, self.drone.n])
        obs["state_self"] = torch.cat(
            [-target_rpos, target_vel.unsqueeze(1).expand(-1, self.drone.n, -1), self.drone_states, identity], dim=-1
        ).unsqueeze(2)
        obs["state_others"] = self.drone_rpos
        obs["state_frame"] = target_state.unsqueeze(1).expand(-1, self.drone.n, 1, -1)
        
        obstacle_pos, _ = self.get_env_poses(self.obstacles.get_world_poses())
        obstacles_vel = self.obstacles.get_velocities()[...,:3]
        # obstacle_rpos + vel
        obs["obstacles"] = torch.concat([vmap(cpos)(drone_pos, obstacle_pos), obstacles_vel.unsqueeze(1).expand(-1, self.drone.n, -1, -1)], dim=-1)

        state = TensorDict({}, [self.num_envs])
        state["state_drones"] = obs["state_self"].squeeze(2)    # [num_envs, drone.n, drone_state_dim]
        state["state_frame"] = target_state                # [num_envs, 1, target_rpos_dim]
        state["obstacles"] = torch.concat([obstacle_pos, obstacles_vel], dim=-1)            # [num_envs, num_obstacles, obstacles_dim]
        return TensorDict(
            {
                "drone.obs": obs,
                "drone.state": state,
                "info": self.info,
            },
            self.batch_size,
        )

    def _compute_reward_and_done(self):
        drone_pos, _ = self.drone.get_world_poses()
        target_pos, _ = self.target.get_world_poses()
        target_pos = target_pos.unsqueeze(1)

        target_dist = torch.norm(target_pos - drone_pos, dim=-1)

        capture_flag = (target_dist < self.catch_radius)
        self.info['capture_episode'].add_(torch.sum(capture_flag, dim=1).unsqueeze(-1))
        self.info['capture'].set_(torch.from_numpy(self.info['capture_episode'].to('cpu').numpy() > 0.0).type(torch.float32).to(self.device))
        self.info['cover_rate'].set_((torch.sum(self.info['capture'] >= 0.95) / self.num_envs).unsqueeze(-1).expand_as(self.info['capture']))
        self.info['capture_per_step'].set_(self.info['capture_episode'] / self.step_spec)
        catch_reward = 10 * capture_flag.sum(-1).unsqueeze(-1).expand_as(capture_flag)

        # speed penalty
        if self.cfg.use_speed_penalty:
            drone_vel = self.drone.get_velocities()
            drone_speed_norm = torch.norm(drone_vel[..., :3], dim=-1)
            speed_reward = - 100 * (drone_speed_norm > self.cfg.v_drone)
        else:
            speed_reward = 0.0

        # collison with obstacles
        coll_reward = torch.zeros(self.num_envs, self.num_agents, device=self.device)
        
        obstacle_pos, _ = self.obstacles.get_world_poses()
        for i in range(self.num_obstacles):
            relative_pos = drone_pos[..., :2] - obstacle_pos[:, i, :2].unsqueeze(-2)
            norm_r = torch.norm(relative_pos, dim=-1)
            if_coll = (norm_r < (self.collision_radius + self.obstacle_size)).type(torch.float32)
            coll_reward -= if_coll # sparse

        # distance reward
        min_dist = (torch.min(target_dist, dim=-1)[0].unsqueeze(-1).expand_as(target_dist))
        dist_reward_mask = (min_dist > self.catch_radius)
        distance_reward = - 1.0 * min_dist * dist_reward_mask

        if self.cfg.use_collision:
            reward = speed_reward + 1.0 * catch_reward + 1.0 * distance_reward + 5 * coll_reward
        else:
            reward = speed_reward + 1.0 * catch_reward + 1.0 * distance_reward
        
        self._tensordict["return"] += reward.unsqueeze(-1)
        self.returns = self._tensordict["return"].sum(1)
        self.info["return"].set_(self.returns)

        done  = (
            (self.progress_buf >= self.max_episode_length).unsqueeze(-1)
        )
        
        caught = (catch_reward > 0) * 1.0
        self.caught = (self.progress_buf > 0) * ((self.caught + caught.any(-1)) > 0)
        self.progress_std = torch.std(self.progress_buf)

        return TensorDict({
            "reward": {
                "drone.reward": reward.unsqueeze(-1)
            },
            "done": done,
            # "caught": self.caught * 1.0,
            # "return": self._tensordict["return"],
            # "success": self.success*torch.ones(self.num_envs, device=self.device),
            # "random_success": self.random_success*torch.ones(self.num_envs, device=self.device),
            # "train_success": self.train_success*torch.ones(self.num_envs, device=self.device),
            # "mean_level": self.goals.mean_level/self.cfg.v_drone*torch.ones(self.num_envs, device=self.device),
            # "num_level": num_level,
            # "max_score": self.goals.max_score*torch.ones(self.num_envs, device=self.device),
            # "min_score": self.goals.min_score*torch.ones(self.num_envs, device=self.device),
            # "buffer_max": self.goals.buffer_max*torch.ones(self.num_envs, device=self.device),
            # "buffer_min": self.goals.buffer_min*torch.ones(self.num_envs, device=self.device),
            # "keep": self.goals.keep*torch.ones(self.num_envs, device=self.device),
            # "collision_times": self.coll_times.sum(-1), #就是coll_reward
            # "cl_level": self.cl_level*torch.ones(self.num_envs, device=self.device),
            # "progress_std": self.progress_std**torch.ones(self.num_envs, device=self.device)
        }, self.batch_size)

    def _get_dummy_policy_prey(self):
        pos, _ = self.drone.get_world_poses(False)
        prey_pos, _ = self.target.get_world_poses()
        prey_pos = prey_pos.unsqueeze(1)
        
        force = torch.zeros(self.num_envs, 3, device=self.device)

        # predators
        # active mask : if drone is failed, do not get force from it
        drone_vel = self.drone.get_velocities()
        active_mask = (torch.norm(drone_vel[...,:3],dim=-1) > 1e-5).unsqueeze(-1).expand(-1,-1,3)
        prey_pos_all = prey_pos.expand(-1,self.num_agents,-1)
        dist_pos = torch.norm(prey_pos_all - pos,dim=-1).unsqueeze(-1).expand(-1,-1,3)
        direction_p = (prey_pos_all - pos) / (dist_pos + 1e-5)
        # force_p = direction_p * (1 / (dist_pos + 1e-5)) * active_mask
        force_p = direction_p * (1 / (dist_pos + 1e-5))
        force += torch.sum(force_p, dim=1)

        # arena
        # 3D
        prey_env_pos, _ = self.get_env_poses(self.target.get_world_poses())
        force_r = torch.zeros_like(force)
        force_r[...,0] = 1 / (prey_env_pos[:,0] - (- self.size_list) + 1e-5) - 1 / (self.size_list - prey_env_pos[:,0] + 1e-5)
        force_r[...,1] = 1 / (prey_env_pos[:,1] - (- self.size_list) + 1e-5) - 1 / (self.size_list - prey_env_pos[:,1] + 1e-5)
        force_r[...,2] += 1 / (prey_env_pos[:,2] - 0 + 1e-5) - 1 / (2 * self.size_list - prey_env_pos[:,2] + 1e-5)
        force += force_r

        # obstacles
        obstacle_pos, _ = self.obstacles.get_world_poses()
        dist_pos = torch.norm(prey_pos[..., :3] - obstacle_pos[..., :3],dim=-1).unsqueeze(-1).expand(-1, -1, 3) # expand to 3-D
        direction_o = (prey_pos[..., :3] - obstacle_pos[..., :3]) / (dist_pos + 1e-5)
        force_o = direction_o * (1 / (dist_pos + 1e-5))
        force[..., :3] += torch.sum(force_o, dim=1)

        # set force_z to 0
        return force.type(torch.float32)
    