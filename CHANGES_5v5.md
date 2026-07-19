# 5v5 多猎物改造记录

## 概述

将 HideAndSeek 环境从 **5 drone vs 1 prey** 改造为 **5 drone vs 5 prey**（真正 5v5），同时修复了原代码中的 shape 不匹配 bug、添加了 curriculum learning（猎物速度从慢到快自动增长）、适配了 PartialAttentionEncoder 的观测维度。

---

## 修改文件清单

| 文件 | 改动量 | 说明 |
|------|--------|------|
| `omni_drones/envs/hide_and_seek/hideandseek.py` | ~200 行 | 核心改动：环境逻辑、观测、奖励、猎物策略、初始化 |
| `cfg/task/HideAndSeek.yaml` | 1 行 | `v_prey: 1.3 → 0.5` |
| `omni_drones/learning/mappo.py` | 3 行 | TP_groundtruth 适配多猎物维度 |

---

## 一、`hideandseek.py` 详细改动

### 1.1 独立函数：`is_perpendicular_line_intersecting_segment`

**位置**：第 48 行

**改动前签名**：
```python
def is_perpendicular_line_intersecting_segment(a, b, c):
    # a: [batch, num_drones, 3]
    # b: [batch, 1, 3]          ← 单猎物
    # c: [batch, num_cylinders, 3]
```

**改动后签名**：
```python
def is_perpendicular_line_intersecting_segment(a, b, c):
    # a: [batch, num_drones, 3]
    # b: [batch, num_prey, 3]   ← 多猎物
    # c: [batch, num_cylinders, 3]
```

**改动内容**：对 `a` 和 `b` 增加 unsqueeze 操作以支持 `(E, D, P, C)` 维度的 broadcasting：
```python
a_exp = a.unsqueeze(2)   # (E, D, 1, 3)
b_exp = b.unsqueeze(1)   # (E, 1, P, 3)
c_exp = c.unsqueeze(1).unsqueeze(2)  # (E, 1, 1, C, 3)
```

**返回值**：`(batch, num_drones, num_prey, num_cylinders)` — 每对 drone-prey-cylinder 的投影判断。

---

### 1.2 独立函数：`is_line_blocked_by_cylinder`

**位置**：第 72 行

**改动前**：
```python
# drone_pos: [num_envs, num_agents, 3]
# target_pos: [num_envs, 1, 3]        ← 单猎物
# cylinder_pos: [num_envs, num_cylinders, 3]
def is_line_blocked_by_cylinder(...):
    diff = drone_pos - target_pos           # (E, D, 3)
    diff2 = cylinder_pos - target_pos       # (E, C, 3)
    numerator = torch.abs(
        torch.matmul(diff[..., 0].unsqueeze(-1), diff2[..., 1].unsqueeze(1)) -
        torch.matmul(diff[..., 1].unsqueeze(-1), diff2[..., 0].unsqueeze(1))
    )
```

**改动后**：
```python
# drone_pos: [num_envs, num_agents, 3]
# target_pos: [num_envs, num_prey, 3]   ← 多猎物
# cylinder_pos: [num_envs, num_cylinders, 3]
def is_line_blocked_by_cylinder(...):
    diff  = drone_pos.unsqueeze(2) - target_pos.unsqueeze(1)     # (E, D, P, 3)
    diff2 = cylinder_pos.unsqueeze(1).unsqueeze(2) - target_pos.unsqueeze(1).unsqueeze(3)  # (E, 1, P, C, 3)
    # 关键修复：diff2[...].unsqueeze(1) 会导致 broadcasting 时维度翻倍
    # 改用直接广播：(E, D, P, 1) * (E, 1, P, C) → (E, D, P, C)
    numerator = torch.abs(
        diff[..., 0].unsqueeze(-1) * diff2[..., 1] -
        diff[..., 1].unsqueeze(-1) * diff2[..., 0]
    )
```

**修复的 bug**：`diff2[...].unsqueeze(1)` 在旧版中会将 `(E, 1, P, C)` 变成 `(E, 1, 1, P, C)`，与 `diff(..., 0).unsqueeze(-1)` 相乘时，PyTorch broadcasting 右对齐导致 `E` 维度被复制，结果变成 `(E, E, D, P, C)`，产生维度爆炸。去掉 `.unsqueeze(1)`，直接依靠 broadcasting：`(E, D, P, 1) × (E, 1, P, C) → (E, D, P, C)`。

**返回值**：`blocked.any(dim=-1)` → `(num_envs, num_agents, num_prey)`

---

### 1.3 `__init__` 方法

**位置**：第 237 行

| 改动 | 改动前 | 改动后 | 原因 |
|------|--------|--------|------|
| target 视图路径 | `"/World/envs/env_*/target"` | `"/World/envs/env_*/target_*"` | 通配符匹配多个 `target_0`...`target_4` |
| target 视图 shape | `shape=[self.num_envs, -1]` | 不变 | `-1` 自动推断 5 个 target |
| `self.capture` | `torch.zeros(self.num_envs, 3)` | `torch.zeros(self.num_envs, self.num_prey)` | 每个猎物独立记录捕获状态 |
| `self.masked_target_pos` | 无 | `self.mask_value * torch.ones(self.num_envs, self.max_prey - self.num_prey, 3)` | 多猎物 TP net 输入 padding |
| TP net 输入维度 | `1 + 3 + 3 + 3 * max_agents` | `1 + 3*max_prey + 3*max_prey + 3*max_agents` | 多猎物 pos + vel 各占 `3*max_prey` |
| TP net 输出维度 | `3 * future_prediction_step` | `3 * num_prey * future_prediction_step` | 预测所有猎物的未来位置 |

---

### 1.4 `_set_specs` 方法

**位置**：第 333 行

| 改动 | 改动前 | 改动后 |
|------|--------|--------|
| `state_self` 观测维度 | `(1, obs_dim)` | `(1, num_prey * obs_dim)` |
| `state_drones` 状态维度 | `(drone.n, obs_dim)` | `(drone.n, num_prey*(3+tp_dim) + drone_state_dim)` |
| TP spec: `TP_groundtruth` | `(1, 3)` | `(num_prey, 3)` |
| TP spec: `TP_input` | `1+3+3 + num_agents*3` | `1 + 3*num_prey + 3*num_prey + num_agents*3` |

**新增变量**：
```python
tp_dim = 3 * future_prediction_step  # TP 预测的未来步数
obs_per_prey = 3 + tp_dim + time_encoding_dim + 13  # 每猎物的观测特征数
state_self_dim = num_prey * obs_per_prey  # 平铺后的总特征数
state_drone_dim = num_prey * (3 + tp_dim) + 7 + 6 + time_encoding_dim  # 状态平铺
```

**关键设计**：`state_self` 用 `(1, state_self_dim)` 而非 `(num_prey, obs_per_prey)`，因为 SplitEmbedding 需要所有实体有相同维数来做 `torch.cat(dim=-2)`。`state_self` 的 1 是 dummy 实体维，`state_others` 有 `drone.n-1` 个实体，`cylinders` 有 `k` 个实体。

---

### 1.5 `_design_scene` 方法

**位置**：第 437 行

**新增变量定义**（必须在 `_design_scene` 内，因为它在 `super().__init__()` 中被调用）：
```python
self.num_prey = self.cfg.task.num_prey
self.max_prey = self.cfg.task.max_prey
```

**drone 初始位置**：从 4 个补到 5 个
```python
drone_pos = torch.tensor([
    [0.6000,  0.0000, 0.5],
    [0.8000,  0.0000, 0.5],
    [0.8000, -0.2000, 0.5],
    [0.8000,  0.2000, 0.5],
    [1.0000,  0.0000, 0.5],   # ← 新增第5个
], device=self.device)[:self.num_agents]
```

**prey spawn**：从 1 个改为 `num_prey` 个
```python
# 改动前：
objects.DynamicSphere(prim_path="/World/envs/env_0/target", name="target", ...)
# 改动后：
for i in range(self.num_prey):
    objects.DynamicSphere(
        prim_path="/World/envs/env_0/target_{}".format(i),
        name="target_{}".format(i),
        translation=target_pos[i],  # 每个 prey 独立位置
        ...
    )
```

**gravity disable**：同样改为循环处理所有 target。

---

### 1.6 `_reset_idx` 方法

**位置**：第 632 行

**随机采样分支**（`use_eval=0`）：
```python
# 改动前：target_pos 采样 (..., 1) — 单猎物
target_pos = self.init_target_pos_dist.sample((*env_ids.shape, 1))
# 改动后：target_pos 采样 (..., num_prey) — 多猎物
target_pos = self.init_target_pos_dist.sample((*env_ids.shape, self.num_prey))
```

**固定场景分支**（`use_eval=1`）：所有 10 个 scenario 的 `drone_pos` 从 4 个坐标补到 5 个，`target_pos` 统一改为 5 猎物的默认坐标组 `_default_targets`。

**`rejection_sampling_random_cylinder`**：`batch_indices.expand(-1, 1)` → `batch_indices.expand(-1, self.num_prey)`，修复 target_grid 的索引维度。

---

### 1.7 `_compute_state_and_obs` 方法

**位置**：第 875 行

**核心维度变化**：

| 变量 | 改动前 | 改动后 |
|------|--------|--------|
| `target_pos` | `(E, 1, 3)` | `(E, P, 3)` |
| `target_rpos` | `(E, D, 1, 3)` | `(E, D, P, 3)` |
| `self.blocked` | `(E, D)` | `(E, D, P)` |
| `in_detection_range` | `(E, D, 1)` | `(E, D, P)` |
| `self.broadcast_detect` | `(E, 1)` | `(E, P)` |
| `target_rpos_masked` | `(E, D, 1, 3)` | `(E, D, P, 3)` |
| `obs["state_self"]` | `(E, D, 1, obs_dim)` | `(E, D, P, obs_per_prey)` → reshape `(E, D, 1, P*obs_per_prey)` |

**detect 广播机制**：
```python
# 改动前：每个 drone 是否检测到 1 个猎物
detect = in_detection_range * (~ self.blocked.unsqueeze(-1))  # (E, D, 1)
self.broadcast_detect = torch.any(detect, dim=1)  # (E, 1)

# 改动后：每个 drone 是否检测到每个猎物
detect = in_detection_range * (~ self.blocked)  # (E, D, P)
self.broadcast_detect = detect.any(dim=1)  # (E, P) — 每个猎物是否被任何 drone 检测到
```

**TP 预测修复**：
```python
# 改动前（有 broadcasting bug）：
target_rpos_predicted = drone_pos.unsqueeze(2) - self.target_pos_predicted.unsqueeze(1)
# drone: (E, D, 1, 3) vs target: (E, 1, P, F, 3)
# → 维度数不等，broadcasting 导致 (E, E, D, P, F, 3) 的维度爆炸

# 改动后：
target_rpos_predicted = drone_pos.unsqueeze(2).unsqueeze(3) - self.target_pos_predicted.unsqueeze(1)
# drone: (E, D, 1, 1, 3) vs target: (E, 1, P, F, 3)
# → (E, D, P, F, 3) 正确
```

**观测拼接**：每个猎物独立拼接 target_rpos、TP 预测、drone state、time encoding，然后 reshape 为一维：`(E, D, P, obs_per_prey) → (E, D, 1, P*obs_per_prey)`

**`state["state_drones"]`**：target_rpos 平铺 `(E, D, P*3)`，不与 drone state 重复展开（和 `state_self` 不同）。

---

### 1.8 `_compute_reward_and_done` 方法

**位置**：第 1019 行

**距离奖励**：每 drone 与**最近**猎物的距离
```python
# 改动前：
target_dist = torch.norm(target_pos - drone_pos, dim=-1)  # (E, D) — 1 猎物
# 改动后：
target_dist = torch.norm(target_pos.unsqueeze(1) - drone_pos.unsqueeze(2), dim=-1)  # (E, D, P)
min_target_dist = target_dist.min(dim=-1).values  # (E, D) — 最近猎物
```

**捕捉奖励**：每猎物独立判断，合作奖励
```python
# 改动前：capture 是 (E, D) 每个 drone 是否捕获
self.capture = (target_dist < self.catch_radius)  # (E, D)

# 改动后：capture 是 (E, D, P) 每个 drone 对每个猎物
self.capture_matrix = (target_dist < self.catch_radius)  # (E, D, P)
broadcast_capture = torch.any(masked_capture, dim=1)  # (E, P) — 每个猎物是否被任何drone捕获
catch_reward_per_drone = self.catch_reward_coef * broadcast_capture.unsqueeze(1)  # (E, 1, P)
catch_reward = catch_reward_per_drone.sum(dim=-1)  # (E, D) — 所有猎物捕捉奖励求和
```

**检测奖励**：
```python
# broadcast_detect: (E, P) — 每个猎物是否被检测到
any_detect = self.broadcast_detect.any(dim=-1, keepdim=True)  # (E, 1)
detect_reward = self.detect_reward_coef * any_detect  # (E, 1)
```

**Curriculum Learning**（保留）：
```python
if torch.any(done):
    if self.stats["success"].mean() >= 0.98:
        self.v_prey += 0.05
        self.v_prey = min(1.3, self.v_prey)
```
当 success rate ≥ 98% 时，猎物速度 +0.05，直到上限 1.3 m/s。

---

### 1.9 `_get_dummy_policy_prey` 方法

**位置**：第 1163 行

**改动前**：单个猎物，受力为标量 `(E, 1, 3)`。

**改动后**：每个猎物独立计算斥力 `(E, P, 3)`。

```python
# 改动前：
force = torch.zeros(self.num_envs, 1, 3)  # 单猎物
force_p = drone斥力...force += force_p.sum(dim=1)  # 汇聚到单猎物

# 改动后：
force = torch.zeros(self.num_envs, self.num_prey, 3)  # 多猎物
force_p = drone斥力...  # (E, D, P, 3)
force += force_p.sum(dim=1)  # (E, P, 3) — 汇聚到各猎物
```

**arena 边界力**：`out_of_arena` 从 `(E,)` 改为 `(E, P)`，每猎物独立。
**圆柱斥力**：`cylinders_mask` broadcast 从 `(E, 1, C)` 改为 `(E, P, C)`。

---

### 1.10 `_pre_sim_step` 方法

**位置**：第 784 行

```python
# 改动前：
target_vel[..., :3] = self.v_prey * forces_target / (torch.norm(forces_target, dim=1).unsqueeze(1) + 1e-5)

# 改动后：norm 改为沿最后一维（pos 维）计算
target_vel[..., :3] = self.v_prey * forces_target / (torch.norm(forces_target, dim=-1, keepdim=True) + 1e-5)
```

**修复原因**：`forces_target` 旧 shape 是 `(E, 1, 3)`，`norm(dim=1)` 沿 prey 维 → 总是 1。新 shape 是 `(E, P, 3)`，`norm(dim=1)` 会错误地把 P 个猎物的力混在一起。改为 `dim=-1` 沿力矩维度计算。

---

### 1.11 `_draw_catch` 方法

**位置**：第 1284 行

适配可视化中的 `self.capture` 和 `self.blocked` 形状变化。

---

## 二、`HideAndSeek.yaml` 配置改动

```yaml
# 改动前
v_prey: 1.3

# 改动后
v_prey: 0.5  # curriculum learning 从低速 0.5 m/s 开始，success ≥ 98% 后自动 +0.05 到 1.3
```

**原因**：`1.3 m/s > 1.0 m/s (v_drone)`，无人机永远追不上猎物，positive reward 永远不会出现。改为 0.5 让无人机在早期能成功围捕。

---

## 三、`mappo.py` TP 训练适配

**位置**：`train_op` 方法，第 407 行

```python
# 新增：多猎物维度平铺
if TP_groundtruth.dim() == 4:  # (batch, time, num_prey, 3)
    TP_groundtruth = TP_groundtruth.reshape(*TP_groundtruth.shape[:-2], -1)  # → (batch, time, num_prey*3)
```

**原因**：TP_groundtruth 从单猎物的 `(batch, time, 3)` 变为 `(batch, time, num_prey, 3)`，而 unfold + windows 代码期望 3D 输入。将 prey 维度合并到 pos 维度避免修改下游代码。

---

## 四、运行命令

```bash
cd ~/hcx/Multi-UAV-pursuit-evasion && \
python scripts/train.py headless=true wandb.mode=offline \
  total_frames=50000000 task=HideAndSeek \
  task.use_eval=0 task.use_random_cylinder=1 \
  task.v_prey=0.5 task.collision_coef=20
```

| 参数 | 含义 |
|------|------|
| `task.use_eval=0` | 随机初始化位置（非固定场景） |
| `task.use_random_cylinder=1` | 随机障碍物数量 |
| `task.v_prey=0.5` | 猎物初始速度 0.5 m/s，curriculum 自动增长到 1.3 |
| `task.collision_coef=20` | 碰撞惩罚从 100 降到 20，减少负奖励压制 |
