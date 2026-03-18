# 环境配置记录

## isaacsim 安装

    # 设置 ROS 版本为 jazzy
    export ROS_DISTRO=jazzy

    # 选择 RMW 实现（推荐 FastDDS）
    export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

    # 将 Isaac Sim 内置的库路径添加到环境变量中
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/huangyucheng/miniconda3/envs/isaac/lib/python3.11/site-packages/isaacsim/exts/isaacsim.ros2.bridge/jazzy/lib

## 


# 测试运行记录

## 训练示例1 (Shadowing WholeBody)

    conda activate isaac && cd InstinctLab

    # 可视化retargeting数据
    python scripts/amass_visualize.py

    # 训练
    python scripts/instinct_rl/train.py --task=Instinct-Shadowing-WholeBody-Plane-G1-Play-v0 --headless
    tensorboard --logdir logs/instinct_rl/g1_shadowing
    # 恢复训练，自动加载最新的 checkpoint
    python scripts/instinct_rl/train.py --task=Instinct-Shadowing-WholeBody-Plane-G1-Play-v0 --headless --resume --load_run 20260307_190718
    python scripts/instinct_rl/train.py --task=Instinct-Shadowing-WholeBody-Plane-G1-Play-v0 --resume --load_run 20260307_190718
    #  恢复训练，指定具体的权重文件
    python scripts/instinct_rl/train.py --task=Instinct-Shadowing-WholeBody-Plane-G1-Play-v0 --headless --resume --load_run 20260307_190718 --checkpoint model_2000.pt

    可视化：带坐标系的是参考机器人
    # 推理
    python scripts/instinct_rl/play.py --task=Instinct-Shadowing-WholeBody-Plane-G1-Play-v0 --load_run 20260307_190718


### 训练指标观察指南 (TensorBoard)
1. **核心生存指标 (优先看)**
   - `Train/mean_episode_length`: 必须稳步上升。数值越大代表机器人坚持模仿的时间越长，不因误差过大而倒地。
   - `Train/mean_reward`: 整体趋势应向上，代表综合模仿精度和动作平滑度在提升。

2. **模仿精度指标 (决定动作像不像)**
   - `Episode_Reward/rewards_link_pos_imitation_gauss`: 最关键！代表手脚等末端位置的跟随精度。
   - `Episode_Reward/rewards_base_position_imitation_gauss`: 躯干中心位置的跟随精度。

3. **动作质量指标 (决定动作稳不稳)**
   - `Episode_Reward/rewards_action_rate_l2`: 惩罚项（负值）。绝对值应逐渐减小，代表动作从“高频抖动”变得平滑。
   - `Episode_Reward/rewards_undesired_contacts`: 惩罚项。数值回升代表非必要碰撞（如膝盖着地、摔倒）在减少。

4. **失败原因诊断**
   - 查看 `Episode_Termination/link_pos_too_far`: 若接近 1.0，说明机器人主要死于手脚动作跟不上参考轨迹。


## 训练任务2 (Parkour)


### Parkour 任务训练架构 (AMP 机制)

Parkour 任务采用了 **WasabiPPO (AMP - Adversarial Motion Priors)** 框架，有三个协同工作的神经网络。

#### 1. 核心网络架构与目标

| 网络名称 | 训练目标 (Objective) | 核心 Loss 函数 |
| :--- | :--- | :--- |
| **策略网络 (Policy/Actor)** | 在完成越障任务的同时，使动作风格逼近人类参考数据。 | **PPO Loss** |
| **判别器 (Discriminator)** | 学习区分机器人的实时动作序列与人类参考动作序列。 | **Binary Cross-Entropy (BCE)** + **Gradient Penalty** |
| **价值网络 (Critic)** | 预测当前状态下的长期期望回报，辅助策略网络更新。 | **Mean Squared Error (MSE)** |

##### 详细训练目标与 Loss 函数说明

这三个网络共同构成了 **AMP (Adversarial Motion Priors)** 框架：

**A. 策略网络 (Policy Network / Actor)**
*   **训练目标**：在复杂地形上完成任务（如达到目标速度、不摔倒），同时动作要符合参考风格。
*   **Loss 函数**：使用的是 **PPO Loss**。策略网络通过**最大化包含风格奖励的优势函数 ($Adv$)** 来学习。
    *   $Loss_{actor} = - \mathbb{E} \left[ \min(ratio \cdot Adv, \text{clip}(ratio, 1-\epsilon, 1+\epsilon) \cdot Adv) \right] - \alpha \cdot H(policy)$
    *   **优势函数 ($Adv$) 与奖励的关系**：$Adv$ 是对 **Reward 的时空累积表现** 的评估。在计算时，首先将任务奖励与风格奖励加权合并为总奖励 $r_t$，再通过 GAE（广义优势估计）计算出 $Adv$。
        1.  **任务奖励**（Task Reward）：如速度跟踪、生存奖励、足端不穿模等，促使机器人完成越障目标。
        2.  **风格奖励**（Style Reward / AMP Reward）：来自判别器，公式为 $r_{style} = -\log(1 - D(s, s'))$。判别器 $D$ 越认为当前动作序列 $(s, s')$ 接近真人参考数据，给出的奖励越高。
    *   **GAE 优势函数具体计算逻辑**：
        根据 `instinct_rl/instinct_rl/storage/rollout_storage.py` 中的实现，优势函数 $Adv$ 是通过**倒序遍历**时间步计算的：
        1.  **计算 TD 误差 ($\delta_t$)**：$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$。它表示当前奖励加上未来的预估价值与当前预估价值的差。
        2.  **累积优势 ($Adv_t$)**：$Adv_t = \delta_t + (\gamma \lambda) Adv_{t+1}$。这里 $\gamma$ 是折扣因子（如 0.99），$\lambda$ 是 GAE 参数（如 0.95）。
        3.  **时空累积的含义**：这种递归计算方式意味着 $Adv_t$ 不仅取决于当前的奖励 $r_t$，还包含了未来所有步的奖励衰减总和。它衡量了在状态 $s_t$ 下采取动作 $a_t$ 相比于平均水平（$V(s_t)$）在**未来长期内**能多拿多少分。
    *   **目标函数说明**：与 SAC 最大化 $Q(s, a)$ 不同，PPO 通过最大化 $Adv$（即 $Q(s, a) - V(s)$）来更新。减去基准线 $V(s)$ 能显著降低梯度方差，使训练更稳定。
    *   $\alpha \cdot H(policy)$ 是熵正则项，用于鼓励策略探索，防止模型过早陷入局部最优。

**B. 判别器网络 (Discriminator)**
*   **训练目标**：区分“机器人的动作序列”和“人类参考动作序列”。它像一个严格的裁判，寻找机器人动作中不自然的地方。
*   **Loss 函数**：使用的是 GAN 风格的二分类 Loss（代码中通常为 MSE 或 BCE）。
    *   $Loss_{disc} = (D(s_{robot}) - 0)^2 + (D(s_{human}) - 1)^2 + \lambda \cdot GP$
    *   $GP$ 是梯度惩罚（Gradient Penalty），用于稳定训练，防止判别器过强导致策略无法学习。

**C. 价值网络 (Critic Network)**
*   **训练目标**：预测当前状态下的长期期望总回报（Value Estimation）。它辅助 Actor 学习，减少训练中的方差。
*   **Loss 函数**：使用的是 **MSE Loss**。
    *   $Loss_{critic} = (V(s) - R_{target})^2$
    *   $R_{target}$ 是通过 TD 误差或 GAE 计算的实际回报目标。

#### 2. 训练输入空间 (Input Space)

**A. 策略网络 (Policy) 的输入 (Observations)**
*   **本体感知 (Proprioception)**：基座角速度、重力投影、关节位置/速度、上一步动作（均包含 **8 帧历史**，用于捕捉动态特征）。
*   **任务指令 (Commands)**：目标线速度 $x, y$ 和目标角速度 $z$。
*   **视觉感知 (Exteroception)**：**深度图 (Depth Image)**。通过头部相机获取，经过裁剪 (Crop)、缩放 (Resize)、高斯模糊和归一化处理，提供前方地形的几何特征。

**B. 判别器 (Discriminator) 的输入 (AMP Observations)**
*   **输入特征**：仅包含运动学状态，**不含**任务指令和视觉信息。
*   **当前状态序列**：机器人的重力投影、关节位置/速度、基座线/角速度（包含 **10 帧历史**）。
*   **参考状态序列**：从 `run_walk` 动作库（AMASS/LAFAN1）中随机抽取的相同维度的状态序列。 怎么理解这个随机抽取？（训练数据量大概多少？代码中有没有参考？）

**C. 价值网络 (Critic) 的输入**
*   **特权信息**：在策略网络输入的基础上，额外增加了 **base_lin_vel (基座实际线速度)**。这作为“特权信息”有助于 Critic 更准确地评估状态价值，显著降低训练方差。

#### 3. 训练策略定义位置

*   **环境与奖励定义**：`source/instinctlab/instinctlab/tasks/parkour/config/parkour_env_cfg.py`
    *   定义了复杂地形生成器 (`ROUGH_TERRAINS_CFG`)。
    *   定义了任务奖励（速度跟踪）与风格奖励（AMP 奖励）的权重分配。
*   **算法与超参数定义**：`source/instinctlab/instinctlab/tasks/parkour/config/g1/agents/instinct_rl_amp_cfg.py`
    *   指定算法为 `WasabiPPO`，并配置判别器的网络结构、学习率及梯度惩罚系数。




## 训练任务3 (Perceptive Shadowing)

### 任务训练架构 (机制)

Perceptive Shadowing 任务旨在让机器人通过视觉感知（深度图）在非平整地形上精确模仿人类参考动作。它采用了带视觉编码器的 **PPO** 算法框架。

#### 1. 核心网络架构与目标

| 网络名称 | 角色 | 训练目标 (Objective) | 核心 Loss 函数 |
| :--- | :--- | :--- | :--- |
| **策略网络 (Policy/Actor)** | 执行者 | 结合视觉感知与参考动作，输出关节控制量，实现地形自适应的动作模仿。 | **PPO Clipped Surrogate Loss** |
| **价值网络 (Critic)** | 评价员 | 评估当前状态（含地形高度图）的价值，辅助策略网络更新。 | **MSE Loss** (或 Clipped Value Loss) |

*   **算法特性**：使用标准 PPO 算法。策略网络包含一个 **Conv2d 视觉编码器**，用于处理深度图像。
*   **训练目标**：最大化模仿奖励（关节、基座、末端位姿对齐）与任务奖励（生存、平滑度），同时利用视觉信息避开或适应地形起伏。

#### 2. 训练输入空间 (Input Space)

**A. 策略网络 (Policy) 的输入 (Observations)**
*   **参考动作 (Reference)**：`joint_pos_ref`, `joint_vel_ref`, `position_ref`, `rotation_ref`（来自 `motion_reference` 数据集）。
*   **视觉感知 (Exteroception)**：**深度图 (Depth Image)**。由头部相机获取，分辨率经过下采样（如 18x32），提供前方地形的几何细节。
*   **本体感知 (Proprioception)**：重力投影、基座角速度、关节位置/速度、上一步动作（均包含 **8 帧历史**）。

**B. 价值网络 (Critic) 的输入**
*   **参考与本体信息**：包含策略网络的所有参考动作和本体感知项。
*   **特权/地形信息**：
    *   **高度扫描 (Height Scan)**：机器人足端周围 1.6m x 1.0m 范围内的地面高度采样点。
    *   **基座线速度 (Base Lin Vel)**：基座在世界坐标系下的实际线速度。
    *   **关键链接位姿**：手脚等关键 link 的位置与旋转。
*   **注意**：在此任务配置中，Critic 通常不直接看深度图，而是通过更直观的 `height_scan` 获取地形信息。

### 3. 训练策略定义位置

*   **环境与奖励定义**：`source/instinctlab/instinctlab/tasks/shadowing/perceptive/perceptive_env_cfg.py`
    *   定义了视觉传感器、高度扫描仪以及模仿奖励项。
*   **算法与编码器定义**：`source/instinctlab/instinctlab/tasks/shadowing/perceptive/config/g1/agents/instinct_rl_ppo_cfg.py`
    *   定义了 `Conv2dHeadEncoderCfg`（视觉编码器结构）和 PPO 超参数。
*   **地形匹配配置**：`source/instinctlab/instinctlab/tasks/shadowing/perceptive/config/g1/perceptive_shadowing_cfg.py`
    *   定义了 `TerrainMotionCfg`，确保加载的动作与地形 mesh 严格对齐。




### 运行指令：

    # GMR格式转换为训练输入数据的标准格式
    python scripts/GMR_to_instinct.py \
    --src "/home/huangyucheng/桌面/Project Instinct/InstinctLab/MOTION_data/test_motion_data_GMR" \
    --tgt "/home/huangyucheng/桌面/Project Instinct/InstinctLab/MOTION_data/test_motion_data_stairs" \
    --urdf "source/instinctlab/instinctlab/assets/resources/unitree_g1/urdf/g1_29dof_torsobase_popsicle.urdf" 

    转换流程：
    (1) 坐标系中心点的物理偏移: base_pos_w 和 base_quat_w 从 pelvis 到 torso_link（躯干）的变换。
    (2) 关节数值的取反: 不同软件（如 HoloSoma vs IsaacSim）对关节旋转正方向的定义（左手系 vs 右手系，或轴向定义）可能相反。默认对象：waist_yaw, waist_roll, waist_pitch。
    (3) 关节顺序的重排


    # 可视化retargeting数据 （纯动力学检查）
    python scripts/amass_visualize.py --num_envs 1 --motion_path "/home/huangyucheng/桌面/Project Instinct/InstinctLab/MOTION_data/test_motion_data_stairs/stairs031_retargeted.npz" --print_foot_pos --print_interval 20 --print_foot_pos
    python scripts/amass_visualize.py --num_envs 1 --motion_path "/home/huangyucheng/桌面/Project Instinct/InstinctLab/MOTION_data/test_motion_data_stairs/" --interactive 
    11：三级，弯腰。 13：四级，弯腰。15：三级，弯腰。20：四级，弯腰。27: 四级。28：四级。29:四级。31：四级。

    0.41
    


    # 生成地形阶梯stl文件
    python scripts/generate_staircase_terrain.py

    # 训练
    python scripts/instinct_rl/train.py --headless --task=Instinct-Perceptive-Shadowing-G1-v0
    python scripts/instinct_rl/train.py \
    --task=Instinct-Perceptive-Shadowing-G1-v0 \
    --num_envs=1 \
    env.scene.terrain.terrain_generator.num_rows=1 \
    env.scene.terrain.terrain_generator.num_cols=1 \
    'env.events.reset_robot.params.randomize_pose_range.x=[0.0,0.0]' \
    'env.events.reset_robot.params.randomize_pose_range.y=[0.0,0.0]' \
    'env.events.reset_robot.params.randomize_pose_range.z=[0.0,0.0]' \
    'env.events.reset_robot.params.randomize_joint_pos_range=[0.0,0.0]' \
    env.scene.motion_reference.motion_buffers.TerrainMotion.env_starting_stub_sampling_strategy=independent \
    'env.scene.motion_reference.motion_buffers.TerrainMotion.motion_start_from_middle_range=[0.0,0.0]' \
    env.scene.motion_reference.motion_buffers.TerrainMotion.motion_bin_length_s=null \
    env.scene.motion_reference.motion_buffers.TerrainMotion.fix_origin_index=0 \
    env.curriculum.beyond_adaptive_sampling=null \
    env.events.bin_fail_counter_smoothing=null \
    env.scene.height_scanner.debug_vis=True 


    env.terminations.base_pg_too_far=null \
    env.terminations.link_pos_too_far=null \
    env.terminations.base_pos_too_far=null \
    env.terminations.illegal_reset_contact=null



    ****训练配置****：
    G1 robot 位置初始的随机化扰动：/home/huangyucheng/桌面/Project Instinct/InstinctLab/source/instinctlab/instinctlab/motion_reference/motion_files/terrain_motion.py

    # 查看训练曲线
    tensorboard --logdir=tensorboard --logdir=logs/instinct_rl/g1_perceptive_shadowing



    # 恢复训练，自动加载最新的 checkpoint
    python scripts/instinct_rl/train.py --task=Instinct-Perceptive-Shadowing-G1-v0 --headless --resume --load_run 20260309_164036
    #  恢复训练，指定具体的权重文件
    python scripts/instinct_rl/train.py --task=Instinct-Perceptive-Shadowing-G1-v0 --headless --resume --load_run 20260309_164036 --checkpoint model_2000.pt

    可视化：带坐标系的是参考机器人
    # 推理
    python scripts/instinct_rl/play.py --task=Instinct-Perceptive-Shadowing-G1-Play-v0 --load_run 20260309_164036 --num_envs 1 


    #注：修改InstinctLab/source/instinctlab/instinctlab/tasks/shadowing/perceptive/config/g1/perceptive_shadowing_cfg.py的MOTION_FOLDER路径
    python scripts/instinct_rl/play.py --task=Instinct-Perceptive-Shadowing-G1-Play-v0 --load_run 20260121_085042 --num_envs 1 




    tmux new -s train
    tmux attach -t train
    tmux kill-session -t train



  



