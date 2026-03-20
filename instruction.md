# 环境配置记录

## isaacsim 安装

    # 设置 ROS 版本为 jazzy
    export ROS_DISTRO=jazzy

    # 选择 RMW 实现（推荐 FastDDS）
    export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

    # 将 Isaac Sim 内置的库路径添加到环境变量中
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/huangyucheng/miniconda3/envs/isaac/lib/python3.11/site-packages/isaacsim/exts/isaacsim.ros2.bridge/jazzy/lib

## isaaclab 安装



# 测试运行记录

## 训练示例1 (Shadowing WholeBody)

> **补充说明：BeyondMimic 与 WholeBody Shadowing 的区别**
> 
> 在 `Project Instinct` 中，平地上的盲追踪（无视觉/地形感知）分为两个任务版本，它们的核心区别在于**目标指令（Commands）的参考坐标系**以及**观察空间（Observations）的构建**：
> 
> 1. **BeyondMimic Shadowing** (`Instinct-BeyondMimic-Plane-G1-v0`)：
>    - **目标指令 (Commands)**：`position_ref_command` 的 `anchor_frame` 是 `"robot"`。这意味着机器人追踪的关键点（如手、脚、躯干）的 6D Pose 目标，是**相对于机器人当前身体坐标系**来计算偏差的。它更倾向于让机器人学习“如何移动我的身体来达到相对目标”。
>    - **观察空间 (Observations)**：
>      - 包含了 `base_lin_vel` (**基座线速度**)：即躯干在机体坐标系下的 $(v_x, v_y, v_z)$。这让策略能直接知道自己跑多快，从而在无历史帧的情况下判断是否跟上了参考动作。
>      - 无历史帧输入（只看当前帧）。
> 
> 2. **Whole Body Shadowing** (`Instinct-Shadowing-WholeBody-Plane-G1-v0`)：
>    - **目标指令 (Commands)**：除了相对于 `"robot"` 的指令外，增加了一个核心指令 `position_b_ref_command`（`anchor_frame` 为 `"reference"`）。这意味着关键点的 6D Pose 目标是**相对于参考动作（Reference Motion）的坐标系**计算的，让策略更好地理解全局轨迹，实现更精确的全身追踪。
>    - **观察空间 (Observations)**：
>      - 去除了 `base_lin_vel`，加入了 **`projected_gravity` (重力投影)**：即世界坐标系下的重力向量 $\mathbf{g}_w = [0, 0, -1]^T$ 通过机器人当前基座姿态四元数 $q_{base}$ 的逆变换，投影到机器人机体坐标系（Body Frame）下的三维向量。计算公式为 $\mathbf{g}_b = \text{quat\_rotate\_inv}(q_{base}, \mathbf{g}_w)$。该项提供了机器人相对于重力矢量的实时倾斜状态，是维持动态平衡的关键输入。
>      - （注：这两种纯盲追踪任务目前都**没有**使用历史帧，只依赖当前帧状态。历史帧通常在带感知的跑酷等复杂任务中通过 `flatten` 过去 8-10 帧的方式启用，用于隐式推断加速度和接触状态）。
> 

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
    --tgt "/home/huangyucheng/桌面/Project Instinct/InstinctLab/MOTION_data/test_motion_data_walk" \
    --urdf "source/instinctlab/instinctlab/assets/resources/unitree_g1/urdf/g1_29dof_torsobase_popsicle.urdf" 

    转换流程：
    (1) 坐标系中心点的物理偏移: base_pos_w 和 base_quat_w 从 pelvis 到 torso_link（躯干）的变换。
    (2) 关节数值的取反: 不同软件（如 HoloSoma vs IsaacSim）对关节旋转正方向的定义（左手系 vs 右手系，或轴向定义）可能相反。默认对象：waist_yaw, waist_roll, waist_pitch。
    (3) 关节顺序的重排


    # 可视化retargeting数据 （纯动力学检查）
    python scripts/amass_visualize.py --num_envs 1 --motion_path "/home/huangyucheng/桌面/Project Instinct/InstinctLab/MOTION_data/test_motion_data_stairs/stairs027_retargeted.npz" --print_foot_pos --print_interval 20 --print_foot_pos
    python scripts/amass_visualize.py --num_envs 1 --motion_path "/home/huangyucheng/桌面/Project Instinct/InstinctLab/MOTION_data/test_motion_data_stairs/" --interactive 
    11：三级，弯腰。 13：四级，弯腰。15：三级，弯腰。20：四级，弯腰。27: 四级。28：四级。29:四级。31：四级。
    # 可视化retargeting数据 （动作和地形模型对齐）
    python scripts/instinct_rl/play_vis.py --task=Instinct-Perceptive-Shadowing-G1-Play-v0 --load_run 20260319_115839 --num_envs 1 --freeze_policy --viewer_eye 5.8 3.0 0.2  --viewer_lookat 0.0 0.0 0.5

    python scripts/amass_visualize.py --num_envs 1 --motion_path "/home/huangyucheng/桌面/Project Instinct/InstinctLab/MOTION_data/test_motion_data_walk/SLP101_stageii_retargeted.npz" --print_foot_pos --print_interval 20 --print_foot_pos



    # 生成地形阶梯stl文件
    python scripts/generate_staircase_terrain.py --output_name stairs_terrain_scale_0.74.stl

    # 训练
    python scripts/instinct_rl/train.py --task=Instinct-Perceptive-Shadowing-G1-v0  --headless
    tmux new -s train
    tmux attach -t train
    tmux kill-session -t train
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
    权重文件保存频率：InstinctLab/source/instinctlab/instinctlab/tasks/shadowing/perceptive/config/g1/agents/instinct_rl_ppo_cfg.py
    # 查看训练曲线
    tensorboard --logdir=tensorboard --logdir=logs/instinct_rl/g1_perceptive_shadowing



    # 恢复训练，自动加载最新的 checkpoint （指定具体的权重文件： --checkpoint model_2000.pt）
    python scripts/instinct_rl/train.py --task=Instinct-Perceptive-Shadowing-G1-v0 --headless --resume --load_run 20260319_145701

 
    # 推理 （可视化：带坐标系的是参考机器人）
    ##注：修改InstinctLab/source/instinctlab/instinctlab/tasks/shadowing/perceptive/config/g1/perceptive_shadowing_cfg.py的MOTION_FOLDER路径
    python scripts/instinct_rl/play.py --task=Instinct-Perceptive-Shadowing-G1-Play-v0 --load_run 20260318_231039 --num_envs 1 --checkpoint model_25000.pt
    python scripts/instinct_rl/play.py --task=Instinct-Perceptive-Shadowing-G1-Play-v0 --load_run 20260319_174712 --num_envs 1 --checkpoint model_46000.pt
    python scripts/instinct_rl/play.py --task=Instinct-Perceptive-Shadowing-G1-Play-v0 --load_run 20260320_151148 --num_envs 1  --checkpoint model_5800.pt




    



  










    motion_files:
    - motion_file: stairs027_retargeted.npz
    terrain_id: 0
    - motion_file: stairs028_retargeted.npz
    terrain_id: 0
    - motion_file: stairs029_retargeted.npz
    terrain_id: 0
    - motion_file: stairs031_retargeted.npz
    terrain_id: 0
    terrains:
    - terrain_file: stairs_terrain.stl
    terrain_id: 0



    # TensorBoard 训练曲线分析（Instinct-Perceptive-Shadowing-G1-v0）
    重点：不要只看总 reward。优先看 Episode_Termination、Episode_Monitor、Episode_Reward、Episode/Curriculum，再看 Loss 和 Train。

    1. Train/*
    Train/mean_reward_0
    含义：平均回报，反映整体训练趋势。
    怎么看：只适合看总体是否在变好，不适合定位具体问题。两条轨迹 reward 接近，也可能失败原因完全不同。

    Train/mean_episode_length
    含义：平均 episode 长度。
    怎么看：越长通常说明越不容易提前失败。如果一条轨迹的这个值明显更低，通常说明更早触发 termination。

    Train/median_episode_length, min_episode_length, max_episode_length
    含义：episode 长度分布。
    怎么看：如果 mean 上升但 median 很低，说明只有少量环境学会，大部分环境仍然早死。

    2. Loss/*
    Loss/surrogate_loss
    含义：PPO 策略更新目标。
    怎么看：主要用来判断训练是否还在有效更新，不用单独追求某个绝对值。

    Loss/value_loss
    含义：价值函数误差。
    怎么看：长期很高，说明 critic 学不好，通常会拖慢策略收敛；如果换轨迹后只有这个明显变差，说明回报更难预测。

    Loss/entropy
    含义：策略随机性。
    怎么看：下降太快，可能过早收缩探索；一直很高，可能策略迟迟学不到稳定动作。

    Loss/learning_rate
    含义：自适应 PPO 学习率。
    怎么看：如果很快降到很小，通常说明 KL 压力大，策略更新受限；两条轨迹对比时，这条很有用。

    Train/grad_norm
    含义：梯度范数。
    怎么看：异常尖峰通常意味着优化不稳定。

    3. Episode_Termination/*
    这是最重要的一组。它直接回答“训练为什么失败”。

    Episode_Termination/link_pos_too_far
    含义：关键 link（主要是脚/手）偏离 reference 太远。
    怎么看：最值得重点看。如果这项高，说明不是简单的总 reward 低，而是关键身体部位先跟丢了。两条轨迹收敛差异大时，优先比较这条。

    Episode_Termination/base_pg_too_far
    含义：机身姿态（projected gravity）偏差过大。
    怎么看：高了通常表示台阶过程中的俯仰/翻滚控制困难，或者视觉感知无法及时支持姿态调整。

    Episode_Termination/base_pos_too_far
    含义：root/base 位置偏离 reference 太远。
    怎么看：高了说明全局根部轨迹没跟住，常见于步幅、相位、落脚节奏不对。

    Episode_Termination/out_of_border
    含义：机器人走出地形边界。
    怎么看：如果高，优先怀疑 terrain 对齐、起始 origin、步态漂移，而不是 imitation 本身。

    Episode_Termination/time_out
    含义：跑到 episode 上限才结束。
    怎么看：高通常是好事，说明能活到最后。

    Episode_Termination/dataset_exhausted
    含义：motion 播放完了。
    怎么看：高通常说明真的跟到了后段，而不是中途就失败。

    4. Episode_Monitor/*
    这是定位“到底哪里跟不好”的核心指标，比总 reward 更直接。

    Episode_Monitor/shadowing_link_pos_w_link_pos_error
    含义：世界坐标系下关键 links 的平均位置误差。
    怎么看：最推荐与 link_pos_too_far 一起看。能直接反映脚、手这些关键 link 的跟踪质量。

    Episode_Monitor/shadowing_link_pos_b_link_pos_error
    含义：base 坐标系下关键 links 的平均位置误差。
    怎么看：如果 world 误差高、base 误差不高，说明整体 root 没跟住；如果两者都高，说明局部姿态也没跟住。

    Episode_Monitor/shadowing_position_base_pos_error
    含义：base 位置误差。
    怎么看：判断整机根部的轨迹误差，适合配合 base_pos_too_far 一起看。

    Episode_Monitor/shadowing_rotation_base_rot_error
    含义：base 朝向误差。
    怎么看：高了说明姿态控制难，尤其在上台阶、过渡、弯腰、抬腿阶段更明显。

    Episode_Monitor/shadowing_joint_pos_joint_pos_error
    含义：关节角误差。
    怎么看：高了说明静态姿态跟踪就已经有问题。

    Episode_Monitor/shadowing_joint_vel_joint_vel_error
    含义：关节速度误差。
    怎么看：非常关键。如果它明显高于 joint_pos_error，通常说明问题在动态节奏、接触时机、爆发速度，而不是单纯姿态不对。

    Step_Monitor/*
    含义：逐 step 的即时误差。
    怎么看：适合找“在哪个时间段突然崩”。如果某条轨迹总是在同一段突然抬升，说明难点集中在那个 motion bin。

    5. Episode_Reward/*
    这组用来拆解“回报是靠什么起来的，又被什么拖下去的”。

    rewards_link_pos_imitation_gauss/sum
    含义：关键 links 位置模仿奖励。
    怎么看：最重要的正奖励之一。两条轨迹对比时，这条低往往对应 link 跟踪更差。

    rewards_link_rot_imitation_gauss/sum
    含义：关键 links 姿态模仿奖励。
    怎么看：如果 link_pos 还行但这个低，说明位置到了，姿态没到。

    rewards_link_lin_vel_imitation_gauss/sum
    rewards_link_ang_vel_imitation_gauss/sum
    含义：关键 links 线速度、角速度模仿奖励。
    怎么看：动态难轨迹最该看这两条。速度项差，通常意味着动作切换、落脚、抬腿阶段更难学。

    rewards_base_position_imitation_gauss/sum
    rewards_base_rot_imitation_gauss/sum
    含义：root/base 的位置和姿态模仿奖励。
    怎么看：如果这两条还行，但 link 奖励差，说明整体走向没问题，细节 limb 跟踪有问题。

    rewards_action_rate_l2/sum
    含义：动作变化惩罚。
    怎么看：绝对值过大说明策略动作过抖，可能为了跟快动作在频繁修正。

    rewards_joint_limit/sum
    含义：关节接近极限的惩罚。
    怎么看：如果某条轨迹这项更差，常说明该动作本身更逼近机器人硬件极限。


    rewards_undesired_contacts/sum （数值异常高）
    含义：非期望接触惩罚。
    怎么看：高了通常是身体碰撞、绊台阶、手臂或躯干误碰环境。
    具体判定逻辑（见 perceptive_env_cfg.py）：
    使用正则 `^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)(?!left_wrist_yaw_link$)(?!right_wrist_yaw_link$).+$`
    即：白名单放行了双脚（ankle）和双手（wrist）。
    任何其他部位（如小腿/膝盖磕碰台阶边缘、躯干摔地、手肘撞墙、甚至严重的四肢自我碰撞）只要受力超过 1.0N，都会触发此惩罚。
    如果此项负分极大，说明机器人频繁发生“膝盖磕台阶”或“躯干摔地”。
    Episode_Reward/rewards_undesired_contacts/max_episode_len_s
    Episode_Reward/rewards_undesired_contacts/sum
    Episode_Reward/rewards_undesired_contacts/timestep

    rewards_applied_torque_limits_by_ratio/sum  （数值异常高）
    含义：扭矩接近限制的惩罚。
    怎么看：如果高，说明控制带宽或 actuator 能力接近上限，往往对应更激烈的 motion。



    6. Episode/Curriculum/*
    这一组非常重要，用来定位“整条轨迹的哪一段最难”。

    Episode/Curriculum/beyond_adaptive_sampling/sampling_entropy
    含义：当前 motion bin 采样分布的熵。
    怎么看：越低表示采样越集中，说明 curriculum 已经认为某些 bin 明显更难。

    Episode/Curriculum/beyond_adaptive_sampling/sampling_top1_prob
    含义：最难 bin 当前被采样到的概率。
    怎么看：越高表示训练越聚焦在单个困难时间段。

    Episode/Curriculum/beyond_adaptive_sampling/sampling_top1_bin
    含义：最难 bin 在整条 motion 中的大致相对位置。
    怎么看：例如 0.40 表示难点大约在整条轨迹 40% 的位置附近。然后可以去 play 对应轨迹，重点观察那一段发生了什么。

    7. 两条不同轨迹收敛差异时，建议优先比较的顺序
    第一优先级：
    Episode_Termination/link_pos_too_far
    Episode_Monitor/shadowing_link_pos_w_link_pos_error
    Episode_Monitor/shadowing_joint_vel_joint_vel_error

    第二优先级：
    Episode_Termination/base_pg_too_far
    Episode_Monitor/shadowing_rotation_base_rot_error
    Episode_Reward/rewards_link_lin_vel_imitation_gauss/sum
    Episode_Reward/rewards_link_ang_vel_imitation_gauss/sum

    第三优先级：
    Episode/Curriculum/beyond_adaptive_sampling/sampling_entropy
    Episode/Curriculum/beyond_adaptive_sampling/sampling_top1_prob
    Episode/Curriculum/beyond_adaptive_sampling/sampling_top1_bin

    第四优先级：
    Loss/value_loss
    Loss/learning_rate
    Train/mean_episode_length

    8. 常见结论模板
    如果一条轨迹主要是 link_pos_too_far 高：
    说明关键 limb 先跟丢，优先怀疑脚/手的时序、落脚点、台阶匹配。

    如果一条轨迹主要是 base_pg_too_far 高：
    说明姿态稳定性更难，优先怀疑上台阶过程中的 pitch/roll 控制。

    如果 joint_vel_error 明显高于 joint_pos_error：
    说明难点主要在动态速度和接触切换，而不是静态姿态。

    如果 sampling_top1_prob 越来越高、entropy 越来越低：
    说明 curriculum 已经锁定某一段最难，应该去可视化这段 motion。

    如果 mean_reward 在涨，但 mean_episode_length 不涨：
    说明策略可能只是在局部 reward 上改善，但并没有真正延长存活时间。

