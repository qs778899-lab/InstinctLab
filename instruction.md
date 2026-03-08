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

## 训练示例1 (动作模仿)

    conda activate isaac && cd InstinctLab
    python scripts/instinct_rl/train.py --task=Instinct-Shadowing-WholeBody-Plane-G1-Play-v0 --headless
    tensorboard --logdir logs/instinct_rl/g1_shadowing
    # 恢复训练，自动加载最新的 checkpoint
    python scripts/instinct_rl/train.py --task=Instinct-Shadowing-WholeBody-Plane-G1-Play-v0 --headless --resume --load_run 20260307_190718
    python scripts/instinct_rl/train.py --task=Instinct-Shadowing-WholeBody-Plane-G1-Play-v0 --resume --load_run 20260307_190718
    #  恢复训练，指定具体的权重文件
    python scripts/instinct_rl/train.py --task=Instinct-Shadowing-WholeBody-Plane-G1-Play-v0 --headless --resume --load_run 20260307_190718 --checkpoint model_2000.pt

    可视化：带坐标系的是参考机器人

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