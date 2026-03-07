# 环境配置记录

## isaacsim 安装

    # 设置 ROS 版本为 jazzy
    export ROS_DISTRO=jazzy

    # 选择 RMW 实现（推荐 FastDDS）
    export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

    # 将 Isaac Sim 内置的库路径添加到环境变量中
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/huangyucheng/miniconda3/envs/isaac/lib/python3.11/site-packages/isaacsim/exts/isaacsim.ros2.bridge/jazzy/lib

## 