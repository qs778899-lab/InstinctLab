from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def register_virtual_obstacle_to_sensor(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    sensor_cfgs: list[SceneEntityCfg] | SceneEntityCfg,
    obstacle_names: list[str] | str | None = None, #vertical
):
    """Make each sensor accessible to the terrain virtual obstacle by providing `sensor.register_virtual_obstacles` with
    `terrain.virtual_obstacles` dict.

    """
    if isinstance(sensor_cfgs, SceneEntityCfg):
        sensor_cfgs = [sensor_cfgs]

    virtual_obstacles: dict = env.scene.terrain.virtual_obstacles

    #vertical
    if obstacle_names is None:
        selected_virtual_obstacles = virtual_obstacles
    else:
        if isinstance(obstacle_names, str):
            obstacle_names = [obstacle_names]
        selected_virtual_obstacles = {
            obstacle_name: virtual_obstacles[obstacle_name]
            for obstacle_name in obstacle_names
            if obstacle_name in virtual_obstacles
        }

    for sensor_cfg in sensor_cfgs:
        sensor = env.scene[sensor_cfg.name]
        if not hasattr(sensor, "register_virtual_obstacles"):
            raise ValueError(f"Sensor {sensor_cfg.name} does not support virtual obstacles.")

        sensor.register_virtual_obstacles(selected_virtual_obstacles) #vertical
