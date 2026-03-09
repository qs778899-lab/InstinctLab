from __future__ import annotations

import torch
from dataclasses import MISSING
from typing import TYPE_CHECKING, Literal

from isaaclab.utils import configclass

if TYPE_CHECKING:
    from collections.abc import Callable

from instinctlab.motion_reference.motion_reference_cfg import MotionBufferCfg

from .amass_motion_cfg import AmassMotionCfg
from .terrain_motion import TerrainMotion


@configclass
class TerrainMotionCfg(AmassMotionCfg):
    """Configuration for terrain motion data, which is typically terrain-dependent."""

    class_type: type = TerrainMotion

    metadata_yaml: str = MISSING
    """YAML file containing the motion matching configuration.
    Please refer to the `MotionMatchedTerrainCfg` for the expected structure.
    """

    max_origins_per_motion: int = 16
    """ Due to the subterrain design, each terrain_id will be generated to multiple subterrains. Thus, each motion
    could be put on any of these subterrains. However, to improve the sample efficiency, we need to vectorize the
    sampling of the origins. This parameter controls the maximum number of origins per motion.
    """

    fix_origin_index: int = -1
    """If set to a non-negative integer (e.g., 0), always use this fixed index into the available origins instead of sampling randomly.
    Useful for debugging to eliminate position randomization from terrain origin selection.
    Set to -1 to disable and sample randomly (default).
    """

    def __post_init__(self) -> None:
        """Post-initialization to ensure the motion matching YAML file is set."""
        assert (
            self.filtered_motion_selection_filepath is None
        ), "TerrainMotionCfg does not support filtered_motion_selection_filepath. Please use metadata_yaml instead."
