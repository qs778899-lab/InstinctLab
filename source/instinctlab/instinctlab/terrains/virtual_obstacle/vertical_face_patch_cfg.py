from __future__ import annotations

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.utils import configclass

from .vertical_face_patch import VerticalFacePatch
from .virtual_obstacle_base import VirtualObstacleCfg


@configclass
class VerticalFacePatchCfg(VirtualObstacleCfg):
    """Configuration for extracting vertical terrain faces as thin box patches."""

    class_type: type = VerticalFacePatch

    normal_z_threshold: float = 0.2
    """Maximum absolute z component of the face normal to be considered vertical."""

    coplanar_angle_threshold_deg: float = 10.0
    """Maximum normal mismatch for merging adjacent triangles into one patch."""

    coplanar_distance_threshold: float = 0.01
    """Maximum point-to-plane distance when merging adjacent triangles."""

    min_triangle_area: float = 1.0e-5
    """Minimum triangle area to keep as a valid vertical-face candidate."""

    min_patch_width: float = 0.05
    """Minimum width of the merged patch in meters."""

    min_patch_height: float = 0.04
    """Minimum height of the merged patch in meters."""

    patch_thickness: float = 0.02
    """Thickness of the debug/rendered thin box representing the vertical face."""

    visualizer: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/verticalFacePatches",
        markers={
            "patch": sim_utils.MeshCuboidCfg(
                size=(1.0, 1.0, 1.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.25, 0.25), opacity=0.18),
            ),
        },
    )
