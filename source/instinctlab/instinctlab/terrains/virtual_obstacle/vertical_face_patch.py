from __future__ import annotations

import math
from collections import deque

import numpy as np
import torch
import trimesh

import isaaclab.utils.math as math_utils
from isaaclab.markers import VisualizationMarkers

from .virtual_obstacle_base import VirtualObstacleBase


class VerticalFacePatch(VirtualObstacleBase):
    """Extract near-vertical terrain faces and represent them as thin OBB patches."""

    def __init__(self, cfg):
        super().__init__(cfg)

    def generate(self, mesh: trimesh.Trimesh, device: torch.device | str = "cpu") -> None:
        self.device = device if isinstance(device, torch.device) else torch.device(device)

        face_normals = np.asarray(mesh.face_normals)
        face_areas = np.asarray(mesh.area_faces)
        face_centers = np.asarray(mesh.triangles_center)

        candidate_mask = (np.abs(face_normals[:, 2]) <= self.cfg.normal_z_threshold) & (
            face_areas >= self.cfg.min_triangle_area
        )

        if not np.any(candidate_mask):
            self._set_empty_buffers()
            return

        adjacency = np.asarray(mesh.face_adjacency)
        cos_threshold = math.cos(math.radians(self.cfg.coplanar_angle_threshold_deg))
        adjacency_list: dict[int, list[int]] = {}

        for face_a, face_b in adjacency:
            if not (candidate_mask[face_a] and candidate_mask[face_b]):
                continue

            normal_dot = float(np.dot(face_normals[face_a], face_normals[face_b]))
            if normal_dot < cos_threshold:
                continue

            plane_dist_a = abs(np.dot(face_centers[face_b] - face_centers[face_a], face_normals[face_a]))
            plane_dist_b = abs(np.dot(face_centers[face_a] - face_centers[face_b], face_normals[face_b]))
            if max(plane_dist_a, plane_dist_b) > self.cfg.coplanar_distance_threshold:
                continue

            adjacency_list.setdefault(int(face_a), []).append(int(face_b))
            adjacency_list.setdefault(int(face_b), []).append(int(face_a))

        candidate_indices = np.flatnonzero(candidate_mask)
        visited: set[int] = set()
        patches: list[dict[str, np.ndarray | float]] = []

        for face_idx in candidate_indices:
            face_idx = int(face_idx)
            if face_idx in visited:
                continue

            group = self._collect_component(face_idx, adjacency_list, candidate_mask, visited)
            patch = self._fit_patch(mesh, np.asarray(group, dtype=np.int64), face_normals)
            if patch is not None:
                patches.append(patch)

        if not patches:
            self._set_empty_buffers()
            return

        centers = np.stack([patch["center"] for patch in patches], axis=0)
        rotations = np.stack([patch["rotation"] for patch in patches], axis=0)
        half_extents = np.stack([patch["half_extents"] for patch in patches], axis=0)

        self.patch_centers = torch.tensor(centers, dtype=torch.float32, device=self.device)
        self.patch_rotations = torch.tensor(rotations, dtype=torch.float32, device=self.device)
        self.patch_half_extents = torch.tensor(half_extents, dtype=torch.float32, device=self.device)
        self.patch_quats = math_utils.quat_from_matrix(self.patch_rotations)

    def disable_visualizer(self) -> None:
        if hasattr(self, "_patch_visualizer"):
            self._patch_visualizer.set_visibility(False)

    def visualize(self):
        if not hasattr(self, "patch_centers") or self.patch_centers.numel() == 0:
            return

        if not hasattr(self, "_patch_visualizer"):
            self._patch_visualizer = VisualizationMarkers(self.cfg.visualizer)

        scales = 2.0 * self.patch_half_extents
        self._patch_visualizer.visualize(
            translations=self.patch_centers,
            orientations=self.patch_quats,
            scales=scales,
        )
        self._patch_visualizer.set_visibility(True)

    def get_points_penetration_offset(self, points: torch.Tensor) -> torch.Tensor:
        if not hasattr(self, "patch_centers") or self.patch_centers.numel() == 0:
            return torch.zeros_like(points)

        local_points = torch.einsum(
            "kji,nkj->nki",
            self.patch_rotations,
            points[:, None, :] - self.patch_centers[None, :, :],
        )  # (N, K, 3)
        margin = self.patch_half_extents[None, :, :] - torch.abs(local_points)
        inside_mask = torch.all(margin > 0.0, dim=-1)  # (N, K)
        if not torch.any(inside_mask):
            return torch.zeros_like(points)

        min_margin, axis_idx = torch.min(margin, dim=-1)  # (N, K)
        best_patch = torch.argmax(torch.where(inside_mask, min_margin, torch.full_like(min_margin, -1.0)), dim=-1)
        best_inside = inside_mask[torch.arange(points.shape[0], device=points.device), best_patch]

        offsets = torch.zeros_like(points)
        if not torch.any(best_inside):
            return offsets

        inside_indices = torch.nonzero(best_inside, as_tuple=False).squeeze(-1)
        selected_patch = best_patch[inside_indices]
        selected_axis = axis_idx[inside_indices, selected_patch]
        selected_margin = min_margin[inside_indices, selected_patch]
        selected_local = local_points[inside_indices, selected_patch]
        selected_rot = self.patch_rotations[selected_patch]

        axis_dirs_local = torch.zeros((inside_indices.shape[0], 3), dtype=points.dtype, device=points.device)
        axis_dirs_local.scatter_(1, selected_axis.unsqueeze(-1), 1.0)
        axis_sign = torch.sign(
            selected_local[torch.arange(inside_indices.shape[0], device=points.device), selected_axis]
        ).unsqueeze(-1)
        axis_sign[axis_sign == 0.0] = 1.0
        axis_dirs_world = torch.einsum("nij,nj->ni", selected_rot, axis_dirs_local * axis_sign)
        offsets[inside_indices] = axis_dirs_world * selected_margin.unsqueeze(-1)
        return offsets

    def _set_empty_buffers(self) -> None:
        self.patch_centers = torch.empty((0, 3), dtype=torch.float32, device=self.device)
        self.patch_rotations = torch.empty((0, 3, 3), dtype=torch.float32, device=self.device)
        self.patch_half_extents = torch.empty((0, 3), dtype=torch.float32, device=self.device)
        self.patch_quats = torch.empty((0, 4), dtype=torch.float32, device=self.device)

    @staticmethod
    def _collect_component(
        start_face: int,
        adjacency_list: dict[int, list[int]],
        candidate_mask: np.ndarray,
        visited: set[int],
    ) -> list[int]:
        queue = deque([start_face])
        group: list[int] = []

        while queue:
            face_idx = queue.popleft()
            if face_idx in visited or not candidate_mask[face_idx]:
                continue
            visited.add(face_idx)
            group.append(face_idx)
            queue.extend(adjacency_list.get(face_idx, []))

        return group

    def _fit_patch(self, mesh: trimesh.Trimesh, face_indices: np.ndarray, face_normals: np.ndarray) -> dict | None:
        unique_vertices = np.unique(mesh.faces[face_indices].reshape(-1))
        points = np.asarray(mesh.vertices[unique_vertices], dtype=np.float64)
        if points.shape[0] < 3:
            return None

        normal = np.mean(face_normals[face_indices], axis=0)
        normal_norm = np.linalg.norm(normal)
        if normal_norm < 1.0e-8:
            return None
        normal = normal / normal_norm

        world_z = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        vertical_axis = world_z - np.dot(world_z, normal) * normal
        if np.linalg.norm(vertical_axis) < 1.0e-8:
            fallback = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            vertical_axis = fallback - np.dot(fallback, normal) * normal
        vertical_axis = vertical_axis / np.linalg.norm(vertical_axis)
        horizontal_axis = np.cross(vertical_axis, normal)
        horizontal_axis = horizontal_axis / np.linalg.norm(horizontal_axis)

        u_coords = points @ horizontal_axis
        v_coords = points @ vertical_axis
        n_coords = points @ normal

        width = float(u_coords.max() - u_coords.min())
        height = float(v_coords.max() - v_coords.min())
        if width < self.cfg.min_patch_width or height < self.cfg.min_patch_height:
            return None

        center = (
            horizontal_axis * (0.5 * (u_coords.max() + u_coords.min()))
            + vertical_axis * (0.5 * (v_coords.max() + v_coords.min()))
            + normal * float(np.mean(n_coords))
        )
        rotation = np.stack([normal, horizontal_axis, vertical_axis], axis=1)
        half_extents = np.array(
            [0.5 * self.cfg.patch_thickness, 0.5 * width, 0.5 * height],
            dtype=np.float64,
        )
        return {
            "center": center.astype(np.float32),
            "rotation": rotation.astype(np.float32),
            "half_extents": half_extents.astype(np.float32),
        }
