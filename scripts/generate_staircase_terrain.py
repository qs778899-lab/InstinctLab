"""
Generate a staircase terrain STL file for InstinctLab Perceptive Shadowing training.

Staircase geometry is derived from holosoma retargeting scene:
  demo_data/climb/mocap_climb_seq_8/create_step_obj.py

Original holosoma coordinate system (robot starts near Y=0, stairs at Y=[2.0, 3.5]):
  - X: [0.14, 1.73], width = 1.59 m
  - Y: [2.0, 3.5], depth per step = 0.5 m
  - Z: [0.0, 1.32], height per step = 0.44 m
  - Robot initial position: pelvis at (0, 0, ~0.79)

After GMR_to_instinct.py conversion, the robot trajectory in InstinctLab coords:
  - Robot starts at Y≈+0.2, walks toward -Y
  - Reaches stair top at Y≈-1.3, Z≈1.32 m

InstinctLab terrain requirements (from mesh_terrains_cfg.py):
  - Terrain origin (0, 0, 0) must be at the terrain surface center.
  - The terrain must have a flat border at the bottom for border_height detection.
  - Terrain is cropped to cfg.size (e.g. 8m x 8m) by the loader.

Coordinate mapping (holosoma -> InstinctLab terrain-local):
  - In holosoma: robot at Y=0, stairs start at Y=2.0 (robot walks +Y)
  - In InstinctLab npz: robot at Y=+0.2, stairs at Y≈-0.3 to -1.8 (robot walks -Y)
  - Holosoma Y axis is flipped: instinct_Y = -(holosoma_Y - 2.0) + 0.2
    => holosoma Y=2.0 -> instinct Y=+0.2 (stair bottom near robot start)
    => holosoma Y=3.5 -> instinct Y=-1.3 (stair top)
  - X: holosoma X center = (0.14+1.73)/2 = 0.935, instinct X center ≈ 0.935
    => shift to center at 0: instinct_X = holosoma_X - 0.935
  - Z: no change needed (ground = 0)

Terrain-local origin for InstinctLab:
  The terrain origin should be placed at the flat ground level where the robot starts.
  We place origin at (0, 0, 0) = ground level at the center of the terrain tile.
"""

import argparse
import os

import numpy as np

try:
    import trimesh
except ImportError:
    raise ImportError("Please install trimesh: pip install trimesh")


# ──────────────────────────────────────────────
# Staircase parameters (from create_step_obj.py)
# ──────────────────────────────────────────────
NUM_STEPS = 3
STEP_DEPTH = 0.5       # Y direction (per step), meters
STEP_HEIGHT = 0.44     # Z direction (per step), meters
STEP_WIDTH_X = 1.59    # X direction (1.73 - 0.14), meters

# Holosoma coordinate: stair bottom face starts at Y=2.0, robot at Y=0
# After GMR conversion: robot at Y≈+0.2, stair bottom at Y≈+0.2
# => stair bottom Y offset in terrain-local coords:
STAIR_START_Y = 0.2    # where the first step begins (terrain-local, +Y side)

# Border parameters (flat ground around the staircase)
BORDER_SIZE = 2.0      # meters of flat ground on each side
BORDER_THICKNESS = 0.1 # thickness of the bottom border slab

# Terrain tile half-size (must fit within cfg.size/2, typically 4m for 8m tile)
TERRAIN_HALF_X = 4.0
TERRAIN_HALF_Y = 4.0


def create_box_mesh(x_min, x_max, y_min, y_max, z_min, z_max) -> trimesh.Trimesh:
    """Create a box mesh from min/max bounds."""
    box = trimesh.creation.box(
        extents=[x_max - x_min, y_max - y_min, z_max - z_min]
    )
    # trimesh.creation.box is centered at origin, translate to correct position
    box.apply_translation([
        (x_min + x_max) / 2,
        (y_min + y_max) / 2,
        (z_min + z_max) / 2,
    ])
    return box


def generate_staircase_terrain(
    num_steps: int = NUM_STEPS,
    step_depth: float = STEP_DEPTH,
    step_height: float = STEP_HEIGHT,
    step_width_x: float = STEP_WIDTH_X,
    stair_start_y: float = STAIR_START_Y,
    border_size: float = BORDER_SIZE,
    border_thickness: float = BORDER_THICKNESS,
) -> trimesh.Trimesh:
    """
    Generate a staircase terrain mesh in InstinctLab terrain-local coordinates.

    Coordinate convention:
      - Origin (0,0,0) is at the flat ground surface at the terrain center.
      - Robot approaches from +Y side, walks toward -Y, climbing stairs.
      - Stairs extend from Y=stair_start_y toward -Y.
      - X is centered at 0.

    The mesh includes:
      1. A flat bottom border slab (for border_height detection by InstinctLab).
      2. The staircase boxes stacked in -Y direction.
    """
    meshes = []

    # ── 1. Flat bottom border slab ──────────────────────────────────────────
    # This is the flat ground that InstinctLab uses to detect border_height.
    # It covers the full terrain tile at Z=[-border_thickness, 0].
    border = create_box_mesh(
        x_min=-TERRAIN_HALF_X, x_max=TERRAIN_HALF_X,
        y_min=-TERRAIN_HALF_Y, y_max=TERRAIN_HALF_Y,
        z_min=-border_thickness, z_max=0.0,
    )
    meshes.append(border)

    # ── 2. Staircase boxes ──────────────────────────────────────────────────
    # Steps are stacked: each step is a box sitting on top of the previous one.
    # Step 1 (lowest): Y=[stair_start_y - step_depth, stair_start_y], Z=[0, step_height]
    # Step 2:          Y=[stair_start_y - 2*step_depth, stair_start_y - step_depth], Z=[step_height, 2*step_height]
    # ...
    # Each step box also includes the full column below it (so the mesh is solid).
    x_half = step_width_x / 2.0

    for i in range(num_steps):
        y_max = stair_start_y - i * step_depth
        y_min = stair_start_y - (i + 1) * step_depth
        z_min = 0.0
        z_max = (i + 1) * step_height

        step_box = create_box_mesh(
            x_min=-x_half, x_max=x_half,
            y_min=y_min, y_max=y_max,
            z_min=z_min, z_max=z_max,
        )
        meshes.append(step_box)

    # ── 3. Merge all meshes ─────────────────────────────────────────────────
    terrain = trimesh.util.concatenate(meshes)
    return terrain


def main():
    parser = argparse.ArgumentParser(
        description="Generate staircase terrain STL for InstinctLab Perceptive Shadowing."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/huangyucheng/桌面/Project Instinct/InstinctLab/MOTION_data/test_motion_data_stairs",
        help="Output directory for the terrain STL file.",
    )
    parser.add_argument("--num_steps", type=int, default=NUM_STEPS)
    parser.add_argument("--step_depth", type=float, default=STEP_DEPTH)
    parser.add_argument("--step_height", type=float, default=STEP_HEIGHT)
    parser.add_argument("--step_width_x", type=float, default=STEP_WIDTH_X)
    parser.add_argument("--stair_start_y", type=float, default=STAIR_START_Y,
                        help="Y coordinate (terrain-local) where the first step begins.")
    parser.add_argument("--output_name", type=str, default="stairs_terrain.stl")
    args = parser.parse_args()

    terrain = generate_staircase_terrain(
        num_steps=args.num_steps,
        step_depth=args.step_depth,
        step_height=args.step_height,
        step_width_x=args.step_width_x,
        stair_start_y=args.stair_start_y,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, args.output_name)
    terrain.export(output_path)
    print(f"[OK] Terrain STL saved to: {output_path}")
    print(f"     Vertices: {len(terrain.vertices)}, Faces: {len(terrain.faces)}")
    print(f"     Bounds X: [{terrain.bounds[0,0]:.3f}, {terrain.bounds[1,0]:.3f}]")
    print(f"     Bounds Y: [{terrain.bounds[0,1]:.3f}, {terrain.bounds[1,1]:.3f}]")
    print(f"     Bounds Z: [{terrain.bounds[0,2]:.3f}, {terrain.bounds[1,2]:.3f}]")
    print()
    print("Staircase geometry summary:")
    print(f"  Steps: {args.num_steps}")
    print(f"  Step depth (Y): {args.step_depth} m")
    print(f"  Step height (Z): {args.step_height} m")
    print(f"  Step width (X): {args.step_width_x} m (centered at X=0)")
    print(f"  Stair start Y: {args.stair_start_y} m (robot approaches from +Y)")
    print(f"  Total stair depth: {args.num_steps * args.step_depth} m")
    print(f"  Total stair height: {args.num_steps * args.step_height} m")


if __name__ == "__main__":
    main()
