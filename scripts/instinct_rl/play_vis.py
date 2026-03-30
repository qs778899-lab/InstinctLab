"""Script to visualize an RL agent/reference scene with Instinct-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import math
import subprocess
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Visualize an RL agent/reference scene with Instinct-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=3000, help="Length of the recorded video (in steps).")
parser.add_argument("--video_start_step", type=int, default=0, help="Start step for the simulation.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--exportonnx", action="store_true", default=False, help="Export policy as ONNX model.")
parser.add_argument("--debug", action="store_true", default=False, help="Enable debug mode.")
parser.add_argument("--no_resume", default=None, action="store_true", help="Force play in no resume mode.")
# custom play arguments
parser.add_argument("--env_cfg", action="store_true", default=False, help="Load configuration from file.")
parser.add_argument("--agent_cfg", action="store_true", default=False, help="Load configuration from file.")
parser.add_argument("--sample", action="store_true", default=False, help="Sample actions instead of using the policy.")
parser.add_argument("--zero_act_until", type=int, default=0, help="Zero actions until this timestep.")
parser.add_argument(
    "--no_terminate", action="store_true", default=False, help="Do not remove termination conditions in simulation."
)
parser.add_argument(
    "--motion_file",
    type=str,
    default=None,
    help="Play a specific motion npz file (absolute path or basename). If set, the motion buffer is restricted to this file.",
)
parser.add_argument(
    "--aux_reward",
    action="store_true",
    default=False,
    help="Whether to assign auxiliary rewards to each of the env's reward term.",
)
parser.add_argument(
    "--freeze_policy",
    action="store_true",
    default=False,
    help="Keep policy actions at zero for the whole run to only visualize the reference scene.",
)
parser.add_argument(
    "--robot_offset",
    type=float,
    nargs=3,
    default=[0.0, 1.0, 2.0],
    metavar=("X", "Y", "Z"),
    help="Offset the simulated robot so the reference robot stays on the terrain.",
)
parser.add_argument(
    "--reference_offset",
    type=float,
    nargs=3,
    default=[0.0, 0.0, 0.0],
    metavar=("X", "Y", "Z"),
    help="Offset applied to the visualized reference robot.",
)
parser.add_argument(
    "--viewer_asset",
    type=str,
    default="robot_reference",
    help="Asset name for the viewer to follow after visualization overrides are applied.",
)
parser.add_argument(
    "--viewer_eye",
    type=float,
    nargs=3,
    default=None,
    metavar=("X", "Y", "Z"),
    help="Override viewer eye position.",
)
parser.add_argument(
    "--viewer_lookat",
    type=float,
    nargs=3,
    default=None,
    metavar=("X", "Y", "Z"),
    help="Override viewer lookat position.",
)
parser.add_argument(
    "--keep_play_defaults",
    action="store_true",
    default=False,
    help="Disable play_vis visualization overrides and keep the original play task settings.",
)
parser.add_argument(
    "--print_debug_layout",
    action="store_true",
    default=False,
    help="Print observation/action layout debug information.",
)
parser.add_argument(
    "--print_foot_pos",
    action="store_true",
    default=False,
    help="Print foot positions every N motion frames (env0).",
)
parser.add_argument(
    "--print_interval",
    type=int,
    default=20,
    help="Interval in motion frames for printing foot positions.",
)
parser.add_argument(
    "--manual_next_motion_key",
    type=str,
    default="N",
    help="Keyboard key for manually switching to the next motion when num_envs==1.",
)
parser.add_argument(
    "--manual_prev_motion_key",
    type=str,
    default="B",
    help="Keyboard key for manually switching to the previous motion when num_envs==1.",
)
# append Instinct-RL cli arguments
cli_args.add_instinct_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import numpy as np
import os
import time
import torch

from instinct_rl.runners import OnPolicyRunner

import isaaclab.utils.math as math_utils
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import load_yaml
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

# Import extensions to set up environment tasks
import instinctlab.tasks  # noqa: F401
from instinctlab.managers.reward_manager import MultiRewardManager
from instinctlab.utils.wrappers import InstinctRlVecEnvWrapper
from instinctlab.utils.wrappers.instinct_rl import InstinctRlOnPolicyRunnerCfg

# wait for attach if in debug mode
if args_cli.debug:
    # import typing; typing.TYPE_CHECKING = True
    import debugpy

    ip_address = ("0.0.0.0", 6789)
    print("Process: " + " ".join(sys.argv[:]))
    print("Is waiting for attach at address: %s:%d" % ip_address, flush=True)
    debugpy.listen(ip_address)
    debugpy.wait_for_client()
    debugpy.breakpoint()


def _print_policy_observation_debug(vec_env: InstinctRlVecEnvWrapper, tag: str) -> None:
    """Print joint ordering and flattened policy observation layout once for debugging."""
    _, extras = vec_env.get_observations()
    observations = extras["observations"]
    policy_obs = observations["policy"]
    critic_obs = observations.get("critic")
    obs_format = vec_env.get_obs_format()

    base_env = vec_env.unwrapped
    robot = base_env.scene["robot"]
    motion_reference = base_env.scene["motion_reference"]

    robot_joint_names = list(robot.joint_names)
    reference_joint_names = list(motion_reference.joint_names)

    print(f"\n[DEBUG {tag}] robot joint order ({len(robot_joint_names)}):")
    for joint_idx, joint_name in enumerate(robot_joint_names):
        print(f"  {joint_idx:02d}: {joint_name}")

    print(f"\n[DEBUG {tag}] motion-reference joint order ({len(reference_joint_names)}):")
    for joint_idx, joint_name in enumerate(reference_joint_names):
        print(f"  {joint_idx:02d}: {joint_name}")

    for group_name, group_format in obs_format.items():
        print(f"\n[DEBUG {tag}] {group_name} observation term order / flat slices:")
        offset = 0
        for term_name, shape in group_format.items():
            flat_dim = math.prod(shape)
            print(f"  {term_name:20s} shape={str(shape):>12s} flat=[{offset}:{offset + flat_dim}]")
            offset += flat_dim

    def _print_joint_term(group_name: str, group_obs: torch.Tensor | None, term_name: str, joint_names: list[str]) -> None:
        if group_obs is None:
            return
        group_format = obs_format.get(group_name)
        if group_format is None or term_name not in group_format:
            return

        offset = 0
        for format_term_name, shape in group_format.items():
            flat_dim = math.prod(shape)
            if format_term_name == term_name:
                term_flat = group_obs[0, offset : offset + flat_dim].detach().cpu()
                print(f"\n[DEBUG {tag}] {group_name}.{term_name} flat shape = {tuple(term_flat.shape)}")
                if flat_dim == len(joint_names):
                    reshaped = term_flat.view(1, len(joint_names))
                elif flat_dim % len(joint_names) == 0:
                    reshaped = term_flat.view(flat_dim // len(joint_names), len(joint_names))
                else:
                    print(
                        f"[DEBUG {tag}] {group_name}.{term_name}: flat dim {flat_dim} is not divisible by"
                        f" num_joints {len(joint_names)}"
                    )
                    return

                print(
                    f"[DEBUG {tag}] {group_name}.{term_name} reshaped as {tuple(reshaped.shape)}, "
                    "showing last frame:"
                )
                for joint_idx, joint_name in enumerate(joint_names):
                    print(f"  {joint_idx:02d}: {joint_name:30s} {float(reshaped[-1, joint_idx]): .6f}")
                return
            offset += flat_dim

    for obs_group_name, group_obs in (("policy", policy_obs), ("critic", critic_obs)):
        for joint_term_name in ("joint_pos_ref", "joint_vel_ref", "joint_pos", "joint_vel", "last_action"):
            _print_joint_term(obs_group_name, group_obs, joint_term_name, robot_joint_names)


def _print_policy_action_debug(vec_env: InstinctRlVecEnvWrapper, policy, tag: str) -> None:
    """Print policy action ordering and a sample action vector once for debugging."""
    obs, _ = vec_env.get_observations()
    with torch.no_grad():
        actions = policy(obs)

    robot_joint_names = list(vec_env.unwrapped.scene["robot"].joint_names)
    action_sample = actions[0].detach().cpu()

    print(f"\n[DEBUG {tag}] policy action shape = {tuple(actions.shape)}")
    print(f"[DEBUG {tag}] action dimensions map to robot joint order ({len(robot_joint_names)}):")

    if action_sample.numel() != len(robot_joint_names):
        print(
            f"[DEBUG {tag}] action dim {action_sample.numel()} does not match robot joint count {len(robot_joint_names)}"
        )
        return

    for action_idx, joint_name in enumerate(robot_joint_names):
        print(f"  {action_idx:02d}: {joint_name:30s} {float(action_sample[action_idx]): .6f}")


def _apply_visualization_overrides(env_cfg) -> None:
    """Tune the play config for reference-motion visualization."""
    if args_cli.keep_play_defaults:
        return

    viewer_cfg = getattr(env_cfg, "viewer", None)
    if viewer_cfg is not None:
        viewer_cfg.asset_name = args_cli.viewer_asset
        if args_cli.viewer_eye is not None:
            viewer_cfg.eye = list(args_cli.viewer_eye)
        if args_cli.viewer_lookat is not None:
            viewer_cfg.lookat = list(args_cli.viewer_lookat)

    motion_reference_cfg = getattr(getattr(env_cfg, "scene", None), "motion_reference", None)
    if motion_reference_cfg is not None and hasattr(motion_reference_cfg, "visualizing_robot_offset"):
        motion_reference_cfg.visualizing_robot_offset = tuple(args_cli.reference_offset)

    reset_robot_event = getattr(getattr(env_cfg, "events", None), "reset_robot", None)
    if reset_robot_event is not None and "position_offset" in reset_robot_event.params:
        reset_robot_event.params["position_offset"] = list(args_cli.robot_offset)

    print(
        "[INFO] play_vis overrides:",
        f"viewer_asset={getattr(viewer_cfg, 'asset_name', None)}",
        f"robot_offset={args_cli.robot_offset}",
        f"reference_offset={args_cli.reference_offset}",
    )


def _setup_single_env_manual_motion_switch(vec_env: InstinctRlVecEnvWrapper):
    """Set up keyboard-based, button-like motion switching for single-env play."""
    try:
        import carb.input as carb_input
        import omni.appwindow as omni_appwindow
        from carb.input import KeyboardEventType
    except ModuleNotFoundError:
        print(
            "[WARN] carb/omni appwindow is unavailable in this Python runtime. "
            "Manual prev/next motion switching is disabled."
        )
        return None

    base_env = vec_env.unwrapped
    if base_env.num_envs != 1:
        return None

    motion_reference = base_env.scene["motion_reference"]
    if len(motion_reference.motion_buffers) != 1:
        print("[WARN] Manual motion switch supports a single motion buffer only. Skip enabling.")
        return None

    buffer_name = next(iter(motion_reference.motion_buffers.keys()))
    buffer = motion_reference.motion_buffers[buffer_name]
    if not hasattr(buffer, "_all_motion_files") or len(buffer._all_motion_files) <= 1:
        print("[INFO] Manual motion switch disabled: only one motion file is available.")
        return None

    next_key_name = args_cli.manual_next_motion_key.upper().strip()
    prev_key_name = args_cli.manual_prev_motion_key.upper().strip()
    next_key_enum = getattr(carb_input.KeyboardInput, next_key_name, None)
    prev_key_enum = getattr(carb_input.KeyboardInput, prev_key_name, None)
    if next_key_enum is None:
        print(f"[WARN] Unknown next key '{args_cli.manual_next_motion_key}', fallback to N.")
        next_key_enum = carb_input.KeyboardInput.N
        next_key_name = "N"
    if prev_key_enum is None:
        print(f"[WARN] Unknown prev key '{args_cli.manual_prev_motion_key}', fallback to B.")
        prev_key_enum = carb_input.KeyboardInput.B
        prev_key_name = "B"

    env_ids = torch.tensor([0], device=base_env.device, dtype=torch.long)
    assigned_ids = buffer.env_ids_to_assigned_ids(env_ids).to(buffer.buffer_device)
    total_motions = len(buffer._all_motion_files)

    switch_state = {"pending_delta": 0}

    def _print_current_motion(tag: str):
        motion_identifier = motion_reference.get_current_motion_identifiers(env_ids)[0]
        print(f"[INFO] {tag}: {motion_identifier}")

    def on_keyboard_input(e):
        if e.type != KeyboardEventType.KEY_PRESS:
            return True
        if e.input == next_key_enum:
            switch_state["pending_delta"] = 1
        elif e.input == prev_key_enum:
            switch_state["pending_delta"] = -1
        return True

    app_window = omni_appwindow.get_default_app_window()
    keyboard = app_window.get_keyboard()
    input_iface = carb_input.acquire_input_interface()
    subscription = input_iface.subscribe_to_keyboard_events(keyboard, on_keyboard_input)

    def process_pending_switch():
        delta = switch_state["pending_delta"]
        if delta == 0:
            return
        switch_state["pending_delta"] = 0

        curr_idx = int(buffer._assigned_env_motion_selection[assigned_ids][0].item())
        next_idx = (curr_idx + delta) % total_motions

        # Keep the selected trajectory fixed during this reset and only advance by one.
        original_sampler = buffer._sample_assigned_env_motion_selection
        try:
            def _keep_motion_selection(_assigned_ids):
                return None

            buffer._sample_assigned_env_motion_selection = _keep_motion_selection
            buffer._assigned_env_motion_selection[assigned_ids] = next_idx
            if hasattr(buffer, "_motion_buffer_start_time_s"):
                buffer._motion_buffer_start_time_s[assigned_ids] = 0.0
            base_env._reset_idx(torch.tensor([0], device=base_env.device, dtype=torch.long))
        finally:
            buffer._sample_assigned_env_motion_selection = original_sampler

        _print_current_motion("Switched to")

    _print_current_motion("Current motion")
    print(
        f"[INFO] Press '{prev_key_name}' for previous and '{next_key_name}' for next motion "
        "in metadata.yaml order."
    )
    return process_pending_switch, input_iface, subscription


def main():
    """Play with Instinct-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    _apply_visualization_overrides(env_cfg)
    agent_cfg: InstinctRlOnPolicyRunnerCfg = cli_args.parse_instinct_rl_cfg(args_cli.task, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "instinct_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    agent_cfg.load_run = args_cli.load_run
    if agent_cfg.load_run is not None:
        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        if os.path.isabs(agent_cfg.load_run):
            resume_path = get_checkpoint_path(
                os.path.dirname(agent_cfg.load_run), os.path.basename(agent_cfg.load_run), agent_cfg.load_checkpoint
            )
        else:
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        log_dir = os.path.dirname(resume_path)
    elif not args_cli.no_resume:
        raise RuntimeError(
            f"\033[91m[ERROR] No checkpoint specified and play.py resumes from a checkpoint by default. Please specify"
            f" a checkpoint to resume from using --load_run or use --no_resume to disable this behavior.\033[0m"
        )
    else:
        print(f"[INFO] No experiment directory specified. Using default: {log_root_path}")
        log_dir = os.path.join(log_root_path, agent_cfg.run_name + "_play")
        resume_path = "model_scratch.pt"

    if args_cli.env_cfg:
        env_cfg = load_yaml(os.path.join(log_dir, "params", "env.yaml"))
    if args_cli.agent_cfg:
        agent_cfg_dict = load_yaml(os.path.join(log_dir, "params", "agent.yaml"))
    else:
        agent_cfg_dict = agent_cfg.to_dict()

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == args_cli.video_start_step,
            "video_length": args_cli.video_length,
            "disable_logger": True,
            "name_prefix": f"model_{resume_path.split('_')[-1].split('.')[0]}",
        }
        print("[INFO] Recording videos during playing.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # Optional: restrict to a specific motion file if provided
    if args_cli.motion_file is not None:
        base_env = env.unwrapped
        motion_reference = base_env.scene["motion_reference"]
        if len(motion_reference.motion_buffers) != 1:
            print(
                "[WARN] --motion_file is currently supported only when a single motion buffer is present. Ignored."
            )
        else:
            buffer_name = next(iter(motion_reference.motion_buffers.keys()))
            buffer = motion_reference.motion_buffers[buffer_name]
            # Find by absolute path match or basename match
            target = args_cli.motion_file
            all_files = getattr(buffer, "_all_motion_files", None)
            if not all_files:
                # ensure the list is built
                try:
                    buffer._refresh_motion_file_list()  # type: ignore
                    all_files = buffer._all_motion_files
                except Exception:
                    all_files = []
            target_idx = None
            import os as _os
            target_base = _os.path.basename(target)
            for i, f in enumerate(all_files):
                if f == target or _os.path.basename(f) == target_base:
                    target_idx = i
                    break
            if target_idx is None:
                print(f"[ERROR] --motion_file not found in buffer list: {target}")
            else:
                # Restrict trajectories to this single index
                import torch as _torch
                try:
                    buffer.enable_trajectories(_torch.tensor([target_idx], device=buffer.buffer_device))  # type: ignore
                except Exception:
                    buffer.enable_trajectories(slice(target_idx, target_idx + 1))
                print(f"[INFO] Playing specific motion: {all_files[target_idx]}")

    # react to custom play arguments
    if args_cli.no_terminate:
        # NOTE: This is only applicable with shadowing task
        env.unwrapped.termination_manager._term_cfgs = [
            env.unwrapped.termination_manager._term_cfgs[
                env.unwrapped.termination_manager._term_names.index("dataset_exhausted")
            ]
        ]
        env.unwrapped.termination_manager._term_names = ["dataset_exhausted"]

    # wrap around environment for instinct-rl
    env = InstinctRlVecEnvWrapper(env)

    if args_cli.print_debug_layout:
        _print_policy_observation_debug(env, tag="InstinctLab")

    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg_dict, log_dir=None, device=agent_cfg.device)
    if agent_cfg.load_run is not None:
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    if args_cli.sample:
        policy = ppo_runner.alg.actor_critic.act
    else:
        policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    if args_cli.print_debug_layout:
        _print_policy_action_debug(env, policy, tag="InstinctLab")

    # export policy to onnx/jit
    if agent_cfg.load_run is not None:
        export_model_dir = os.path.join(log_dir, "exported")
        if args_cli.exportonnx:
            assert env.unwrapped.num_envs == 1, "Exporting to ONNX is only supported for single environment."
            if not os.path.exists(export_model_dir):
                os.makedirs(export_model_dir)
            obs, _ = env.get_observations()
            ppo_runner.export_as_onnx(obs, export_model_dir)

    # reset environment
    obs, _ = env.get_observations()
    # Always print which motion is currently playing (env 0) for clarity.
    try:
        motion_reference = env.unwrapped.scene["motion_reference"]
        current_motion = motion_reference.get_current_motion_identifiers([0])[0]
        print(f"[INFO] Current motion (env0): {current_motion}")
    except Exception:
        pass
    manual_motion_switch_ctx = _setup_single_env_manual_motion_switch(env) #批量播放

    #print foot
    # prepare state for foot position printing
    _last_printed_motion_frame = -1
    # identify the first (and only) motion buffer name for later use
    try:
        _buffer_name_for_print = next(iter(env.unwrapped.scene["motion_reference"].motion_buffers.keys()))
    except Exception:
        _buffer_name_for_print = None

    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        #批量播放 run everything in inference mode
        with torch.inference_mode():
            # Process manual motion switch in inference mode to avoid
            # in-place writes on inference tensors during reset.
            if manual_motion_switch_ctx is not None:
                manual_motion_switch_ctx[0]()

            #print foot
            # Optional printing of foot positions by motion frame index
            if args_cli.print_foot_pos and _buffer_name_for_print is not None:
                try:
                    base_env = env.unwrapped
                    motion_reference = base_env.scene["motion_reference"]
                    buffer = motion_reference.motion_buffers[_buffer_name_for_print]
                    # motion frame index based on motion start time + current motion time and framerate
                    assigned_motion_idx = int(buffer._assigned_env_motion_selection[0].item())
                    motion_fps = float(buffer._all_motion_sequences.framerate[assigned_motion_idx].item())
                    motion_start_time_s = float(buffer._motion_buffer_start_time_s[0].item())
                    motion_time_s = float(motion_reference._timestamp[0].item())
                    motion_frame_idx = int(round((motion_start_time_s + motion_time_s) * motion_fps))
                    if motion_frame_idx != _last_printed_motion_frame and motion_frame_idx % int(args_cli.print_interval) == 0:
                        _last_printed_motion_frame = motion_frame_idx
                        motion_name = buffer._all_motion_files[assigned_motion_idx].split("/")[-1]
                        frame = motion_reference.reference_frame
                        links = motion_reference.cfg.link_of_interests
                        pos_w = frame.link_pos_w[0, 0]  # [num_links, 3]
                        print(f"--- Frame {motion_frame_idx} (Motion: {motion_name}) ---")
                        for i, link_name in enumerate(links):
                            if ("ankle" in link_name) or ("foot" in link_name):
                                print(f"Link: {link_name}, Pos: {pos_w[i].detach().cpu().numpy()}")
                except Exception:
                    # Be silent if printing state is temporarily unavailable (e.g., just switched/reset)
                    pass
            
            
            # agent stepping
            actions = policy(obs)
            if args_cli.freeze_policy or timestep < args_cli.zero_act_until:
                actions[:] = 0.0
            # env stepping
            obs, rewards, dones, infos = env.step(actions)
        timestep += 1

        # override reward terms if auxiliary reward is enabled
        if args_cli.aux_reward:
            # NOTE: This is only applicable when reward_term has `.reward` to be overridden
            aux_rewards = ppo_runner.alg.compute_auxiliary_reward(infos["observations"])
            for aux_reward_name, aux_reward in aux_rewards.items():
                aux_term_cfg = env.unwrapped.reward_manager.get_term_cfg(aux_reward_name)  # type: ignore
                aux_term_cfg.func.reward[:] = aux_reward * getattr(ppo_runner.alg, aux_reward_name + "_coef", 1.0)

        # exit the loop if video_length is meet
        if args_cli.video:
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

    # close the simulator
    env.close()

    #批量播放
    if manual_motion_switch_ctx is not None:
        _, input_iface, subscription = manual_motion_switch_ctx
        input_iface.unsubscribe_from_keyboard_events(subscription)

    if args_cli.video:
        subprocess.run(
            [
                "code",
                "-r",
                os.path.join(log_dir, "videos", "play", f"model_{resume_path.split('_')[-1].split('.')[0]}-step-0.mp4"),
            ]
        )


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
