"""Script to play a checkpoint if an RL agent from Instinct-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import math
import subprocess
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Play an RL agent with Instinct-RL.")
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
    "--aux_reward",
    action="store_true",
    default=False,
    help="Whether to assign auxiliary rewards to each of the env's reward term.",
)

# virtual obstacles
# append Instinct-RL cli arguments
cli_args.add_instinct_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

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
from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import load_yaml
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab_tasks.utils.hydra import hydra_task_config

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

# virtual obstacles
@hydra_task_config(args_cli.task, "env_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: InstinctRlOnPolicyRunnerCfg | None = None):
    """Play with Instinct-RL agent."""
    # update env_cfg based on cli arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if args_cli.disable_fabric:
        env_cfg.sim.use_fabric = False
    else:
        env_cfg.sim.use_fabric = True
        
    # Re-parse agent cfg manually using Instinct-RL's parser as it handles checkpoint paths differently
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

    if hasattr(env_cfg, "apply_play_overrides"):
        env_cfg.apply_play_overrides()

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

    # print order
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

    # print order
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
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            if timestep < args_cli.zero_act_until:
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
