import copy
import os
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import imageio
import cv2
import einops
import numpy as np
import torch
from termcolor import cprint
import h5py

from sailor.dreamer.tools import add_to_cache


def get_obs_stacked(obs_list, obs_horizon):
    # maintain the last obs_horizon observations, repeat the first observation if there are not enough
    obs_stacked = []  # rolling buffer
    all_obs_stacked = []  # list of all stacked observations

    # fill the first observation obs_horizon times
    for i in range(obs_horizon - 1):
        obs_stacked.append(obs_list[0])

    for i in range(len(obs_list)):
        # insert current observation
        obs_stacked.append(obs_list[i])

        # push to all_obs_stacked
        if obs_horizon == 1:
            all_obs_stacked.append(np.expand_dims(obs_stacked[-1], axis=-1))
        else:
            all_obs_stacked.append(np.stack(obs_stacked, axis=-1))

        # remove the oldest observation
        obs_stacked.pop(0)

    return all_obs_stacked


def get_act_stacked(act_list, pred_horizon):
    # append the current and next pred_horizon-1 actions
    # for the last pred_horizon-1 actions, repeat the last action
    act_stacked = []  # rolling buffer
    all_act_stacked = []  # list of all stacked actions

    # fill in the first pred_horizon-1 actions
    for i in range(pred_horizon):
        if i < len(act_list):
            act_stacked.append(act_list[i])
        else:
            act_stacked.append(act_list[-1])

    for i in range(pred_horizon, len(act_list) + pred_horizon):
        # push to all_act_stacked
        if pred_horizon == 1:
            all_act_stacked.append(np.expand_dims(act_stacked[-1], axis=-1))
        else:
            all_act_stacked.append(np.stack(act_stacked, axis=-1))

        # pop the oldest action
        act_stacked.pop(0)

        # append the current action (or last action)
        if i < len(act_list):
            act_stacked.append(act_list[i])
        else:
            act_stacked.append(act_list[-1])

    return all_act_stacked


def add_env_obs_to_dict(
    obs,
    obs_traj: dict,
    base_action,
    residual_action,
    action,
    rewards,
    dones,
    pixel_keys: list,
    step_idx,
    max_traj_len,
):
    """
    obs shape: num_envs x ...
    obs_traj: List of len num_envs of dictionaries
    """
    num_envs = len(obs_traj)
    assert (
        num_envs == obs["state"].shape[0]
    ), "Number of envs should be same as first dim of obs input"

    for env_idx in range(num_envs):
        obs_traj_env = obs_traj[env_idx]

        # If the previous entry for this environment had is_terminal as True, then skip this environment
        if len(obs_traj_env["is_terminal"]) > 0 and obs_traj_env["is_terminal"][-1]:
            continue

        obs_traj_env["state"].append(obs["state"][env_idx])
        obs_traj_env["base_action"].append(base_action[env_idx])
        obs_traj_env["residual_action"].append(residual_action[env_idx])
        obs_traj_env["action"].append(action[env_idx])
        obs_traj_env["reward"].append(rewards[env_idx])
        obs_traj_env["is_first"].append(step_idx == 0)
        obs_traj_env["is_last"].append(step_idx == max_traj_len - 1)
        obs_traj_env["is_terminal"].append(dones[env_idx])
        for key in pixel_keys:
            obs_traj_env[key].append(obs[key][env_idx])


def save_collected_traj_video(obs_traj, rollout_idx, logdir):
    """
    Save the trajectory as a video
    """
    max_frames = max(
        [len(obs_traj[i]["agentview_image"]) for i in range(len(obs_traj))]
    )

    # Collect frames and pad with zeros if the length is less than max_frames
    frames = []
    for i in range(len(obs_traj)):
        frames.append(obs_traj[i]["agentview_image"])
        if len(frames[-1]) < max_frames:
            frames[-1] += [
                np.zeros_like(frames[-1][0])
                for _ in range(max_frames - len(frames[-1]))
            ]

    frames = np.array(frames)  # Shape (n_envs, time, frame_height, frame_width, 3)
    frames = einops.rearrange(
        frames,
        "n_envs n_imgs frame_height frame_width c -> n_imgs frame_height (n_envs frame_width) c",
    )
    savedir = logdir / "collected_traj_videos/"
    os.makedirs(savedir, exist_ok=True)
    
    imageio.mimwrite(
        str(savedir / f"rollout_{rollout_idx}.mp4"),
        frames,
        fps=30,
        quality=8,
    )
    

def add_traj_to_cache(
    env_idx, obs_traj, pixel_keys, pred_horizon, obs_horizon, train_eps
):
    """
    Traj IDX: ID of teh collected trajectory
    obs_traj: Dictionary of observations collected
    pixel_keys: List of pixel keys in the observation
    """
    # Assign unique eps_name
    eps_name = f"traj_{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}_{env_idx}"

    # Stack Observations for State and Pixel Keys
    stacked_obs = {}
    # state: [state_dim, obs_horizon] [9, 2]
    stacked_obs["state"] = get_obs_stacked(obs_traj["state"], obs_horizon)
    # image: [H, W, C, obs_horizon] [64, 64, 3, 2]
    for key in pixel_keys:
        stacked_obs[key] = get_obs_stacked(obs_traj[key], obs_horizon)

    # Stack base and residual actions
    # [action_dim, pred_horizon] [7, 8]
    stacked_base_acts = get_act_stacked(obs_traj["base_action"], pred_horizon)
    stacked_residual_acts = get_act_stacked(obs_traj["residual_action"], pred_horizon)

    # Stack Actions
    stacked_actions = get_act_stacked(obs_traj["action"], pred_horizon)

    # Fill the transitions in self.train_eps
    for idx in range(len(obs_traj["state"])):
        transition = defaultdict(np.array)
        for key in stacked_obs.keys():
            transition[key] = stacked_obs[key][idx]

        transition["base_action"] = stacked_base_acts[idx]
        transition["residual_action"] = stacked_residual_acts[idx]
        transition["action"] = stacked_actions[idx]
        transition["reward"] = np.array(obs_traj["reward"][idx], dtype=np.float32)

        transition["is_first"] = np.array(obs_traj["is_first"][idx], dtype=np.bool_)
        transition["is_last"] = np.array(obs_traj["is_last"][idx], dtype=np.bool_)
        transition["is_terminal"] = np.array(
            obs_traj["is_terminal"][idx], dtype=np.bool_
        )

        add_to_cache(train_eps, eps_name, transition)

    return eps_name


def mixed_sample(
    batch_size,
    expert_dataset,
    train_dataset,
    device,
    remove_obs_stack=True,
    sqil_discriminator=False,
):
    """
    Sample 50% from expert dataset and 50% from self.train_eps
    If remove_obs_stack is True, keep only latest obs in the batch
    """
    assert batch_size % 2 == 0, "Batch Size should be even."

    expert_batch = next(expert_dataset)
    train_batch = next(train_dataset)

    # Merge the two batches
    data_batch = {}
    for key in expert_batch.keys():
        if key in train_batch.keys():
            expert_batch[key] = torch.tensor(expert_batch[key], dtype=torch.float32)
            train_batch[key] = torch.tensor(train_batch[key], dtype=torch.float32)
            data_batch[key] = torch.cat(
                [expert_batch[key], train_batch[key]], dim=0
            ).to(device)

    # # SQIL discriminator, +1 for expert, -1 for all other
    if sqil_discriminator:
        data_batch["reward"] = torch.cat(
            [
                torch.ones_like(expert_batch["reward"]),
                -torch.ones_like(train_batch["reward"]),
            ],
            dim=0,
        ).to(device)

    if remove_obs_stack:
        data_batch = select_latest_obs(data_batch)

    return data_batch


def select_latest_obs(obs: dict):
    # Removes the stacked observations, keeping only the latest one
    # Returns a copy of the observations with removed stacked dimensions
    obs_out = {}
    obs_out["state"] = copy.deepcopy(obs["state"][..., -1])
    if "agentview_image" in obs.keys():
        obs_out["agentview_image"] = copy.deepcopy(obs["agentview_image"][..., -1])
    if "robot0_eye_in_hand_image" in obs.keys():
        obs_out["robot0_eye_in_hand_image"] = copy.deepcopy(
            obs["robot0_eye_in_hand_image"][..., -1]
        )
    # Keep all other things same
    for key in obs.keys():
        if key not in obs_out.keys():
            obs_out[key] = obs[key]
    return obs_out


def collect_onpolicy_trajs(
    num_steps,
    max_traj_len,
    base_policy,
    train_env,
    pred_horizon,
    obs_horizon,
    train_eps,
    state_only,
    save_dir=None,
    save_episodes=False,
    discard_if_not_success=False,
    vis=False,
    is_stacked_obs=True,
):
    start_time = time.time()
    """
    Collect num_trajs trajectories using base_policy in train_env
    """
    if num_steps == 0:
        print("Collecting 0 steps.")
        return

    num_envs = train_env.num_envs
    print(f"Collecting {num_steps} steps, Num Envs: {num_envs}")

    obs = train_env.reset()
    if state_only:
        pixel_keys = []
    else:
        pixel_keys = sorted([key for key in obs.keys() if "image" in key])
    
    n_step_collected = 0
    eps_names = []
    while n_step_collected < num_steps:
        obs_traj = [
            {
                "state": [],
                "base_action": [],
                "residual_action": [],
                "action": [],
                "reward": [],
                "is_first": [],
                "is_last": [],
                "is_terminal": [],
                "success": False,
                **{key: [] for key in pixel_keys},
            }
            for _ in range(num_envs)
        ]  # List of independent dictionaries of data for each environment

        # Reset the environment and the policy
        # obs: dict_keys(['agentview_image', 'robot0_eye_in_hand_image', 
        #                 'state', 'is_first', 'is_last', 'is_terminal'])
        # each with shape (num_envs, ...)
        obs = train_env.reset()
        base_policy.reset()

        dones = np.zeros(num_envs, dtype=bool)

        # Collect data for obs_list
        print(f"Collected {n_step_collected}/{num_steps}, collecting more...")
        for step_idx in range(max_traj_len):
            if np.all(dones):
                break
            
            # generate actions with noises
            # action_dict: dict_keys(['base_action', 'residual_action'])
            action_dict = base_policy.get_action(obs)
            action = np.clip(
                action_dict["base_action"] + action_dict["residual_action"], -1, 1
            )
            
            # generate obs with obatined action
            obs_next, rewards, dones, infos = train_env.step(action)
            import pdb; pdb.set_trace()

            # Add obs and action to obs_list
            add_env_obs_to_dict(
                obs=obs,
                obs_traj=obs_traj,
                base_action=action_dict["base_action"],
                residual_action=action_dict["residual_action"],
                action=action,
                rewards=rewards,
                dones=dones,
                pixel_keys=pixel_keys,
                step_idx=step_idx,
                max_traj_len=max_traj_len,
            )

            all_successes = infos["success"]  # num_envs x 1
            for env_idx in range(num_envs):
                obs_traj[env_idx]["success"] = (
                    all_successes[env_idx] or obs_traj[env_idx]["success"]
                )

            obs = obs_next

        # Uncomment to visualize the collected trajectory
        if vis:
            rollout_vis_dir = "/home/bli678/projects/egowm/demos/robomimic_square_rollout"
            os.makedirs(rollout_vis_dir, exist_ok=True)
            save_collected_traj_video(obs_traj, rollout_idx=n_step_collected, logdir=Path(rollout_vis_dir))
            # breakpoint()

        for env_idx in range(num_envs):
            if discard_if_not_success == True and not obs_traj[env_idx]["success"]:
                cprint(
                    f"Skipping adding Env IDX: {env_idx}, Traj Len: {len(obs_traj[env_idx]['state'])}, Total Reward: {np.sum(obs_traj[env_idx]['reward'])}",
                    "red",
                )
                continue
            
            # add to cache
            if is_stacked_obs:
                eps_name = add_traj_to_cache(
                    env_idx=env_idx,
                    obs_traj=obs_traj[env_idx],
                    pixel_keys=pixel_keys,
                    pred_horizon=pred_horizon,
                    obs_horizon=obs_horizon,
                    train_eps=train_eps,
                )
            else:
                eps_name = f"traj_{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}_{env_idx}"
                
                obs_traj_env = obs_traj[env_idx]
                add_to_cache(train_eps, eps_name, obs_traj_env)
            
            eps_names.append(eps_name)
            cprint(
                f"Added Env IDX: {env_idx}, Traj Len: {len(obs_traj[env_idx]['state'])}, Total Reward: {np.sum(obs_traj[env_idx]['reward'])}",
                "green",
            )
            n_step_collected += len(obs_traj[env_idx]["state"])

        if n_step_collected >= num_steps:
            break

    print(
        f"Time taken to collect {n_step_collected}/{num_steps} steps: {time.time() - start_time:.2f} seconds"
    )

    if save_dir is not None and save_episodes:
        print("Saving Episodes to Disk: ", eps_names, " at ", save_dir)
        save_episodes(
            directory=save_dir, episodes={name: train_eps[name] for name in eps_names}
        )
    return n_step_collected


##############################################
# New added
##############################################

def vis_rollout_data(rollout_data, save_dir, rollout_idx=0):
    max_frames = max([len(rollout_data[i]["states"]) for i in range(len(rollout_data))])
    
    frames = []
    for i in range(len(rollout_data)):
        obs_i = rollout_data[i]['obs']
        frames_i = [obs_i[j]['agentview_image'] for j in range(len(obs_i))]
        if len(frames_i) < max_frames:
            frames_i += [
                frames_i[-1]
                for _ in range(max_frames - len(frames_i))
            ]
        frames.append(frames_i)
    
    frames = np.array(frames)  # [n_envs, time, frame_height, frame_width, 3]
    frames = einops.rearrange(
        frames,
        "n_envs n_imgs frame_height frame_width c -> n_imgs frame_height (n_envs frame_width) c",
    )
    
    save_path = os.path.join(save_dir, f"rollout_{rollout_idx:04d}.mp4")
    imageio.mimwrite(
        str(save_path),
        frames,
        fps=30,
        quality=8,
    )
    print(f"Saved rollout video at {save_path}")
    

def generate_onpolicy_trajs(
    num_demos,
    max_traj_len,
    base_policy,
    train_env,
    vis=False,
    output_path=None,
):
    start_time = time.time()
    """
    Collect num_trajs trajectories using base_policy in train_env
    """
    num_envs = train_env.num_envs
    print(f"Collecting {num_demos} rollout episodes, Num Envs: {num_envs}")
    
    n_episodes_collected = 0
    n_rounds = 0
    while n_episodes_collected < num_demos:
        cprint(f"=========Starting Round {n_rounds} of data collection=========", "yellow")
        
        rollout_data = [{
            'states': [],
            'actions': [],
            'dones': [],
            'obs': [],
            'next_obs': [],
            'rewards': [],
        }
            for _ in range(num_envs)
        ]
        
        # Reset the environment and the policy
        # obs: dict_keys(['agentview_image', 'robot0_eye_in_hand_image', 
        #                 'state', 'is_first', 'is_last', 'is_terminal'])
        # each with shape (num_envs, ...)
        obs = train_env.reset()
        base_policy.reset()
        dones = np.zeros(num_envs, dtype=bool)
        env_is_terminal = np.zeros(num_envs, dtype=bool)

        # Collect data
        print(f"Collected {n_episodes_collected}/{num_demos}, collecting more...")
        for step_idx in range(max_traj_len):   
            if np.all(dones):
                break
            
            # generate actions with noises
            # action_dict: dict_keys(['base_action', 'residual_action'])
            # action: [num_envs, action_dim]
            action_dict = base_policy.get_action(obs)
            action = np.clip(
                action_dict["base_action"] + action_dict["residual_action"], -1, 1
            )
            
            # generate obs with obatined action
            obs_next, rewards, dones, infos = train_env.step(action)

            # append data to rollout_data
            for env_idx in range(num_envs):
                env_i = train_env.envs[env_idx]
                
                # If the previous entry for this environment had is_terminal as True, then skip this environment
                if env_is_terminal[env_idx]:
                    continue
                
                obs_i = {k: obs[k][env_idx] for k in obs.keys()}
                obs_next_i = {k: obs_next[k][env_idx] for k in obs_next.keys()}
                
                rollout_data[env_idx]['states'].append(env_i.sim.get_state().flatten())
                rollout_data[env_idx]['actions'].append(action[env_idx])
                rollout_data[env_idx]['obs'].append(obs_i)
                rollout_data[env_idx]['next_obs'].append(obs_next_i)
                rollout_data[env_idx]['rewards'].append(rewards[env_idx])
                rollout_data[env_idx]['dones'].append(dones[env_idx])
                
                env_is_terminal[env_idx] = obs_i['is_terminal']

            obs = obs_next
        
        # Visualization
        if vis:
            save_dir = '/home/bli678/projects/egowm/demos/robomimic_square_rollout'
            os.makedirs(save_dir, exist_ok=True)
            vis_rollout_data(rollout_data, save_dir, rollout_idx=n_rounds)    
        
        # Save collected data
        for env_idx in range(num_envs):
            rollout_data_env = rollout_data[env_idx]
            env_i = train_env.envs[env_idx]
            cprint(
                f"Added Env IDX: {env_idx}, Traj Len: {len(rollout_data_env['states'])}, Total Reward: {np.sum(rollout_data_env['rewards'])}",
                "green",
            )
            n_episodes_collected += 1

            # Save collected data
            if output_path is not None:
                save_rollout_data(rollout_data_env, env_i, output_path)
        
        if n_episodes_collected >= num_demos:
            break
        
        n_rounds += 1
    
    cprint(
        f"Collected {n_episodes_collected}/{num_demos} episodes, Time: {time.time() - start_time:.2f} seconds",
        "yellow", attrs=["bold"]
    )

            
        
def save_rollout_data(rollout_data, rollout_env, output_dataset):
    """
    Save rollout data to HDF5 dataset in robomimic format.
    
    Args:
        rollout_data (dict): Dictionary containing states, actions, obs, rewards, dones
        rollout_env: Environment instance to extract metadata from
        output_dataset (str): Path to output HDF5 file
    """
    # Create or open the output HDF5 file
    if os.path.exists(output_dataset):
        f_out = h5py.File(output_dataset, "a")  # append mode
    else:
        f_out = h5py.File(output_dataset, "w")  # write mode
    
    # Create data group if it doesn't exist
    if "data" not in f_out:
        f_out.create_group("data")
    
    data_grp = f_out["data"]
    
    # Generate unique demo name
    existing_demos = list(data_grp.keys())
    if existing_demos:
        # Extract demo numbers and find the next one
        demo_numbers = []
        for demo_name in existing_demos:
            if demo_name.startswith("demo_"):
                demo_num = int(demo_name.split("_")[1])
                demo_numbers.append(demo_num)
        next_demo_num = max(demo_numbers) + 1 if demo_numbers else 0
    else:
        next_demo_num = 0
    
    demo_name = f"demo_{next_demo_num}"
    
    # Create demo group
    demo_grp = data_grp.create_group(demo_name)
    
    # Convert lists to numpy arrays
    states = np.array(rollout_data['states'])
    actions = np.array(rollout_data['actions'])
    rewards = np.array(rollout_data['rewards'])
    dones = np.array(rollout_data['dones'])
    
    # Save states, actions, rewards
    demo_grp.create_dataset("states", data=states)
    demo_grp.create_dataset("actions", data=actions)
    demo_grp.create_dataset("rewards", data=rewards)
    demo_grp.create_dataset("dones", data=dones)
    
    # Save observations - create obs group and save each observation key
    if len(rollout_data['obs']) > 0:
        obs_grp = demo_grp.create_group("obs")
        # Get all observation keys from the first observation
        obs_keys = rollout_data['obs'][0].keys()
        for obs_key in obs_keys:
            # Stack all observations for this key across time steps
            obs_data = np.array([obs[obs_key] for obs in rollout_data['obs']])
            obs_grp.create_dataset(obs_key, data=obs_data)
    
    # Save next observations - create next_obs group and save each observation key
    if len(rollout_data['next_obs']) > 0:
        next_obs_grp = demo_grp.create_group("next_obs")
        # Get all observation keys from the first next observation
        next_obs_keys = rollout_data['next_obs'][0].keys()
        for obs_key in next_obs_keys:
            # Stack all next observations for this key across time steps
            next_obs_data = np.array([obs[obs_key] for obs in rollout_data['next_obs']])
            next_obs_grp.create_dataset(obs_key, data=next_obs_data)
    
    # Save episode metadata
    current_model_xml = rollout_env.sim.model.get_xml()
    demo_grp.attrs["model_file"] = current_model_xml
    demo_grp.attrs["num_samples"] = len(rollout_data['actions'])