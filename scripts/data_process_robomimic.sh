TASK=square
python3 -m robomimic.scripts.dataset_states_to_obs \
    --done_mode 1 \
    --dataset datasets/robomimic_datasets/${TASK}/ph/demo_v141.hdf5 \
    --output_name image_64_shaped_done1_v141.hdf5 \
    --camera_names agentview robot0_eye_in_hand \
    --camera_height 64 \
    --camera_width 64 \
    --shaped \
    # --n 1