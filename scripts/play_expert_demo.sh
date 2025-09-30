SUITE="robomimic" # [robomimic | maniskill | robocasa]
TASK="square" # Any of the tasks in the paper for the respective suite
NUM_EXP_TRAJS=10 # Number of trajectories to visualize
# conda activate ${SUITE}_env

python3 train_sailor.py --wandb_exp_name "test_mppi" \
    --viz_expert_buffer True \
    --configs cfg_dp_mppi ${SUITE} debug\
    --task "${SUITE}__${TASK}" \
    --num_exp_trajs ${NUM_EXP_TRAJS}