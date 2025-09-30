SUITE="robomimic" # [robomimic | maniskill | robocasa]
TASK="can" # Any of the tasks in the paper for the respective suite
# conda activate ${SUITE}_env
python3 train_sailor.py \
    --wandb_exp_name "test" \
    --configs cfg_dp_mppi ${SUITE} debug \
    --task "${SUITE}__${TASK}" \
    --num_exp_trajs 10