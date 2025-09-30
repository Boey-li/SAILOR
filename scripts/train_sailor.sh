SUITE="robomimic" # [robomimic | maniskill | robocasa]
TASK="square" # Any task of the respective suite
NUM_EXP_TRAJS=50
SEED=0
python3 train_sailor.py \
    --configs cfg_dp_mppi ${SUITE}\
    --wandb_project SAILOR_${SUITE} \
    --wandb_exp_name "seed${SEED}" \
    --task "${SUITE}__${TASK}" \
    --num_exp_trajs ${NUM_EXP_TRAJS} \
    --seed ${SEED}