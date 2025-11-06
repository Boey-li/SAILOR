SUITE="robomimic" # [robomimic | maniskill | robocasa]
TASK="square" # Any task of the respective suite
NUM_EXP_TRAJS=1 #50
SEED=0
BASE_POLICY_PRETRAINED_CKPT="/coc/flash7/bli678/projects/egowm/external/SAILOR/scratch_dir/logs/robomimic__square/seed0_demos50/seed0/DP_Pretrain_base_policy_latest.pt"

python3 train_sailor_wm.py \
    --configs cfg_dp_mppi ${SUITE}\
    --wandb_project SAILOR_${SUITE} \
    --wandb_exp_name "seed${SEED}" \
    --task "${SUITE}__${TASK}" \
    --num_exp_trajs ${NUM_EXP_TRAJS} \
    --seed ${SEED} \
    --use_wandb False \
    --set dp.pretrained_ckpt ${BASE_POLICY_PRETRAINED_CKPT} \
    --batch_size 2 \
    --batch_length 1 \
    --set train_dp_mppi_params.use_discrim True \
