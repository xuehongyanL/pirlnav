export GLOG_minloglevel=2
export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet

config="configs/experiments/rl_ft_objectnav.yaml"

DATA_PATH="data/datasets/objectnav/objectnav_hm3d/objectnav_hm3d_v1"
TENSORBOARD_DIR="tb/objectnav_il_rl_ft/ovrl_resnet50/seed_1/"
EVAL_CKPT_PATH_DIR="data/new_checkpoints/objectnav_il_rl_ft/ovrl_resnet50/seed_1/"
EVAL_CKPT_PATH_DIR="checkpoints/objectnav_rl_ft_hd.ckpt"

echo "In ObjectNav IL DDP"
python -u -m run \
--exp-config $config \
--run-type eval \
TENSORBOARD_DIR $TENSORBOARD_DIR \
EVAL_CKPT_PATH_DIR $EVAL_CKPT_PATH_DIR \
NUM_UPDATES 20000 \
NUM_ENVIRONMENTS 8 \
RL.DDPPO.pretrained False \
TASK_CONFIG.DATASET.SPLIT "val" \
TASK_CONFIG.DATASET.DATA_PATH "$DATA_PATH/{split}/{split}.json.gz" \
