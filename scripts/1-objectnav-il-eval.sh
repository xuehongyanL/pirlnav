export GLOG_minloglevel=2
export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet

config="configs/experiments/il_objectnav.yaml"

dataset="objectnav_hm3d_hd"

DATA_PATH="data/datasets/objectnav/objectnav_hm3d/objectnav_hm3d_v1"
TENSORBOARD_DIR="tb/objectnav_il/ovrl_resnet50/seed_1/"
EVAL_CKPT_PATH_DIR="data/new_checkpoints/objectnav_il/${dataset}/ovrl_resnet50/seed_1/"

mkdir -p $TENSORBOARD_DIR
set -x

echo "In ObjectNav IL eval"
python -u -m run \
--exp-config $config \
--run-type eval \
TENSORBOARD_DIR $TENSORBOARD_DIR \
EVAL_CKPT_PATH_DIR $EVAL_CKPT_PATH_DIR \
NUM_ENVIRONMENTS 20 \
RL.DDPPO.force_distributed True \
TASK_CONFIG.DATASET.SPLIT "val" \
EVAL.USE_CKPT_CONFIG False \
TASK_CONFIG.DATASET.DATA_PATH "$DATA_PATH/{split}/{split}.json.gz" \
