# bash experiments/cifar-10.sh
# experiment settings
DATASET=cifar-10
N_CLASS=200

# save directory
OUTDIR=outputs/${DATASET}/2-task

# hard coded inputs
GPUID='0'
CONFIG=configs/cifar-10_prompt.yaml
REPEAT=1

###############################################################

# process inputs
mkdir -p $OUTDIR

# L2P++
#
# prompt parameter args:
#    arg 1 = e-prompt pool size (# tasks)
#    arg 2 = e-prompt pool length
#    arg 3 = -1 -> shallow, 1 -> deep
python -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
    --learner_type prompt --learner_name L2P \
    --prompt_param 30 20 -1 \
    --log_dir ${OUTDIR}/l2p++