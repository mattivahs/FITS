#!/bin/bash

# LQR Experiment.

SYS='quadrotor_2D'

TASK='tracking'

ALGO='lqr'

SF='vanillaCBF'

if [ "$SYS" == 'cartpole' ]; then
    SYS_NAME=$SYS
else
    SYS_NAME='quadrotor'
fi

python3 ./cbf_experiment.py \
    --task ${SYS_NAME} \
    --algo ${ALGO} \
    --safety_filter ${SF} \
    --overrides \
        ./config_overrides/${SYS}/${SYS}_${TASK}.yaml \
        ./config_overrides/${SYS}/${ALGO}_${SYS}_${TASK}.yaml
