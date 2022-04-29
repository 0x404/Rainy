#! /bin/bash

# test train workflow and evalute workflow
python3 launch.py --config configs/test.py --do_train --train_max_step 10
[ $? != '0' ] && exit 1

# test init from checkpoint
checkpoint=$(ls checkpoints/test | head -n 1)
python3 launch.py --config configs/test.py --do_train --train_max_step 10 --init_checkpoint checkpoints/test/$checkpoint

