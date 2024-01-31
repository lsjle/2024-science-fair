#/bin/sh

CUDA_VISIBLE_DEVICES='3' nohup python allinonetrustful.py > stdout.log 2> allinone.err < /dev/null &
