#!/bin/bash sh
echo "Starting RQ1 experiments"
nohup python -u -m coverage.rq1.rq1_runner --size 800 --dataset_models speech-commands_deepspeech  > logs/rq1.speech`date "+%Y%m%d%H%M%S"`.out 2>&1 &
sleep 3
nohup python -u -m coverage.rq1.rq1_runner --size 800 --dataset_models mnist_lenet5 fashion-mnist_lenet5 cifar10_vgg16 driving_dave-orig driving_dave-dropout  > logs/rq1.5-models`date "+%Y%m%d%H%M%S"`.out 2>&1 &
sleep 3
nohup python -u -m coverage.rq1.rq1_runner --size 800 --dataset_models cifar10_alexnet cifar10_resnet20 > logs/rq1.alexnet-resnet20-models`date "+%Y%m%d%H%M%S"`.out 2>&1 &
sleep 3
nohup python -u -m coverage.rq1.rq1_runner --size 800 --dataset_models cifar100_resnet32 --start_group 0 --group_count 250 > logs/rq1.cifar100-1.`date "+%Y%m%d%H%M%S"`.out 2>&1 &
sleep 3
nohup python -u -m coverage.rq1.rq1_runner --size 800 --dataset_models cifar100_resnet32 --start_group 250 --group_count 250 > logs/rq1.cifar100-2.`date "+%Y%m%d%H%M%S"`.out 2>&1 &
sleep 3
