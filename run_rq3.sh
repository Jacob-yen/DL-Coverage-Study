#!/bin/bash sh

nohup python -u -m coverage.rq3.rq3_runner --size 800 --dataset_models mnist_lenet5 fashion-mnist_lenet5 cifar10_vgg16 cifar10_alexnet --repeat_times 100 > logs/rq3.4-models.`date "+%Y%m%d%H%M%S"`.out 2>&1 &
sleep 2
nohup python -u -m coverage.rq3.rq3_runner --size 800 --dataset_models cifar10_resnet20 --repeat_times 100 > logs/rq3.resnet20.`date "+%Y%m%d%H%M%S"`.out 2>&1 &
sleep 2
nohup python -u -m coverage.rq3.rq3_runner --size 800 --dataset_models cifar100_resnet32 --repeat_times 100 --split_ids 4 5 > logs/rq3.cifar100-1.`date "+%Y%m%d%H%M%S"`.out 2>&1 &
sleep 2
nohup python -u -m coverage.rq3.rq3_runner --size 800 --dataset_models cifar100_resnet32 --repeat_times 100 --split_ids 7 9 > logs/rq3.cifar100-2.`date "+%Y%m%d%H%M%S"`.out 2>&1 &
sleep 2
nohup python -u -m coverage.rq3.rq3_runner --size 800 --dataset_models cifar100_resnet32 --repeat_times 100 --split_ids 2 > logs/rq3.cifar100-3.`date "+%Y%m%d%H%M%S"`.out 2>&1 &
