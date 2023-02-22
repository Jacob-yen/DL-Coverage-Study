#!/bin/bash sh
nohup python -u -m coverage.rq2.rq2_runner --size 800 --dataset_models cifar10_resnet20 --sample_indices 1 2 3 4 5 --repeat_times 100 > logs/rq2.resnet20-1.`date "+%Y%m%d%H%M%S"`.out 2>&1 &
sleep 2
nohup python -u -m coverage.rq2.rq2_runner --size 800 --dataset_models cifar10_resnet20 --sample_indices 6 7 8 9 10 --repeat_times 100 > logs/rq2.resnet20-2.`date "+%Y%m%d%H%M%S"`.out 2>&1 &
sleep 2
nohup python -u -m coverage.rq2.rq2_runner --size 800 --dataset_models mnist_lenet5 fashion-mnist_lenet5 cifar10_vgg16 cifar10_alexnet --repeat_times 100 > logs/rq2.4-models.`date "+%Y%m%d%H%M%S"`.out 2>&1 &
sleep 2
nohup python -u -m coverage.rq2.rq2_runner --size 800 --dataset_models cifar100_resnet32 --repeat_times 100 --sample_indices 1 2 3 > logs/rq2.cifar100-1.`date "+%Y%m%d%H%M%S"`.out 2>&1 &
sleep 2
nohup python -u -m coverage.rq2.rq2_runner --size 800 --dataset_models cifar100_resnet32 --repeat_times 100 --sample_indices 4 5 6 > logs/rq2.cifar100-2.`date "+%Y%m%d%H%M%S"`.out 2>&1 &
sleep 2
nohup python -u -m coverage.rq2.rq2_runner --size 800 --dataset_models cifar100_resnet32 --repeat_times 100 --sample_indices 7 8 9 10 > logs/rq2.cifar100-3.`date "+%Y%m%d%H%M%S"`.out 2>&1 &
