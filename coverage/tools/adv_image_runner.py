import os
import sys
import argparse

from coverage.tools import dataloader

if __name__ == "__main__":

    """This script is only used for fashion-mnist,mnist,cifar10,cifar100,imagenet"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", help="", type=str, default="mnist",
                        choices=['mnist', 'cifar10', 'cifar100', 'imagenet', "fashion-mnist"])
    parser.add_argument("--attack_list", "-attack", help="attack type of ['fgsm', 'cw', 'jsma', 'bim']", type=str,
                        nargs='+',choices=['fgsm', 'cw', 'jsma', 'bim'] )
    parser.add_argument("--network", "-n", help="model", type=str, default="mnist")
    parser.add_argument("--start", "-s", help="start class", type=int, default=0)
    parser.add_argument("--end", "-e", help="end class", type=int, default="1000")
    args = parser.parse_args()

    dataset_model = f"{args.dataset}_{args.network}"
    for attack in args.attack_list:
        # for i in range(0,1000,100):
        #     s,e = i,i+100
        if args.dataset == "imagenet":
            class_selections = range(args.start, args.end+1, 20)
            for s,e in zip(class_selections,class_selections[1:]):
                command = f"python -u -m  coverage.tools.generate_adv_image " \
                          f"--dataset {args.dataset} --network {args.network} " \
                          f"--attack_list {attack} --start {s} --end {e}"
                print(command)
                os.system(command)
        else:
            class_num = dataloader.class_num(args.dataset)
            s, e = 0, class_num

            command = f"python -u -m  coverage.tools.generate_adv_image " \
                      f"--dataset {args.dataset} --network {args.network} " \
                      f"--attack_list {attack} --start {s} --end {e}"
            print(command)
            os.system(command)
