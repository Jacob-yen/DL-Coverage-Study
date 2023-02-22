import configparser
import os
import shutil
from datetime import datetime, date
import argparse
from coverage import root_dir
import numpy as np
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # NOTE: uses 1000 as test_size as default
    parser.add_argument("--size", "-size", help="the number of each group", type=int, default=800)
    parser.add_argument("--dataset_models", "-dms", help="dataset models to run", type=str, nargs='+')
    parser.add_argument("--repeat_times", help="number of selected classes", type=int, default=2)
    parser.add_argument("--split_ids", help="split_ids", type=int, nargs='+', choices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    args = parser.parse_args()
    print(args)

    general_targets = ['cw', 'bim', 'jsma', 'fgsm']
    driving_targets = ['nature', 'black', 'light']  # ,'black','light'
    speech_commands_target = ['ctcm']

    exp_cfg = configparser.ConfigParser()
    exp_cfg.read(f"{root_dir}/config/exp.conf")
    rq3_path = exp_cfg['parameters'].get("rq3_path")
    total_group_nums = exp_cfg['parameters'].getint("group_nums")
    # test_total_group_nums = 10
    repeat_one_run = total_group_nums
    script_repeat_time = int(total_group_nums / repeat_one_run)

    exp_date = str(date.today())
    s0 = datetime.now()
    split_ids = np.random.choice(a=list(range(10)), size=5, replace=False) \
        if args.split_ids is None else args.split_ids
    for dataset_network in args.dataset_models:

        s = datetime.now()
        dataset, network = tuple(dataset_network.split("_"))
        dataset_network = f"{dataset}_{network}"
        exp_path = os.path.join(root_dir, rq3_path, exp_date)
        dataset_network_dir = os.path.join(exp_path,dataset_network)
        if not os.path.exists(dataset_network_dir):
            os.makedirs(dataset_network_dir)
        if dataset == 'driving':
            targets = driving_targets
        elif dataset == 'speech-commands':
            targets = speech_commands_target
        else:
            targets = general_targets
        for selected_target in targets:
            print(f"#####Testing {network} on {dataset} target {selected_target}#####")
            for split_id in split_ids:
                command = f"python -u -m coverage.rq3.rq3_script --sample_capacity {args.size} " \
                          f"--dataset_network {dataset_network} --exp_date {exp_date} --repeat_times {args.repeat_times} " \
                          f"--attack {selected_target} --split_id {split_id}"
                print(command)
                os.system(command)
        print(f"Time cost for {network} {dataset}: {datetime.now() - s}")
    print(f"Overall Time cost for {len(args.dataset_models)} models: {datetime.now() - s0}")
