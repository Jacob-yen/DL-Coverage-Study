import configparser
import os
import shutil
from datetime import datetime, date
import argparse
from coverage import root_dir

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # NOTE: uses 1000 as test_size as default
    parser.add_argument("--size", "-size", help="the number of each group", type=int, default=800)
    parser.add_argument("--dataset_models", "-dms", help="dataset models to run", type=str, nargs='+')
    parser.add_argument("--start_group", "-sg", help="start_group", type=int, default=0)
    parser.add_argument("--group_count", "-gc", help="group_count", type=int)
    args = parser.parse_args()
    print(args)

    general_targets = ['cw', 'bim', 'jsma', 'fgsm']
    driving_targets = ['black', 'light']  # ,'black','light'
    speech_commands_target = ['ctcm']

    exp_cfg = configparser.ConfigParser()
    exp_cfg.read(f"{root_dir}/config/exp.conf")
    nc_result_path = exp_cfg['parameters'].get("rq1_path")
    total_group_nums = exp_cfg['parameters'].getint("group_nums")
    # test_total_group_nums = 10
    args.group_count = total_group_nums if args.group_count is None else args.group_count
    exp_date = str(date.today())
    s0 = datetime.now()
    for dataset_network in args.dataset_models:
        s = datetime.now()
        dataset, network = tuple(dataset_network.split("_"))
        dataset_network = f"{dataset}_{network}"
        exp_path = os.path.join(root_dir, nc_result_path, exp_date)
        dataset_network_dir = os.path.join(exp_path, dataset_network)
        if not os.path.exists(dataset_network_dir):
            os.makedirs(dataset_network_dir)
        #     shutil.rmtree(dataset_network_dir)
        # os.makedirs(dataset_network_dir)
        if dataset == 'driving':
            targets = driving_targets
        elif dataset == 'speech-commands':
            targets = speech_commands_target
        else:
            targets = general_targets
        for selected_target in targets:
            print(f"#####Testing {network} on {dataset} target {selected_target}#####")
            command = f"python -u -m coverage.rq1.rq1_script " \
                      f"--size {args.size} --dataset {dataset} --network {network} --exp_date {exp_date} " \
                      f"--target {selected_target} --start_group {args.start_group} --group_count {args.group_count}"
            print(command)
            os.system(command)
        print(f"Time cost for {network} {dataset}: {datetime.now() - s}")
    print(f"Overall Time cost for {len(args.dataset_models)} models: {datetime.now() - s0}")
