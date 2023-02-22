import argparse
import numpy as np
import pandas as pd
from pandas import DataFrame
from coverage import root_dir
from coverage.tools.coverage_utils import SurpriseCoverage, execute_sampling
from coverage.tools import dataloader, model_utils, common_utils
from collections import defaultdict
from datetime import datetime, date
import configparser
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"


def split_data_by_ratio(clean_x, clean_y, adv_x, adv_y, group_id):
    """
    get group data from clean data and adversarial data
    Args:
        clean_x:
        clean_y:
        adv_x:
        adv_y:
        p: ratio of adversarial
        labels: files that store the indices information of group
        group_id:

    Returns:
        input_dict
    """
    _loop_time_mark = datetime.now()
    if console_args.target == 'nature':
        mixed_x = clean_x[nature_samples[group_id]]
        mixed_y = clean_y[nature_samples[group_id]]
    else:
        normal_x = clean_x[mix_normal_indices[group_id]]
        normal_y = clean_y[mix_normal_indices[group_id]]

        adversarial_x = adv_x[mix_adv_indices[group_id]]
        adversarial_y = adv_y[mix_adv_indices[group_id]]

        mixed_x = np.concatenate([normal_x, adversarial_x])
        mixed_y = np.concatenate([normal_y, adversarial_y])
    print(f"### Split data done: {datetime.now() - _loop_time_mark}")
    print(f"### Shape of {console_args.target} datasets: {mixed_x.shape}")
    return mixed_x, mixed_y


if __name__ == "__main__":
    start = datetime.now()

    parser = argparse.ArgumentParser()
    parser.add_argument("--size", "-size", help="the number of each group", type=int, default=800)
    parser.add_argument("--exp_date", "-exp_date", help="experiment date", type=str,default="test0115")
    parser.add_argument("--dataset", "-d", help="Dataset", type=str, default="mnist")
    parser.add_argument("--network", "-n", help="Network", type=str, default="lenet5")
    parser.add_argument("--start_group", "-start_group", help="The number of start_group", type=int, default=0)
    parser.add_argument("--group_count", "-group_count", help="The number of group to tun", type=int, default=10)
    parser.add_argument("--target", "-target", help="Target input set (test or adversarial set)", type=str,
                        default="cw")
    console_args = parser.parse_args()
    print(console_args)

    exp_cfg = configparser.ConfigParser()
    coverage_parameters = {"n_bucket": 1000}
    exp_cfg.read(f"{root_dir}/config/exp.conf")
    total_group_nums = exp_cfg['parameters'].getint("group_nums")
    coverage_parameters["kmnc_k_section"] = exp_cfg['parameters'].getint("kmnc_k_section")
    coverage_parameters["tknc_k_value"] = exp_cfg['parameters'].getint("tknc_k_value")
    coverage_parameters["nc_threshold"] = exp_cfg['parameters'].getfloat("nc_threshold")
    coverage_parameters["idc_relevant_neurons"] = exp_cfg['parameters'].getint("idc_relevant_neurons")

    rq1_path = exp_cfg['parameters'].get("rq1_path")
    sa_dir_name = exp_cfg['parameters'].get("sa_intermediate")
    sa_intermedia_path = os.path.join(root_dir, sa_dir_name)
    idc_dir_name = exp_cfg['parameters'].get("idc_intermediate")
    idc_intermedia_path = os.path.join(root_dir, idc_dir_name)
    coverage_parameters["idc_intermedia_path"] = idc_intermedia_path
    console_args.exp_date = str(date.today()) if console_args.exp_date is None else console_args.exp_date
    dataset_network = f"{console_args.dataset}_{console_args.network}"
    dataset_network_dir = os.path.join(root_dir, rq1_path, console_args.exp_date,dataset_network)

    common_utils.create_path(sa_intermedia_path, idc_intermedia_path, rq1_path, dataset_network_dir)

    s0 = datetime.now()
    classifier = model_utils.load_model(network=console_args.network, dataset=console_args.dataset)
    num_classes = dataloader.class_num(console_args.dataset)
    test_sizes = dataloader.test_sizes[console_args.dataset]
    adv_size = dataloader.adv_sizes[console_args.dataset]
    classification = True if dataset_network not in ["driving_dave-orig","driving_dave-dropout"] else False
    sampling_indices = common_utils.sampling_indices_dict(groups_num=500, dataset_model=dataset_network,
                                                          test_size=console_args.size)
    adv_samples_percent = sampling_indices['p']
    mix_adv_indices = sampling_indices["mix_adv_indices"]
    mix_normal_indices = sampling_indices["mix_normal_indices"]
    nature_samples = sampling_indices["nature_samples"]

    boundary = common_utils.load_boundary(dataset=console_args.dataset, network=console_args.network)

    # direct use `size_per_class` correctly classified images
    # clean images are standardized and need to be preprocessed.
    x_test, y_test = dataloader.load_dataset(console_args.dataset)
    x_test = dataloader.preprocess_dataset(console_args.dataset, console_args.network, x_test)
    print(f"INFO: {console_args.dataset, console_args.network} "
          f"value range of clean images :[{np.min(x_test)},{np.max(x_test)}]")


    # the adversarial inputs are already preprocessed.
    adv_x, adv_y = dataloader.load_adversarial_images(console_args.dataset, console_args.network,
                                                      console_args.target, mode="full")
    print(f"INFO: {console_args.dataset, console_args.network} "
          f"value range of adv images :[{np.min(adv_x)},{np.max(adv_x)}]")

    # I skip loading train set here. We don't need train-set because we have generated SA and IDC intermediate files
    skip_train = True
    if skip_train:
        x_train = y_train = None
    else:
        # note that the y_train is not in one-vector format. It's just an array of class ids.
        x_train, y_train = dataloader.load_train_set(console_args.dataset)
        x_train = dataloader.preprocess_dataset(console_args.dataset, console_args.network, x_train)
        print(f"INFO: {console_args.dataset, console_args.network} "
              f"value range of train images :[{np.min(x_train)},{np.max(x_train)}]")

    coverage_dict = defaultdict(float)
    df_titles = ["Group ID", "p", "NC", "NBC", "SNAC", "TKNC", 'KMNC', "LSC", "DSC", "MDSC",
                 "IDC", "error_rate"]
    df_path = os.path.join(dataset_network_dir,
                           f"nc_{console_args.dataset}_{console_args.network}-{console_args.size}samples"
                           f"-500groups-{console_args.target}_"
                           f"sg{console_args.start_group}_gc{console_args.group_count}.xlsx")
    adv_lsa, adv_dsa, adv_mdsa = common_utils.cached_sa(dataset_network=dataset_network,
                                                        attack_type=console_args.target,
                                                        test_size=adv_size,classification=classification)
    clean_lsa, clean_dsa, clean_mdsa = common_utils.cached_sa(dataset_network=dataset_network,
                                                              attack_type="normal",
                                                              test_size=test_sizes,classification=classification)
    sa_dict = dict()
    sa_dict["clean_lsa"], sa_dict["adv_lsa"] = clean_lsa, adv_lsa
    sa_dict["clean_dsa"], sa_dict["adv_dsa"] = clean_dsa, adv_dsa
    sa_dict["clean_mdsa"], sa_dict["adv_mdsa"] = clean_mdsa, adv_mdsa

    sa_dict["lsa_boundary"] = SurpriseCoverage.filter_outliers("LSA",np.concatenate([clean_lsa,adv_lsa]).copy())
    if classification:
        sa_dict["dsa_boundary"] = SurpriseCoverage.filter_outliers("DSA",np.concatenate([clean_dsa,adv_dsa]).copy())
        sa_dict["mdsa_boundary"] = SurpriseCoverage.filter_outliers("MDSA",np.concatenate([clean_mdsa,adv_mdsa]).copy())

    # set dataframe columns
    # use list to get the list of generator in python.
    # if not os.path.exists(df_path):
    df = DataFrame(columns=df_titles)
    row_id = 0
    # else:
    #     df = pd.read_excel(df_path, index_col=0, header=0)
    #     row_id = df.shape[0]
    print(
        f"INFO: Getting results of group from {console_args.start_group} to {console_args.start_group + console_args.group_count}")
    for group_id in range(console_args.start_group, console_args.start_group + console_args.group_count):
        print(f"current progress: {group_id - console_args.start_group} of {console_args.group_count}")
        # p is the percentage of the number of adversarial samples to the total (1000)
        if console_args.target == 'nature':
            percent = 0
            raise ValueError("currently we don't support nature for rq1")
        else:
            percent = adv_samples_percent[group_id]
        # TODO: currently we don't support nature for rq1
        normal_indices = mix_normal_indices[group_id]
        adv_indices = mix_adv_indices[group_id]

        selected_x, selected_y = split_data_by_ratio(x_test, y_test, adv_x, adv_y, group_id)
        loop_time_mark = datetime.now()

        row = execute_sampling(dataset_network=dataset_network, classifier=classifier, x=selected_x, y=selected_y,
                               train_inputs=x_train, train_labels=y_train, boundary=boundary, sa_dict=sa_dict,
                               coverage_parameters=coverage_parameters, normal_indices=normal_indices,
                               adv_indices=adv_indices,classification=classification)

        row_str = [round(rate, 4) for rate in row]
        sampling_row = [f"group_{group_id}", round(percent, 4)]
        sampling_row.extend(row_str)
        df.loc[row_id] = sampling_row
        row_id += 1
    df.to_excel(df_path)

    elapsed = (datetime.now() - start)
    print("Total Time used: ", elapsed)
