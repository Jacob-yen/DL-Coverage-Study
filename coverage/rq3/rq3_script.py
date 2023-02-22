import argparse
import configparser
import os
import numpy as np
from datetime import datetime, date

from pandas import DataFrame

from coverage import root_dir
import coverage.tools.dataloader as dataloader
from coverage.tools import common_utils
import coverage.tools.model_utils as model_utils
from coverage.tools.coverage_utils import execute_sampling, SurpriseCoverage


def get_aggregated_indices(labels, select_idx):
    sampled_indices_list = []
    for class_id in select_idx:
        sampled_indices = np.nonzero(labels == class_id)[0]
        sampled_indices_list.append(sampled_indices)
    aggregated_indices = np.concatenate(sampled_indices_list)
    return aggregated_indices


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_capacity", help="number of images", type=int, default=800)
    parser.add_argument("--repeat_times", help="number of selected classes", type=int, default=2)
    parser.add_argument("--dataset_network", help="selected class id", type=str, default="cifar100_resnet32")
    parser.add_argument("--attack", help="adversarial attack", type=str, default="cw")
    parser.add_argument("--exp_date", help="data_of_exp", type=str,)
    parser.add_argument("--split_id", help="id number of select split", type=int, default=1)
    console_args = parser.parse_args()
    print(console_args)

    dataset_network = console_args.dataset_network

    exp_cfg = configparser.ConfigParser()
    coverage_parameters = {"n_bucket": 1000}
    exp_cfg.read(f"{root_dir}/config/exp.conf")
    total_group_nums = exp_cfg['parameters'].getint("group_nums")
    coverage_parameters["kmnc_k_section"] = exp_cfg['parameters'].getint("kmnc_k_section")
    coverage_parameters["tknc_k_value"] = exp_cfg['parameters'].getint("tknc_k_value")
    coverage_parameters["nc_threshold"] = exp_cfg['parameters'].getfloat("nc_threshold")
    coverage_parameters["idc_relevant_neurons"] = exp_cfg['parameters'].getint("idc_relevant_neurons")

    rq3_path = exp_cfg['parameters'].get("rq3_path")
    sa_dir_name = exp_cfg['parameters'].get("sa_intermediate")
    sa_intermedia_path = os.path.join(root_dir, sa_dir_name)
    idc_dir_name = exp_cfg['parameters'].get("idc_intermediate")
    idc_intermedia_path = os.path.join(root_dir, idc_dir_name)
    coverage_parameters["idc_intermedia_path"] = idc_intermedia_path
    console_args.exp_date = str(date.today()) if console_args.exp_date is None else console_args.exp_date
    dataset_network_dir = os.path.join(root_dir, rq3_path, console_args.exp_date, dataset_network)
    common_utils.create_path(sa_intermedia_path, idc_intermedia_path, rq3_path, dataset_network_dir)

    dataset_name, network_name = tuple(dataset_network.split("_"))
    num_classes = dataloader.class_num(dataset_name)
    test_sizes = dataloader.test_sizes[dataset_name]

    s0 = datetime.now()
    # load model and boundary
    classifier = model_utils.load_model(network=network_name, dataset=dataset_name)
    boundary = common_utils.load_boundary(dataset_name, network_name)
    # direct use `size_per_class` correctly classified images
    x_test, y_test = dataloader.load_dataset(dataset_name)
    x_test = dataloader.preprocess_dataset(dataset_name, network_name, x_test)
    print(f"INFO: {dataset_name, network_name} value range of clean images :[{np.min(x_test)},{np.max(x_test)}]")

    # the adversarial inputs are already preprocessed.
    adv_x, adv_y = dataloader.load_adversarial_images(dataset_name, network_name, console_args.attack, mode="full")
    print(f"INFO: {dataset_name, network_name} value range of adv images :[{np.min(adv_x)},{np.max(adv_x)}]")

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
    print(f"Data & Model preparing time:{datetime.now() - s0}")

    sampling_indices = common_utils.sampling_indices_dict(500, dataset_model=dataset_network,
                                                          test_size=console_args.sample_capacity)
    correct_indices = sampling_indices['pure_correct_indices']
    pure_correct_labels = y_test[correct_indices].copy()

    # we divide the classes into ten splits
    section_num = 10
    class_ids = np.arange(num_classes)
    section_length = int(num_classes / section_num)

    adv_lsa, adv_dsa, adv_mdsa = common_utils.cached_sa(dataset_network=dataset_network,
                                                        attack_type=console_args.attack,
                                                        test_size=test_sizes)
    clean_lsa, clean_dsa, clean_mdsa = common_utils.cached_sa(dataset_network=dataset_network,
                                                              attack_type="normal",
                                                              test_size=test_sizes)
    sa_dict = dict()
    sa_dict["clean_lsa"], sa_dict["adv_lsa"] = clean_lsa, adv_lsa
    sa_dict["clean_dsa"], sa_dict["adv_dsa"] = clean_dsa, adv_dsa
    sa_dict["clean_mdsa"], sa_dict["adv_mdsa"] = clean_mdsa, adv_mdsa
    sa_dict["lsa_boundary"] = SurpriseCoverage.filter_outliers("LSA",np.concatenate([clean_lsa,adv_lsa]).copy())
    sa_dict["dsa_boundary"] = SurpriseCoverage.filter_outliers("DSA",np.concatenate([clean_dsa,adv_dsa]).copy())
    sa_dict["mdsa_boundary"] = SurpriseCoverage.filter_outliers("MDSA",np.concatenate([clean_mdsa,adv_mdsa]).copy())

    start_class_id = int(section_length * console_args.split_id)
    top_idx = class_ids[start_class_id:start_class_id + section_length]
    print(f"Selecting spilt:{console_args.split_id},classes:{top_idx}")
    df_titles = ["Sampling_Name", "correct_proportion", "NC", "NBC", "SNAC", "TKNC", 'KMNC', "LSC", "DSC", "MDSC",
                 "IDC", "error_rate"]
    df_path = os.path.join(dataset_network_dir,
                           f"{console_args.dataset_network}_{console_args.attack}_size{console_args.sample_capacity}"
                           f"_class_ratio-split{console_args.split_id}.xlsx")

    df = DataFrame(columns=df_titles)
    row_id = 0

    _aggregated_correct_idx = get_aggregated_indices(pure_correct_labels, top_idx)
    aggregated_correct_idx = correct_indices[_aggregated_correct_idx]
    aggregated_wrong_idx = get_aggregated_indices(adv_y, top_idx)

    s0 = datetime.now()
    for rid in range(console_args.repeat_times):
        if len(aggregated_correct_idx) >= console_args.sample_capacity:
            adv_minimum = 0
        else:
            adv_minimum = console_args.sample_capacity - len(aggregated_correct_idx)
        adv_maximum = int(console_args.sample_capacity * 0.7)
        assert adv_maximum > adv_minimum, f"Maximum {adv_maximum} <= Minimum {adv_minimum}. " \
                                          f"Only {len(aggregated_correct_idx)} correct inputs are found."
        wrong_num = np.random.randint(low=adv_minimum, high=adv_maximum + 1)
        correct_num = console_args.sample_capacity - wrong_num
        print(f"Repeat times: {rid} of {console_args.repeat_times}, correct: {correct_num}, wrong: {wrong_num}")
        select_correct_idx = np.random.choice(a=aggregated_correct_idx, size=correct_num, replace=False)
        select_wrong_idx = np.random.choice(a=aggregated_wrong_idx, size=wrong_num, replace=False)
        select_correct_inputs, select_correct_labels = \
            x_test[select_correct_idx].copy(), y_test[select_correct_idx].copy()
        select_wrong_inputs, select_wrong_labels = \
            adv_x[select_wrong_idx].copy(), adv_y[select_wrong_idx].copy()
        selected_x = np.concatenate([select_correct_inputs, select_wrong_inputs])
        selected_y = np.concatenate([select_correct_labels, select_wrong_labels])
        row = execute_sampling(dataset_network=dataset_network, classifier=classifier, x=selected_x, y=selected_y,
                               train_inputs=x_train, train_labels=y_train, boundary=boundary, sa_dict=sa_dict,
                               coverage_parameters=coverage_parameters, normal_indices=select_correct_idx,
                               adv_indices=select_wrong_idx,classification=True)

        row_str = [round(rate, 2) for rate in row]
        sampling_row = [f"sample{console_args.split_id}_repeat_{rid}",
                        round(correct_num / console_args.sample_capacity, 2)]
        sampling_row.extend(row_str)
        df.loc[row_id] = sampling_row
        row_id += 1
    df.to_excel(df_path)

    elapsed = (datetime.now() - s0)
    print(f"RQ2 Time used for {dataset_network}-{console_args.attack} ", elapsed)
