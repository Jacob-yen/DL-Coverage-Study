import os
import argparse
from coverage import root_dir
from coverage.tools import dataloader, model_utils, common_utils
import numpy as np
from datetime import datetime

from coverage.tools import dataloader
from coverage.tools.common_utils import ScoreUtils
from coverage.tools.coverage_utils import StructuralCoverage,SurpriseCoverage,ImportanceDrivenCoverage

if __name__ == "__main__":
    dataset = "mnist"
    network = "lenet5"
    n_bucket = 1000
    kmnc_k_section = 1000
    tknc_k_value = 10
    nc_threshold = 0.5
    idc_relevant_neurons = 8
    sa_dir_name = "sa_intermediate"
    sa_intermedia_path = os.path.join(root_dir, sa_dir_name)
    idc_dir_name = "files/idc_intermediate"
    idc_intermedia_path = os.path.join(root_dir, idc_dir_name)
    dataset_network = f"{dataset}_{network}"

    args = argparse.Namespace()
    args.save_path = sa_intermedia_path
    args.dataset = dataset
    args.network = network
    is_classification = True
    args.is_classification = is_classification
    args.num_classes = dataloader.class_num(dataset)
    # var_threshold is used in Surprise Coverage. 1e-5 is the default value recommended by the author.
    args.var_threshold = 1e-5

    common_utils.create_path(sa_intermedia_path, idc_intermedia_path)
    boundary = common_utils.load_boundary(dataset=dataset, network=network)
    model = model_utils.load_model(network=network, dataset=dataset)
    # direct use `size_per_class` correctly classified images
    # clean images are standardized and need to be preprocessed.
    x_test, y_test = dataloader.load_dataset(dataset)
    x_test, y_test = x_test[:100], y_test[:100]
    x_test = dataloader.preprocess_dataset(dataset, network, x_test)
    test_size = len(x_test)
    print(f"INFO: {dataset, network} value range of clean images :[{np.min(x_test)},{np.max(x_test)}]")

    x_train, y_train = dataloader.load_train_set(dataset)
    x_train = dataloader.preprocess_dataset(dataset, network, x_train)
    print(f"INFO: {dataset, network} value range of train images :[{np.min(x_train)},{np.max(x_train)}]")

    # #######################
    # # Structural coverage #
    # #######################
    s_coverage = StructuralCoverage(model=model, k_section=kmnc_k_section, k_value=tknc_k_value,
                                    boundary=boundary, threshold=nc_threshold)
    # neuron coverage includes four types of coverage:DNC,TKNC,NBC,SANC
    predictions = s_coverage.update_coverage(input_data=x_test.copy())
    _coverage_rate, _ = s_coverage.get_coverage()
    print(_coverage_rate)

    # ###########################
    # # Non-structural coverage #
    # ###########################

    sa_coverage = SurpriseCoverage(model=model, dataset=dataset, network=network, args=args)

    used_cached = False
    os.makedirs(os.path.join(root_dir, "files/cached_sa", dataset_network), exist_ok=True)
    if used_cached:
        clean_lsa, clean_dsa  = common_utils.cached_sa(dataset_network=dataset_network,
                                                         attack_type="normal",
                                                         test_size=test_size,
                                                         classification=is_classification)
    else:
        # if it is the first time to run, then save the intermediate results
        clean_lsa = sa_coverage.origin_lsa(x_train, x_test, f"{network}_normal_{test_size}")

        lsa_file = os.path.join(root_dir, "files/cached_sa",dataset_network,
                                f"lsa_{dataset_network}_normal_{test_size}.npy")
        # save clean lsa
        np.save(lsa_file, clean_lsa)
        if is_classification:
            clean_dsa = sa_coverage.origin_dsa(x_train, x_test, f"{network}_normal_{test_size}")
            dsa_file = os.path.join(root_dir, "files/cached_sa",dataset_network,
                                    f"dsa_{dataset_network}_normal_{test_size}.npy")
            # save clean dsa
            np.save(dsa_file, clean_dsa)

    lsa_lower, lsa_upper = SurpriseCoverage.filter_outliers("LSA", clean_lsa)
    if is_classification:
        dsa_lower, dsa_upper = SurpriseCoverage.filter_outliers("DSA", clean_dsa)

    lsc = sa_coverage.surprise_coverage(target_sa=clean_lsa, lower=lsa_lower, upper=lsa_upper, n_bucket=n_bucket)

    dsc = sa_coverage.surprise_coverage(target_sa=clean_dsa, lower=dsa_lower, upper=dsa_upper, n_bucket=n_bucket)

    print("LSC", lsc)
    print("DSC", dsc)


    ##############################
    # Importance Driven coverage #
    ##############################
    idc_coverage = ImportanceDrivenCoverage(model=model, dataset_network=dataset_network,
                                            num_relevant_neurons=idc_relevant_neurons, train_inputs=x_train,
                                            train_labels=y_train, idc_save_path=idc_intermedia_path)

    idc, _, _ = idc_coverage.test(x_test)
    print("IDC", idc)

