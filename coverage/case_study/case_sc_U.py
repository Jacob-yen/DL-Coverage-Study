import os
import sys
import numpy as np
from datetime import datetime
from coverage import root_dir
from pandas import DataFrame
from tqdm import tqdm
import matplotlib.pyplot as plt
from coverage.tools import model_utils, dataloader, common_utils
from coverage.tools.common_utils import ScoreUtils
from coverage.tools.coverage_utils import SurpriseCoverage


def get_surprise_coverage(clean_sa, adv_sa, clean_idx, adv_idx, n_bucket, lower, upper):
    target_sa = [clean_sa[i] for i in clean_idx]
    selected_sa_adv = [adv_sa[i] for i in adv_idx]
    target_sa.extend(selected_sa_adv)
    return SurpriseCoverage.surprise_coverage(target_sa=target_sa.copy(), n_bucket=n_bucket, lower=lower, upper=upper)


if __name__ == "__main__":
    # set dataset_model, adv
    dataset_models = ["mnist_lenet5", "fashion-mnist_lenet5", "cifar10_alexnet", "cifar10_resnet20",
                      "cifar10_vgg16", "cifar100_resnet32", "speech-commands_deepspeech", "driving_dave-orig","driving_dave-dropout"][3:4]

    # attacks = ["cw", "fgsm", "bim", "jsma"]
    # cov_names = ["lsc", "dsc", "mdsc"]
    group_num = 500
    df_titles = ["p", "max_value", "three_sigma", "1%_value","3%_value","5%_value","7%_value","10%_value"]
    boundary_types = ["max_value", "three_sigma", "1%_value","3%_value","5%_value","7%_value","10%_value"]
    case_study_dir = os.path.join(root_dir, "results/case_study_sc_20220129_lsc_dsc")
    if not os.path.exists(case_study_dir):
        os.makedirs(case_study_dir)
    for dataset_model in dataset_models:
        dataset, network = tuple(dataset_model.split("_"))
        print("=============",dataset_model,"=============")
        # load dataset
        s0 = datetime.now()
        # load model
        num_classes = dataloader.class_num(dataset)
        test_sizes = dataloader.test_sizes[dataset]
        adv_size = dataloader.adv_sizes[dataset]
        # load indices_dict
        sampling_indices = common_utils.sampling_indices_dict(groups_num=500, dataset_model=dataset_model,
                                                              test_size=800)
        adv_samples_percent = sampling_indices['p']
        mix_adv_indices = sampling_indices["mix_adv_indices"]
        mix_normal_indices = sampling_indices["mix_normal_indices"]
        nature_samples = sampling_indices["nature_samples"]

        # need to get error rate
        if dataset_model in ["speech-commands_deepspeech", "driving_dave-orig","driving_dave-dropout"]:
            cov_names = ["lsc", "dsc"]
            classifier = model_utils.load_model(network=network, dataset=dataset)
            # direct use `size_per_class` correctly classified images
            # clean images are standardized and need to be preprocessed.
            x_test, y_test = dataloader.load_dataset(dataset)
            x_test = dataloader.preprocess_dataset(dataset, network, x_test)
            print(f"INFO: {dataset, network} value range of clean images :[{np.min(x_test)},{np.max(x_test)}]")

        if dataset_model == "speech-commands_deepspeech":
            classification = True
            attacks = ["ctcm"]
            cov_names = ["lsc", "dsc", "mdsc"]
        elif dataset_model in ["driving_dave-orig","driving_dave-dropout"]:
            classification = False
            attacks = ["light", "black"]
            cov_names = ["lsc"]
        else:
            classification = True
            cov_names = ["lsc", "dsc"]
            attacks = ["cw","fgsm","bim","jsma"]
        for attack in attacks:
            print(f"<<<<<<<<<<<<Attack {attack}>>>>>>>>>>>>>>>>>")
            if dataset_model in ["speech-commands_deepspeech", "driving_dave-orig","driving_dave-dropout"]:
                adv_x, adv_y = dataloader.load_adversarial_images(dataset, network, attack, mode="full")
                print(f"INFO: {dataset, network} value range of adv images :[{np.min(adv_x)},{np.max(adv_x)}]")

            sa_dict = dict()
            adv_lsa, adv_dsa, adv_mdsa = common_utils.cached_sa(dataset_network=dataset_model, attack_type=attack,
                                                                test_size=adv_size, classification=classification)
            clean_lsa, clean_dsa, clean_mdsa = common_utils.cached_sa(dataset_network=dataset_model,
                                                                      attack_type="normal",
                                                                      test_size=test_sizes,
                                                                      classification=classification)
            sa_dict["clean_lsc"], sa_dict["adv_lsc"] = clean_lsa, adv_lsa
            sa_dict["clean_dsc"], sa_dict["adv_dsc"] = clean_dsa, adv_dsa
            sa_dict["clean_mdsc"], sa_dict["adv_mdsc"] = clean_mdsa, adv_mdsa
            sa_boundary = {cov_n: dict() for cov_n in cov_names}
            sa_res_dict = {cov_n: DataFrame(columns=df_titles) for cov_n in cov_names}

            for cov_name in cov_names:
                sa_list = []
                clean_sa, adv_sa = sa_dict[f"clean_{cov_name}"], sa_dict[f"adv_{cov_name}"]
                total_sa = np.concatenate([clean_sa, adv_sa]).copy()
                # for each test set.
                print(f"SA of {cov_name.upper()}: [{np.min(total_sa)},{np.max(total_sa)}]")
                # get different upper

                # ====get sc when U is the max value
                upper2 = np.max(total_sa)
                sa_boundary[cov_name]["max_value"] = (lower, upper2)
                print("MAX Value", lower, upper1)
                # ====get sc when U is the 1%
                count = int(len(total_sa) * 0.01)
                position = np.argsort(total_sa)[-count]
                upper3 = total_sa[position]
                print("1% quantile", lower, upper3)
                sa_boundary[cov_name]["1%_value"] = (lower, upper3)
                # ====get sc when U is the 3%
                count = int(len(total_sa) * 0.03)
                position = np.argsort(total_sa)[-count]
                upper4 = total_sa[position]
                print("3% value", lower, upper4)
                sa_boundary[cov_name]["3%_value"] = (lower, upper4)
                # ====get sc when U is the 5%
                count = int(len(total_sa) * 0.05)
                position = np.argsort(total_sa)[-count]
                upper5 = total_sa[position]
                print("5% value", lower, upper5)
                sa_boundary[cov_name]["5%_value"] = (lower, upper5)
                # ====get sc when U is the 7%
                count = int(len(total_sa) * 0.07)
                position = np.argsort(total_sa)[-count]
                upper6 = total_sa[position]
                print("7% quantile", lower, upper6)
                sa_boundary[cov_name]["7%_value"] = (lower, upper6)
                # ====get sc when U is the 10%
                count = int(len(total_sa) * 0.1)
                position = np.argsort(total_sa)[-count]
                upper7 = total_sa[position]
                print("10% value", lower, upper2)
                sa_boundary[cov_name]["10%_value"] = (lower, upper7)
                print(cov_name)
                for idx,boundary_type in enumerate(boundary_types):
                    # plt.subplot(2,len(boundary_types),idx+1)
                    # plt.scatter(list(range(len(total_sa))), total_sa)
                    # plt.hlines(sa_boundary[cov_name][boundary_type][0], 0, len(total_sa), colors="red")
                    # plt.hlines(sa_boundary[cov_name][boundary_type][1], 0, len(total_sa),colors="red")
                    # plt.title(f"LSA:" + boundary_type)
                    # plt.show()
                    print(boundary_type,round(sa_boundary[cov_name][boundary_type][1],2))
            # sys.exit(0)
            for group_id in tqdm(range(group_num)):
                if dataset_model in ["speech-commands_deepspeech", "driving_dave-orig","driving_dave-dropout"]:
                    normal_x = x_test[mix_normal_indices[group_id]].copy()
                    normal_y = y_test[mix_normal_indices[group_id]].copy()

                    adversarial_x = adv_x[mix_adv_indices[group_id]].copy()
                    adversarial_y = adv_y[mix_adv_indices[group_id]].copy()

                    mixed_x = np.concatenate([normal_x, adversarial_x])
                    mixed_y = np.concatenate([normal_y, adversarial_y])

                    error_rate = ScoreUtils.get_model_error_rate(x_test=mixed_x, y_test=mixed_y, model=classifier,
                                                                 dataset=dataset, network=network)
                else:
                    error_rate = adv_samples_percent[group_id]
                normal_indices = mix_normal_indices[group_id]
                adv_indices = mix_adv_indices[group_id]

                for cov_name in cov_names:
                    clean_sa, adv_sa = sa_dict[f"clean_{cov_name}"], sa_dict[f"adv_{cov_name}"]
                    row = [adv_samples_percent[group_id]]
                    for boundary_type in boundary_types:
                        l, u = sa_boundary[cov_name][boundary_type]
                        sc = get_surprise_coverage(clean_sa=clean_sa, adv_sa=adv_sa, clean_idx=normal_indices,
                                                   adv_idx=adv_indices,
                                                   lower=l, upper=u, n_bucket=1000)
                        row.append(sc)
                    row.append(error_rate)
                    sa_res_dict[cov_name].loc[group_id] = row

            for cov_name in cov_names:
                df = sa_res_dict[cov_name].copy()
                df = df.sort_values(by='error_rate', ascending=True)
                for idx,boundary_type in enumerate(boundary_types):
                    sc_list = df[boundary_type].values.tolist()
                    plt.subplot(2,len(boundary_types),idx+len(boundary_types)+1)
                    plt.plot(list(range(len(sc_list))), sc_list)
                    plt.title(f"LSC_{boundary_type}")
                    # plt.show()
                df.to_excel(os.path.join(case_study_dir, f"case_study_{cov_name}_{dataset_model}_{attack}.xlsx"))
            plt.tight_layout()
            # plt.show()
        print(f"Time cost for {dataset_model}:{datetime.now() - s0}")
