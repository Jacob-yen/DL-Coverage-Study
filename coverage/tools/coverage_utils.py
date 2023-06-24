import os
from collections import namedtuple
from functools import partial
from multiprocessing import Pool
from typing import List
import keras
import numpy as np
from keras.models import Model
from datetime import datetime
from sklearn import cluster
from sklearn.metrics import silhouette_score
from tqdm import tqdm
from collections import defaultdict

from coverage.tools import dataloader
from coverage.tools.common_utils import ScoreUtils
from coverage.tools.idc.lrp_toolbox.model_io import write, read
from coverage.tools.surprise_adequacy.sa import fetch_lsa, sa_selected_layers, get_sc, fetch_dsa, fetch_mdsa
from coverage.tools.surprise_adequacy.surprise_adequacy import SurpriseAdequacyConfig, LSA, DSA

Args = namedtuple("Args",
                  ["save_path", "dataset", "network", "var_threshold",
                   "n_bucket", "num_classes", "is_classification", ])
na = np.newaxis


def clustering(cluster_num, output):
    kmeans_cluster = cluster.KMeans(n_clusters=cluster_num)
    cluster_labels = kmeans_cluster.fit_predict(
        np.array(output).reshape(-1, 1))
    silhouette_avg = silhouette_score(
        np.array(output).reshape(-1, 1), cluster_labels)
    return silhouette_avg, kmeans_cluster


class BaseCoverage:

    def __init__(self, model, k_section, k_value, boundary, threshold):
        """
        @param model: the prediction model
        @param k_section: section number used in coverage KMNC
        @param k_value: the number of top neurons used in TKNC
        @param boundary: boundary values of each neuron used in coverage KMNC
        @param threshold: threshold of neuron coverage.
        If activation value > threshold, the neuron is regarded as activated
        """

        self.k_value = k_value
        self.model = model
        self.k_section = k_section
        self.boundary = boundary
        self.threshold = threshold
        self.layer_names = [layer.name
                            for layer in self.model.layers
                            if 'flatten' not in layer.name and 'input' not in layer.name]

        self.intermediate_layer_model = Model(inputs=self.model.input,
                                              outputs=[self.model.get_layer(layer_name).output
                                                       for layer_name in self.layer_names])

        output_layers = [self.model.get_layer(
            layer_name).output for layer_name in self.layer_names]
        output_layers.append(self.model.output)
        #
        # self.intermediate_layer_model_list = [
        #     Model(inputs=self.model.input, outputs=output_layers)]
        self.inter_model = Model(
            inputs=self.model.input, outputs=output_layers)
        self.layer_index2name, self.layer_name2index = self.get_layer_mapping()
        self.model_layer_dict_nc = None
        self.model_layer_dict_nbc_lower = None
        self.model_layer_dict_nbc_upper = None
        self.model_layer_dict_snac = None
        self.model_layer_dict_tknc = None
        self.model_layer_dict_kmnc = None
        self.initialize()

    def initialize(self):
        self.model_layer_dict_nc = self.init_coverage_tables()
        self.model_layer_dict_nbc_lower = self.init_coverage_tables()
        self.model_layer_dict_nbc_upper = self.init_coverage_tables()
        self.model_layer_dict_snac = self.init_coverage_tables()
        self.model_layer_dict_tknc = self.init_coverage_tables()
        self.model_layer_dict_kmnc = self.init_kmnc_covered_array()

    def init_kmnc_covered_array(self):
        covered_array_dict = dict()
        for layer_id, layer_name in enumerate(self.layer_names):
            for neuron_idx in range(self.model.get_layer(layer_name).output.shape[-1]):
                covered_array_dict[(layer_id, neuron_idx)] = np.zeros(
                    self.k_section).astype('uint8')
        return covered_array_dict

    def get_layer_mapping(self):

        layer_index2name = dict()
        layer_name2index = dict()
        index = 0
        # ‘model.layers’ returns different results from enumerate(model.layers)
        # can't use enumerate here
        for layer in self.model.layers:
            if 'flatten' in layer.name or 'input' in layer.name:
                continue
            else:
                layer_index2name[index] = layer.name
                layer_name2index[layer.name] = index
            index += 1
        return layer_index2name, layer_name2index

    def init_coverage_tables(self):
        model_layer_dict = defaultdict(bool)
        for layer in self.model.layers:
            if 'flatten' in layer.name or 'input' in layer.name:
                continue
            """
            Copied from deepxplore. They only use last dimension
            https://github.com/peikexin9/deepxplore/blob/8795b89a458cbfde733820798802eed549ce66db/MNIST/utils.py#L56
            """
            for index in range(layer.output_shape[-1]):
                model_layer_dict[(layer.name, index)] = False
        return model_layer_dict

    def neuron_amount(self):
        return len(self.model_layer_dict_nc)

    def neuron_covered(self):
        covered_neurons = len(
            [v for v in self.model_layer_dict_nc.values() if v])
        boundary_covered_neurons_lower = len(
            [v for v in self.model_layer_dict_nbc_lower.values() if v])
        boundary_covered_neurons_upper = len(
            [v for v in self.model_layer_dict_nbc_upper.values() if v])
        strong_covered_neurons = len(
            [v for v in self.model_layer_dict_snac.values() if v])
        topk_covered_neurons = len(
            [v for v in self.model_layer_dict_tknc.values() if v])
        total_neurons = len(self.model_layer_dict_nc)
        return covered_neurons, boundary_covered_neurons_lower, boundary_covered_neurons_upper, \
            strong_covered_neurons, topk_covered_neurons, total_neurons

    def update_coverage(self, **kwargs):
        pass

    @staticmethod
    def scaler(intermediate_layer_output):
        def _scale(layer_outputs, rmax=1, rmin=0):
            value_range = (layer_outputs.max() - layer_outputs.min())
            if value_range == 0:
                return np.zeros(shape=layer_outputs.shape)
            else:
                """
                copied from DeepXplore:
                https://github.com/peikexin9/deepxplore/blob/8795b89a458cbfde733820798802eed549ce66db/MNIST/utils.py#L100
                """
                X_std = (layer_outputs - layer_outputs.min()) / value_range
                X_scaled = X_std * (rmax - rmin) + rmin
                return X_scaled

        scaled_output = []
        for out_for_input in intermediate_layer_output:
            scaled_output.append(_scale(out_for_input))
        return np.array(scaled_output)


class StructuralCoverage(BaseCoverage):

    def __init__(self, model, k_section, k_value, boundary, threshold):
        BaseCoverage.__init__(self, model, k_section,
                              k_value, boundary, threshold)

    def update_neuron_dict(self, layer_name, neuron_indices, cov_name):
        if cov_name == 'nc':
            update_dict = self.model_layer_dict_nc
        elif cov_name == 'nbc_lower':
            update_dict = self.model_layer_dict_nbc_lower
        elif cov_name == 'nbc_upper':
            update_dict = self.model_layer_dict_nbc_upper
        elif cov_name == 'snac':
            update_dict = self.model_layer_dict_snac
        elif cov_name == 'tknc':
            update_dict = self.model_layer_dict_tknc
        else:
            raise ValueError(f"No such coverage: {cov_name}")
        for neuron_index in neuron_indices:
            if (layer_name, neuron_index) not in update_dict.keys():
                raise ValueError((layer_name, neuron_index))
            update_dict[(layer_name, neuron_index)] = True

    def update_coverage(self, input_data):

        # Get the activation traces of the inner layers and the output of the final layer
        inter_layer_outputs: List[np.ndarray] = self.inter_model.predict(
            input_data, batch_size=128, verbose=1)
        dnn_output = inter_layer_outputs.pop()
        for idx, intermediate_layer_output in enumerate(tqdm(inter_layer_outputs)):
            time_mark = datetime.now()
            # if there only exists one neuron. skip it
            if intermediate_layer_output.shape[-1] <= 1:
                continue
            # scaled = intermediate_layer_output
            scaled_nc = self.scaler(intermediate_layer_output.copy())
            input_size, neuron_count = scaled_nc.shape[0], scaled_nc.shape[-1]
            layer_index, layer_name = idx, self.layer_index2name[idx]

            # get input_neuron matrix. shape is like (input_size,neuron_count)
            """
            The representation of neuron activation is followed by deepxplore[1].
            I'm sure the implementation below (employing vectorization) is equivalent to the way they use in deepxplore.

            [1] https://github.com/peikexin9/deepxplore/blob/master/MNIST/utils.py#L90
            """
            activations_mat_nc = np.mean(scaled_nc.reshape(
                (input_size, -1, neuron_count)), axis=1)

            if np.isnan(activations_mat_nc).any() or np.isinf(activations_mat_nc).any():
                raise ValueError(f"Found NaN or INF in output matrix!")

            # NC
            nc_check_array = np.max(activations_mat_nc, axis=0)
            # np.where returns tuple. get first because check_array is 1-dim
            nc_activated_index = np.where(nc_check_array > self.threshold)[0]
            # print(nc_activated_index)
            self.update_neuron_dict(layer_name, nc_activated_index, 'nc')
            # print(f"Time for NC: {datetime.now() - time_mark}")
            time_mark = datetime.now()

            lower_array = np.array([self.boundary[(idx, neuron_id)][0]
                                    for neuron_id in range(neuron_count)])
            upper_array = np.array([self.boundary[(idx, neuron_id)][1]
                                    for neuron_id in range(neuron_count)])

            # print(lower_array.shape,upper_array.shape)
            if np.isnan(lower_array).any() or np.isinf(lower_array).any() or \
                    np.isnan(upper_array).any() or np.isinf(upper_array).any():
                raise ValueError(f"Boundary for neurons in layer ({idx},{self.layer_index2name[idx]})"
                                 f" contain NaN or INF!")

            # scaled shape is like (1000,32,32,16) or (1000,512,16) ,etc.
            # input_size would <= batch_size. some input would be deleted.
            # output of some inputs would contain nan. if min==max
            input_size, neuron_count = intermediate_layer_output.shape[
                0], intermediate_layer_output.shape[-1]

            activations_mat = np.mean(intermediate_layer_output.reshape(
                (input_size, -1, neuron_count)), axis=1)

            if np.isnan(activations_mat).any() or np.isinf(activations_mat).any():
                raise ValueError(f"Found NaN or INF in output matrix!")

            # NBC
            nbc_lower_check_array = np.min(
                activations_mat, axis=0) - lower_array
            nbc_upper_check_array = np.max(
                activations_mat, axis=0) - upper_array
            nbc_lower_index = np.where(nbc_lower_check_array < 0)[0]
            nbc_upper_index = np.where(nbc_upper_check_array > 0)[0]
            self.update_neuron_dict(
                layer_name, nbc_upper_index, 'nbc_upper')
            self.update_neuron_dict(
                layer_name, nbc_lower_index, 'nbc_lower')
            # print(f"Time for NBC: {datetime.now() - time_mark}")
            time_mark = datetime.now()

            # SNAC
            snac_upper_check_array = np.max(
                activations_mat, axis=0) - upper_array
            snac_upper_index = np.where(snac_upper_check_array > 0)[0]
            self.update_neuron_dict(layer_name, snac_upper_index, 'snac')
            # print(f"Time for SNAC: {datetime.now() - time_mark}")
            time_mark = datetime.now()

            # TKNC
            neurons_ranks = np.argsort(activations_mat, axis=1)
            K_value = neuron_count if neuron_count < self.k_value else self.k_value
            top_k_neurons_ranks_flatten = neurons_ranks[..., -
                                                        K_value:].flatten()
            # use set() because we don't count repeatedly
            top_k_neurons_ranks_flatten_set = set(top_k_neurons_ranks_flatten)
            # print(top_k_neurons_ranks_flatten_set)
            self.update_neuron_dict(
                layer_name, top_k_neurons_ranks_flatten_set, 'tknc')
            # print(f"Time for TKNC: {datetime.now() - time_mark}")
            time_mark = datetime.now()

            # KMNC
            sec_length_array = (upper_array - lower_array) / self.k_section
            # get section id for each mean
            mean_sub_lower_array = activations_mat - lower_array
            # use nonzero to get indices of elements which are not equal to 0.
            # nonzero returns tuple
            non_zero_indices = np.nonzero(sec_length_array)[0]
            total_sec_ids = mean_sub_lower_array[...,
                                                 non_zero_indices] / sec_length_array[non_zero_indices]
            total_sec_ids = total_sec_ids.astype('int')

            # filter values: mean < lower
            # iter_idx is used to get the result of neuron_idx
            time_mark = datetime.now()
            for iter_idx, neuron_idx in enumerate(non_zero_indices):
                sec_ids = total_sec_ids[..., iter_idx].copy()
                sec_ids = sec_ids[sec_ids >= 0]
                # filter values: mean > upper
                sec_ids = sec_ids[sec_ids <= self.k_section]
                # if sec_id == k_section, it indicates that mean == upper. sec_id should be k_section - 1
                sec_ids[sec_ids == self.k_section] = self.k_section - 1
                # 0 represents uncovered and 1 represents covered
                self.model_layer_dict_kmnc[(idx, neuron_idx)][sec_ids] = 1
            # print(f"Time for KMNC: {datetime.now() - time_mark}")
        return dnn_output

    def get_coverage(self):
        # For NC,NBC,SANC TKNC
        covered_neurons, boundary_covered_neurons_lower, boundary_covered_neurons_upper, strong_covered_neurons, \
            topk_covered_neurons, total_neurons = self.neuron_covered()
        boundary_covered_neurons = boundary_covered_neurons_lower + \
            boundary_covered_neurons_upper
        nc, nbc, snac, tknc = covered_neurons / total_neurons, boundary_covered_neurons / (2 * total_neurons), \
            strong_covered_neurons / total_neurons, topk_covered_neurons / total_neurons

        # For KMNC
        total_covered_secs = 0
        for (_, _), covered_array in self.model_layer_dict_kmnc.items():
            total_covered_secs += np.sum(covered_array)
        kmnc = total_covered_secs / (total_neurons * self.k_section)
        """
        deprecate code. all size option stores the results of batch_size
        """
        coverage_rate = {'nc': nc, 'nbc': nbc,
                         'snac': snac, 'tknc': tknc, 'kmnc': kmnc}
        covered_items = {'nc': covered_neurons, 'nbc': boundary_covered_neurons, 'snac': strong_covered_neurons,
                         'tknc': topk_covered_neurons, 'kmnc': total_covered_secs}
        return coverage_rate, covered_items


class SurpriseCoverage:
    def __init__(self, args, model, dataset, network):
        self.model = model
        self.dataset = dataset
        self.network = network
        # layer_name is used for SA-family metrics
        self.sa_layer_names = sa_selected_layers[f"{dataset}_{network}"]
        self.args = args
        self.sa_config = SurpriseAdequacyConfig(saved_path=self.args.save_path,
                                                is_classification=self.args.is_classification,
                                                layer_names=self.sa_layer_names, ds_name=dataset, net_name=network,
                                                num_classes=self.args.num_classes)

    def origin_lsa(self, x_train, x_target, target_name, y_train=None):
        target_lsa = fetch_lsa(model=self.model, x_train=x_train, x_target=x_target, target_name=target_name,
                               layer_names=self.sa_layer_names, args=self.args)
        return target_lsa

    def origin_dsa(self, x_train, x_target, target_name, y_train=None):
        target_dsa = fetch_dsa(model=self.model, x_train=x_train, x_target=x_target, target_name=target_name,
                               layer_names=self.sa_layer_names, args=self.args)
        return target_dsa

    def mdsa(self, x_train, x_target, target_name, y_train=None):
        target_mdsa = fetch_mdsa(model=self.model, x_train=x_train, x_target=x_target, target_name=target_name,
                                 layer_names=self.sa_layer_names, args=self.args)
        return target_mdsa

    @staticmethod
    def filter_outliers(criteria, data_points):
        # # drop 10% points
        # data_std = np.std(data_points)
        # data_mean = np.mean(data_points)
        # cut_off = data_std * 3
        # lower = data_mean - cut_off
        # upper = data_mean + cut_off
        # lower = 0 if lower < 0 else lower
        # # lower = np.amin(data_points)
        # print(f"{criteria} BOUNDARY: [{lower},{upper}]")
        #
        # return lower, upper

        # drop 10% points
        lower = np.min(data_points)
        upper = np.max(data_points)
        # cut_off = data_std * 3
        # lower = data_mean - cut_off
        # upper = data_mean + cut_off
        # lower = 0 if lower < 0 else lower
        # # lower = np.amin(data_points)
        print(f"New Min Max {criteria} BOUNDARY: [{lower},{upper}]")

        return lower, upper

    @staticmethod
    def surprise_coverage(target_sa, n_bucket, lower, upper):
        target_cov = get_sc(lower=lower, upper=upper, k=n_bucket, sa=target_sa)
        return target_cov


class ImportanceDrivenCoverage:
    def __init__(self, model, dataset_network, num_relevant_neurons, train_inputs,
                 train_labels, idc_save_path, use_lrp=None):
        self.idc_save_path = idc_save_path
        self.covered_combinations = set()
        self.subject_layer_name = idc_subjects_layer[dataset_network]
        self.model = model
        self.dataset_network = dataset_network
        self.dataset, self.network = tuple(dataset_network.split("_"))
        self.train_size = dataloader.train_sizes[self.dataset]
        if self.dataset_network == "driving_dave-orig":
            # driving_dave-orig would crash. set the neuron number here.(in debugging!)
            self.num_relevant_neurons = 8
        else:
            self.num_relevant_neurons = num_relevant_neurons
        self.train_inputs = np.array(train_inputs)
        self.train_labels = train_labels
        self.temp_model = Model(inputs=self.model.input,
                                outputs=[layer.output for layer in model.layers
                                         if layer.name == self.subject_layer_name])
        if use_lrp is None:
            self.use_lrp = True if self.dataset_network in lrp_support else False
        else:
            self.use_lrp = use_lrp

        if not os.path.exists(self.idc_save_path):
            os.makedirs(self.idc_save_path)

    def get_measure_state(self):
        return self.covered_combinations

    def set_measure_state(self, covered_combinations):
        self.covered_combinations = covered_combinations

    def test(self, test_inputs):
        ###########################
        # 1.Find Relevant Neurons #
        ###########################
        totalR_path = os.path.join(self.idc_save_path,
                                   f"{self.dataset_network}_training_{self.train_size}.npy")
        if os.path.exists(totalR_path):
            print("Loading previous R values")
            totalR = np.load(totalR_path, allow_pickle=True)
            relevant_neurons = np.argsort(
                totalR)[::-1][:self.num_relevant_neurons]
        else:
            print("Relevance scores must be calculated. Doing it now!")
            s0 = datetime.now()
            if self.use_lrp:
                print("Using LRP model to get relevant neurons")
                # Convert keras model into txt
                model_path = os.path.join(
                    self.idc_save_path, f"{self.dataset_network}")
                write(self.model, model_path,
                      num_channels=test_inputs[0].shape[-1], fmt='keras_txt')
                lrpmodel = read(model_path + '.txt', 'txt')
                lrpmodel.drop_softmax_output_layer()
                lrpmodel.set_exp_name(self.dataset_network)
                # NOTE:
                #  The structures of lrp_model and original model may be inconsistent.
                #  Using the same index may cause mismatch issue.
                #  We use the subject_layer_name to select subject layer from lrp_model
                for idx, lrp_layer in enumerate(lrpmodel.modules):
                    if lrp_layer.name == self.subject_layer_name:
                        lrp_subject_layer = idx
                        break
                else:
                    raise ValueError(
                        f"No such layer named {self.subject_layer_name} in lrp model")
                relevant_neurons, least_relevant_neurons, total_R = \
                    self.lrp_find_relevant_neurons(lrpmodel=lrpmodel, inps=self.train_inputs,
                                                   subject_layer=lrp_subject_layer, num_rel=self.num_relevant_neurons)
            else:
                print("Using Keras layer outputs to get relevant neurons")
                relevant_neurons, least_relevant_neurons, total_R = \
                    self.find_relevant_neurons(model=self.model, inputs=self.train_inputs,
                                               num_rel=self.num_relevant_neurons)
            print("Finding relevant neurons,", datetime.now() - s0)
            print(
                f"INFO: Relevant neurons of {self.dataset_network}", relevant_neurons)
            np.save(totalR_path, total_R)

        ######################################
        # 2.Quantize Relevant Neuron Outputs #
        ######################################
        qtized_path = os.path.join(self.idc_save_path, f"{self.dataset_network}_"
                                                       f"{self.subject_layer_name}_"
                                                       f"{self.num_relevant_neurons}_"
                                                       f"training_{self.train_size}.npy")
        if os.path.exists(qtized_path):
            print("Loading previous qtized results")
            qtized = np.load(qtized_path, allow_pickle=True)
            print(qtized)
        else:
            print("Clustering results NOT FOUND; Calculating them now!")
            train_layer_out = self.temp_model.predict(
                self.train_inputs, verbose=1)
            s1 = datetime.now()
            qtized = self.quantize_silhouette(
                train_layer_out, relevant_neurons)
            print("Clustering time,", datetime.now() - s1)
            np.save(qtized_path, qtized)

        ######################
        # 3.Measure coverage #
        ######################
        print("Calculating IDC coverage")
        test_layer_out = self.temp_model.predict(np.array(test_inputs))
        s2 = datetime.now()
        coverage, covered_combinations, max_comb = self.measure_idc(relevant_neurons=relevant_neurons,
                                                                    test_layer_out=test_layer_out,
                                                                    qtized=qtized,
                                                                    covered_combinations=self.covered_combinations)
        print("measure_idc", datetime.now() - s2)
        self.set_measure_state(covered_combinations)
        return coverage, covered_combinations, max_comb

    @staticmethod
    def quantize_silhouette(out_vectors, relevant_neurons):
        quantized_ = []
        inputs_size, neuron_nums = out_vectors.shape[0], out_vectors.shape[-1]
        # outputs shape: (inputs_size,-1, neuron_num)
        outputs = out_vectors.reshape(inputs_size, -1, neuron_nums)
        # get neuron_output_matrix with shape (inputs_size, neuron_num)
        relevant_neuron_outputs = np.mean(outputs, axis=1)
        for relevant_neuron in tqdm(relevant_neurons):
            relevant_neuron_output = relevant_neuron_outputs[..., relevant_neuron].copy(
            )
            # If it is a convolutional layer no need for 0 output check
            filtered_output = relevant_neuron_output[relevant_neuron_output != 0]
            if not len(filtered_output) < 10:
                clusterSize = list(range(2, 5))  # [2, 3, 4]
                # multi-process version
                partial_cluster = partial(clustering, output=filtered_output)
                with Pool(processes=5) as p:
                    results = p.map(partial_cluster, clusterSize)
                silhouette_scores = [t[0] for t in results]
                maxSilhouette_score_idx = np.argmax(silhouette_scores)
                bestKMean = results[maxSilhouette_score_idx][1]
                values = bestKMean.cluster_centers_.squeeze()

                # single-process version
                # clustersDict = {}
                # for clusterNum in clusterSize:
                #     kmeans = cluster.KMeans(n_clusters=clusterNum)
                #     clusterLabels = kmeans.fit_predict(np.array(filtered_output).reshape(-1, 1))
                #     silhouetteAvg = silhouette_score(np.array(filtered_output).reshape(-1, 1), clusterLabels)
                #     clustersDict[silhouetteAvg] = kmeans
                # maxSilhouetteScore = max(clustersDict.keys())
                # bestKMean = clustersDict[maxSilhouetteScore]
                # values = bestKMean.cluster_centers_.squeeze()
            else:
                # values = [0]
                # print(f"ERROR: Unable to cluster for rel neuron {relevant_neuron} "
                #       f"when number of data points <= 10. {len(filtered_output)} found!")
                raise Exception(f"ERROR: Unable to cluster for rel neuron {relevant_neuron} "
                                f"when number of data points <= 10. {len(filtered_output)} found!")
            values = list(values)
            values = ImportanceDrivenCoverage.limit_precision(values)
            if len(values) == 0:
                values.append(0)
            quantized_.append(values)
        return quantized_

    @staticmethod
    def limit_precision(values, prec=2):
        limited_values = []
        for v in values:
            limited_values.append(round(v, prec))

        return limited_values

    @staticmethod
    def measure_idc(relevant_neurons, test_layer_out, qtized, covered_combinations: set):
        test_size, neuron_num = test_layer_out.shape[0], test_layer_out.shape[-1]
        reshaped_test_layer_out = test_layer_out.reshape(
            test_size, -1, neuron_num)
        # shape (test_size,neuron_num)
        neurons_output_mat = np.mean(reshaped_test_layer_out, axis=1)
        combinations_mat = np.zeros_like(neurons_output_mat)
        for idx, r in enumerate(relevant_neurons):
            # qtized[idx]: type->list,length->number of cluster
            centroids = np.array(qtized[idx])
            # shape (test_size, number of cluster)
            centroids_mat = np.tile(centroids, (test_size, 1))
            # get output of a specific neuron
            # shape (test_size,)
            neuron_output = neurons_output_mat[..., r].copy()
            # shape (test_size,1)
            neuron_output_reshape = neuron_output.reshape(-1, 1)
            # get distance between neuron activation and centroids.
            # automatically broadcast
            dis = abs(neuron_output_reshape - centroids_mat)
            dis_sort = np.argsort(dis, axis=1)
            nearest_centroids_idx = dis_sort[..., 0]
            centers_vals = [centroids[i] for i in nearest_centroids_idx]
            combinations_mat[..., idx] = np.array(centers_vals)
        # for each test case
        # update their centroids values into
        for comb in combinations_mat:
            covered_combinations.add(tuple(comb))

        max_comb = 1
        for q in qtized:
            max_comb *= len(q)

        covered_num = len(covered_combinations)
        coverage = float(covered_num) / max_comb
        print(float(covered_num), max_comb)
        return coverage * 100, covered_combinations, max_comb

    @staticmethod
    def find_relevant_neurons(model, inputs, num_rel):
        last_layer = model.layers[-1]
        # get last three layers
        total_layers = [
            layer.name for layer in model.layers if "input" not in layer.name]
        last_three_layers = total_layers[-3:]
        temp_model = Model(inputs=model.input,
                           outputs=[layer.output for layer in model.layers if layer.name in last_three_layers])
        layers_outputs = temp_model.predict(inputs, verbose=1)
        # check if the last layer is softmax-activation;
        # driving-dave-orig use Lambda as activation
        if isinstance(last_layer, (keras.layers.Activation, keras.layers.Lambda)):
            # if last layer is softmax-activation-> penultimate layer is logits layer
            # W,B is the weights and bias of layer that produce logits
            W, B = model.layers[-2].get_weights()
            logits_input = layers_outputs[-3]
            logits = layers_outputs[-2]
        elif isinstance(last_layer, keras.layers.Dense) and \
                hasattr(last_layer, 'activation') and \
                'softmax' in last_layer.activation.__name__.lower():
            # last_layer is dense and its activation function is softmax
            # get logits from last dense layer
            W, B = model.layers[-1].get_weights()
            logits_input = layers_outputs[-2]
            logits = np.dot(logits_input, W) + B
        else:
            # raise Exception() -> "last layer should be activation or dense"
            raise TypeError(
                f"last layer should be activation or dense(with softmax). {type(last_layer)} found")
        # get Rinit
        ypreds = logits
        labels = np.argmax(ypreds, axis=1)
        # an elegant way to generate one-hot matrix
        eye_matrix = np.eye(ypreds.shape[1])
        mask = eye_matrix[labels].copy()
        Rinit = ypreds * mask

        # get R for subject layer;
        # In our study,subject layer is dense layer
        Y = logits
        X = logits_input
        Zs = Y + 1e-16 * ((Y >= 0) * 2 - 1.)
        Z = W[na, :, :] * X[:, :, na]
        R = (Z * (Rinit / Zs)[:, na, :]).sum(axis=2)
        totalR = np.sum(R.copy(), axis=0)
        return np.argsort(totalR)[::-1][:num_rel], np.argsort(totalR)[:num_rel], totalR

    @staticmethod
    def lrp_find_relevant_neurons(lrpmodel, inps, subject_layer, num_rel):
        # get prediction matrix
        s0 = datetime.now()
        ypreds = lrpmodel.forward(inps)
        print(f"LRP forward time:{datetime.now() - s0}")
        labels = np.argmax(ypreds, axis=1)
        # an elegant way to generate one-hot matrix
        eye_matrix = np.eye(ypreds.shape[1])
        mask = eye_matrix[labels].copy()
        Rinits = ypreds * mask
        R_inp, R_all = lrpmodel.lrp(Rinits, subject_layer=subject_layer)
        # use cumulative relevant scores
        totalR = np.sum(R_all[0].copy(), axis=0)
        return np.argsort(totalR)[::-1][:num_rel], np.argsort(totalR)[:num_rel], totalR


def execute_sampling(dataset_network, classifier, x, y, train_inputs, train_labels, boundary, coverage_parameters,
                     sa_dict, normal_indices, adv_indices, classification):
    dataset_name, network_name = tuple(dataset_network.split("_"))
    n_bucket = coverage_parameters["n_bucket"]
    kmnc_k_section = coverage_parameters["kmnc_k_section"]
    tknc_k_value = coverage_parameters["tknc_k_value"]
    nc_threshold = coverage_parameters["nc_threshold"]
    idc_relevant_neurons = coverage_parameters["idc_relevant_neurons"]
    idc_intermedia_path = coverage_parameters["idc_intermedia_path"]

    clean_lsa, adv_lsa = sa_dict["clean_lsa"], sa_dict["adv_lsa"]
    clean_dsa, adv_dsa = sa_dict["clean_dsa"], sa_dict["adv_dsa"]
    clean_mdsa, adv_mdsa = sa_dict["clean_mdsa"], sa_dict["adv_mdsa"]


    #
    # #######################
    # # Structural coverage #
    # #######################
    # time_marker = datetime.now()
    # s_coverage = StructuralCoverage(model=classifier, k_section=kmnc_k_section, k_value=tknc_k_value,
    #                                 boundary=boundary, threshold=nc_threshold)
    # # neuron coverage includes four types of coverage:DNC,TKNC,NBC,SANC
    # predictions = s_coverage.update_coverage(input_data=x.copy())
    # _coverage_rate, _ = s_coverage.get_coverage()
    # nc_cost = datetime.now() - time_marker
    #
    # total_error_rate = ScoreUtils.analyze_error_rate(labels=y, predictions=predictions, dataset=dataset_name,
    #                                                  network=network_name)

    total_error_rate = ScoreUtils.get_model_error_rate(dataset=dataset_name, network=network_name, model=classifier,
                                                       x_test=x.copy(), y_test=y)
    '''
    ##############################
    # Importance Driven coverage #
    ##############################
    time_marker = datetime.now()
    if dataset_network == "speech-commands_deepspeech":
        idc = 0
        print(f"INFO: Skip {dataset_network} for IDC. ")
    else:
        idc_coverage = ImportanceDrivenCoverage(model=classifier, dataset_network=dataset_network,
                                                num_relevant_neurons=idc_relevant_neurons, train_inputs=train_inputs,
                                                train_labels=train_labels, idc_save_path=idc_intermedia_path)

        idc, _, _ = idc_coverage.test(x.copy())
    idc_cost = datetime.now() - time_marker
    '''


    ###########################
    # Non-Structural coverage #
    ###########################
    time_marker = datetime.now()


    lsa_lower, lsa_upper = sa_dict["lsa_boundary"]
    lsc = get_surprise_coverage(clean_lsa, adv_lsa, normal_indices, adv_indices, n_bucket,
                                lower=lsa_lower, upper=lsa_upper)


    if classification:
        dsa_lower, dsa_upper = sa_dict["dsa_boundary"]
        mdsa_lower, mdsa_upper = sa_dict["mdsa_boundary"]


        dsc = get_surprise_coverage(clean_dsa, adv_dsa, normal_indices, adv_indices, n_bucket,
                                    lower=dsa_lower, upper=dsa_upper)
        #
        # mdsc = get_surprise_coverage(clean_mdsa, adv_mdsa, normal_indices, adv_indices, n_bucket,
        #                              lower=mdsa_lower, upper=mdsa_upper)

    else:
        dsc = mdsc = 0
    sa_cost = datetime.now() - time_marker
    # print("mdsc",mdsc)
    r = [0, 0, 0, 0, 0, lsc, dsc, 0, 0, total_error_rate]
    # r = [0, 0, 0, 0, 0, lsc, dsc, mdsc, 0, total_error_rate]
    # r = [0, 0, 0, 0, 0, 0, 0, 0, idc, total_error_rate]
    # print(f"Time cost: StructuralCoverage:{nc_cost} IDC: {idc_cost} SACoverage:{sa_cost}")
    print(f"Time cost: SACoverage:{sa_cost}")
    # print(f"Time cost: IDC: {idc_cost}")
    # print(f"Time cost: MDSACoverage: {sa_cost}")
    return r


def get_surprise_coverage(clean_sa, adv_sa, clean_idx, adv_idx, n_bucket, lower, upper):
    target_sa = [clean_sa[i] for i in clean_idx]
    selected_sa_adv = [adv_sa[i] for i in adv_idx]
    target_sa.extend(selected_sa_adv)
    return SurpriseCoverage.surprise_coverage(target_sa=target_sa, n_bucket=n_bucket, lower=lower, upper=upper)


idc_subjects_layer = {
    # In IDC, we choose "penultimate" layer.
    # If the last layer is softmax(activation), we use antepenultimate layer
    # If last layer is dense, we use penultimate layer
    "fashion-mnist_lenet5": "dense_2",
    'mnist_lenet5': 'dense_2',
    'cifar10_alexnet': 'dropout_2',
    'cifar10_vgg16': 'dropout_1',
    'cifar10_resnet20': 'flatten_1',
    'cifar100_resnet32': 'flatten_1',
    'imagenet_vgg19': 'fc2',
    'imagenet_resnet50': 'avg_pool',
    "driving_dave-orig": "fc3",
    "driving_dave-dropout": "fc2"
}
lrp_support = ["mnist_lenet5", "driving_dave-orig"]
