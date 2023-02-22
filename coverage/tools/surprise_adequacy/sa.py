from warnings import warn

import os

from multiprocessing import Pool

import numpy as np
from tqdm import tqdm
from keras.models import Model
from scipy.stats import gaussian_kde
from coverage.tools.surprise_adequacy.sa_utils import *
from coverage.tools.common_utils import ScoreUtils
from coverage.tools.deepspeech.deepspeech_utils import DSDataUtils


def _aggr_output(x):
    return [np.mean(x[..., j]) for j in range(x.shape[-1])]


def _get_saved_path(base_path, dataset, network, train_size: int, dtype, layer_names):
    """Determine saved path of ats and pred
    Args:
        base_path (str): Base save path.
        dataset (str): Name of dataset.
        dtype (str): Name of dataset type (e.g., train, test, fgsm, ...).
        layer_names (list): List of layer names.
    Returns:
        ats_path: File path of ats.
        pred_path: File path of pred (independent of layers)
    """

    joined_layer_names = "_".join(layer_names)
    return (
        os.path.join(
            base_path,
            dataset + "_" + network + "_" + dtype + "_" +
            str(train_size) + "_" + joined_layer_names + "_ats" + ".npy",
        ),
        os.path.join(base_path, dataset + "_" + network + "_" +
                     dtype + "_" + str(train_size) + "_pred" + ".npy"),
    )


def get_ats(
        model,
        dataset,
        name,
        layer_names,
        save_path=None,
        batch_size=128,
        is_classification=True,
        num_classes=10,
        num_proc=10,
        dataset_name=None,
):
    """Extract activation traces of dataset from model.
    Args:
        model (keras model): Subject model.
        dataset (list): Set of inputs fed into the model.
        name (str): Name of input set.
        layer_names (list): List of selected layer names.
        save_path (tuple): Paths of being saved ats and pred.
        batch_size (int): Size of batch when serving.
        is_classification (bool): Task type, True if classification task or False.
        num_classes (int): The number of classes (labels) in the dataset.
        num_proc (int): The number of processes for multiprocessing.
    Returns:
        ats (list): List of (layers, inputs, neuron outputs).
        pred (list): List of predicted classes.
    """

    temp_model = Model(
        inputs=model.input,
        outputs=[model.get_layer(
            layer_name).output for layer_name in layer_names],
    )

    prefix = info("[" + name + "] ")
    if is_classification:
        p = Pool(num_proc)
        print(prefix + "Model serving")
        # pred = model.predict_classes(dataset, batch_size=batch_size, verbose=1)
        predict = model.predict(dataset, batch_size=batch_size, verbose=1)
        if dataset_name == "speech-commands":
            pred_words = ScoreUtils.speech_commands_prediction(predict)
            pred = [DSDataUtils.get_words_idx(s) for s in pred_words]
        else:
            pred = np.argmax(predict, axis=1)

        if len(layer_names) == 1:
            layer_outputs = [
                temp_model.predict(dataset, batch_size=batch_size, verbose=1)
            ]
        else:
            layer_outputs = temp_model.predict(
                dataset, batch_size=batch_size, verbose=1
            )

        print(prefix + "Processing ATs")
        ats = None
        for layer_name, layer_output in zip(layer_names, layer_outputs):
            print("Layer: " + layer_name)
            # (primarily for convolutional layers - note that kim et al used ndim==3)
            # I think here should be 2.
            # The output shape may be like (batch_size,channel1,channel2),
            # and we should change it to (batch_size,channel2)
            if layer_output[0].ndim >= 2:
                # For convolutional layers
                layer_matrix = np.array(
                    p.map(_aggr_output, [layer_output[i]
                                         for i in range(len(dataset))])
                )
            else:
                layer_matrix = np.array(layer_output)

            if ats is None:
                ats = layer_matrix
            else:
                ats = np.append(ats, layer_matrix, axis=1)
                layer_matrix = None
    else:
        p = Pool(num_proc)
        pred = []
        print(prefix + "Model serving")
        if len(layer_names) == 1:
            layer_outputs = [
                temp_model.predict(dataset, batch_size=batch_size, verbose=1)
            ]
        else:
            layer_outputs = temp_model.predict(
                dataset, batch_size=batch_size, verbose=1
            )

        print(prefix + "Processing ATs")
        ats = None
        for layer_name, layer_output in zip(layer_names, layer_outputs):
            print("Layer: " + layer_name)
            if layer_output[0].ndim == 3:
                # For convolutional layers
                layer_matrix = np.array(
                    p.map(_aggr_output, [layer_output[i]
                                         for i in range(len(dataset))])
                )
            else:
                layer_matrix = np.array(layer_output)

            if ats is None:
                ats = layer_matrix
            else:
                ats = np.append(ats, layer_matrix, axis=1)
                layer_matrix = None

    # if save_path is not None:
    #     np.save(save_path[0], ats)
    #     np.save(save_path[1], pred)

    return ats, pred


def find_closest_at(at, train_ats):
    """The closest distance between subject AT and training ATs.
    Args:
        at (list): List of activation traces of an input.
        train_ats (list): List of activation traces in training set (filtered)

    Returns:
        dist (int): The closest distance.
        at (list): Training activation trace that has the closest distance.
    """

    dist = np.linalg.norm(at - train_ats, axis=1)
    return (min(dist), train_ats[np.argmin(dist)])


def _get_train_target_ats(model, x_train, x_target, target_name, layer_names, args):
    """Extract ats of train and target inputs. If there are saved files, then skip it.
    Args:
        model (keras model): Subject model.
        x_train (list): Set of training inputs.
        x_target (list): Set of target (test or adversarial) inputs.
        target_name (str): Name of target set.
        layer_names (list): List of selected layer names.
        args: keyboard console_args.
    Returns:
        train_ats (list): ats of train set.
        train_pred (list): pred of train set.
        target_ats (list): ats of target set.
        target_pred (list): pred of target set.
    """
    train_size = len(x_train)
    saved_train_path = _get_saved_path(
        args.save_path, args.dataset, args.network, train_size, "train", layer_names)
    if os.path.exists(saved_train_path[0]):
        print(infog("Found saved {} ATs, skip serving".format("train")))
        # In case train_ats is stored in a disk
        train_ats = np.load(saved_train_path[0])
        train_pred = np.load(saved_train_path[1])
    else:
        train_ats, train_pred = get_ats(
            model,
            x_train,
            "train",
            layer_names,
            num_classes=args.num_classes,
            is_classification=args.is_classification,
            save_path=saved_train_path,
            dataset_name=args.dataset,
        )
        print(infog("train ATs is saved at " + saved_train_path[0]))
        if saved_train_path is not None:
            np.save(saved_train_path[0], train_ats)
            np.save(saved_train_path[1], train_pred)

    saved_target_path = _get_saved_path(
        args.save_path, args.dataset, args.network, train_size, target_name, layer_names
    )

    if True:
        target_ats, target_pred = get_ats(
            model,
            x_target,
            target_name,
            layer_names,
            num_classes=args.num_classes,
            is_classification=args.is_classification,
            save_path=saved_target_path,
            dataset_name=args.dataset,
        )
        print(infog(target_name + " ATs is saved at " + saved_target_path[0]))
    return train_ats, train_pred, target_ats, target_pred


def generate_at(model, x_train, args, layer_names):
    train_size = len(x_train)
    saved_train_path = _get_saved_path(
        args.save_path, args.dataset, args.network, train_size, "train", layer_names)
    if os.path.exists(saved_train_path[0]):
        print(infog("Found saved {} ATs, skip serving".format("train")))
        print("Skip training ats generation")
    else:
        train_ats, train_pred = get_ats(
            model,
            x_train,
            "train",
            layer_names,
            num_classes=args.num_classes,
            is_classification=args.is_classification,
            save_path=saved_train_path,
        )
        print(infog("train ATs is saved at " + saved_train_path[0]))
        if saved_train_path is not None:
            np.save(saved_train_path[0], train_ats)
            np.save(saved_train_path[1], train_pred)


def fetch_dsa(model, x_train, x_target, target_name, layer_names, args):
    # """Distance-based SA
    # Args:
    #     model (keras model): Subject model.
    #     x_train (list): Set of training inputs.
    #     x_target (list): Set of target (test or adversarial) inputs.
    #     target_name (str): Name of target set.
    #     sa_layer_names (list): List of selected layer names.
    #     console_args: keyboard console_args.
    # Returns:
    #     dsa (list): List of dsa for each target input.
    # """

    assert args.is_classification

    prefix = info("[" + target_name + "] ")
    train_ats, train_pred, target_ats, target_pred = _get_train_target_ats(
        model, x_train, x_target, target_name, layer_names, args
    )

    class_matrix = {}
    all_idx = []
    for i, label in enumerate(train_pred):
        if label not in class_matrix:
            class_matrix[label] = []
        class_matrix[label].append(i)
        all_idx.append(i)

    dsa = []

    print(prefix + "Fetching DSA")
    for i, at in enumerate(tqdm(target_ats)):
        label = target_pred[i]
        a_dist, a_dot = find_closest_at(at, train_ats[class_matrix[label]])
        b_dist, _ = find_closest_at(
            a_dot, train_ats[list(set(all_idx) - set(class_matrix[label]))]
        )
        dsa.append(a_dist / b_dist)

    return dsa


def fetch_mdsa(model, x_train, x_target, target_name, layer_names, args):
    """
    @param model: Subject model.
    @param x_train: Set of training inputs.
    @param x_target: Set of target (test or adversarial) inputs.
    @param target_name: name of targeted test inputs
    @param layer_names: List of selected layer names.
    @param args: keyboard console_args.
    @return: List of mdsa for each target input.
    """

    assert args.is_classification

    prefix = info("[" + target_name + "] ")
    train_ats, train_pred, target_ats, target_pred = _get_train_target_ats(
        model, x_train, x_target, target_name, layer_names, args
    )

    class_matrix = {}
    all_idx = []
    for i, label in enumerate(train_pred):
        if label not in class_matrix:
            class_matrix[label] = []
        class_matrix[label].append(i)
        all_idx.append(i)
    mdsa = []

    print(prefix + "Fetching MDSA")
    train_size = len(x_train)
    mdsa_inter_path = os.path.join(
        args.save_path, f"{args.dataset}_{args.network}_{train_size}_mdsa_inter.npz")
    if os.path.exists(mdsa_inter_path):
        inter_dict = np.load(mdsa_inter_path, allow_pickle=True)
        to_keep_dict, mu_dict, Sinv_dict = inter_dict["to_keep"][(
        )], inter_dict["mu"][()], inter_dict["Sinv"][()]
    else:
        # generate to_keep
        # here, train_ats should be like (test_size, cols_nums)
        to_keep_dict = dict()
        mu_dict = dict()
        Sinv_dict = dict()
        for label in range(args.num_classes):
            _to_keep = np.ones(train_ats.shape[1], dtype=np.bool_)
            # print("INFO",train_ats[class_matrix[label]].shape)
            col_vectors = np.transpose(train_ats[class_matrix[label]])
            # print("INFO",col_vectors.shape)
            for i in range(col_vectors.shape[0]):
                # print(np.var(col_vectors[i]))
                if np.var(col_vectors[i]) < args.var_threshold:
                    _to_keep[i] = False
            refined_ats = col_vectors[_to_keep, :]
            to_keep_dict[label] = _to_keep
            _mu = np.mean(refined_ats, axis=1).transpose()
            mu_dict[label] = _mu.copy()
            _Sinv = np.linalg.inv(np.cov(refined_ats))
            Sinv_dict[label] = _Sinv.copy()
        np.savez(mdsa_inter_path, to_keep=to_keep_dict,
                 mu=mu_dict, Sinv=Sinv_dict)

    for i, at in enumerate(tqdm(target_ats)):
        to_keep = to_keep_dict[target_pred[i]]
        col_vector = at.transpose()
        refined_col_vector = col_vector[to_keep].transpose()
        label = target_pred[i]
        mu, Sinv = mu_dict[label], Sinv_dict[label]
        tmp = np.dot((refined_col_vector - mu).transpose(), Sinv)
        mdsa.append(np.sqrt(np.dot(tmp, (refined_col_vector - mu))).item())

    return mdsa


def _get_kdes(train_ats, train_pred, class_matrix, args):
    """Kernel density estimation
    Args:
        train_ats (list): List of activation traces in training set.
        train_pred (list): List of prediction of train set.
        class_matrix (list): List of index of classes.
        args: Keyboard console_args.
    Returns:
        kdes (list): List of kdes per label if classification task.
        removed_cols (list): List of removed columns by variance threshold.
    """

    removed_cols = []
    if args.is_classification:
        for label in range(args.num_classes):
            col_vectors = np.transpose(train_ats[class_matrix[label]])
            for i in range(col_vectors.shape[0]):
                if (
                        np.var(col_vectors[i]) < args.var_threshold
                        and i not in removed_cols
                ):
                    removed_cols.append(i)
        print(sorted(removed_cols))
        kdes = {}
        for label in tqdm(range(args.num_classes), desc="kde"):
            refined_ats = np.transpose(train_ats[class_matrix[label]])
            refined_ats = np.delete(refined_ats, removed_cols, axis=0)
            print(refined_ats.shape)
            print(label)
            if refined_ats.shape[0] == 0:
                print(
                    warn("ats were removed by threshold {}".format(
                        args.var_threshold))
                )
                break
            kdes[label] = gaussian_kde(refined_ats)

    else:
        if np.isnan(train_ats).any():
            print("Found nan in train ats")
        col_vectors = np.transpose(train_ats)
        for i in range(col_vectors.shape[0]):
            if np.var(col_vectors[i]) < args.var_threshold:
                removed_cols.append(i)
        print(len(removed_cols))
        refined_ats = np.transpose(train_ats)
        refined_ats = np.delete(refined_ats, removed_cols, axis=0)
        if refined_ats.shape[0] == 0:
            print(warn("ats were removed by threshold {}".format(args.var_threshold)))
        kdes = [gaussian_kde(refined_ats)]
        print(gaussian_kde(refined_ats))
        # print(type(kdes[0]))
        # if np.isnan(kdes[0]).any():
        #     raise Exception("Found NaN in kde")

    print(infog("The number of removed columns: {}".format(len(removed_cols))))

    return kdes, removed_cols


def _get_lsa(kde, at, removed_cols):
    refined_at = np.delete(at, removed_cols, axis=0)
    # print(refined_at)
    # print(np.transpose(refined_at))
    transpose_refined_at = np.transpose(refined_at)
    _logpdf = -kde.logpdf(transpose_refined_at)
    res = np.asscalar(_logpdf)
    if np.isnan(res).any() or np.isinf(res).any():
        raise Exception()
    return np.asscalar(-kde.logpdf(np.transpose(refined_at)))


def fetch_lsa(model, x_train, x_target, target_name, layer_names, args):
    def check_nan(x):
        import math
        if isinstance(x, np.ndarray):
            if np.isnan(x).any() or np.isinf(x).any():
                raise Exception("nan")
        if isinstance(x, list):
            for xi in x:
                if math.isnan(xi) or math.isinf(xi):
                    raise Exception("nan")
        print("No nan found")

    # """Likelihood-based SA
    # Args:
    #     model (keras model): Subject model.
    #     x_train (list): Set of training inputs.
    #     x_target (list): Set of target (test or[] adversarial) inputs.
    #     target_name (str): Name of target set.
    #     sa_layer_names (list): List of selected layer names.
    #     console_args: Keyboard console_args.
    # Returns:
    #     lsa (list): List of lsa for each target input.
    # """

    prefix = info("[" + target_name + "] ")
    train_ats, train_pred, target_ats, target_pred = _get_train_target_ats(
        model, x_train, x_target, target_name, layer_names, args
    )

    check_nan(train_ats)
    check_nan(train_pred)
    check_nan(target_ats)
    check_nan(target_pred)

    class_matrix = {}
    if args.is_classification:
        for i, label in enumerate(train_pred):
            if label not in class_matrix.keys():
                class_matrix[label] = []
            class_matrix[label].append(i)

    kdes, removed_cols = _get_kdes(train_ats, train_pred, class_matrix, args)

    lsa = []
    print(prefix + "Fetching LSA")
    if args.is_classification:
        for i, at in enumerate(tqdm(target_ats)):
            label = target_pred[i]
            kde = kdes[label]
            lsa.append(_get_lsa(kde, at, removed_cols))
    else:
        kde = kdes[0]
        for at in tqdm(target_ats):
            lsa.append(_get_lsa(kde, at, removed_cols))

    return lsa


def get_sc(lower, upper, k, sa):
    """Surprise Coverage
    Args:
        lower (int): Lower bound.
        upper (int): Upper bound.
        k (int): The number of buckets.
        sa (list): List of lsa or dsa.
    Returns:
        cov (int): Surprise coverage.
    """

    buckets = np.digitize(sa, np.linspace(lower, upper, k))
    return len(list(set(buckets))) / float(k) * 100


# sa_selected_layers = {
#     'alexnet' : [""],
#     'lenet5': ['dense_3'],
#     'vgg16': ['dense_1'],
#     'resnet20': ['activation_19'],
#     'resnet32': ['activation_28'],
#     'vgg19': ['block5_conv4'],
#     'resnet50': ['activation_49'],
#     'deepspeech': ['dense_1'],
#     'dave-orig': ['fc4'],
# }

sa_selected_layers = {
    'cifar10_alexnet': ["dense_2"],  # -3
    # 'cifar10_alexnet': ["dense_1"],  # -3
    "fashion-mnist_lenet5": ["dense_3"],  # -2
    'mnist_lenet5': ['dense_3'],  # -2
    'cifar10_vgg16': ['dense_1'],  # -3
    'cifar10_resnet20': ['flatten_1'],  # -1
    'cifar100_resnet32': ['flatten_1'],  # -1
    'imagenet_vgg19': ['block5_conv4'],  # -6
    'imagenet_resnet50': ['activation_49'],  # -3
    'speech-commands_deepspeech': ['dense_1'],
    'driving_dave-orig': ['fc4'],
    'driving_dave-dropout': ['fc3'],
}
