import os
import numpy as np
import keras.datasets.fashion_mnist as fashion_mnist
import keras.datasets.mnist as mnist
import keras.datasets.cifar10 as cifar10
import keras.datasets.cifar100 as cifar100
from coverage import root_dir
import coverage.tools.deepspeech.deepspeech_utils as ds_utils
from coverage.tools.deepspeech.deepspeech_utils import DSDataUtils
from coverage.tools.deepspeech.deepspeech_utils import dataGen_mfcc_ctc

IMAGENET_PATH = ""

def class_num(name):
    if name in ['mnist', 'cifar10', "fashion-mnist"]:
        return 10
    elif name == 'cifar100':
        return 100
    elif name == 'imagenet':
        return 1000
    elif name == 'speech-commands':
        return 30
    elif name == 'driving':
        return None
    else:
        raise ValueError(f"No num classes for {name}")


def subtract_mean(x, means):
    x[..., 0] -= means[0]
    x[..., 1] -= means[1]
    x[..., 2] -= means[2]
    return x


def imagenet_preprocess(x):
    """
    Refer to keras.applications
    https://github.com/keras-team/keras/blob/df03bb5b1cc9fd297b0f19e08d916a4faedea267/keras/applications/imagenet_utils.py#L60
    x should be 'RGB' mode.
    """
    mean = [103.939, 116.779, 123.68]
    # change 'RGB' mode to 'BGR' mode.
    x = x[..., ::-1]
    x[..., 0] -= mean[0]
    x[..., 1] -= mean[1]
    x[..., 2] -= mean[2]
    return x


def load_train_set(dataset):
    # load unprocessed data
    if dataset == "mnist":
        (_train_x, _train_y), _ = mnist.load_data()
        # Insert a new axis in the array
        _train_x = _train_x[..., np.newaxis]
        _train_x = _train_x.astype("float32")
        _train_x /= 255.0
    elif dataset == "fashion-mnist":
        (_train_x, _train_y), _ = fashion_mnist.load_data()
        # Insert a new axis in the array
        _train_x = _train_x[..., np.newaxis]
        _train_x = _train_x.astype("float32")
        _train_x /= 255.0
    elif dataset == "cifar10":
        (_train_x, _train_y), _ = cifar10.load_data()
        _train_x = _train_x.astype("float32")
        _train_x /= 255.0
    elif dataset == "cifar100":
        (_train_x, _train_y), _ = cifar100.load_data()
        _train_x = _train_x.astype("float32")
        _train_x /= 255.0
    elif dataset == "imagenet":
        raise NotImplementedError("Currently we don't support to load train set of imagenet for it's too large")
    elif dataset == "driving":
        _train_x = np.load(os.path.join(root_dir, "files/original_datasets/driving_dave-orig_train.npy"))
        _train_y = None
    elif dataset == "speech-commands":
        batch_size = train_sizes[dataset]
        trGen = dataGen_mfcc_ctc(ds_utils.trfile, batchSize=batch_size)
        [_train_x, _train_y, _, _], _ = next(DSDataUtils.gen_ctc_byclass(trGen))
    else:
        raise ValueError(f"This dataset {dataset} is not supported by current implementation.")

    # if the shape of y is (`num_classes`,1)
    # squeeze y
    if _train_y is not None and len(_train_y.shape) > 1:
        _train_y = _train_y.squeeze()
    return _train_x, _train_y


def load_dataset(dataset):
    # load unprocessed data
    if dataset == "mnist":
        _, (_test_x, _test_y) = mnist.load_data()
        # Insert a new axis in the array
        _test_x = _test_x[..., np.newaxis]
        _test_x = _test_x.astype("float32")
        _test_x /= 255.0
    elif dataset == "fashion-mnist":
        _, (_test_x, _test_y) = fashion_mnist.load_data()
        # Insert a new axis in the array
        _test_x = _test_x[..., np.newaxis]
        _test_x = _test_x.astype("float32")
        _test_x /= 255.0
    elif dataset == "cifar10":
        _, (_test_x, _test_y) = cifar10.load_data()
        _test_x = _test_x.astype("float32")
        _test_x /= 255.0
    elif dataset == "cifar100":
        _, (_test_x, _test_y) = cifar100.load_data()
        _test_x = _test_x.astype("float32")
        _test_x /= 255.0
    elif dataset == "imagenet":
        # cropped_imagenet = np.load(IMAGENET_PATH)
        imagenet_path = os.path.join(root_dir, "datasets", "imagenet_validation_5000.npz")
        cropped_imagenet = np.load(imagenet_path)
        _test_x, _test_y = cropped_imagenet['x_data'], cropped_imagenet['y_data']
        _test_x = _test_x.astype("float32")
    elif dataset == "driving":
        data = np.load(os.path.join(root_dir, "files/original_datasets/driving_dave-orig_test.npz"))
        _test_x, _test_y = data["inputs"], data["labels"]
    elif dataset == "speech-commands":
        batch_size = test_sizes[dataset]
        testGen = dataGen_mfcc_ctc(ds_utils.testfile, batchSize=batch_size)
        [_test_x, _test_y, _, _], _ = next(DSDataUtils.gen_ctc_byclass(testGen))
    else:
        raise ValueError(f"This dataset {dataset} is not supported by current implementation.")

    # if the shape of y is (`num_classes`,1)
    # squeeze y
    if _test_y is not None and len(_test_y.shape) > 1:
        _test_y = _test_y.squeeze()
    return _test_x, _test_y


def preprocess_dataset(dataset, network, x):
    # preprocess data
    if dataset in ["mnist", "fashion-mnist", "driving", "speech-commands"]:
        return x
    elif dataset == "cifar10" and network == 'vgg16':
        return x
    elif dataset == "cifar10" and network in ['resnet20', 'alexnet']:
        means = [0.49139968, 0.48215827, 0.44653124]
        x = subtract_mean(x, means)
    elif dataset == "cifar100":
        means = [0.5070746, 0.48654896, 0.44091788]
        x = subtract_mean(x, means)
    elif dataset == "imagenet":
        # argument x should be 'RGB' mode.
        # The returned x is 'BGR' mode.
        x = imagenet_preprocess(x)
    else:
        raise ValueError(f"This dataset {dataset} is not supported by current implementation.")
    return x


def load_adversarial_images(dataset, network, method, mode="full"):
    # assert method == "fgsm", "only test fgsm now"
    dataset_model = f"{dataset}_{network}"
    if dataset == "driving":
        adv_path = os.path.join(root_dir, "files/adversarial_examples", mode)
        file_name = f"driving_dave-orig_{method}_{mode}.npz"
        data_dict = np.load(os.path.join(adv_path, "driving_dave-orig", file_name))
        adv_inputs, adv_labels = data_dict['inputs'], data_dict['labels']
        if adv_labels is not None and len(adv_labels.shape) > 1:
            adv_labels = adv_labels.squeeze()
        return adv_inputs, adv_labels
    elif dataset in ['mnist', 'cifar10', 'cifar100', 'imagenet', "fashion-mnist"]:
        adv_path = os.path.join(root_dir, "files/adversarial_examples", mode)
        file_name = f"{dataset_model}_{method}_{mode}.npz"
        data_dict = np.load(os.path.join(adv_path, dataset_model, file_name))
        adv_inputs, adv_labels = data_dict['inputs'], data_dict['labels']
        if adv_labels is not None and len(adv_labels.shape) > 1:
            adv_labels = adv_labels.squeeze()
        return adv_inputs, adv_labels
    elif dataset == "speech-commands":
        batch_size = adv_sizes[dataset]
        advGen = dataGen_mfcc_ctc(ds_utils.advfile, batchSize=batch_size)
        [adv_inputs, adv_labels, _, _], _ = next(DSDataUtils.gen_ctc_byclass(advGen))
        return adv_inputs, adv_labels
    else:
        raise NotImplementedError(f"{dataset_model} is not supported currently.")


train_sizes = {
    "fashion-mnist": 60000,
    "mnist": 60000,
    "cifar10": 50000,
    "cifar100": 50000,
    "driving": 101167,
    "speech-commands": 51776
}

test_sizes = {
    "fashion-mnist": 10000,
    "mnist": 10000,
    "cifar10": 10000,
    "cifar100": 10000,
    "driving": 5614,
    "speech-commands": 6471
}

adv_sizes = {
    "fashion-mnist": 10000,
    "mnist": 10000,
    "cifar10": 10000,
    "cifar100": 10000,
    "driving": 5614,
    "speech-commands": 6471
}

if __name__ == "__main__":
    pass
    # import sys
    # import numpy as np
    #
    # dataset, network, attack = sys.argv[1], sys.argv[2], sys.argv[3]
    # inputs_name = f"{attack}_{dataset}_image_{network}_0.npy"
    # labels_name = f"{attack}_{dataset}_label_{network}_0.npy"
    # inputs = np.load(inputs_name)
    # labels = np.load(labels_name)
    # np.savez(f"{dataset}_{network}_{attack}_{len(inputs)}.npz", x_test=inputs.copy(), y_test=labels.copy())

    #
    # import prettytable as pt
    # import keras
    # from coverage.tools.common_utils import ScoreUtils
    # import coverage.tools.model_utils as model_utils
    # from coverage.tools.coverage_utils import StructuralCoverage, SurpriseCoverage, Args
    # from collections import Counter, defaultdict
    #
    # adv_tables = pt.PrettyTable()
    # adv_tables.title = 'Adversarial attack accuracy of each model'
    # adv_tables.field_names = ["Dataset&Model", "cw","fgsm","bim","jsma"]
    #
    # # remove the failed adv images
    # for dataset_network in ["mnist_lenet5", "cifar10_vgg16", 'cifar10_resnet20',"cifar100_resnet32"]:
    #     dataset_name, network_name = tuple(dataset_network.split("_"))
    #     num_classes = class_num(dataset_name)
    #     row = [dataset_network]
    #     # load model
    #     classifier = model_utils.load_model(network=network_name, dataset=dataset_name)
    #     # load adv data
    #     for method in ["cw", "fgsm", "bim", "jsma"]:
    #         x_data, y_data = load_adversarial_images(dataset=dataset_name, network=network_name, method=method)
    #         # prediction
    #         preds = classifier.predict(x_data,verbose=1)
    #         pred_res = np.argmax(preds,axis=1)
    #         # get misclassified images indices
    #         correct_indices = np.nonzero(pred_res == y_data)[0]
    #         wrong_indices = np.array(list(set(np.arange(preds.shape[0])) - set(correct_indices)))
    #         acc = len(correct_indices)/preds.shape[0]
    #
    #         left_x,left_y = x_data[wrong_indices].copy(), y_data[wrong_indices].copy()
    #         file_name = f"{dataset_network}_{method}.npz"
    #         adv_path = os.path.join(root_dir, "data", "adv_images")
    #         file_path = os.path.join(adv_path, dataset_network, file_name)
    #         np.savez(file_path,x_test=left_x[:5000].copy(),y_test=left_y[:5000].copy())
    #         left_y_vec = keras.utils.to_categorical(left_y, num_classes)
    #         score = classifier.evaluate(left_x, left_y_vec, verbose=0)
    #         row.append(f"{acc}/{score[1]}")
    #     adv_tables.add_row(row)
    # print(adv_tables)

    # save to file
