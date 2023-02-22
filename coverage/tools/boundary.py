"""
Run command:
nohup python -u -m coverage.tools.boundary --dataset fashion-mnist --network lenet5 > logs/adv_fm_lenet5.log 2>&1 &
"""

import os
import pickle
from collections import defaultdict

import keras.layers
import numpy as np
import argparse
from datetime import datetime
from keras import Model
from tqdm import tqdm
from coverage.tools import model_utils, dataloader


def get_batch(input_data, batch_size):
    batch_num = int(np.ceil(len(input_data) / batch_size))
    for i in range(batch_num):
        yield input_data[i * batch_size:(i + 1) * batch_size]


if __name__ == "__main__":
    start = datetime.now()

    cur_dir = os.path.abspath(os.path.dirname(__file__))
    parent_dir = os.path.dirname(cur_dir)

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", help="Dataset", type=str, default="cifar10")
    parser.add_argument("--network", "-n", help="Network", type=str, default="alexnet")
    parser.add_argument("--batch_size", "-b", help="The number of batch_size", type=int, default=128)
    args = parser.parse_args()
    print(args)

    # load model
    exclude_layers = ["input", "flatten"]
    model = model_utils.load_model(network=args.network, dataset=args.dataset)
    # construct function model
    layers = [layer for layer in model.layers if all(ex not in layer.name for ex in exclude_layers)]
    layer_names = [layer.name for layer in layers]
    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=[model.get_layer(layer_name).output
                                              for layer_name in layer_names])
    # construct boundary dict
    boundary_dict = defaultdict(list)
    for layer_idx, layer in enumerate(layers):
        for neuron_idx in range(layer.output_shape[-1]):
            boundary_dict[(layer_idx, neuron_idx)] = [np.inf, -np.inf]

    # load train set
    # check whether we need to load it from different file
    if args.dataset == "imagenet":
        raise NotImplemented("ImageNet dataset is too large and we need to load it iteratively."
                             "Currently it's not supported")
    else:
        x_train_raw, y_train = dataloader.load_train_set(args.dataset)
        x_train = dataloader.preprocess_dataset(args.dataset, args.network, x_train_raw)
        print(f"INFO: {args.dataset, args.network} value range of clean images :[{np.min(x_train)},{np.max(x_train)}]")

    # update boundary dict
    for batch_idx, batch in enumerate(get_batch(x_train, args.batch_size)):

        intermediate_layer_outputs = intermediate_layer_model.predict(batch)
        for layer_i, intermediate_layer_output in enumerate(tqdm(intermediate_layer_outputs)):
            input_size, neuron_count = intermediate_layer_output.shape[0], intermediate_layer_output.shape[-1]

            """
            The representation of neuron activation is followed by deepxplore[1].
            I'm sure the implementation below (employing vectorization) is equivalent to the way they use in deepxplore.

            [1] https://github.com/peikexin9/deepxplore/blob/master/MNIST/utils.py#L90
            """
            # get input_neuron matrix. shape is like (input_size,neuron_count)
            activations_mat_nc = np.mean(intermediate_layer_output.reshape((input_size, -1, neuron_count)), axis=1)

            for neuron_idx in range(activations_mat_nc.shape[-1]):
                neuron_outputs = activations_mat_nc[..., neuron_idx]
                min_value, max_value = np.min(neuron_outputs), np.max(neuron_outputs)
                boundary_dict[(layer_i, neuron_idx)][0] = np.minimum(min_value, boundary_dict[(layer_i, neuron_idx)][0])
                boundary_dict[(layer_i, neuron_idx)][1] = np.maximum(max_value, boundary_dict[(layer_i, neuron_idx)][1])

    # dump boundary dict
    save_path = os.path.join(parent_dir, "boundary_values")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(os.path.join(save_path, f"boundary_{args.dataset}_{args.network}.pkl"), "wb") as file:
        pickle.dump(boundary_dict, file)
