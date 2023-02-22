import os
import keras
from coverage.tools.driving.driving_models import *
from coverage import root_dir


def load_model(network, dataset):
    model_name = f"{dataset}_{network}.h5"
    if dataset == "imagenet":
        if network == "resnet50":
            m = keras.applications.resnet50.ResNet50(weights='imagenet')
        elif network == 'vgg19':
            m = keras.applications.vgg19.VGG19(weights='imagenet')
        else:
            raise NotImplementedError(f"ImageNet for network {network} is not implemented")
    elif network == 'dave-orig':
        img_rows, img_cols = 100, 100
        input_shape = (img_rows, img_cols, 3)
        # define input tensor as a placeholder
        input_tensor = Input(shape=input_shape)
        m = Dave_orig(input_tensor=input_tensor)
        m.load_weights(os.path.join(root_dir, "files/models", 'driving_dave-orig.h5'))
        m.compile(loss='mse', optimizer='adadelta')
    elif network == 'dave-dropout':
        img_rows, img_cols = 100, 100
        input_shape = (img_rows, img_cols, 3)
        # define input tensor as a placeholder
        input_tensor = Input(shape=input_shape)
        m = Dave_dropout(input_tensor=input_tensor)
        m.load_weights(os.path.join(root_dir, "files/models", 'driving_dave-dropout.h5'))
        m.compile(loss='mse', optimizer='adadelta')
    elif os.path.exists(os.path.join(root_dir,"files/models", model_name)):
        print(os.path.join(root_dir,"files/models", model_name))
        m = keras.models.load_model(os.path.join(root_dir,"files/models", model_name))
    else:
        raise NotImplementedError(f"No such model for {model_name}")

    if dataset in ['mnist','cifar10','cifar100','imagenet',"fashion-mnist"]:
        m.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    return m


if __name__ == '__main__':
    pass
    # dataset_networks_tuples = [("mnist", "lenet5", 10), ("cifar10", "vgg16", 10), ("cifar10", "resnet20", 10),
    #                            ("cifar100", "resnet32", 100), ("imagenet", "vgg19", 1000),
    #                            ("imagenet", "resnet50", 1000),
    #                            ("speech-commands", "deepspeech", None), ("driving", "dave-orig", None),
    #                            ("driving", "chauffeur", None), ]
    # for dn_t in dataset_networks_tuples:
    #     print(dn_t[1])
    #     model = load_model(dataset=dn_t[0],network=dn_t[1])