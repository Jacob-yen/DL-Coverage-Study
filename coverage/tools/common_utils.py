import pickle
import keras
import keras.backend as K

from coverage.tools import dataloader
from coverage.tools.deepspeech.deepspeech_utils import DSDataUtils
from coverage.tools.deepspeech.deepspeech_text import Alphabet
import os
import numpy as np
from coverage import root_dir



class ScoreUtils:
    @staticmethod
    def get_general_error_rate(x_test, y_test, model, num_classes, input_size_list=(100, 500, 1000)):
        Y_test = keras.utils.to_categorical(y_test, num_classes)
        score = model.evaluate(x_test, Y_test, verbose=1)
        print('INFO: Test accuracy:', score[1])
        print('INFO: Number of wrong prediction: ', len(x_test) * (1 - score[1]))
        # sheet.cell(row=row, column=column).value = (1 - score[1])
        return 1 - score[1]

    @staticmethod
    def get_driving_dave_orig_error_rate(x_test, y_test, model, ):
        pred = model.predict(x_test).reshape(-1)
        y_test = np.squeeze(y_test)
        print(x_test.shape, y_test.shape, pred.shape)
        mse = np.sum(np.square(pred - y_test)) / x_test.shape[0]
        print('INFO: Test accuracy:', 1 - mse)
        return mse

    @staticmethod
    def get_driving_chauffeur_score(x_test, y_test, model, ):
        yhats = []
        label = []
        model = model.make_stateful_predictor()
        for i in range(x_test.shape[0]):
            yhat = model(x_test[i])
            yhats.append(yhat)
            label.append(y_test[i])
        mse = 0.
        count = 0
        if len(yhats) != len(label):
            print("yhat and label have different lengths")
            return -1
        for i in range(len(yhats)):
            count += 1
            predicted_steering = yhats[i]
            steering = label[i]
            mse += (float(steering) - float(predicted_steering)) ** 2.
        mse = mse / count
        print('INFO: Test accuracy:', 1 - mse)

        return mse

    @staticmethod
    def get_deepspeech_error_rate(x_test, y_test, model):
        alphabet = Alphabet(os.path.join(root_dir, 'files/original_datasets/speech-commands_deepspeech/alphabet.txt'))
        import keras.backend as K
        total = 0
        diff = 0
        correct = 0
        y_true = [[alphabet._label_to_str[y] for y in x] for x in y_test]
        y_true = [''.join(x).strip() for x in y_true]

        y_pred = model.predict(x_test, verbose=1)
        # print(f"output shape of model:{y_pred.shape}")
        # print(y_pred[0])
        # print(np.argsort(y_pred[0], axis=1))
        input_len = np.ones(y_pred.shape[0]) * y_pred.shape[1]
        # print("input_len", input_len.shape)
        # print(input_len[0])
        # print(input_len[1])
        y_pred = K.get_value(K.ctc_decode(y_pred, input_length=input_len)[0][0])
        # print("after get value", y_pred.shape)
        y_pred = [[alphabet._label_to_str[y] for y in x if y >= 0] for x in y_pred]
        # print("example,", y_pred[0])
        y_pred = [''.join(x).strip() for x in y_pred]
        # print("example,", y_pred[0])
        y_pred = [DSDataUtils.get_bestmatch_keywords_using_wer(x) for x in y_pred]
        # print("example,", y_pred[0])
        # print(len(y_pred))
        for a, b in list(zip(y_true, y_pred)):
            total += 1
            if a == b:
                correct += 1
        print("INFO: Test accuracy", correct / total)
        return 1 - correct / total

    @staticmethod
    def get_model_error_rate(x_test, y_test, model, dataset, network, option_kw=None):
        if dataset in ["fashion-mnist", 'mnist', 'cifar10', 'cifar100', 'imagenet']:
            return ScoreUtils.get_general_error_rate(x_test=x_test, y_test=y_test, model=model,
                                                     num_classes=dataloader.class_num(dataset))
        elif dataset == "driving":
            if network == "chauffeur":
                return ScoreUtils.get_driving_chauffeur_score(x_test=x_test, y_test=y_test, model=model)
            else:
                return ScoreUtils.get_driving_dave_orig_error_rate(x_test=x_test, y_test=y_test, model=model)
        elif dataset == "speech-commands":
            return ScoreUtils.get_deepspeech_error_rate(x_test=x_test, y_test=y_test, model=model)
        else:
            raise Exception(f"No such error function for {dataset}-{network}")

    @staticmethod
    def classification_error_rate(labels, predictions):
        assert len(labels.shape) == 1, "labels shouldn't be one-hot vector format"
        pred_labels = np.argmax(predictions, axis=1)
        error_rate = np.sum(labels != pred_labels) / len(labels)
        return error_rate

    @staticmethod
    def regression_error_rate(labels, predictions):
        labels = np.squeeze(labels)
        predictions = np.squeeze(predictions)
        mse = np.sum(np.square(predictions - labels)) / len(predictions)
        return mse

    @staticmethod
    def speech_commands_prediction(predictions):
        alphabet = Alphabet(os.path.join(root_dir, 'files/original_datasets/speech-commands_deepspeech/alphabet.txt'))
        input_len = np.ones(predictions.shape[0]) * predictions.shape[1]
        y_pred = K.get_value(K.ctc_decode(predictions, input_length=input_len)[0][0])
        y_pred = [[alphabet._label_to_str[y] for y in x if y >= 0] for x in y_pred]
        y_pred = [''.join(x).strip() for x in y_pred]
        y_pred = [DSDataUtils.get_bestmatch_keywords_using_wer(x) for x in y_pred]
        return y_pred

    @staticmethod
    def speech_commands_error_rate(labels, predictions):
        alphabet = Alphabet(os.path.join(root_dir, 'files/original_datasets/speech-commands_deepspeech/alphabet.txt'))
        total = 0
        correct = 0
        y_true = [[alphabet._label_to_str[y] for y in x] for x in labels]
        y_true = [''.join(x).strip() for x in y_true]
        y_pred = ScoreUtils.speech_commands_prediction(predictions)
        for a, b in list(zip(y_true, y_pred)):
            total += 1
            if a == b:
                correct += 1
        return 1 - correct / total

    @staticmethod
    def analyze_error_rate(labels, predictions, dataset, network, ):
        if dataset in ["fashion-mnist", 'mnist', 'cifar10', 'cifar100', 'imagenet']:
            return ScoreUtils.classification_error_rate(labels, predictions)
        elif dataset == "driving":
            return ScoreUtils.regression_error_rate(labels, predictions)
        elif dataset == "speech-commands":
            return ScoreUtils.speech_commands_error_rate(labels, predictions)
        else:
            raise NotImplementedError(f"No such implementation for {dataset}-{network}")


def load_boundary(dataset, network):
    # it can be generated by running coverage/tools/boundary.py
    with open(os.path.join(root_dir, "files", "boundary_values", f"boundary_{dataset}_{network}.pkl"), "rb") as file:
        return pickle.load(file)


def sampling_indices_dict(groups_num, dataset_model, test_size=1000):
    with open(os.path.join(root_dir, "files", f"total_sampling_{groups_num}groups_{test_size}tests.pkl"),
              "rb") as file:
        return pickle.load(file)[dataset_model]


def create_path(*args):
    for file_path in args:
        if not os.path.exists(file_path):
            os.makedirs(file_path)


def cached_sa(dataset_network, attack_type, test_size, classification=True):
    # file is like dsa_mnist_lenet5_normal_10000.npy
    lsa_file = os.path.join(root_dir, "files/cached_sa",
                            f"{dataset_network}/lsa_{dataset_network}_{attack_type}_{test_size}.npy")
    lsa = np.load(lsa_file, allow_pickle=True)
    if classification:
        dsa_file = os.path.join(root_dir, "files/cached_sa",
                                f"{dataset_network}/dsa_{dataset_network}_{attack_type}_{test_size}.npy")
        dsa = np.load(dsa_file, allow_pickle=True)

    else:
        dsa = None

    return lsa, dsa
