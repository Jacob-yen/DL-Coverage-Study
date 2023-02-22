import csv
import numpy as np
import os
import sys
from coverage import root_dir
# sys.path.append("../..")
use_ds2 = True  # Deep Speech 2

# Number of MFCC features
n_input = 40  # MFCC, maybe need add Delta

# The number of frames in the context
n_context = 0

feature_len = 100  # all input is 1s wavfiel 1000ms/10ms = 100.

feature_dim = n_input * (n_context * 2 + 1)

# alphabet = Alphabet('/data/cradle/dataset/alphabet.txt')
# print('alphabet.size() ', alphabet.size())
# print(alphabet._label_to_str)
# # The number of characters in the target language plus one
#
# n_character = alphabet.size() + 1  # +1 for CTC blank label

max_labellen = 6

n_hidden = 128

trfile = os.path.join(root_dir,'files/original_datasets/speech-commands_deepspeech/speech-commands_train.csv')
cvfile = os.path.join(root_dir,'files/original_datasets/speech-commands_deepspeech/speech-commands_dev.csv')
testfile = os.path.join(root_dir,'files/original_datasets/speech-commands_deepspeech/speech-commands_test.csv')
advfile = os.path.join(root_dir,'files/original_datasets/speech-commands_deepspeech/speech-commands_adv.csv')

BATCH_SIZE_INFERENCE = 1000
VERBOSE = 1

allwords = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go", "zero", "one", "two",
            "three",
            "four", "five", "six", "seven", "eight", "nine", "bed", "bird", "cat", "dog", "happy", "house",
            "marvin",
            "sheila", "tree", "wow"]

class DSDataUtils:
    def __init__(self):
        pass

    @staticmethod
    def loadfromecsv(fname):
        r = []
        with open(fname) as docfile:
            reader = csv.reader(docfile)
            for line in reader:
                r.append(line)
        return r

    @staticmethod
    def get_files_mfcc(wav_filenames):
        from coverage.tools.deepspeech.deepspeech_audio import audiofile_to_input_vector
        mfccs = []
        lens = []
        for audio_fname in wav_filenames:
            this_mfcc = audiofile_to_input_vector(audio_fname, n_input, n_context)
            if len(this_mfcc) != feature_len:
                needlen = feature_len - len(this_mfcc)
                a = ([[0 for x in range(feature_dim)] for y in range(needlen)])
                this_mfcc = np.concatenate((this_mfcc, np.array(a)))
            # print(this_mfcc.shape)
            this_mfcc = np.reshape(this_mfcc, (feature_len, n_input, 1))
            mfccs.append(this_mfcc)
            lens.append(len(this_mfcc))
        a_mfccs = np.array(mfccs)  # shape, (batch, time_step_len, feature_len)
        a_lens = np.array(lens)  # shape, (batch, 1), value == time_step_len
        # print('MFCCs shape', a_mfccs.shape, a_lens.shape)
        return a_mfccs, a_lens

    @staticmethod
    def ctc_lambda_func(args):
        import keras.backend as K
        y_pred, labels, input_length, label_length = args
        return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

    @staticmethod
    def gen_ctc_byclass(dataGen, return_wavfilenames=False):
        batch_size = dataGen.batchSize
        while True:
            a_mfccs, a_mfcclens, a_label, a_labelen, wavfilenames = dataGen.__next__()
            if not return_wavfilenames:
                yield [a_mfccs, a_label, a_mfcclens, a_labelen], np.ones(batch_size)
            else:
                yield [a_mfccs, a_label, a_mfcclens, a_labelen], np.ones(batch_size), wavfilenames

    @staticmethod
    def get_bestmatch_keywords_using_wer(str):
        from coverage.tools.deepspeech.deepspeech_text import wer
        if str in allwords:
            return str
        r = []
        str1 = ' '.join(list(str))
        for o in allwords:
            o1 = ' '.join(list(o))
            # print (type(o), type(str), o1, str1)
            r.append(wer(o1, str1))
        idx = int(np.argmin(np.array(r), axis=0))
        # print (idx)
        # print(str, allwords[idx])
        return allwords[idx]

    @staticmethod
    def get_words_idx(s):
        assert s in allwords, f"Error: {s} not in {allwords}"
        return allwords.index(s)


class dataGen_mfcc_ctc:
    def __init__(self, csvfile, batchSize=128):
        self.batchPointer = 0
        self.data = DSDataUtils.loadfromecsv(csvfile)[1:]
        self.batchSize = batchSize
        self.totallen = int(len(self.data))

        self.numSteps = int(self.totallen / self.batchSize + 1)
        print('dataGen_speechcmd_mfcc: init', len(self.data))

    def __next__(self):
        if (self.batchPointer + self.batchSize) >= len(self.data):
            self.batchPointer = 0
            thislen = self.batchSize
        else:
            thislen = self.batchSize
        a_mfccs, a_mfcclens, a_label, a_labelen, wav_filenames = self.getNextSplitData(
            self.data[self.batchPointer: self.batchPointer + thislen])
        self.batchPointer += thislen
        if self.batchPointer >= len(self.data):
            self.batchPointer = 0
            np.random.shuffle(self.data)
        return a_mfccs, a_mfcclens, a_label, a_labelen, wav_filenames

    def _getLabel(self, transcripts):
        from coverage.tools.deepspeech.deepspeech_text import Alphabet
        alphabet = Alphabet(os.path.join(root_dir,'files/original_datasets/speech-commands_deepspeech/alphabet.txt'))
        print('alphabet.size() ', alphabet.size())
        a_label = np.asarray(
            [[alphabet.label_from_string(a) for a in c + ' ' * (max_labellen - len(c))] for c in transcripts])
        a_labelen = np.asarray([len(c) for c in transcripts])
        return a_label, a_labelen

    def getNextSplitData(self, fileinfos):
        wav_filenames = list(zip(*fileinfos))[0]
        # wav_filesizes = list(zip(*fileinfos))[1]
        transcripts = list(zip(*fileinfos))[2]
        a_mfccs, a_mfcclens = DSDataUtils.get_files_mfcc(wav_filenames)
        a_label, a_labelen = self._getLabel(transcripts)
        return a_mfccs, a_mfcclens, a_label, a_labelen, wav_filenames
