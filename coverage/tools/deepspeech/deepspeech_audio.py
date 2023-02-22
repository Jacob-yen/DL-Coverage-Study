from __future__ import absolute_import, print_function

import scipy.io.wavfile as wav
import sys
from coverage import root_dir
try:
    from deepspeech.utils import audioToInputVector
    # import deepspeech

except ImportError:
    print("No deepspeech")
    import numpy as np
    from python_speech_features import mfcc
    from six.moves import range

    class DeprecationWarning:
        displayed = False

    def audioToInputVector(audio, fs, numcep, numcontext):
        if DeprecationWarning.displayed is not True:
            DeprecationWarning.displayed = True
            print('------------------------------------------------------------------------', file=sys.stderr)
            print('WARNING: libdeepspeech failed to load, resorting to deprecated code',      file=sys.stderr)
            print('         Refer to README.md for instructions on installing libdeepspeech', file=sys.stderr)
            print('------------------------------------------------------------------------', file=sys.stderr)

        # Get mfcc coefficients
        features = mfcc(audio, samplerate=fs, numcep=numcep)

        # We only keep every second feature (BiRNN stride = 2)
        features = features[::2]

        # One stride per time step in the input
        num_strides = len(features)

        # Add empty initial and final contexts
        empty_context = np.zeros((numcontext, numcep), dtype=features.dtype)
        features = np.concatenate((empty_context, features, empty_context))

        # Create a view into the array with overlapping strides of size
        # numcontext (past) + 1 (present) + numcontext (future)
        window_size = 2*numcontext+1
        train_inputs = np.lib.stride_tricks.as_strided(
            features,
            (num_strides, window_size, numcep),
            (features.strides[0], features.strides[0], features.strides[1]),
            writeable=False)

        # Flatten the second and third dimensions
        train_inputs = np.reshape(train_inputs, [num_strides, -1])

        # Whiten inputs
        # Copy the strided array so that we can write to it safely
        train_inputs = np.copy(train_inputs)
        train_inputs = (train_inputs - np.mean(train_inputs))/np.std(train_inputs)

        # Return results
        return train_inputs


def audiofile_to_input_vector(audio_filename, numcep, numcontext):
    r"""
    Given a WAV audio file at ``audio_filename``, calculates ``numcep`` MFCC features
    at every 0.01s time step with a window length of 0.025s. Appends ``numcontext``
    context frames to the left and right of each time step, and returns this data
    in a numpy array.
    """
    import os
    # basedir = os.path.abspath(os.path.dirname(__file__))
    # Load wav files
    # fs, audio = wav.read(os.path.join(basedir,audio_filename))
    fs, audio = wav.read(os.path.join(root_dir,"files/original_datasets/speech-commands_deepspeech", audio_filename))
    # print('fs, numcep, numcontext:', fs, numcep, numcontext)
    return audioToInputVector(audio, fs, numcep, numcontext)
