import csv
import math
import os

import numpy as np
import pandas as pd
from category_encoders import OrdinalEncoder
from keras.utils import to_categorical
from scipy import stats
from scipy.stats import entropy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from scipy.signal import butter, filtfilt, medfilt


class Preprocess:

    def __init__(self, binary_label=None):
        """ initialise label binarizer """
        self.encoder = OrdinalEncoder()
        with open('config/norm_mat.csv') as file:
            norm_mat = pd.read_csv(file, header=None)
        self.norm_mins = norm_mat.iloc[0, :]
        self.norm_ranges = norm_mat.iloc[1, :]

    def window(self, df, window_size=100, step_size=100, xyz=None):
        """ return windowed data with stepped overlap """
        if xyz is None:
            xyz = [1, 2, 3]

        current_frames = 0
        # Instantiate dataframe to hold windowed output
        windowed_output = pd.DataFrame()
        # Window data, apply filter, and extract additional features where required
        for chunk in self.__chunker(df, window_size, step_size):
            if chunk.shape[0] == window_size:  # only interested in whole chunks
                # Get timestamp for whole frame
                timestamps = chunk.values[:, 0]
                frame_timestamp = timestamps[len(timestamps)//2]
                # take x, y, z values
                frame = chunk.take(xyz, axis=1)
                frame = self.apply_filter(frame).values.flatten()
                if not self.raw_data_only:
                    # extract extra features
                    energy = self.__energy(frame)
                    rms = self.__root_mean_square(frame)
                    iqr = self.__interquartile_range(frame)
                    entropy = self.__entropy(frame)
                    std_d = self.__std_d(frame)
                    median = self.__median(frame)
                    variance = self.__variance(frame)
                    rngv = self.__rangeValue(frame)
                    skew = self.__skewness(frame)
                    kurt = self.__kurtosis(frame)
                    sdft = self.__sumDiscreteFourierTransform(frame)
                    ddft = self.__dominantDiscreteFourierTransform(frame)

                    # append extra features and to frames existing xyz data
                    frame = np.append(frame, np.array([
                        energy,
                        rms,
                        iqr,
                        entropy,
                        std_d,
                        median,
                        variance,
                        rngv,
                        skew,
                        kurt,
                        sdft,
                        ddft
                    ]))
                frame = np.append(frame, np.array([frame_timestamp]))
                if current_frames > 0:
                    windowed_output = np.vstack((windowed_output, frame))
                else:
                    windowed_output = frame
                current_frames += 1
                self.__window_progress_report(
                    df, current_frames, step_size)
        return windowed_output

    def __chunker(self, seq, window_size, step_size):
        """ return a window / chunk of incoming sequence data """
        return (seq[pos:pos + window_size] for pos in range(0, len(seq), step_size))

    def __window_progress_report(self, df, current_frames, step_size):
        """ print progress of windowing """
        if current_frames % 100 == 0:
            estimate_frames = df.shape[0] / step_size
            print("[WINDOW PROGRESS]: {} / {}".format(current_frames, estimate_frames))

    def ohe(self, y, num_classes=4):
        """ return features and one-hot encoded label for given dataframe """
        df = pd.DataFrame(y, columns=['labels'])
        encoded_labels = self.encoder.fit_transform(df)
        x = to_categorical(encoded_labels, num_classes=num_classes)
        return x

    def split_data(self, df):
        """ return features X and labels y from dataframe """
        label_column_idx = len(df.columns) - 1
        X = df.values[:, :label_column_idx]
        y = df.values[:, label_column_idx]
        return X, y

    def train_test_split(self, X, y, normalise=False, test_size=0.33, random_state=0):
        """ return train and test data given X and y """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state)

        if normalise:
            print("[normalised]")
            X_train = self.normalise(X_train)
            X_test = self.normalise(X_test)

        return X_train, X_test, y_train, y_test

    def filter(self, data):
        b_low, a_low = butter(3, 0.3, fs=100)
        b_high, a_high = butter(4, 0.3, 'highpass', fs=100)

        low_filtered_x = filtfilt(b_low, a_low, medfilt(data['ax'], 5))
        low_filtered_y = filtfilt(b_low, a_low, medfilt(data['ay'], 5))
        low_filtered_z = filtfilt(b_low, a_low, medfilt(data['az'], 5))

        high_filtered_x = filtfilt(b_high, a_high, data['ax'])
        high_filtered_y = filtfilt(b_high, a_high, data['ay'])
        high_filtered_z = filtfilt(b_high, a_high, data['az'])

        gravity = pd.DataFrame(list(zip(low_filtered_x, low_filtered_y, low_filtered_z)))
        body_acceleration = pd.DataFrame(list(zip(high_filtered_x, high_filtered_y, high_filtered_z)))

        return np.hstack([gravity, body_acceleration])

    def stack_data(self, data):
        loaded = list()
        loaded.append(data[:, 0:100])
        loaded.append(data[:, 100:200])
        loaded.append(data[:, 200:300])
        loaded.append(data[:, 300:400])
        loaded.append(data[:, 400:500])
        loaded.append(data[:, 500:600])
        loaded = np.dstack(loaded)

        X = loaded
        return X

    def normalise(self, X):
        # return (X-np.asarray(self.norm_mins))/np.asarray(self.norm_ranges)
        return X

    def decode(self, Y):
        """ return decoded numerical classes to text representation """
        encoded_labels = [np.argmax(y, axis=None, out=None) for y in Y]
        df = pd.DataFrame(encoded_labels, columns=['labels'])
        return self.encoder.inverse_transform(df['labels'])[
            'labels'].tolist()

    def reshape_data(self, X):
        """ return reshaped X for lstm input """
        n_steps, n_length, n_features = 5, 20, 6
        return X.reshape((X.shape[0], n_steps, n_length, n_features))

