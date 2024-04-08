# Author:Nerako at Heriot-Watt University
# This project is used for personal Graduation project.
# Project: Using a Brain-Computer Interface with a Human Support Robot for Object Grasping

# This code is used for some helper functions

# -----------------------2023.10-2024.4-------------------------
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pyedflib
import pandas as pd

### Training settings:
def parse_option():
    parser = argparse.ArgumentParser('BCI control robot')



    args, unparsed = parser.parse_known_args()
    return args

# ### Plot the channel
# def plot_channel(edf_file_path):
#     channels = ["Af3", "F7.", "F3.", "Fc5", "T7.", "P7.", "O1.", "O2.", "P8.", "T8.", "Fc6", "F4.", "F8.", "Af4"]
#
#     with pyedflib.EdfReader(edf_file_path) as f:
#         # Get all the channel labels in the file
#         all_channel_labels = f.getSignalLabels()
#
#         # Adjusting the channels list to match the actual channel names in the file
#         channels_adjusted = [ch if ch in all_channel_labels else ch + '.' for ch in channels]
#
#         # Finding the indices of the channels of interest
#         indices_of_interest = [all_channel_labels.index(ch) for ch in channels_adjusted]
#
#         eeg_data = np.array([f.readSignal(i) for i in indices_of_interest])
#
#     # Get the sampling rate (assuming that all channels have the same sampling rate)
#     sampling_rate = f.getSampleFrequency(indices_of_interest[0])
#
#     time_axis = np.arange(eeg_data.shape[1]) / sampling_rate
#     fig, axes = plt.subplots(len(channels), 1, figsize=(15, 20), sharex=True)
#
#     fig.suptitle("plot channel", fontsize=14)
#
#     for i, channel in enumerate(channels):
#         axes[i].plot(time_axis, eeg_data[i], label=channel)
#         axes[i].legend(fontsize=8)
#         # axes[i].set_ylabel('Amplitude', fontsize=10)
#         # Remove the set_title to declutter the plot
#         # axes[i].set_title(channel, fontsize=10)
#
#     axes[-1].set_xlabel('Time (s)', fontsize=12)
#     plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for the title
#     plt.show()

### min-max normalization
class MinMaxNormalize:
    def __init__(self, min_val=None, max_val=None):
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, x):
        if self.min_val is None or self.max_val is None:
            self.min_val = x.min(0, keepdim=True)[0]
            self.max_val = x.max(0, keepdim=True)[0]
        return (x - self.min_val) / (self.max_val - self.min_val)


def positional_encoding(x_train):
    n_electrodes = x_train.shape[1]
    encoded_data = np.zeros_like(x_train)

    for i in range(n_electrodes):
        if i % 2 == 0:
            encoded_data[:, i] = np.sin(x_train[:, i] / (10000 ** (2 * i / n_electrodes)))
        else:
            encoded_data[:, i] = np.cos(x_train[:, i] / (10000 ** (2 * (i - 1) / n_electrodes)))

    return encoded_data


def plot_channel(edf_file_path):
    channels = ["T7.", "F7.", "Fc5", "Af3", "F3.", "P7.", "O1.", "O2.", "P8.", "F4.", "Af4", "F8.", "Fc6", "T8."]

    with pyedflib.EdfReader(edf_file_path) as f:
        # Get all the channel labels in the file
        all_channel_labels = f.getSignalLabels()

        # Adjusting the channels list to match the actual channel names in the file
        channels_adjusted = [ch if ch in all_channel_labels else ch + '.' for ch in channels]

        # Finding the indices of the channels of interest
        indices_of_interest = [all_channel_labels.index(ch) for ch in channels_adjusted]

        eeg_data = np.array([f.readSignal(i) for i in indices_of_interest])

    # 在绘图之前，对EEG数据应用位置编码
    encoded_eeg_data = positional_encoding(eeg_data)

    # Get the sampling rate (assuming that all channels have the same sampling rate)
    sampling_rate = f.getSampleFrequency(indices_of_interest[0])

    # 下面的代码不变，只是把 eeg_data 替换为 encoded_eeg_data
    time_axis = np.arange(encoded_eeg_data.shape[1]) / sampling_rate
    fig, axes = plt.subplots(len(channels), 1, figsize=(15, 20), sharex=True)

    fig.suptitle("plot channel", fontsize=14)

    for i, channel in enumerate(channels):
        axes[i].plot(time_axis, eeg_data[i], label=channel)
        axes[i].legend(fontsize=8)
        # axes[i].set_ylabel('Amplitude', fontsize=10)
        # Remove the set_title to declutter the plot
        # axes[i].set_title(channel, fontsize=10)

    axes[-1].set_xlabel('Time (s)', fontsize=12)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for the title

    plt.show()