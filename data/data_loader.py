# Author:Nerako at Heriot-Watt University
# This project is used for personal Graduation project.
# Project: Using a Brain-Computer Interface with a Human Support Robot for Object Grasping

# This code is used for load data.

# -----------------------2023.10-2024.4-------------------------
import pandas as pd
import pyedflib
import torch.utils.data
from scipy.io import arff
import os
import mne
import matplotlib.pyplot as pl
import numpy as np
import utils
import warnings
import matplotlib.pyplot as plt
from io import StringIO
from pyedflib import highlevel

### code for processing data in arff format
class load_arff_data(torch.utils.data.Dataset):
    def __init__(self, arff_path, transform=None):
        self.data ,self.meta = arff.loadarff(arff_path)
        df = pd.DataFrame(self.data)
        numpy_array = df.to_numpy(dtype=np.float32)
        self.transform = transform if transform is not None else utils.MinMaxNormalize()

        self.pin_name = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]
        self.eyeDetection = ["eye close","eye open"]

        self.x_train = torch.tensor(numpy_array[:, :-1], dtype=torch.float)
        self.y_train = torch.tensor(numpy_array[:, -1], dtype=torch.long)

        self.path = arff_path

        self.x_train = self.transform(self.x_train)

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, index):
        sample = self.x_train[index]
        label = self.y_train[index]
        return sample, label

### code for processing data in csv format
class load_csv_data(torch.utils.data.Dataset):
    def __init__(self, csv_path, transform=None):
        self.data_dir = csv_path
        # self.transform = transform
        self.transform = transform if transform is not None else utils.MinMaxNormalize()
        self.pin_name = ["Fp1", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "Fp2"]
        self.label = ["OpenHand_left", "Touch_left", "Movement_left", "Lift_left", "Replace_left", "Release_left",
                      "OpenHand_right", "Touch_right", "Movement_right", "Lift_right", "Replace_right", "Release_right"]

        self.data_files = self._get_file_pairs()
        self.x_train, self.y_train = self._load_all_data()
        self.x_train = self.transform(self.x_train)

    def _get_file_pairs(self):
        data_files = []
        event_files = []

        for file in os.listdir(self.data_dir):
            if file.endswith("_data.csv"):
                data_files.append(os.path.join(self.data_dir, file))
            elif file.endswith("_events.csv"):
                event_files.append(os.path.join(self.data_dir, file))

        data_files.sort()
        event_files.sort()

        return list(zip(data_files, event_files))

    def _load_all_data(self):
        all_data = []
        all_labels = []

        for data_path, event_path in self.data_files:
            # Load specified columns
            data = pd.read_csv(data_path, usecols=["id"] + self.pin_name)
            events = pd.read_csv(event_path)

            # Filter rows where at least one event is non-zero
            events_filtered = events[(events.iloc[:, 1:] != 0).any(axis=1)]
            data_filtered = data[data['id'].isin(events_filtered['id'])]

            # Append to list
            all_data.append(data_filtered.iloc[:, 1:].values)  # Ignoring 'id' column
            labels_for_file = events_filtered.iloc[:, 1:].values

            labels_processed = []
            for label in labels_for_file:
                if not np.all(label == 0):
                    label_doubled = np.concatenate([label, label])
                    labels_processed.append(label_doubled)

            if labels_processed:
                all_labels.append(np.stack(labels_processed))

        # Concatenate all data and labels
        all_data_concat = torch.FloatTensor(np.concatenate(all_data, axis=0))
        all_labels_concat = torch.FloatTensor(np.concatenate(all_labels, axis=0))

        if self.transform:
            all_data_concat = self.transform(all_data_concat)

        return all_data_concat, all_labels_concat

    def __len__(self):
        return self.x_train.shape[0]

    def __getitem__(self, idx):
        return self.x_train[idx], self.y_train[idx]

    # def __getitem__(self, idx):
    #     data_path, event_path = self.data_files[idx]
    #
    #     # Update: Load the specified column based on the column name
    #     data = pd.read_csv(data_path, usecols=["id"] + self.pin_name)
    #     events = pd.read_csv(event_path)
    #
    #     # Rows with at least one non-zero item in the event are kept and filtered data synchronously
    #     events_filtered = events[(events.iloc[:, 1:] != 0).any(axis=1)]
    #     data_filtered = data[data['id'].isin(events_filtered['id'])]
    #
    #     # Ignore the id column and convert to a tensor
    #     data_tensor = torch.FloatTensor(data_filtered.iloc[:, 1:].values)  # Update: Only the data of the specified column is kept
    #     events_tensor = torch.FloatTensor(events_filtered.iloc[:, 1:].values)  # Ignoring the id column
    #
    #     if self.transform:
    #         data_tensor = self.transform(data_tensor)
    #
    #     return data_tensor, events_tensor

### code for processing data in mat format
class load_mat_data(torch.utils.data.Dataset):
    def __init__(self, mat_path):
        self.pin_name = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]



### code for processing data in edf format
class load_edf_data(torch.utils.data.Dataset):
    def __init__(self, data_root_dir, transform=None):
        # self.data_dir = edf_path
        self.data_root_dir = data_root_dir
        self.transform = transform if transform is not None else utils.MinMaxNormalize()
        self.pin_name = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]
        self.channels = ["Af3", "F7.", "F3.", "Fc5", "T7.", "P7.", "O1.", "O2.", "P8.", "T8.", "Fc6", "F4.", "F8.", "Af4"]
        self.label = ["OpenHand_left", "Touch_left", "Movement_left", "Lift_left", "Replace_left", "Release_left",
                      "OpenHand_right", "Touch_right", "Movement_right", "Lift_right", "Replace_right", "Release_right"]

        self.edf_files = self._find_all_edf_files(data_root_dir)
        # self.edf_files = [os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir) if f.endswith('.edf')]
        self.x_train, self.y_train = self._load_data()
        self.x_train = self.transform(self.x_train)

    def _find_all_edf_files(self, root_dir):
        edf_files = []
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if file.endswith('.edf'):
                    edf_files.append(os.path.join(root, file))
        return edf_files

    def _load_data(self):
        all_signals = []
        all_labels = []

        for edf_file in self.edf_files:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)

                raw = mne.io.read_raw_edf(edf_file, preload=True, verbose=False)
                # Adjust channel names to include '.' if necessary, based on the plot_channel function
                channels_adjusted = [ch + '.' if ch + '.' in raw.info['ch_names'] else ch for ch in self.channels]
                picks = mne.pick_channels(raw.info['ch_names'], include=channels_adjusted, ordered=False)
                signals = raw.get_data(picks=picks)
                all_signals.append(signals)

                # Extract events from annotations
                annotations = raw.annotations
                # Map annotations description to labels and filter out None values
                label_list = [self._map_annotation_to_label(desc) for desc in annotations.description]
                label_list = [label for label in label_list if label is not None]

                # Check if label_list is not empty before stacking
                if label_list:
                    labels_stacked = np.stack(label_list)
                    all_labels.append(labels_stacked)
                else:
                    # Handle the case where all annotations are T0 or unrecognized
                    # Potentially you might want to skip this file altogether
                    continue

        # Concatenating signals from all files
        if all_signals and all_labels:
            x_train = np.concatenate(all_signals, axis=1)
            y_train = np.concatenate(all_labels, axis=0)
            return torch.tensor(x_train.T, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)
        else:
            return torch.tensor([], dtype=torch.float32), torch.tensor([], dtype=torch.long)

    def _map_annotation_to_label(self, description):
        # Implement a mapping from annotation descriptions to your label schema
        # This is a placeholder mapping, adjust according to your annotation descriptions
        if description == 'T1':
            return np.array([1] * 6 + [0] * 6)
        elif description == 'T2':
            return np.array([0] * 6 + [1] * 6)
        else:
            return None  # Ignore T0 and unknown annotations

    def __len__(self):
        return len(self.x_train)

    def __getitem__(self, idx):
        return self.x_train[idx], self.y_train[idx]

    def preview_data(self, n=5):
        """Preview the first n samples from x_train and y_train."""
        print("x_train preview (first 5 samples):", self.x_train[:n])
        print("y_train preview (first 5 labels):", self.y_train[:n])

if __name__ == '__main__':
    # arff_data = load_arff_data("./data/EEG_Eyes.arff")
    # csv_data_dir = "./dataset/grasp-and-lift-eeg-detection/train"
    # dataset = load_csv_data(csv_data_dir)
    # dataset = load_edf_data("./dataset/eeg-motor-dataset/files/S001")
    #

    dataset = load_edf_data("./dataset/eeg-motor-dataset/files")
    print(dataset.x_train)
    print(dataset.y_train)