import torch
import torch.nn as nn
import numpy as np
import gzip
import matplotlib.pyplot as plt
from typing import List


class Reader:
    """
    Reads in MNIST data.  See http://yann.lecun.com/exdb/mnist/
    """
    @staticmethod
    def csv_to_tensor(file_path: str, input_type: torch.dtype = torch.FloatTensor, output_type: torch.dtype = torch.FloatTensor, skip_header: bool = True) -> torch.tensor:
        data = []
        output = []
        with open(file_path) as file:
            for line in file:
                if skip_header:
                    skip_header = False
                    continue
                pixel_data = list(map(int, line.split(',')))
                data.append(pixel_data[1:])
                output.append(Reader.to_one_hot(pixel_data[0]))
        return [torch.tensor(data, dtype = input_type), torch.tensor(output, dtype = output_type)]

    @staticmethod
    def data_to_tensor(file_path: str, image_size: int, n_images: int, data_type: torch.dtype = torch.float) -> torch.tensor:
        data = None
        with gzip.open(file_path, "rb") as file:
            i = 0
            file.read(16)
            buffer = file.read(image_size * image_size * n_images)
            data = np.frombuffer(buffer, dtype = np.uint8).astype(np.float32)
            data = data.reshape(n_images, image_size * image_size)
        return torch.tensor(data, dtype = data_type)

    @staticmethod
    def labels_to_tensor(file_path: str, n_labels: int, data_type: torch.dtype = torch.float) -> torch.tensor:
        data = None
        d = []
        with gzip.open(file_path, "rb") as file:
            file.read(8)
            buffer = file.read(n_labels)
            data = np.frombuffer(buffer, dtype = np.uint8).astype(np.int)
            for i in range(len(data)):
                d.append(Reader.to_one_hot(data[i]))
        return torch.tensor(d, dtype = data_type)

    @staticmethod
    def to_one_hot(num: int) -> np.ndarray:
        t = np.zeros([10])
        t[num] = 1
        return t


class Plotter:
    """
    Plots data from the MNIST dataset.
    """
    @staticmethod
    def plot(tensor_list: List[torch.tensor], image_size: int, n_rows: int, n_columns: int, actual_label_list: List, predicted_label_list: List):
        fig = plt.figure(figsize=(8, 8))
        for i in range(0, len(tensor_list)):
            img = tensor_list[i].numpy().reshape([image_size, image_size])
            ax = fig.add_subplot(n_rows, n_columns, i + 1)

            title = ""
            if actual_label_list is not None:
                title += "A: " + str(actual_label_list[i])
            if predicted_label_list is not None:
                title += " P: " + str(predicted_label_list[i])
            ax.set_title(title, fontsize=10)

            plt.imshow(img)
        plt.show()

if __name__ == "__main__":
    """
    train_path = "data/train.csv"
    test_path = "data/test.csv"
    train_input, train_output = Reader.csv_to_tensor(train_path, input_type = torch.float, output_type = torch.float)
    test_input, test_output = Reader.csv_to_tensor(test_path, input_type = torch.float, output_type = torch.float)

    print(train_input.shape)
    print(train_output.shape)
    print(test_input.shape)
    print(test_output.shape)
    """

    n_train = 60000
    n_test = 10000

    image_size = 28

    train_data_path = "data/train-images.gz"
    train_labels_path = "data/train-labels.gz"
    test_data_path = "data/t10k-images.gz"
    test_labels_path = "data/t10k-labels.gz"

    train_data = Reader.data_to_tensor(train_data_path, image_size, n_train)
    train_labels = Reader.labels_to_tensor(train_labels_path, n_train)
    test_data = Reader.data_to_tensor(test_data_path, image_size, n_test)
    test_labels = Reader.labels_to_tensor(test_labels_path, n_test)

    print(train_data.shape)
    print(train_labels.shape)
    print(test_data.shape)
    print(test_labels.shape)

    print(test_labels[0])

    #plt.imshow(test_data[10, :, :].numpy())
    #plt.show()