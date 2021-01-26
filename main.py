import torch
import torch.nn as nn
from mnist_data import Reader, Plotter
import numpy as np
from network import NeuralNetwork

model_save_path = "state_dicts/model.torch"
nn = NeuralNetwork(model_save_path)

n_epochs = 50000
learning_rate = 0.001
image_size = 28
n_train = 60000
n_test = 10000

train_data_path = "data/train-images.gz"
train_labels_path = "data/train-labels.gz"
test_data_path = "data/t10k-images.gz"
test_labels_path = "data/t10k-labels.gz"

train_data = Reader.data_to_tensor(train_data_path, image_size, n_train).cuda()
train_labels = Reader.labels_to_tensor(train_labels_path, n_train).cuda()
test_data = Reader.data_to_tensor(test_data_path, image_size, n_test).cuda()
test_labels = Reader.labels_to_tensor(test_labels_path, n_test).cuda()

def main():
    nn.train(n_epochs, learning_rate, train_data, train_labels)

    print(nn.loss_function(nn.model(test_data), test_labels))

    plot_data = []
    plot_actual_labels = []
    plot_predicted_labels = []
    n_rows = 3
    n_cols = 3
    for i in range(0, n_rows * n_cols):
        random_index = np.random.randint(0, n_test)
        plot_data.append(test_data[random_index, :].cpu())
        plot_actual_labels.append(test_labels.cpu().numpy()[random_index].argmax())
        plot_predicted_labels.append(nn.model(test_data[random_index]).detach().cpu().numpy().argmax())

    Plotter.plot(plot_data, 28, n_rows, n_cols, plot_actual_labels, plot_predicted_labels)

def test_model(state_dict_path: str):
    nn.load_model(state_dict_path)
    print(nn.loss_function(nn.model(test_data), test_labels))

    plot_data = []
    plot_actual_labels = []
    plot_predicted_labels = []
    n_rows = 5
    n_cols = 5
    for i in range(0, n_rows * n_cols):
        random_index = np.random.randint(0, n_test)
        plot_data.append(test_data[random_index, :].cpu())
        plot_actual_labels.append(test_labels.cpu().numpy()[random_index].argmax())
        plot_predicted_labels.append(nn.model(test_data[random_index]).detach().cpu().numpy().argmax())

    Plotter.plot(plot_data, 28, n_rows, n_cols, plot_actual_labels, plot_predicted_labels)

if __name__ == "__main__":
    #main()
    test_model("state_dicts/model-50000-copy.torch")