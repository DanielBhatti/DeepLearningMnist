import torch
import torch.optim as optim
import torch.nn as nn

from mnist_data import Reader, Plotter
from typing import List
import time
import numpy as np
from torchsummary import summary

class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)

class NeuralNetwork:
    def __init__(self, model_save_path: str):
        self.model = self._create_model()
        self.loss_function = nn.MSELoss()
        self.model_save_path = model_save_path

    def _create_model(self):
        m = nn.Sequential(
            Lambda(self._preprocess),
            nn.Conv2d(1, 32, kernel_size=5, stride=1),
            nn.Sigmoid(),
            nn.Conv2d(32, 32, kernel_size=5, stride=1, bias=False),
            nn.BatchNorm2d(32),
            nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.Sigmoid(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm2d(64),
            nn.Sigmoid(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25),
            nn.Flatten(1, 3),
            nn.Linear(576, 64, bias=False),
            nn.BatchNorm1d(64),
            nn.Linear(64, 84),
            nn.BatchNorm1d(84),
            nn.Sigmoid(),
            nn.Dropout(0.25),
            nn.Linear(84, 10),
            nn.Softmax()
        ).cuda()
        return m

    def _create_old_model(self):
        m = nn.Sequential(
            nn.Linear(784, 392),
            nn.Sigmoid(),
            nn.Linear(392, 200),
            nn.Sigmoid(),
            nn.Linear(200, 10),
            nn.Sigmoid()
        ).cuda()
        return m

    def _preprocess(self, x):
        return x.view(-1, 1, 28, 28)

    def _print_layer(self, x):
        print(x.shape)
        return x

    def train(self, n_epochs: int, learning_rate: float, t_input: torch.tensor, t_output: torch.tensor):
        optimizer = optim.Adam(self.model.parameters(), lr = learning_rate)
        start_time = time.time()
        for epoch in range(1, n_epochs + 1):
            #for index in np.random.randint(0, t_input.shape[0], int(0.1 * t_input.shape[0]))
            t_predicted = self.model(t_input)
            l = self.loss_function(t_output, t_predicted)

            optimizer.zero_grad()
            l.backward()
            optimizer.step()

            if n_epochs < 100 or epoch % int(n_epochs / 100) == 0:
                print(f"Epoch {epoch}, Loss {l}, Elapsed Time: {time.time() - start_time}")

        torch.save(self.model.state_dict(), self.model_save_path)

    def load_model(self, state_dict_path: str):
        state_dict = torch.load(state_dict_path)
        self.model.load_state_dict(state_dict)

if __name__ == "__main__":
    nn = NeuralNetwork("state_dicts/testmodel.torch")
    summary(nn.model, (1, 784))