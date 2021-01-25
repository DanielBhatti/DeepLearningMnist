import torch
import torch.optim as optim
import torch.nn as nn
from mnist_data import Reader, Plotter
from typing import List
import time
import numpy as np

class NeuralNetwork:
    def __init__(self, model_save_path: str):
        self.model = self._create_model()
        self.loss_function = nn.MSELoss()
        self.model_save_path = model_save_path

    def _create_model(self):
        m = nn.Sequential(
            nn.Linear(784, 392),
            nn.Sigmoid(),
            nn.Linear(392, 200),
            nn.Sigmoid(),
            nn.Linear(200, 10),
            nn.Sigmoid()
        ).cuda()
        return m

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

            if epoch % 1000 == 0:
                print(f"Epoch {epoch}, Loss {l}, Elapsed Time: {time.time() - start_time}")

        torch.save(self.model.state_dict(), self.model_save_path)

if __name__ == "__main__":
    print('a')