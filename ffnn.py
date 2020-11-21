#!/usr/bin/env python

import torch
import torch.nn as nn
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset


# This is just a minor refactoring of code from
# https://www.cheminformania.com/building-a-simple-qsar-model-using-a-feed-forward-neural-network-in-pytorch/

class Net(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_rate, out_size):
        super(Net, self).__init__()
        # Three layers and a output layer
        self.fc1 = nn.Linear(input_size, hidden_size)  # 1st Full-Connected Layer
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, out_size)  # Output layer
        # Layer normalization for faster training
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.ln3 = nn.LayerNorm(hidden_size)
        # LeakyReLU will be used as the activation function
        self.activation = nn.LeakyReLU()
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):  # Forward pass: stacking each layer together
        # Fully connected =&amp;gt; Layer Norm =&amp;gt; LeakyReLU =&amp;gt; Dropout times 3
        out = self.fc1(x)
        out = self.ln1(out)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.ln2(out)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.fc3(out)
        out = self.ln3(out)
        out = self.activation(out)
        out = self.dropout(out)
        # Final output layer
        out = self.fc_out(out)
        return out


class FFNN:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_select = None
        self.device = None

    def fit(self, X_train, y_train):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.scaler = StandardScaler()
        y_train_scaled = self.scaler.fit_transform(y_train.reshape([-1, 1]))

        self.feature_select = VarianceThreshold(threshold=0.05)
        X_train_select = self.feature_select.fit_transform(list(X_train))

        X_train_select = torch.tensor(X_train_select, device=self.device).float()

        y_train_scaled = torch.tensor(y_train_scaled, device=self.device).float()

        train_dataset = TensorDataset(X_train_select, y_train_scaled)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=256,
                                                   shuffle=True)

        input_size = X_train_select.size()[-1]  # The input size should fit our fingerprint size
        hidden_size = 1024  # The size of the hidden layer
        dropout_rate = 0.80  # The dropout rate
        output_size = 1  # This is just a single task, so this will be one
        learning_rate = 0.001  # The learning rate for the optimizer
        self.model = Net(input_size, hidden_size, dropout_rate, output_size)

        if torch.cuda.is_available():
            self.model.cuda()
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        self.model.train()  # Ensure the network is in "train" mode with dropouts active
        epochs = 200
        for e in range(epochs):
            running_loss = 0
            for fps, labels in train_loader:
                # Training pass
                optimizer.zero_grad()  # Initialize the gradients, which will be recorded during the forward pa

                output = self.model(fps)  # Forward pass of the mini-batch
                loss = criterion(output, labels)  # Computing the loss
                loss.backward()  # calculate the backward pass
                optimizer.step()  # Optimize the weights

                running_loss += loss.item()

    def predict(self, X_test):
        X_test_select =  self.feature_select.transform(list(X_test))
        X_test_select = torch.tensor(X_test_select, device=self.device).float()
        self.model.eval()  # Switch to evaluation mode, where dropout is switched off
        y_pred_test = self.model(X_test_select)
        return [x[0] for x in self.scaler.inverse_transform(y_pred_test.detach().cpu())]
