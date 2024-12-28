import torch
from torch import nn, Tensor
import torch.nn.functional as F


def initialize_weights(model):
    if type(model) in [nn.Linear]:
        nn.init.xavier_uniform_(model.weight)
        nn.init.zeros_(model.bias)
    elif type(model) in [nn.LSTM, nn.RNN, nn.GRU]:
        nn.init.orthogonal_(model.weight_hh_l0)
        nn.init.xavier_uniform_(model.weight_ih_l0)
        nn.init.zeros_(model.bias_hh_l0)
        nn.init.zeros_(model.bias_ih_l0)


class MANN(nn.Module):
    def __init__(self, num_classes, samples_per_class, hidden_dim):
        super(MANN, self).__init__()
        self.num_classes = num_classes
        self.samples_per_class = samples_per_class

        self.layer1 = torch.nn.LSTM(num_classes + 784, hidden_dim, batch_first=True)
        self.layer2 = torch.nn.LSTM(hidden_dim, num_classes, batch_first=True)

    def forward(self, input_images, input_labels):
        B = input_images.shape[0]
        K = self.samples_per_class - 1
        N = self.num_classes

        concatenated = torch.cat((input_images, input_labels), dim=-1)
        concatenated = concatenated.view(B, (K + 1) * N, 784 + N)

        mask = torch.ones_like(concatenated)
        mask[:, -N:, 784:] = 0
        concatenated = concatenated * mask

        output, _ = self.layer1(concatenated)
        output, _ = self.layer2(output)

        return output.view(B, K + 1, N, N)

    def loss_function(self, preds, labels):

        query_preds = preds[:, -1, :, :]  # Shape: [B, N, N]

        # Extract the true labels for the query set
        query_labels = labels[:, -1, :, :]  # Shape: [B, N, N]

        # Reshape predictions and labels for cross entropy loss
        B, N, _ = query_preds.shape
        query_preds = query_preds.reshape(B * N, N)  # Shape: [B*N, N]
        query_labels = query_labels.argmax(dim=-1).view(B * N)  # Shape: [B*N]

        # Compute the Cross Entropy Loss for the query set
        loss = F.cross_entropy(query_preds, query_labels)

        return loss
