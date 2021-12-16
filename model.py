import torch
import torch.nn as nn

class LinearModel(nn.Module):
    def __init__(self, in_dim=16*8, out_dim=2):
        super(LinearModel, self).__init__()
        # self.relu = nn.ReLU(inplace=True)
        # self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.fc1 = nn.Linear(in_features=in_dim, out_features=2*in_dim)
        self.fc2 = nn.Linear(in_features=2*in_dim, out_features=in_dim)
        self.fc3 = nn.Linear(in_features=in_dim, out_features=out_dim)
        initialize_weights(self.fc1, self.fc2)

    def forward(self, x):
        out = self.fc1(x)
        out = self.sigmoid(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        out = self.fc3(out)
        return out

class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
                            torch.zeros(1,1,self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]

# kaiming_uniform initialization
def initialize_weights(*models):
    for model in models:
        if isinstance(model, nn.Linear):
            nn.init.kaiming_uniform_(model.weight.data)
            nn.init.constant_(model.bias.data, 0)