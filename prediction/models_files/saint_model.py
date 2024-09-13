import torch.nn as nn

class SAINT(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SAINT, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)  # Batch Norm 추가
        self.gelu = nn.GELU()  # GELU activation function
        self.dropout = nn.Dropout(0.3)  # Dropout 추가
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)  # Batch Norm 추가
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 첫 번째 은닉층
        out = self.fc1(x)
        out = self.bn1(out)  # Batch Normalization 적용
        out = self.gelu(out)
        out = self.dropout(out)  # Dropout 적용

        # 두 번째 은닉층
        out = self.fc2(out)
        out = self.bn2(out)  # Batch Normalization 적용
        out = self.gelu(out)
        out = self.dropout(out)  # Dropout 적용

        # 출력층
        out = self.fc3(out)
        return out