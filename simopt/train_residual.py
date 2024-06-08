import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# define residualnet
class ResidualNetwork(nn.Module):
    def __init__(self, input_size):
        super(ResidualNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, input_size)
        
    def forward(self, x):
        residual = x
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x + residual

input_size = 10  # s_t + a_t

model = ResidualNetwork(input_size)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

s_t = torch.randn(100, input_size)
a_t = torch.randn(100, input_size)
s_next_prime = torch.randn(100, input_size)

dataset = TensorDataset(torch.cat((s_t, a_t), dim=1), s_next_prime - s_t)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# loop
num_epochs = 100
for epoch in range(num_epochs):
    for inputs, targets in dataloader:
        predictions = model(inputs)
        
        loss = criterion(predictions, targets)
        
        # opt
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')