import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import torch.nn.init as init
import numpy as np
import os
import random
class TEncoder(nn.Module):
    def __init__(self, channel_in=3, ch=16, z=64, h_dim=512):
        super(TEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.ZeroPad2d((2, 2, 2, 2)),
            nn.Conv2d(channel_in, ch, kernel_size=(8, 8), stride=(4, 4)),
            nn.ReLU(),
            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(ch, ch*2, kernel_size=(4, 4), stride=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(ch*2, ch*32, kernel_size=(11, 11), stride=(1, 1)),
            nn.ReLU(),
        )

        self.conv_mu = nn.Conv2d(ch*32, z, 1, 1)
        
    def forward(self, x):
        #print(torch.max(x))
        x = self.encoder(x)
        x = self.conv_mu(x)
        x = torch.flatten(x, start_dim=1)
        return x
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                # Initialize weights using a random initialization method
                # For example, you can use the Xavier/Glorot initialization
                init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    # Initialize biases to zero
                    init.constant_(m.bias.data, 0.0)

# Assuming you already have the data as tensors: inputs and targets
encodernet_VEP = TEncoder(channel_in=1, ch=32, z=512)
checkpoint = torch.load('/lab/kiran/ckpts/pretrained/atari/1CHAN_NVEP_ATARI_EXPERT_1CHAN_SPACEDEMO_STANDARD_2.0_-1.0_2_nsame_triplet_32_2_0.0001_1.0.pt',map_location='cuda')
encodernet_VEP.load_state_dict(checkpoint['model_state_dict'])
encodernet_VEP.to('cuda')
encodernet_VEP.eval()

encodernet_VIP = TEncoder(channel_in=1, ch=32, z=512)
checkpoint = torch.load('/lab/kiran/ckpts/pretrained/atari/1CHAN_VIP_ATARI_EXPERT_1CHAN_SPACEDEMO_STANDARD_50.0_15.0_0.01_32_0_0.0001_0.pt',map_location='cuda')
encodernet_VIP.load_state_dict(checkpoint['model_state_dict'])
encodernet_VIP.to('cuda')
encodernet_VIP.eval()

encodernet_SOM = TEncoder(channel_in=1, ch=32, z=512)
checkpoint = torch.load('/lab/kiran/ckpts/pretrained/atari/1CHAN_SOM_ATARI_EXPERT_1CHAN_SPACEDEMO_STANDARD_0.1_32_32_0.0001_0.pt',map_location='cuda')
encodernet_SOM.load_state_dict(checkpoint['model_state_dict'])
encodernet_SOM.to('cuda')
encodernet_SOM.eval()

encodernet_TCN = TEncoder(channel_in=1, ch=32, z=512)
checkpoint = torch.load('/lab/kiran/ckpts/pretrained/atari/1CHAN_TCN_ATARI_EXPERT_1CHAN_SPACEDEMO_STANDARD_-1.0_32_0_0.0001_0.pt',map_location='cuda')
encodernet_TCN.load_state_dict(checkpoint['model_state_dict'])
encodernet_TCN.to('cuda')
encodernet_TCN.eval()

encodernet_RANDOM = TEncoder(channel_in=1, ch=32, z=512)
encodernet_RANDOM.initialize_weights()
encodernet_RANDOM.to('cuda')
encodernet_RANDOM.eval()













# Hyperparameters
batch_size = 64
learning_rate = 0.001
num_epochs = 5

# Pretrained Encoder (assuming you have a pretrained model)
pretrained_encoder = encodernet_VIP
pretrained_encoder.eval()

# Create your own neural network with a frozen encoder
class MyNetwork(nn.Module):
    def __init__(self, pretrained_encoder):
        super(MyNetwork, self).__init__()
        self.encoder = pretrained_encoder
        # Freeze the encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Add a linear layer for regression
        self.fc = nn.Linear(in_features=2048, out_features=1).to('cuda')

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.encoder(x.view(-1, 1, 84, 84).to('cuda'))
        x = x.view(batch_size, 2048)
        x = self.fc(x)
        return x

# Instantiate the model
model = MyNetwork(pretrained_encoder)

# Loss function
criterion = nn.MSELoss()

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

data_path = '/lab/tmpig14c/kiran/expert_4stack_atari/expert_4stack_demonattack/5/50/'
obs_path = os.path.join(data_path,'observation')
value_path = os.path.join(data_path,'value_truncated.npy')

obs = np.load(obs_path,allow_pickle=True)
val = np.load(value_path,allow_pickle=True)

obs = torch.from_numpy(obs).to(dtype=torch.float32)
val = torch.from_numpy(val).to(dtype=torch.float32).to('cuda')
# Data Loader
train_dataset = TensorDataset(obs[:800000], val[:800000].view(-1, 1))
test_dataset = TensorDataset(obs[800000:], val[800000:].view(-1, 1))

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

train_losses = []
test_losses = []

for epoch in range(num_epochs):
    model.train()
    for param in model.encoder.parameters():
        param.requires_grad = False
    epoch_train_loss = 0.0

    for batch_inputs, batch_targets in train_dataloader:
        optimizer.zero_grad()
        outputs = model(batch_inputs)
        loss = criterion(outputs, batch_targets)  # Assuming targets are 1-dimensional
        loss.backward()
        optimizer.step()

        epoch_train_loss += loss.item()

    average_train_loss = epoch_train_loss / len(train_dataloader)
    train_losses.append(average_train_loss)
    print(f'Training - Epoch [{epoch + 1}/{num_epochs}], Loss: {average_train_loss:.4f}')

    # Evaluate on the test set
    model.eval()
    epoch_test_loss = 0.0

    with torch.no_grad():
        for batch_inputs, batch_targets in test_dataloader:
            outputs = model(batch_inputs)
            loss = criterion(outputs, batch_targets)
            epoch_test_loss += loss.item()

    average_test_loss = epoch_test_loss / len(test_dataloader)
    test_losses.append(average_test_loss)
    print(f'Testing - Epoch [{epoch + 1}/{num_epochs}], Loss: {average_test_loss:.4f}')



# Plot the regression loss
plt.plot(train_losses, label='Training Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Regression Loss Over Epochs')

plt.savefig(f"./regression/VIP_nn{random.randint(1,1000)}.png")


