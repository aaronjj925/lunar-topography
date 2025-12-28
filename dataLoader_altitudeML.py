import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math
from data_graphing import read_graph
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset 
import torch_directml


class Model(nn.Module):
    def __init__(self, input_features=3, h1=20, h2=20, h3=20, h4=20, h5=20, out_features=1):
        super().__init__() 
        # fcx = fully connected x
        self.fc1 = nn.Linear(input_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, h3)
        self.fc4 = nn.Linear(h3, h4)
        self.fc5 = nn.Linear(h4, h5)
        self.out = nn.Linear(h5, out_features)    
    
    # basically moving forward in nn to go to each layer and reach the output
    def forward(self, x):
        # Combine both inputs along the feature dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.out(x)
        #relu = rectified linear unit = if x < 0, x=0. if x > 0, x=x
        return x

class batchedData(Dataset):
    def __init__(self, df):
        # data loading
        self.lon = (torch.tensor(df['longitude'].values, dtype=torch.float32))
        self.lat = (torch.tensor(df['latitude'].values, dtype=torch.float32))
        self.rad = ((torch.tensor(df['radius'].values, dtype=torch.float32)).unsqueeze(1))
        self.inputs = (degToCoord(self.lon, self.lat))

    def __getitem__(self, i):
        inputs = self.inputs[i]
        rad = self.rad[i]
        return inputs, rad

    def __len__(self):
        # len(dataset)
        return len(self.lon)
    
    def getLonLat(self):
        return (self.lon), (self.lat)

# returns new lon, lat, rad data with outliers of x percent higher/lower removed
def outlier_removal (lon, lat, rad, percent):
    rad = rad.flatten()
    # amount of elements being eliminated
    k = int(percent * rad.numel())
    # gets the indices for the top and bottom values
    bottom_vals, bottom_idx = torch.topk(rad, k, largest=False)
    top_vals, top_idx = torch.topk(rad, k, largest=True)
    outlier_idx = torch.cat([bottom_idx, top_idx])
    # uses a mask to only keep the non outlier values
    mask = torch.ones_like(rad, dtype=torch.bool)
    mask[outlier_idx] = False

    lon = lon.flatten()
    lat = lat.flatten()
    rad = rad[mask]
    lon = lon[mask]
    lat = lat[mask]
    return lon, lat, rad

# switches lon lat degrees to xyz coords
def degToCoord (lon, lat):
    # finding xyz coords after changing lat and lon to 
    lat_rad = torch.deg2rad(lat)
    lon_rad = torch.deg2rad(lon)

    # changing to xyz coordinates to be applicable for larger data regions
    x = torch.cos(lat_rad) * torch.cos(lon_rad)
    y = torch.cos(lat_rad) * torch.sin(lon_rad)
    z = torch.sin(lat_rad)
    # z-score normalization
    x = (x - x.mean()) / x.std()
    y = (y - y.mean()) / y.std()
    z = (z - z.mean()) / z.std()
    inputs = torch.stack([x, y, z], dim=1)
    return inputs


# uses read_graph to get the data frame's values (115435 total data points)
filePath = "C:/Users/16679/Desktop/LOLA_DataFolder/filterMain.csv"
lonMin, lonMax, latMin, latMax = 14, 16, -17, -15
df = read_graph(filePath, lonMin, lonMax, latMin, latMax)
# N is length of full df
N = len(df)
# shuffles dataframe, and then resets row index to be in numerical order afterwards
df = df.sample(frac=1).reset_index(drop=True)
df = df[:100000]
N = 100000
# separates into TVT data sets
# 70/20/10 split
kTrain = int(N * 0.7)
kVal = kTrain + int(N * 0.2)
# Training data
df_train = df[:kTrain]
# Validation data
df_val = df[kTrain:kVal]
# Test data
df_test = df[kVal:]

# set the device
# cpu=cpu, privateuseone:0 = AMD Graphics, 
dml = torch_directml.device(1) 
device = torch.device(dml)
print(device)

# initializing dataset and dataloader for TVT
batch = 10000
train_dataset = batchedData(df_train)
val_dataset = batchedData(df_val)
test_dataset = batchedData(df_test)
train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=batch, shuffle=False)
# just to see how many minibatches per epoch
print(int(N/batch))

# initialize model
model = Model().to(device)


###################################################################################


# initializes Adam optimizer
lr = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4, foreach=False)
criterion = nn.SmoothL1Loss()
# reduces by a factor of 0.1 and patience=# of epochs of no improvement before applying the reduction
'''FIX THE SCHEDULER TO WORK FOR THE MINIBATCHING'''
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=25)
epoch = 250


# epoch for loop
train_losses = []
val_losses = []
previous_lr = lr
start = time.time()

for i in range(epoch):
    batch_losses = []
    # goes through all minibatches per epoch
    for inputs, rad in train_loader:
        # zeros the gradient and then runs the minibatch through the model and criterion
        optimizer.zero_grad()
        inputs = inputs.to(device)
        rad = rad.to(device)
        rad_pred = model(inputs)
        train_loss = criterion(rad_pred, rad)
        # keep track of losses
        batch_losses.append((train_loss.item()))

        train_loss.backward()
        optimizer.step()

    # takes the average loss from the minibatching
    avg_train_loss = sum(batch_losses) / len(batch_losses)
    train_losses.append(avg_train_loss)

    # evaluate on val set
    model.eval()
    val_loss = 0
    batch_losses = []
    with torch.no_grad():
        for inputs, rad in val_loader:
            inputs = inputs.to(device)
            rad = rad.to(device)
            val_pred = model(inputs)
            val_loss = criterion(val_pred, rad)
            batch_losses.append((val_loss.item()))
    # takes the average loss from the minibatching
    avg_val_loss = sum(batch_losses) / len(batch_losses)
    val_losses.append(avg_val_loss)

    if i % 10 == 0:
        print("Epoch #" + str(i))
        print("Training loss: " + str(avg_train_loss))
        print("Validation loss: " + str(avg_val_loss))
    current_lr = optimizer.param_groups[0]['lr']

    scheduler.step(avg_train_loss)
    if current_lr < previous_lr:
        print("Epoch #" + str(i))
        print("New Learning Rate: " + str(current_lr))
        previous_lr = current_lr

    model.train()

end = time.time()
print("Time elapsed: " + str(end-start))

# loss graph
log_train_losses = [math.log(x) for x in train_losses]
log_val_losses = [math.log(x) for x in val_losses]
plt.plot(range(epoch), log_train_losses, label = "training loss")
plt.plot(range(epoch), log_val_losses, label = "validation loss")
plt.xlabel("epoch")
plt.ylabel("log(loss)")
plt.title("Loss vs Epochs")
plt.legend()
plt.show()

# setting model to eval mode and using test data
model.eval()
test_inputs_tot = []
test_rad_act = []
test_rad_pred = []
outputs = []

# goes through all mini batches for one epoch without learning
with torch.no_grad():
    for inputs, rad in test_loader:
        inputs = inputs.to(device)
        rad = rad.to(device)
        outputs = model(inputs)
        test_rad_act.append(rad)
        test_rad_pred.append(outputs.to(device))

# concatenates the list of tensors into one big list
rad_pred = torch.cat(test_rad_pred, dim=0).squeeze()
rad_test = torch.cat(test_rad_act, dim=0).squeeze()

lon_test, lat_test = test_dataset.getLonLat()
# this is the test graph
plt.subplot(1,2,1)
scat = plt.scatter(lon_test, lat_test, c=rad_pred.numpy(), s=5)
plt.title("Predicted Test Radii")
plt.colorbar(scat, label="Radius")
# Actual
plt.subplot(1,2,2)
scat = plt.scatter(lon_test, lat_test, c=rad_test.flatten().numpy(), s=5)
plt.title("Actual Test Radii")
plt.colorbar(scat, label="Radius")

plt.show()