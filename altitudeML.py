import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math
from data_graphing import read_graph
from customLoss import MSE, MAE, MAPE, APE
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau

class Model(nn.Module):
    def __init__(self, input_features=3, h1=32, h2=32, h3=32, h4=32, h5=32, out_features=1):
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

# initialize model
model = Model()
# uses read_graph to get the data frame's values (115435 total data points)
filePath = "C:/Users/16679/Desktop/LOLA_DataFolder/filterMain.csv"
lonMin, lonMax, latMin, latMax = 14, 16, -17, -15
df = read_graph(filePath, lonMin, lonMax, latMin, latMax)
# changing type to torch tensor to perform cat or stack on it
lon = (torch.tensor(df['longitude'].values, dtype=torch.float32))
lat = (torch.tensor(df['latitude'].values, dtype=torch.float32))
rad = (torch.tensor(df['radius'].values, dtype=torch.float32))

# randomly shuffles lon,lat,rad the same exact permutation
N = len(df)
print(N)
split = 5000
perm = torch.randperm(N)
lon_og = lon[perm]
lat_og = lat[perm]
rad_og = (rad[perm]).unsqueeze(1)



# selecting a fraction of the randomized data
lon = lon_og[0:split]
lat = lat_og[0:split]
rad = rad_og[0:split]

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
print(inputs.size())
N = len(lon)

###################################################################################

# 70/20/10 split
kTrain = int(N * 0.7)
kVal = kTrain + int(N * 0.2)


# Training data
lon_train = lon[0:kTrain]
lat_train = lat[0:kTrain]
rad_train = rad[0:kTrain]
train_inputs = inputs[0:kTrain]

# Validation data
lon_val = lon[kTrain:kVal]
lat_val = lat[kTrain:kVal]
rad_val = rad[kTrain:kVal]
val_inputs = inputs[kTrain:kVal]

# Test data
lon_test = lon[kVal:N]
lat_test = lat[kVal:N]
rad_test = rad[kVal:N]
test_inputs = inputs[kVal:N]

# graph of data
plt.hexbin(lon_train, lat_train, C=rad_train, gridsize=100)
plt.title("All Data Points")
plt.colorbar(label='Radius')
plt.show()

# initializes Adam optimizer
lr = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay=1e-4)
criterion = nn.SmoothL1Loss()
# reduces by a factor of 0.1 and patience=# of epochs of no improvement before applying the reduction
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=300)
epoch = 40000


# epoch for loop
train_losses = []
val_losses = []
previous_lr = lr
start = time.time()
for i in range(epoch):
    # zeros the gradient
    optimizer.zero_grad()
    # go forward and get a prediction
    rad_pred = model(train_inputs)
    # measure the loss/error
    train_loss = criterion(rad_pred, rad_train)
    # keep track of losses
    train_losses.append(math.log(train_loss.detach().numpy())) # changes loss to a numpy number
    # back propagation: take the error rate of forward propagation
    # and feed it back through the nn to fine tune the weights
    train_loss.backward()
    optimizer.step()
    scheduler.step(train_loss)
    model.eval()  # turn into evaluation mode
    
    # turning off grad for validating model
    with torch.no_grad():
        val_pred = model(val_inputs)
        val_loss = criterion(val_pred, rad_val)
        val_losses.append(math.log(val_loss.detach().numpy()))
    if i % 500 == 0:
        print("Epoch #" + str(i))
        print("Training loss: " + str(train_loss))
        print("Validation loss: " + str(val_loss))
    current_lr = optimizer.param_groups[0]['lr']

    if current_lr < previous_lr:
        print("Epoch #" + str(i))
        print("New Learning Rate: " + str(current_lr))
        previous_lr = current_lr

    model.train() # turn back into training mode

end = time.time()
print("Time elapsed: " + str(end-start))

# loss graph
plt.plot(range(epoch), train_losses, label = "training loss")
plt.plot(range(epoch), val_losses, label = "validation loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.title("Loss vs Epochs")
plt.legend()
plt.show()


# prediction vs actual of training data on final epoch
plt.subplot(1,2,1)
rad_pred = model.forward(train_inputs)
scat = plt.scatter(lon_train, lat_train, c=rad_pred.detach().numpy(), s=10)
plt.title("Predicted Values of Training Data")
plt.colorbar(scat, label='Radius')
plt.subplot(1,2,2)
scat = plt.scatter(lon_train, lat_train, c=rad_train, s=10)
plt.title("Actual Values of Training Data")
plt.colorbar(scat, label='Radius')
plt.show()

# loss heat map
rad_APE = APE(rad_pred, rad_train)
lon_train, lat_train, rad_APE = outlier_removal(lon_train, lat_train, rad_APE, 0.025)

scat = plt.scatter(lon_train, lat_train, c=rad_APE.detach().numpy(), s=10)
plt.title("Heat map of Absolute Percent Error of Training Data")
plt.colorbar(scat, label='Percentage Error')
plt.show()


# testing the model
rad_pred = model.forward(test_inputs)
plt.subplot(1,2,1)
scat = plt.scatter(lon_test, lat_test, c=rad_pred.detach().numpy(), s=10)
plt.title("Predicted Values of Test Data")
plt.colorbar(scat, label='Radius')
plt.subplot(1,2,2)
scat = plt.scatter(lon_test, lat_test, c=rad_test.detach().numpy(), s=10)
plt.title("Actual Values of Test Data")
plt.colorbar(scat, label='Radius')
plt.show()






# graphing the remaining the data set that wasn't used for train, val, test
# lon_rem = lon_og[split:]
# lat_rem = lat_og[split:]
# rad_rem = rad_og[split:]
# lon_rem_rad = torch.deg2rad(lon_rem)
# lat_rem_rad = torch.deg2rad(lat_rem)
# x = torch.cos(lat_rem_rad) * torch.cos(lon_rem_rad)
# y = torch.cos(lat_rem_rad) * torch.sin(lon_rem_rad)
# z = torch.sin(lat_rem_rad)
# # z-score normalization
# x = (x - x.mean()) / x.std()
# y = (y - y.mean()) / y.std()
# z = (z - z.mean()) / z.std()
# rem_inputs = torch.stack([x, y, z], dim=1)

# # graphing the remaining data's model's predictions
# plt.subplot(1,2,1)
# scat = plt.scatter(lon_rem, lat_rem, c=rad_rem.detach().numpy(), s=10)
# plt.colorbar(scat, label='Radius')
# plt.title("Actual values of Remaining Unused Data")

# rad_rem_pred = model.forward(rem_inputs)
# # getting rid of random outliers that the model predicted way off
# lon_rem, lat_rem, rad_rem_pred = outlier_removal(lon_rem, lat_rem, rad_rem_pred, 0.005)
# plt.subplot(1,2,2)
# scat = plt.scatter(lon_rem, lat_rem, c=rad_rem_pred.detach().numpy(), s=10)
# plt.colorbar(scat, label='Radius')
# plt.title("Prediction values of Remaining Unused Data")
# plt.show()


# 15.52, 15.54, -18, -16
# # interpolation of missing data
# shape = (5000, )
# lon_miss = torch.deg2rad(torch.FloatTensor(*shape).uniform_(lonMin, lonMax))
# lat_miss = torch.deg2rad(torch.FloatTensor(*shape).uniform_(latMin, latMax))
# # Spherical coordinate embedding
# x = torch.cos(lat_miss) * torch.cos(lon_miss)
# y = torch.cos(lat_miss) * torch.sin(lon_miss)
# z = torch.sin(lat_miss)
# # z-score normalization
# x = (x - x.mean()) / x.std()
# y = (y - y.mean()) / y.std()
# z = (z - z.mean()) / z.std()
# miss_inputs = torch.stack([x, y, z], dim=1)
# rad_miss = model.forward(miss_inputs)
# # plotting model's attempt at interpolation
# scat = plt.scatter(torch.rad2deg(lon_miss), torch.rad2deg(lat_miss), c=rad_miss.detach().numpy(), s=10)
# plt.colorbar(scat, label='Radius')
# plt.title("Estimate of Missing Data")
# plt.show()