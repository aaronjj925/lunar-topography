import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

def read_graph(filePath, lonmin, lonmax, latmin, latmax):
    # read csv file
    df = pd.read_csv(filePath)

    # filtering subsections using ranges for lat and lon
    df = df[(df['longitude'] >= lonmin) & (df['longitude'] <= lonmax) &
    (df['latitude'] >= latmin) & (df['latitude'] <= latmax)]

    # set graph variable names
    lon = df['longitude']
    lat = df['latitude']
    rad = df['radius']
    lon = (torch.tensor(lon.values, dtype=torch.float32))
    lat = (torch.tensor(lat.values, dtype=torch.float32))
    rad = (torch.tensor(rad.values, dtype=torch.float32))

    # # scatter plot
    # plt.figure(figsize=(8,6))
    # scat = plt.scatter(lon, lat, c=rad, s=10)
    # plt.colorbar(scat, label='Radius')
    # # scat = plt.scatter(lat, rad, c=lon, s=5)
    # # plt.colorbar(scat, label="longitude")

    # # # label setting
    # plt.xlabel('Longitude')
    # plt.ylabel('Latitude')
    # plt.title('Heatmap of Altitude by Lat/Lon')
    # print(len(lon))
    return df

# read_graph("C:/Users/16679/Desktop/LOLA_DataFolder/filterMain.csv", 14, 16, -17, -15)
# plt.show()
# filterMain goes for around lon=(13.5,19.75), lat=(-14.75,-18.5)