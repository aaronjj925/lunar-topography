# Training a Convolutional Neural Network (CNN) on 

## Purpose
The following code uses the LRO's Lunar Orbiter Laser Altimeter's (LOLA) numerical data to interpolate the lunar terrain of missing data points. Although the LRO has orbiting the Earth for nearly 17 years, there are many spots on the lunar surface without altimetry data. This machine learning model aims to estimate these missing points with minimal error. The training data comprised of small subsections of the lunar surface, where data would be divided into training, validation, and test data

## Script Descriptions
# altitudeML.py
- Primary machine learning code used to tune the model's hyperparameters and test the performance of different architectures, loss functions, and optimizers.

# dataLoader_altitudeML.py
This script kept the same hyperparameters and functions as the previous script, with the difference being in how data is processed. To take in larger quantities of data at a time, Pytorch's DataLoader sub-library was used to send data in batches, to train the model more quickly, while still maintaining progress from past batches. The goal of this was to expand the capabilities of the machine learning model when run on a laptop. However, due to incapabilities with AMD Graphics Cards, this method failed at training on larger datasets.

# customLoss.py
This has a few simple loss functions, primarily used to better understand and test self-made loss functions.

# data_graphing.py
- Code used to graph sections of the lunar surface to see how the altimetry data looks like when all data points of the section are graphed.

# data_parse.py
-Takes in the csv file from the NASA LOLA website, filters out unnecessary data, and saves as a new csv file
