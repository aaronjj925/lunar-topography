# Machine Learning for Lunar Topography Applications

## Purpose
Primary goal is to develop a machine learning model capable of producing a lunar gravity model using data from NASA's Lunar Reconnaissance Orbiter (LRO). This repository contains two methods: altimetry data from the Lunar Orbiter Laser Altimeter (LOLA) and the LRO's Camera (LROC). These methods use Convolutional Neural Networks and Neural Radiance Fields, respectively.

## Requirements
The requirements.txt file is a list of all Python libraries used in both working with the altimetry and image data of the LRO.
First, by using pip freeze > requirements.txt within an environment, all current libraries downloaded in the virtual environment (venv) are saved in the requirements.txt file.
```
pip freeze > requirements.txt
```
This text file of required libraries are already saved within the repository, and can be downloaded within a venv with pip install -r requirements.txt.
```
pip install -r requirements.txt
```
