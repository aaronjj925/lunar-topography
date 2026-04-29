# Required library imports
import requests
import csv
import os
import shutil
import numpy as np
import spiceypy as spice
from datetime import datetime, timedelta
import urllib.request
import urllib.error
import math
import lroc_img_download
import pandas as pd


'''
ALL in METERS

target range = target_center_distance (col: 81) (convert to m)
sc altitude = spacecraft_atlitude (col:80) (convert to m)
center lat = center_latitude (col:70) (degrees)
center lon = center_longitude (col:71) (degrees)
horizontal fov = geometry through lat and lon of camera corners (cols:72-79)

According to Section 4.1.3 of Calibration Reports:
    The NAC focal length was measured to be 2000 ± 4 mm (3σ uncertainty) with no detectable field distortion

    
'''


def dataset_csv_convert(min_lat, max_lat, min_lon, max_lon, 
src_dir, target_dir, start_date, end_date):

    product_ids = []

    # LROC1001 spams from 2009 6/30 to 12/31 (in YYYYDOY, it's 2009 181 to 365)
    url = "https://pds.lroc.im-ldi.com/data/LRO-L-LROC-3-CDR-V1.0/LROLRC_1030/INDEX/CUMINDEX.TAB"
    with requests.get(url, stream=True) as r:

        # decode the streaming data from requests as str
        lines = (line.decode('utf-8') for line in r.iter_lines() if line)
        reader = csv.reader(lines)
        fileName = 'lroc_data.csv'

        with open(fileName, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # all column names
            writer.writerow(["Product ID","Latitude", "Longitude", "Target to Center", "S/C Altitude",
                             "Horizontal FOV", "Vertical FOV", "Focal Length", "Sensor Position X J2K", 
                             "Sensor Position Y J2K", 
                             "Sensor Position Z J2K", "Camera XYZ", "ImageFrameQuaternion Q1", 
                             "ImageFrameQuaternion Q2", "ImageFrameQuaternion Q3", "ImageFrameQuaternion Q4",
                             "dataset_name", "image_path", "Cx", "Cy", "Pixel Width", "Pixel Height", "Time UTC"])
            img_count = 0

            for row in reader:

                # center_lat = latitude of center of image
                center_lat = float(row[69])
                center_lon = float(row[70])
                nac_or_wac = (str(row[4]).strip())[0:3]
                camera_type = (str(row[4]).strip())[0:4]

                time_utc = row[14]
                # Strip leading/trailing whitespace
                time_utc = time_utc.strip()

                # Parse into datetime object
                time_dt = datetime.strptime(time_utc, "%Y-%m-%d %H:%M:%S.%f")

                # stops the script from running any further once it goes past the max cutoff date to save time
                if (time_dt > end_date):
                    break

                
                if (min_lat <= center_lat <= max_lat and min_lon <= center_lon <= max_lon and nac_or_wac == "nac" 
                and time_dt > start_date and time_dt < end_date):

                    # focal lengths of NAC right and left are slightly different
                    if camera_type == "nacr":
                        focal_meters = 701.57e-3 # in meters
                    else: 
                        focal_meters = 699.62e-3 # in meters
                    pixel_pitch = 7e-6 # in meters
                    focal_length = focal_meters/pixel_pitch

                    t_to_c = float(row[80])*1000
                    sc_alt = float(row[79])*1000
                    
                    # normal lat = latitude of camera itself (this one is good) aka sub-spacecraft lat
                    lat = float(row[65])
                    lon = float(row[66])

                    dataset_name = "LROC NAC Dataset"
                    img_path = "path" # placeholder
                    id = (str(row[5])).strip()

                    # approximating fov by assuming area is small to be concave WITH the camera, not the lunar surface
                    UR_lat = float(row[71])
                    UR_lon = float(row[72])
                    LR_lat = float(row[73])
                    LR_lon = float(row[74])
                    LL_lat = float(row[75])
                    LL_lon = float(row[76])
                    UL_lat = float(row[77])
                    UL_lon = float(row[78])
                    # calculating the range of degrees the camera covers
                    h_fov = abs(((UR_lon + LR_lon)/2) - ((UL_lon + LL_lon)/2))
                    v_fov = abs(((UR_lat + UL_lat)/2) - ((LR_lat + LL_lat)/2))
                    
                    # converting sc angle values to radians, and then converting to spherical coordinates
                    lat_rad = np.deg2rad(lat)
                    lon_rad = np.deg2rad(lon)
                    # multiplying by distance from target to center to scale distance past unit circle
                    pos_x = t_to_c * np.cos(lat_rad) * np.cos(lon_rad)
                    pos_y = t_to_c * np.cos(lat_rad) * np.sin(lon_rad)
                    pos_z = t_to_c * np.sin(lat_rad)

                    # calculating quaternion by using SPICE kernels
                    with open("kernelList.txt", "r") as f:
                        kernelList = [line.strip() for line in f.readlines()]
                    spice.furnsh(kernelList)
                    et = spice.str2et(time_utc)
                    # getting the rotation matrix from moon frame to LRO frame
                    rotation_matrix = spice.pxform("IAU_MOON", "LRO_SC_BUS", et)
                    # getting quaternion values from rotation matrix and storing them for the csv file
                    quaternion = spice.m2q(rotation_matrix)
                    q1 = quaternion[1]
                    q2 = quaternion[2]
                    q3 = quaternion[3]
                    q4 = quaternion[0]
                    spice.kclear()

                    camera_xyz = "[x,y,z]"
                    w = int(row[53]) # line samples (amount of samples per line)
                    h = int(row[52]) # image lines (total amount of horizontal lines)
                    cx = w/2
                    cy = h/2
                        
                    product_ids.append(id)
                    writer.writerow([id, lat, lon, t_to_c, sc_alt, h_fov, v_fov, focal_length,
                                      pos_x, pos_y, pos_z, 
                                     camera_xyz, q1, q2, q3, q4, dataset_name, img_path, cx, cy, w, h, time_utc])

                    img_count += 1
                    print(f"Image No. {img_count} Downloaded")
    print(f"Total Images Found and Stored: {img_count}")

    source_file = os.path.join(src_dir, fileName)
    target_file = os.path.join(target_dir, fileName)
    if os.path.isfile(source_file):
        shutil.move(source_file, target_file)
    return product_ids
    


if __name__ == "__main__":
    # Curiosity Paths
    # datetime = YYYY MM DD
    start_date = datetime(2011, 1, 1)
    end_date = datetime(2012, 1, 1)
    # product_ids = dataset_csv_convert(30.0, 33.0, 30.0, 33.0, 
    #                     "/home/aajung/lunar-project/lunar-project/lunar-topography",
    #                     "/home/aajung/lunar-project/lunar-project/lunar-topography/LROC NeRF", 
    #                     start_date, end_date)
    # lroc_img_download.download_and_convert_lroc(product_ids)

    # storing product ids to set the correct path to put the path into the folder
    df = pd.read_csv("/home/aajung/lunar-project/lunar-topography/LROC NeRF/lroc_data.csv")
    print(len(df))
    for i in range(len(df)):
        id = df.iloc[i,0]
        png_filepath = f"~/lunar-project/lroc_png_files/{id}.png"
        df.iloc[i,17] = png_filepath
        print(f"Completed: {i+1}")

    # saving result to new csv file
    df.to_csv('lroc_data(with png_path).csv', index=False)