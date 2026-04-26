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

# function to get all necessary kernels from online for quaternion calculations
def get_kernels(start_date, end_date, path):
    kernelList = []

    #####################################
    # Spacecraft Clock-Kernel
    os.makedirs(f"{path}/sclk", exist_ok=True)
    url = "https://naif.jpl.nasa.gov/pub/naif/pds/data/lro-l-spice-6-v1.0/lrosp_1000/data/sclk/lro_clkcor_2025351_v00.tsc"
    filepath = os.path.join(f"{path}/sclk", "lro_clkcor_2025351_v00.tsc")
    urllib.request.urlretrieve(url, filepath)
    kernelList.append(filepath)

    #####################################
    # Frames-Kernel
    os.makedirs(f"{path}/fk", exist_ok=True)
    url = "https://naif.jpl.nasa.gov/pub/naif/pds/data/lro-l-spice-6-v1.0/lrosp_1000/data/fk/lro_frames_2014049_v01.tf"
    filepath = os.path.join(f"{path}/fk", "lro_frames_2014049_v01.tf")
    urllib.request.urlretrieve(url, filepath)
    kernelList.append(filepath)

    #####################################
    # Leapsecond-Kernel
    os.makedirs(f"{path}/lsk", exist_ok=True)
    url = "https://naif.jpl.nasa.gov/pub/naif/pds/data/lro-l-spice-6-v1.0/lrosp_1000/data/lsk/naif0012.tls"
    filepath = os.path.join(f"{path}/lsk", "naif0012.tls")
    urllib.request.urlretrieve(url, filepath)
    kernelList.append(filepath)

    #####################################
    # Planetary Constant-Kernel
    os.makedirs(f"{path}/pck", exist_ok=True)
    url = "https://naif.jpl.nasa.gov/pub/naif/pds/data/lro-l-spice-6-v1.0/lrosp_1000/data/pck/pck00010.tpc"
    filepath = os.path.join(f"{path}/pck", "pck00010.tpc")
    urllib.request.urlretrieve(url, filepath)
    kernelList.append(filepath)

    #####################################
    # Instrument Kernel
    os.makedirs(f"{path}/ik", exist_ok=True)
    url = "https://naif.jpl.nasa.gov/pub/naif/pds/data/lro-l-spice-6-v1.0/lrosp_1000/data/ik/lro_lroc_v20.ti"
    filepath = os.path.join(f"{path}/ik", "lro_lroc_v20.ti")
    urllib.request.urlretrieve(url, filepath)
    kernelList.append(filepath)

    #####################################
    # C-Kernel Search
    os.makedirs(f"{path}/ck", exist_ok=True)
    
    # looks for suitable nearest start date
    start_date_found = False
    current_date = start_date
    next_date = start_date + timedelta(days=10)
    iterator = 0
    while start_date_found == False:
        iterator == iterator + 1
        if iterator == 20:
            print("Unable to find valid start date file")
            return
        # starts from the current (aka start) and sees if there's a file with that name
        current_yyyydoy = current_date.strftime("%Y%j")
        next_yyyydoy = next_date.strftime("%Y%j")
        filename = f"lrosc_{current_yyyydoy}_{next_yyyydoy}_v01.bc"
        url = "https://naif.jpl.nasa.gov/pub/naif/pds/data/lro-l-spice-6-v1.0/lrosp_1000/data/ck/" + filename
        req = urllib.request.Request(url, method='HEAD')
        # if found, print statement runs and the files begin downloading
        try:
            urllib.request.urlopen(req, timeout=5)
            print(f"Found viable starting file found at {filename}. Downloading first file")
            # download first file while you're at it
            current_yyyydoy = current_date.strftime("%Y%j")
            next_yyyydoy = next_date.strftime("%Y%j")
            filename = f"lrosc_{current_yyyydoy}_{next_yyyydoy}_v01.bc"
            url = "https://naif.jpl.nasa.gov/pub/naif/pds/data/lro-l-spice-6-v1.0/lrosp_1000/data/ck/" + filename
            filepath = os.path.join(f"{path}/ck", filename)
            urllib.request.urlretrieve(url, filepath)
            # going by 1 day difference at a time because date spacing is irregular
            current_date = next_date
            next_date = next_date + timedelta(days=1)
            break
        # if unable to be found, the dates go back one day and checks again 
        # (goes back a day to ensure start date is within time period)
        except Exception as e:
            print(f"Failed to find {filename}. Error: {e}. Still looking for viable starting file")
        current_date = current_date - timedelta(days=1)
        next_date = next_date - timedelta(days=1)


    # goes through and downloads all CK files while current_date is before end_date
    while current_date < end_date:
        current_yyyydoy = current_date.strftime("%Y%j")
        next_yyyydoy = next_date.strftime("%Y%j")
        filename = f"lrosc_{current_yyyydoy}_{next_yyyydoy}_v01.bc"
        url = "https://naif.jpl.nasa.gov/pub/naif/pds/data/lro-l-spice-6-v1.0/lrosp_1000/data/ck/" + filename
        filepath = os.path.join(f"{path}/ck", filename)
        
        try:
            urllib.request.urlretrieve(url, filepath)
        except Exception as e:
            print(f"Failed to download {filename}: {e}")
            if (next_date - current_date) > timedelta(days=12):
                print("Gap between dates, updating current_date by one day and searching.")
                current_date = current_date + timedelta(days=1)
                next_date = current_date + timedelta(days=1)
            else:
                next_date = next_date + timedelta(days=1)
        else: # only if try block worked
            print(f"Downloading {filename}")
            current_date = next_date
            next_date = next_date + timedelta(days=1)
            kernelList.append(filepath)
    
    with open('kernelList.txt', 'w') as f:
        for item in kernelList:
            f.write(f"{item}\n")




def dataset_csv_convert(min_lat, max_lat, min_lon, max_lon, src_dir, target_dir, start_date, end_date):

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

                # center latitude and longitude
                dataset_name = "LROC NAC Dataset"
                img_path = "N/A"
                id = row[5]
                # normal lat = latitude of camera itself (this one is good) aka sub-spacecraft lat
                lat = float(row[65])
                lon = float(row[66])
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
    
        

# personal laptop
# dataset_csv_convert(30.0, 35.0, 30.0, 35.0, "C:/Users/16679/lunar-topography",
#                     "C:/Users/16679/lunar-topography/LROC NeRF")


# Curiosity Paths
# datetime = YYYY MM DD
start_date = datetime(2011, 1, 1)
end_date = datetime(2012, 1, 1)
# get_kernels(start_date, end_date, "/home/aajung/lunar-project/kernels")
dataset_csv_convert(30.0, 33.0, 30.0, 33.0, "/home/aajung/lunar-project/kernels/lunar-project/lunar-topography",
                    "/home/aajung/lunar-project/kernels/lunar-project/lunar-topography/LROC NeRF", start_date, end_date)
