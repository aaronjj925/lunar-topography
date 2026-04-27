import requests
import csv
import os
import shutil
import numpy as np
from datetime import datetime, timedelta
import urllib.request
import urllib.error



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


if __name__ == "__main__":
    start_date = datetime(2011, 1, 1)
    end_date = datetime(2012, 1, 1)
    get_kernels(start_date, end_date, "/home/aajung/lunar-project/kernels")