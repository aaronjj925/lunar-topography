import requests
import csv
import os
import shutil

'''
ALL in METERS

target range = target_center_distance (col: 81) (convert to m)
sc altitude = spacecraft_atlitude (col:80) (convert to m)
center lat = center_latitude (col:70) (degrees)
center lon = center_longitude (col:71) (degrees)
horizontal fov = geometry through lat and lon of camera corners (cols:72-79)

horizontal/vertical fov DONE
pos of camera in world frame
angle orientation of camera in world frame
orientation of camera in world frame as quaternion
dataset_name
path to image
cx,cy (principal point in pixels)
w,h (width and height in pixels)


'''
def dataset_csv_convert(min_lat, max_lat, min_lon, max_lon, src_dir, target_dir):
    url = "https://pds.lroc.im-ldi.com/data/LRO-L-LROC-3-CDR-V1.0/LROLRC_1001/INDEX/CUMINDEX.TAB"

    with requests.get(url, stream=True) as r:

        # Decode the raw byte stream into strings on the fly
        lines = (line.decode('utf-8') for line in r.iter_lines() if line)
        # PDS files are standard comma-delimited tables
        reader = csv.reader(lines)
        fileName = 'lroc_data.csv'

        with open(fileName, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Latitude", "Longitude", "Target to Center", "S/C Altitude",
                             "Horizontal FOV", "Vertical FOV", "Sensor Position X J2K", "Sensor Position Y J2K", 
                             "Sensor Position Z J2K", "Camera Right X", "Camera Up Y",  "Camera Look-at Z",
                             "ImageFrameQuaternion Q1", "ImageFrameQuaternion Q2", "ImageFrameQuaternion Q3", "ImageFrameQuaternion Q4",
                             "dataset_name", "image_path", "Cx", "Cy", "Pixel Width", "Pixel Height"])
            match_count = 0
            
            for row in reader:

                # center latitude and longitude
                dataset_name = "LROC Subsection Dataset"
                img_path = "N/A"
                lat = float(row[69])
                lon = float(row[70])
                
                ######### add another conditional for only getting nac imgs bc wac is too wide
                ######### and im making the assumption that the surface is not concave 
                ######### which id assume is more prone to error with this assumption
                if min_lat <= lat <= max_lat and min_lon <= lon <= max_lon:
                    # Extract the exact telemetry for the matching image
                    t_to_c = float(row[80])*1000
                    sc_alt = float(row[79])*1000

                    # horizontal fov = abs(((UR_lon + LR_lon)/2) - ((UL_lon + LL_lon)/2))
                    # vertical fov = abs(((UR_lat + UL_lat)/2) - ((LR_lat + LL_lat)/2))
                    UR_lat = float(row[71])
                    UR_lon = float(row[72])
                    LR_lat = float(row[73])
                    LR_lon = float(row[74])
                    LL_lat = float(row[75])
                    LL_lon = float(row[76])
                    UL_lat = float(row[77])
                    UL_lon = float(row[78])

                    h_fov = abs(((UR_lon + LR_lon)/2) - ((UL_lon + LL_lon)/2))
                    v_fov = abs(((UR_lat + UL_lat)/2) - ((LR_lat + LL_lat)/2))

                    pos_x = 0
                    pos_y = 0
                    pos_z = 0
                    right_x = 0
                    up_y = 0
                    look_z = 0
                    q1 = 0
                    q2 = 0
                    q3 = 0
                    q4 = 0
                    w = 0
                    h = 0
                    cx = w/2
                    cy = h/2
                        
                    writer.writerow([lat, lon, t_to_c, sc_alt, h_fov, v_fov, pos_x, pos_y, pos_z, 
                                     right_x, up_y, look_z, q1, q2, q3, q4, dataset_name, img_path, cx, cy, w, h])
                    match_count += 1

    source_file = os.path.join(src_dir, fileName)
    target_file = os.path.join(target_dir, fileName)
    if os.path.isfile(source_file):
        shutil.move(source_file, target_file)
    
        

# Execute the stream
dataset_csv_convert(30.0, 45.0, 30.0, 45.0, "C:/Users/16679/lunar-topography",
                    "C:/Users/16679/lunar-topography/LROC NeRF")