# Machine Learning for Lunar Topography Applications

## Purpose
Primary goal is to develop a neural network model capable of producing a lunar gravity model using the NASA LRO's LOLA altimetry data 

## Script/File Descriptions
### kernel_retrieval.py
- Retrieves and downloads all necessary kernel for calculating the quaternion
- Requires a start and end date of mission phase to get the correct C-kernels
- Kernel info is linked below
  - [Spacecraft-Clock Kernel](https://naif.jpl.nasa.gov/pub/naif/pds/data/lro-l-spice-6-v1.0/lrosp_1000/data/sclk/sclkinfo.txt)
  - [Frames Kernel](https://naif.jpl.nasa.gov/pub/naif/pds/data/lro-l-spice-6-v1.0/lrosp_1000/data/fk/fkinfo.txt)
  - [Leapsecond Kernel](https://naif.jpl.nasa.gov/pub/naif/pds/data/lro-l-spice-6-v1.0/lrosp_1000/data/lsk/lskinfo.txt)
  - [Planetary-Constant Kernel](https://naif.jpl.nasa.gov/pub/naif/pds/data/lro-l-spice-6-v1.0/lrosp_1000/data/pck/pckinfo.txt)
  - [Instrument Kernel](https://naif.jpl.nasa.gov/pub/naif/pds/data/lro-l-spice-6-v1.0/lrosp_1000/data/ik/ikinfo.txt)
  - [C-Kernel](https://naif.jpl.nasa.gov/pub/naif/pds/data/lro-l-spice-6-v1.0/lrosp_1000/data/ck/ckinfo.txt)

### kernelList.txt
- List of all paths to kernel downloads from kernel_retrieval.py

### lroc_to_csv.py
- Given a section of the LRO mission from [here](https://pds.lroc.im-ldi.com/data/LRO-L-LROC-3-CDR-V1.0/) (as file number increases, current mission data gets added on top of past data), the script goes through the CUMINDEX.TAB file to find the metadata of the images taken within a specific latitude, longitude range and a start and end date. This data is saved as lroc_data.csv, where the file then access functions from the lroc_img_download.py to download the images as PNGs, saves their paths, and creates a final CSV file that contains all necessary information called lroc_data(with png_path).csv 

### lroc_data.csv
- CSV of all necessary metadata of images for NeRF model training (except for image paths)

### lroc_img_download.py
- lroc_to_csv.py's main functino outputs a list of the product IDs. These product IDs are then used by this script to find all the raw .IMG files of every product ID given in the list, and then converts them to a PNG file and saves them within a folder (uses OpenCV and other techniques that I need to go over more for filtering the image to be more accurate to the original that can be found online)

### lroc_data(with png_path).csv 
- Completed CSV file with all necessary entries filled for NeRF model training



## Sources
- [LROC Image Database](https://pds.lroc.im-ldi.com/data/LRO-L-LROC-3-CDR-V1.0/LROLRC_1001/)
- [NAIF SPICE Toolkit Information](https://naif.jpl.nasa.gov/naif/toolkit.html)
- [LROC SPICE Kernels](https://naif.jpl.nasa.gov/pub/naif/pds/data/lro-l-spice-6-v1.0/lrosp_1000/)
- [SpiceyPy Repo](https://github.com/AndrewAnnex/SpiceyPy)
