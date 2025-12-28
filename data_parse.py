import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# importing csv

def df_filter(file_path, new_path):
    df = pd.read_csv(file_path)

    # dropping all columns except for latitude, longitude, and energy
    df = df.drop('Coordinated_Universal_Time', axis=1)
    df = df.drop('    Pt_Range', axis=1)
    df = df.drop(' Pt_PulseW', axis=1)
    df = df.drop(' Pt_Energy', axis=1) 
    df = df.drop('Pt_noi', axis=1)
    df = df.drop('Pt_Thr', axis=1)
    df = df.drop(' Pt_Gn', axis=1)
    df = df.drop('   Flg', axis=1)
    df = df.drop('S', axis=1)
    df = df.drop('Frm', axis=1)
    df = df.drop('Mission_ET', axis=1)
    df = df.drop('   Subseconds', axis=1)
    df = df.drop('Terrestrial_Dyn_Time', axis=1)
    df = df.drop(' TX_Energy_mJ', axis=1)
    df = df.drop('TX_PulseW', axis=1)
    df = df.drop('SC_Longitude', axis=1)
    df = df.drop(' SC_Latitude', axis=1)
    df = df.drop('   SC_radius', axis=1)
    df = df.drop('     Geoid', axis=1)
    df = df.drop('Offnadir', axis=1)
    df = df.drop('Emission', axis=1)
    df = df.drop(' Sol_INC', axis=1)
    df = df.drop(' Sol_Phs', axis=1)
    df = df.drop(' Earth_Centr.', axis=1)
    df = df.drop('Earth_PW', axis=1)
    df = df.drop('Earth_E.', axis=1)

    # extracting out each column
    latitude = df[' Pt_Latitude']
    longitude = df['Pt_Longitude']
    radius = df['   Pt_Radius']
    # renaming columns to easier-to-type names
    df.rename(columns={' Pt_Latitude': 'latitude'}, inplace=True)
    df.rename(columns={'Pt_Longitude': 'longitude'}, inplace=True)
    df.rename(columns={'   Pt_Radius': 'radius'}, inplace=True)
    # creating a str that represents the path of the newly sorted csv
    save_path = "C:/Users/16679/Desktop/LOLA_DataFolder/" + str(new_path)
    # creates the new "filtered" csv
    df.to_csv(save_path, index=False)
    # ensures that the dataframe values are all numerical
    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
    df['radius'] = pd.to_numeric(df['radius'], errors='coerce')


df1 = "C:/Users/16679/Desktop/LOLA_DataFolder/unfilterMain.csv"
df_filter(df1, "filterMain.csv")