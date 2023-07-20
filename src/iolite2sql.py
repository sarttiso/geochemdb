import sqlite3
import numpy as np
import pandas as pd
from thefuzz import fuzz, process

#%%

def addrow():
    """
    

    Returns
    -------
    None.

    """
    return 

#%%
def parsespot(iolite_spot):
    """
    Take an iolite spot string and create a sample name str and a spot name str.

    Parameters
    ----------
    iolite_spot : str
        Spot name in iolite.

    Returns
    -------
    sample_name : str
        sample name
    spot_name : str
        spot name

    """
    # count number of underscores
    iolite_spot_split = iolite_spot.split('_')
    n_under = len(iolite_spot_split)
    
    # likely a standard
    if n_under == 1:
        sample_name, spot_name = iolite_spot_split[0], iolite_spot_split[1]
    # likely an unknown
    elif n_under >= 2:
        sample_name = iolite_spot_split[0:-2]
        spot_name = iolite_spot_split[-2:]
        
        # join with spaces
        sample_name = ' '.join(sample_name)
            
        # check for z in spot names and remove it if so
        if 'z' in spot_name[1]:
            spot_name[1] = spot_name[1].replace('z', '')
        # concatenate mount prefix and spot number
        spot_name = ''.join(spot_name)
    return sample_name, spot_name
    

def matchsample(sample_name, database_path):
    
    # connect to database
    con = sqlite3.connect(database_path)
    # read samples table
    samples_df = pd.read_sql_query('SELECT * from samples', con)
    
    # use fuzzy matching to get nearest match and score
    sample_match, score = process.extractOne(sample_name, samples_df['name'].values)
    
    return sample_match, score

#%%

# def update(df, db):
#     n_spots = len(df)
    
#     # parse sample and spot names
#     sample_names = []
#     spot_names = []
#     for spot in df['spot'].iteritems():
    

#%%

def parseexcel(excel_path):
    """
    prepare a clean DataFrame to then match with the database

    Parameters
    ----------
    excel_path : str
        Path to an excel file with geochemical data exported from iolite4

    Returns
    -------
    df : pd.DataFrame
        DataFrame with cleaned and standardized column names

    """
    # load dataframe
    df = pd.read_excel(excel_path, sheet_name='Data')
    
    # rename first columns
    df.rename({'Unnamed: 0': 'spot'}, axis=1, inplace=True)
    
    # trim column names
    cols = list(df)
    cols_new = [col.replace('_ppm', '') for col in cols]
    cols_new = [col.replace('_mean', '') for col in cols_new]
    cols_new = [col.replace('Final ', '') for col in cols_new]
    cols_new = [col.replace('(prop)', '') for col in cols_new]
    cols_new = [col.replace('(int)', '') for col in cols_new]
    cols_new = [col.replace('Approx_', '') for col in cols_new]
    cols_new = [col.replace('_PPM', '') for col in cols_new]
    
    # rename columns
    df.rename(columns=dict(zip(cols, cols_new)), inplace=True)
    
    return df

