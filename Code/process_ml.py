#!/usr/bin/env python3
import sys
import logging
import numpy as np
from data_utils import *
from sklearn.utils import check_random_state
from collections import Counter
from metrics import *
from statistics import mean
from pandas import read_csv
import pandas as pd

np.seterr(all='raise')

###########################################################################################
def get_rating_data(dataset):
    ratings_file = dataset_dtls[dataset]['folder_path'] + dataset_dtls[dataset]['original_ratings_file']
    columns = dataset_dtls[dataset]['rat_columns']
    delim = dataset_dtls[dataset]['delim']
    r_skip =  dataset_dtls[dataset]['r_skip'] if 'r_skip' in dataset_dtls[dataset] else None
    if isfile(ratings_file):
        print("Reading from the ratings file %s" %(ratings_file))
        R = read_csv(ratings_file, header = None, skiprows=r_skip, delimiter=delim, names=['UID','IID','RATING','TIMESTAMP'], usecols=columns,quotechar='"', dtype={'UID':np.uint32, 'IID':np.uint32, 'RATING':np.float16, 'TIMESTAMP':np.uint64}, engine='python')
    else:
        print("Rating file %s does NOT exist. Exiting " %(ratings_file))
        exit(0)
    return R #Index of the u_id/m_id in U_map/M_map is the position in r_matrix.which(U_map==ID)
###########################################################################################
def get_items_sideinfo(dataset):
    items_file = dataset_dtls[dataset]['folder_path'] + dataset_dtls[dataset]['original_items_file']
    columns = dataset_dtls[dataset]['itm_columns']
    delim = dataset_dtls[dataset]['delim']
    i_skip = dataset_dtls[dataset]['i_skip'] if 'i_skip' in dataset_dtls[dataset] else None
    if isfile(items_file):
        print("Reading from the items file %s" %(items_file))
        M = read_csv(items_file, header = None, skiprows=i_skip, delimiter=delim, names=['IID','TITLE','TAG'], usecols=columns, quotechar='"', dtype={'IID':np.uint32, 'TITLE':np.unicode, 'TAG':np.unicode}, engine='python')  #create a structured array. easy to do mapping.
        return M
    else:
        print("Items file %s does not exist. Exiting " %(items_file))
        exit(0)
###########################################################################################
def get_tags_info(dataset):
    tags_file = dataset_dtls[dataset]['folder_path'] + dataset_dtls[dataset]['tags_file']
    columns = dataset_dtls[dataset]['tag_columns']
    delim = dataset_dtls[dataset]['delim']
    t_skip = dataset_dtls[dataset]['t_skip'] if 't_skip' in dataset_dtls[dataset] else None
    if isfile(tags_file):
        print("Reading from the tags file %s" %(tags_file))
        T = read_csv(tags_file, header = None, skiprows=t_skip, delimiter=delim, names=['IID','TAG'], usecols=columns, quotechar='"', dtype={'IID':np.uint32, 'TAG':np.unicode}, engine='python')  #create a structured array. easy to do mapping.
        T = T[~T.TAG.isna()]    #get rid of NaNs
        #T = T[pd.to_numeric(T['TAG'].astype(str), errors='coerce').isna()]
        T = T[~T.TAG.str.isnumeric()]   #remove all numeric ones
        T = T[T.TAG.str.len() > 3]  #take only if tag length is more than 3 characters
        T = T[T.groupby("TAG")["TAG"].transform('size') > 200]  #we consider tag information which appears at least 200 times
        return T
    else:
        print("Tags file %s does not exist. Exiting" %(tags_file))
        exit(0)
###########################################################################################
def combine_tags_with_items(M,T):
    print("Adding tags to the item data")
    out = pd.concat((M,T),sort=False).groupby('IID').agg({'TITLE':'first','TAG':'|'.join}).reset_index()
    out['TAG'] = out['TAG'].str.upper().str.split('|').explode().groupby(level=0).unique().str.join('|')
    return out
###########################################################################################
def filter_items_sideinfo(df_i, df_r):
    print("Filtering the items side information data") #to save space, we store side-info of items which are in the ratings data
    return df_i[df_i.IID.isin(df_r.IID.unique())]
###########################################################################################
def filter_rating_data(df,thld,min_rel_count):
    print("Filtering the ratings data. We consider users with at least thld ratings")
    df = df[df.RATING.ge(thld).groupby(df.UID).transform(sum).ge(min_rel_count)]
    df.reset_index(inplace=True,drop=True)
    return df
###########################################################################################
def process_ml(): #main function
    thld = 4 #anything less than 4 is considered as irrelevant and otherwise relevant
    min_rel_count = 10    #we consider users with minium of these many rating counts
    no_ds = 2
    all_datasets = ['dataset'+str(_) for _ in range(3,5)]
    rand = check_random_state(42)
    for dataset in all_datasets:
        folder_path = get_folder_path(dataset)
        #log_file = folder_path + '%s_access.log' %(alg)
        #logging.basicConfig(filename = log_file, format='%(asctime)s : %(message)s', level=logging.INFO)
        RD = get_rating_data(dataset)    #ratings data as a DF
        RD = filter_rating_data(RD,thld,min_rel_count)    #we consider users who has at least 10 relevant ratings
        MD = get_items_sideinfo(dataset)   #movies data as a DF
        GD = get_tags_info(dataset) #tags data as a DF
        MD = combine_tags_with_items(MD,GD)  #join tags with movies
        MD = filter_items_sideinfo(MD,RD)  #MD contains information about all the movies. To save space we store only the ones which appear in RD
        rating_file = folder_path+'ratings.csv'
        RD.to_csv(rating_file,index=False)
        items_file = folder_path+'movies.csv'
        MD.to_csv(items_file,index=False)

if __name__ == '__main__':
    process_ml()
