#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 08:55:46 2020

@author: stereopickle

scaler/data manipulation functions for housing price model
"""
import pandas as pd
import numpy as np
import scipy.stats as st
import math
from sklearn.cluster import KMeans
import pickle


# add year_old feature
def make_yearold(df):
    df['month_sold'] = df.date.apply(lambda x: int(x[4:6]))
    df['yr_sold'] = df.date.apply(lambda x: int(x[0:4]))
    df['year_old'] = np.where(df.yr_renovated == 0, 
    df.yr_sold - df.yr_built, df.yr_sold - df.yr_renovated)
    return df


# Make dummies
def dummies(x, prefix):
    """
    Helper function to make dummies
    Input: series or array, prefix string
    Output: dummy dataframe
    """
    dummy = pd.get_dummies(x, prefix = prefix, drop_first = True)
    return dummy

# remove features
def remove_feat(df, exc_list):
    df = df.drop(exc_list, axis = 1)
    return df

# add interaction
def add_interaction(df, x1, x2):
    """
    Input: x1, x2 are column names (string)
    Output: dataframe
    """
    new_var = 'x'.join([x1, x2])
    df[new_var] = df[x1] * df[x2]
    return df


def kmeans_location(df):
    centroids = pickle.load(open('kcentroids.sav', 'rb'))
    location = list(zip(df['lat'], df['long']))
    km = KMeans(n_clusters = 19, init = centroids, n_init = 1, max_iter = 1).fit(location)
    data = df.copy()
    data['km_location'] = km.labels_
    dummy = dummies(data['km_location'], 'kml')
    data = pd.concat([data, dummy], axis = 1)
    exc_list = ['lat', 'long']
    data = remove_feat(data, exc_list)
    return data


# final transformation
def df_transformation(df, X = "X1"):
    
    df = make_yearold(df)    
    
    # log transformation
    logvals = ['sqft_lot']
    
    for item in logvals: 
        df[item] = np.log(df[item])

    exc_list = ['id', 'date', 'yr_built', 'yr_renovated', 'month_sold', 'yr_sold', 
                'sqft_living', 'sqft_lot15', 'sqft_living15']

    # drop exclusion lists
    df = remove_feat(df, exc_list)
    
    # categorical values to turn into dummies
    catvals = ['floors', 'waterfront', 'view', 'condition', 'grade', 'zipcode']

    for col in df.columns: 
        if col in catvals:
            dummy = dummies(df[col], col[0:3])
            df = pd.concat([df, dummy], axis = 1)
            df = df.drop(col, axis = 1)    
            
    interaction = [('sqft_above', 'wat_1'), ('bathrooms', 'wat_1')]
     
    for item in interaction: 
        df = add_interaction(df, item[0], item[1])
    
    df = kmeans_location(df)
    
    filename = f"{X}_columns.sav"
    cols = pickle.load(open(filename, 'rb'))

    df = df[cols]

    return df




