#!/usr/bin/env python
# coding: utf-8

# # Wrist Movement Data Formating

# In[20]:


import numpy as np
import pandas as pd
from math import sqrt,acos,degrees
import matplotlib.pyplot as plt
from scipy.signal import resample
from tqdm.notebook import tqdm
import os
import csv


# ## EMG Data Extraction (Cumilative Function)

# In[22]:


def extract_dataframe(path,file,save=False):
    with open(path+'/'+file, "r") as f:
        lines = f.readlines()
    emg_labels = ['Frame','Sub Frame',
                 'IM EMG1',
                 'IM EMG2',
                 'IM EMG3',
                 'IM EMG4',
                 'IM EMG5',
                 'IM EMG6',
                 'IM EMG7',
                 'IM EMG8',
                 'IM EMG9',
                 'IM EMG10',
                 'IM EMG11',
                 'IM EMG12']
    emg_labels_ref = ['Frame','Sub Frame',
                 'EMG1',
                 'EMG2',
                 'EMG3',
                 'EMG4',
                 'EMG5',
                 'EMG6',
                 'EMG7',
                 'EMG8',
                 'EMG9',
                 'EMG10',
                 'EMG11',
                 'EMG12']
    marker_labels = ['Frame','Sub Frame',
                  'RSHO_X','RSHO_Y','RSHO_Z',
                  'RUPA_X','RUPA_Y','RUPA_Z',
                  'RELB_X','RELB_Y','RELB_Z',
                  'RM1_X','RM1_Y','RM1_Z',
                 'RFRM_X','RFRM_Y','RFRM_Z',
                 'WRM2_X','WRM2_Y','WRM2_Z',
                 'RWRA_X','RWRA_Y','RWRA_Z',
                 'RWRB_X','RWRB_Y','RWRB_Z',
                 'RFIN_X','RFIN_Y','RFIN_Z']
    
    pronation_movement = ['No Motion','Supination','Pronation']
    flexion_movement = ['No Motion','Extension','Flexion']
    radial_movement = ['No Motion','Ulnar','Radial']
    
    #################
    # EMG Data Frame#
    #################
    
    emg_lines = []
    for line in lines[5:]:
        if line=='\n':
            break
        emg_lines.append(line.split(','))
    emg_df = pd.DataFrame(np.array(emg_lines),columns=lines[3].split(','))
    emg_df = emg_df[emg_labels]
    emg_df.columns = emg_labels_ref
    emg_df = emg_df[emg_df.columns].astype(float)
    duration = emg_df.shape[0]/2000
    
    # Marker Data Frame
    marker_lines = []
    marker_line_start = None
    for i in range(len(lines)):
        if lines[i]=='Trajectories\n':
            marker_line_start = i
            break
    for line in lines[marker_line_start+5:]:
        if line=='\n':
            break
        marker_lines.append(line.split(','))
    marker_df = pd.DataFrame(np.array(marker_lines),columns=marker_labels)
    marker_df = marker_df[marker_df.columns].astype(float)
    
    # Angles Dataframe
    angles_df = compute_wrist_angles(marker_df,degree=True)
    
    pronations = np.array(angles_df['Pronation_Angle'])
    flexions = np.array(angles_df['Flexion_Angle'])
    radials = np.array(angles_df['Radial_Angle'])
    elbows = np.array(angles_df['Elbow_Joint'])
    
    # Resampling to EMG SR(2000 Hz) from Vicon SR(100 Hz)
    pronations = resample_series(pronations,100,2000)
    flexions = resample_series(flexions,100,2000)
    radials = resample_series(radials,100,2000)
    elbows = resample_series(elbows,100,2000)    
    
    pronation_labels,pronation_movement_labels = direction_labels(pronations,500,pronation_movement)
    flexion_labels,flexion_movement_labels = direction_labels(flexions,500,flexion_movement)
    radial_labels,radial_movement_labels = direction_labels(radials,500,radial_movement)
    
    emg_df['Pronation_Angle'] = pronations
    emg_df['Pronation_Label'] = pronation_labels
    
    emg_df['Flexion_Angle'] = flexions
    emg_df['Flexion_Label'] = flexion_labels
    
    emg_df['Radial_Angle'] = radials
    emg_df['Radial_Label'] = radial_labels
    
    emg_df['Elbow_Joint_Angle'] = elbows
    
    
    if(save==True):
        emg_df.to_csv(path+'/computed_'+file)
        
    return emg_df

def resample_series(data,sr_origin,sr_new):
    """
    Upsamples Series Vector to required Freq(Hz)
    data - Series 1D Array
    sr_origin - Origin Sampling Rate
    sr_new - New Sampling Rate
    Return - Resampled Data to Given Sample Rate
    """
    dt = pd.Series(data)
    dt.index = pd.to_datetime(dt.index*(int((sr_origin/10)**7)))
    dt2 = dt.resample(str(1/sr_new)+'S').ffill()
    len_diff = len(dt)*(sr_new/sr_origin) - len(dt2)
    dt2_list = list(np.array(dt2))
    resampled_array = None
    
    # Balencing
    if(len_diff%2==0):
        nd = int(len_diff/2)
        resampled_array = [dt2_list[0]]*nd + dt2_list
        resampled_array = resampled_array + [dt2_list[-1]]*nd
    else:
        nd = int((len_diff+1)/2)
        resampled_array = [dt2_list[0]]*nd + dt2_list
        resampled_array = resampled_array + [dt2_list[-1]]*nd
        resampled_array = resampled_array[1:]
        
    return np.array(resampled_array)


# ## Compute Wrist Angles

# In[10]:


def compute_wrist_angles(df,degree=False):
    # Wrist Segment
    WRM2 = df[['WRM2_X','WRM2_Y','WRM2_Z']]
    RWRA = df[['RWRA_X','RWRA_Y','RWRA_Z']]
    RWRB = df[['RWRB_X','RWRB_Y','RWRB_Z']]
    # Palm Segment
    RFIN = df[['RFIN_X','RFIN_Y','RFIN_Z']]

    # Elbow Segment
    RFRM = df[['RFRM_X','RFRM_Y','RFRM_Z']]
    RM1 = df[['RM1_X','RM1_Y','RM1_Z']]
    RELB = df[['RELB_X','RELB_Y','RELB_Z']]
    # Shoulder Segment
    RSHO = df[['RSHO_X','RSHO_Y','RSHO_Z']]
    RUPA = df[['RUPA_X','RUPA_Y','RUPA_Z']]
    # Bisector Point
    MID = (np.array(RWRB) + np.array(RWRA))/2
    MIDE = (np.array(RFRM) + np.array(RM1))/2

    # Translate Wrist to Elbow Segment Mid
    RWRB_E = RWRB - MIDE

    flexion_angles = angles_lines(RFIN,WRM2,MID,deg=degree)
    radial_angles = angles_lines(RFIN,RWRB,MID,deg=degree)-90
    pronation_angles = angles_lines(RFRM,RWRB_E,MIDE,deg=degree)
    elbow_angles = angles_lines(RSHO,MID,MIDE,deg=degree)
    
    df_labels = ['Flexion_Angle','Radial_Angle','Pronation_Angle','Elbow_Joint']
#     df_labels = ['Pitch','Yaw','Roll','Elbow_Joint']
    ndf = pd.DataFrame(columns=df_labels)
    ndf['Flexion_Angle'] = flexion_angles
    ndf['Radial_Angle'] = radial_angles
    ndf['Pronation_Angle'] = pronation_angles
    ndf['Elbow_Joint'] = elbow_angles
    return ndf

def angles_lines(p1,p2,mid,deg=False):
    u = np.array(p1)-np.array(mid)
    v = np.array(p2)-np.array(mid)
    i1,j1,k1 = u[:,0],u[:,1],u[:,2]
    i2,j2,k2 = v[:,0],v[:,1],v[:,2]
    angles = []
    for t in range(len(i1)):
        cos_t = abs(i1[t]*i2[t]+j1[t]*j2[t]+k1[t]*k2[t])
        cos_t = cos_t/(sqrt(i1[t]**2+j1[t]**2+k1[t]**2)*sqrt(i2[t]**2+j2[t]**2+k2[t]**2))
        if deg==False:
            angles.append(acos(cos_t))  
        if deg==True:
            angles.append(degrees(acos(cos_t)))
    return np.array(angles)


# ## Movement Labelling

# In[11]:


def direction_labels(array,interval=50,movements=None):
    """
    0 - No Motion
    1 - Positive Direction
    2 - Negative Direction
    """
    labels = []
    diff_arr = difference(array,interval)
    data_len = len(array)
    len_diff = int(data_len - len(diff_arr))
    for diff in diff_arr:
        if(abs(diff)<5):
            labels.append(0)
        elif(diff>0):
            labels.append(1)
        elif(diff<0):
            labels.append(2)
    # Balencing
    if(len_diff%2==0):
        nd = int(len_diff/2)
        labels = [labels[0]]*nd + labels
        labels = labels + [labels[-1]]*nd
    else:
        nd = int((len_diff+1)/2)
        labels = [labels[0]]*nd + labels
        labels = labels + [labels[-1]]*nd 
        labels = labels[1:]
        
    if(movements==None):
        return np.array(labels)
    else:
        movement_labels = []
        hot_labels = labels
        for lb in labels:
            movement_labels.append(movements[lb])
        return hot_labels,movement_labels
    
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return diff


