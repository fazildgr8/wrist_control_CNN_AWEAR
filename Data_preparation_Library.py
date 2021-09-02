import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from time import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer
import matplotlib.patches as mpatches
import Geometry3D as G3D
from math import sin, cos, radians
import os
from random import shuffle
from pickle import dump, load
from scipy import signal

def rms_df(df,window=20):
    ldif = window-1
    emg_df = pd.DataFrame(columns=df.columns)
    for column in df.columns:
        rms_arr = window_rms(df[column],window)
        fill_arr = np.ones(ldif)
        emg_df[column] = np.array(list(window_rms(df[column],window)) + list(fill_arr))
    # print(emg_df.shape)
    return emg_df
        
def window_rms(a, window_size):
    a2 = np.power(a,2)
    window = np.ones(window_size)/float(window_size)
    return np.sqrt(np.convolve(a2, window, 'valid'))


def difference(dataset, interval=1):
	diff = []
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return np.array(diff)

def map_it(x,old_range,new_range):
    A,B = old_range
    C,D = new_range
    scale = (D-C)/(B-A)
    offset = -A*(D-C)/(B-A) + C
    x = np.array(x)
    new_x = x*scale + offset
    return new_x

def min_max(df):
    scaler = MinMaxScaler(feature_range=(-1,1))
    scaler.fit(df)
    df = scaler.transform(df)
    dump(scaler, open('min_max.pkl', 'wb'))
    return df

def snr_scale(snr,arr):
    x_watts = arr
    target_snr_db = snr
    # Calculate signal power and convert to dB 
    sig_avg_watts = np.mean(x_watts)
    sig_avg_db = 10 * np.log10(sig_avg_watts)
    # Calculate noise according to [2] then convert to watts
    noise_avg_db = sig_avg_db - target_snr_db
    noise_avg_watts = 10 ** (noise_avg_db / 10)
    return np.sqrt(noise_avg_watts)

def add__Gausian_noise(array,theta):
    if type(array)==pd.DataFrame:
        df = array
        for x in array.columns:
            pure = np.array(df[x])
            noise = np.random.normal(0, snr_scale(theta,pure), pure.shape[0])
            n_signal = pure.reshape(array.shape[0],1) + noise.reshape(array.shape[0],1)
            df[x] = n_signal
        return df
    
    elif array.shape[1]>1:
        for x in range(array.shape[1]):
            pure = np.array(array[:,x])
            noise = np.random.normal(0, snr_scale(theta,pure), pure.shape[0])
            n_signal = pure.reshape(array.shape[0]) + noise.reshape(array.shape[0])
            array[:,x] = n_signal
        return array
        
    else:
        pure = np.array(array)
        noise = np.random.normal(0, snr_scale(theta,pure), pure.shape[0])
        x = np.linspace(0,100,pure.shape[0])
        return pure.reshape(data.shape[0],1) + noise.reshape(data.shape[0],1)

def norm(df):
    if len(df.shape)==1:
        df = np.array(df)
        df  = df.reshape(-1,1)
    # scaler = StandardScaler(with_mean=True,
    #                         with_std=True,
    #                         copy=False).fit(df)
    # dump(scaler, open('standard_scaler_filtered_master.pkl', 'wb'))
    scaler = load(open('standard_scaler_master.pkl', 'rb')) #'standard_scaler_filtered_master.pkl'
    df = scaler.transform(df)

    # mx_scaler = MinMaxScaler(feature_range=(0,1))
    # mx_scaler.fit(df)
    # dump(mx_scaler, open('minmax_scaler_filtered_master.pkl', 'wb'))
    # mx_scaler = load(open('mx_standard_scaler_master.pkl', 'rb')) #'standard_scaler_master.pkl'
    # df = mx_scaler.transform(df)

    return df

def prep_data(df,window,angle_label,interval=0,Normalize=False,rms=False,angle_thresh = 0.001,plot=False):
    
    emg_labels = ['EMG1', 'EMG2', 'EMG3', 'EMG4', 'EMG5', 'EMG6',
           'EMG7', 'EMG8'] # , 'EMG9', 'EMG10', 'EMG11', 'EMG12'
    emg_df = df[emg_labels]
    

    if(Normalize==True):
        emg_df = pd.DataFrame(norm(emg_df),columns=emg_labels)
        
    if(rms==True):
        emg_df = rms_df(emg_df,window)
        
    emg_array = np.array(emg_df)
    all_angle = np.array(df[angle_label])
    segments = []
    X = []
    y = []
    i = 0
    counter = [0,0,0]
    while i < df.shape[0]-window:
        rmin = i
        rmax = i+window
        
        loc_arr = emg_array[rmin:rmax]
        X.append(loc_arr)
        angles = all_angle[rmin:rmax]
        
        diff = difference(angles).mean()
        
        if(abs(diff)<angle_thresh):
            y.append([1,0,0])
            segments.append([rmin,rmax,'1'])
            counter[0]=counter[0]+1
        elif(diff>0):
            y.append([0,1,0])
            segments.append([rmin,rmax,'r'])
            counter[1]=counter[1]+1
        else:
            y.append([0,0,1])
            segments.append([rmin,rmax,'b'])
            counter[2]=counter[2]+1
        i = i + interval
            
    X = np.array(X)
    y = np.array(y)

    
    ## Plot Codes
    if plot==True:
        fig_size = (18,8)
        emg_df = pd.DataFrame(np.array(emg_df),columns=emg_labels)
        emg_df.plot(figsize=fig_size,title='EMG',legend=True)
        a = map_it(interval,(0,1000),(0,1))
        for x in segments:
            plt.axvspan(x[0],x[1],facecolor= x[2], alpha=a)
        plt.show()
        plt.figure(figsize=fig_size)
        plt.plot(all_angle,label=angle_label)
        clear_patch = mpatches.Patch(color='white', label='No Motion')
        blue_patch = mpatches.Patch(color='blue', label='Negative Motion(-)')
        red_patch = mpatches.Patch(color='red', label='Positive Motion(+)')
        plt.legend(handles=[clear_patch,red_patch,blue_patch],loc=2)
        plt.title(angle_label,loc=2)
        plt.xlabel('t [2000hz]')
        plt.ylabel('Angle (deg)')
        for x in segments:
            plt.axvspan(x[0],x[1],facecolor= x[2], alpha=a)
        plt.show()
        for i in range(3):
            print(i,'->',counter[i]*100/X.shape[0],'%')
    return X, y

def prep_data_prosup(df,window,interval=0,Normalize=False,rms=False,angle_thresh = 0.001,plot=False):
    angle_label = 'Pronation_Angle'
    emg_labels = ['EMG1', 'EMG2', 'EMG3', 'EMG4', 'EMG5', 'EMG6','EMG7', 'EMG8']
    emg_df = df[emg_labels]
    emg_df = pd.DataFrame(np.array(emg_df),columns=emg_labels)

    if(Normalize==True):
        emg_df = pd.DataFrame(norm(emg_df),columns=emg_labels)

    if(rms==True):
        emg_df = rms_df(emg_df,window)


    angle_div = int(len(emg_df)/500)
    emg_array = np.array(emg_df)
    all_angle = np.array(df[angle_label])

    split_angles = np.array_split(all_angle,angle_div)

    labels = []
    for arr in split_angles:
        diff_arr = np.diff(arr)
        if abs(diff_arr.mean()) > angle_thresh:
            if diff_arr.mean()>0:
                labels = labels + [1]*arr.shape[0]
            else:
                labels = labels + [2]*arr.shape[0]
        else:
            labels = labels + [0]*arr.shape[0]

    segments = []
    X = []
    y = []
    i = 0
    counter = [0,0,0]
    while i < df.shape[0]-window:
        rmin = i
        rmax = i+window

        loc_arr = emg_array[rmin:rmax]
        X.append(loc_arr)
        max_label = np.bincount(np.array(labels[rmin:rmax])).argmax()

        if(max_label==0):
            y.append([1,0,0])
            segments.append([rmin,rmax,'1'])
            counter[0]=counter[0]+1
        elif(max_label==1):
            y.append([0,1,0])
            segments.append([rmin,rmax,'r'])
            counter[1]=counter[1]+1
        else:
            y.append([0,0,1])
            segments.append([rmin,rmax,'b'])
            counter[2]=counter[2]+1
        i = i + interval

    X = np.array(X)
    y = np.array(y)

    ## Plot Codes
    if plot == True:
        fig_size = (18,8)
        emg_df = pd.DataFrame(np.array(emg_df),columns=emg_labels)
        emg_df.plot(figsize=fig_size,title='EMG',legend=True)
        a = map_it(interval,(0,1000),(0,1))
        for x in segments:
            plt.axvspan(x[0],x[1],facecolor= x[2], alpha=a)
        clear_patch = mpatches.Patch(color='white', label='No Motion')
        blue_patch = mpatches.Patch(color='blue', label='Pronation')
        red_patch = mpatches.Patch(color='red', label='Supination')
        plt.legend(handles=[clear_patch,red_patch,blue_patch],loc=2)
        plt.show()

        plt.figure(figsize=fig_size)
        plt.plot(all_angle)
        plt.title(angle_label)
        plt.xlabel('t [2000hz]')
        plt.ylabel('Angle (deg)')
        plt.legend([angle_label],loc=2)
        for x in segments:
            plt.axvspan(x[0],x[1],facecolor= x[2], alpha=a)
        plt.show()
        for i in range(3):
            print(i,'->',counter[i]*100/X.shape[0],'%')
    return X, y

def prep_data_prosup_bin(df,window,interval=0,Normalize=False,rms=False,angle_thresh = 0.001,plot=False):
    angle_label = 'Pronation_Angle'
    emg_labels = ['EMG1', 'EMG2', 'EMG3', 'EMG4', 'EMG5', 'EMG6','EMG7', 'EMG8']
    emg_df = df[emg_labels]
    emg_df = pd.DataFrame(np.array(emg_df),columns=emg_labels)

    if(Normalize==True):
        emg_df = pd.DataFrame(norm(emg_df),columns=emg_labels)

    if(rms==True):
        emg_df = rms_df(emg_df,window)


    angle_div = int(len(emg_df)/500)
    emg_array = np.array(emg_df)
    all_angle = np.array(df[angle_label])

    split_angles = np.array_split(all_angle,angle_div)

    labels = []
    for arr in split_angles:
        diff_arr = np.diff(arr)
        if abs(diff_arr.mean()) > angle_thresh:
            if diff_arr.mean()>0:
                labels = labels + [1]*arr.shape[0]
            else:
                labels = labels + [2]*arr.shape[0]
        else:
            labels = labels + [0]*arr.shape[0]

    segments = []
    X = []
    y = []
    i = 0
    counter = [0,0,0]
    while i < df.shape[0]-window:
        rmin = i
        rmax = i+window

        loc_arr = emg_array[rmin:rmax]
        # X.append(loc_arr)
        max_label = np.bincount(np.array(labels[rmin:rmax])).argmax()

        if(max_label==0):
            # y.append([1,0,0])
            segments.append([rmin,rmax,'1'])
            counter[0]=counter[0]+1
        elif(max_label==1):
            X.append(loc_arr)
            y.append([1,0])
            segments.append([rmin,rmax,'r'])
            counter[1]=counter[1]+1
        else:
            X.append(loc_arr)
            y.append([0,1])
            segments.append([rmin,rmax,'b'])
            counter[2]=counter[2]+1
        i = i + interval
    X = np.array(X)
    y = np.array(y)

    ## Plot Codes
    if plot == True:
        fig_size = (18,8)
        emg_df = pd.DataFrame(np.array(emg_df),columns=emg_labels)
        emg_df.plot(figsize=fig_size,title='EMG',legend=True)
        a = map_it(interval,(0,1000),(0,1))
        for x in segments:
            plt.axvspan(x[0],x[1],facecolor= x[2], alpha=a)
        clear_patch = mpatches.Patch(color='white', label='No Motion')
        blue_patch = mpatches.Patch(color='blue', label='Pronation')
        red_patch = mpatches.Patch(color='red', label='Supination')
        plt.legend(handles=[clear_patch,red_patch,blue_patch],loc=2)
        plt.show()

        plt.figure(figsize=fig_size)
        plt.plot(all_angle)
        plt.title(angle_label)
        plt.xlabel('t [2000hz]')
        plt.ylabel('Angle (deg)')
        plt.legend([angle_label],loc=2)
        for x in segments:
            plt.axvspan(x[0],x[1],facecolor= x[2], alpha=a)
        plt.show()
        for i in range(3):
            print(i,'->',counter[i]*100/X.shape[0],'%')
    return X, y

def prep_data_DTM(df,window,interval=0,Normalize=False,rms=False,angle_thresh = 0.008,plot=False):

    emg_labels = ['EMG1', 'EMG2', 'EMG3', 'EMG4', 'EMG5', 'EMG6',
           'EMG7', 'EMG8'] # , 'EMG9', 'EMG10', 'EMG11', 'EMG12'
    emg_df = df[emg_labels]
    

    if(Normalize==True):
        emg_df = pd.DataFrame(norm(emg_df),columns=emg_labels)
        
    if(rms==True):
        emg_df = rms_df(emg_df,window)
    
    emg_array = np.array(emg_df)
    all_angle_flexion = np.array(df['Flexion_Angle'])
    all_angle_radial = np.array(df['Radial_Angle'])
    
    segments = []
    X = []
    y = []
    i = 0
    counter = [0,0,0]
    while i < df.shape[0]-window:
        rmin = i
        rmax = i+window
        
        loc_arr = emg_array[rmin:rmax]
        X.append(loc_arr)
        angles_flexion = all_angle_flexion[rmin:rmax]
        angles_radial = all_angle_radial[rmin:rmax]
        diff_flexion = difference(angles_flexion).mean()
        diff_radial = difference(angles_radial).mean()
        
        if(abs(diff_flexion)<angle_thresh and abs(diff_radial)<angle_thresh):
            y.append([1,0,0])
            segments.append([rmin,rmax,'1'])
            counter[0]=counter[0]+1
        elif(diff_flexion>0 and diff_radial>0):
            y.append([0,1,0])
            segments.append([rmin,rmax,'r'])
            counter[1]=counter[1]+1
        else:
            y.append([0,0,1])
            segments.append([rmin,rmax,'b'])
            counter[2]=counter[2]+1
        i = i + interval
            
    X = np.array(X)
    y = np.array(y)

    
    ## Plot Codes
    if plot==True:
        fig_size = (18,8)
        emg_df = pd.DataFrame(np.array(emg_df),columns=emg_labels)
        emg_df.plot(figsize=fig_size,title='EMG',legend=True)
        a = map_it(interval,(0,1000),(0,1))
        for x in segments:
            plt.axvspan(x[0],x[1],facecolor= x[2], alpha=a)
        clear_patch = mpatches.Patch(color='white', label='No Motion')
        blue_patch = mpatches.Patch(color='blue', label='DTM Backward')
        red_patch = mpatches.Patch(color='red', label='DTM Forward')
        plt.legend(handles=[clear_patch,red_patch,blue_patch],loc=2)
        plt.show()
        
        plt.figure(figsize=fig_size)
        plt.plot(all_angle_flexion)
        plt.plot(all_angle_radial)
        plt.title('DTM Angles')
        plt.legend(['Flexion Angle','Radial Angle'],loc=2)
        plt.xlabel('t [2000hz]')
        plt.ylabel('Angle (deg)')
        for x in segments:
            plt.axvspan(x[0],x[1],facecolor= x[2], alpha=a)
        plt.show()
        for i in range(3):
            print(i,'->',counter[i]*100/X.shape[0],'%')
    return X, y

def prep_data_DTM_bin(df,window,interval=0,Normalize=False,rms=False,angle_thresh = 0.008,plot=False):

    emg_labels = ['EMG1', 'EMG2', 'EMG3', 'EMG4', 'EMG5', 'EMG6',
           'EMG7', 'EMG8'] # , 'EMG9', 'EMG10', 'EMG11', 'EMG12'
    emg_df = df[emg_labels]
    

    if(Normalize==True):
        emg_df = pd.DataFrame(norm(emg_df),columns=emg_labels)
        
    if(rms==True):
        emg_df = rms_df(emg_df,window)
    
    emg_array = np.array(emg_df)
    all_angle_flexion = np.array(df['Flexion_Angle'])
    all_angle_radial = np.array(df['Radial_Angle'])
    
    segments = []
    X = []
    y = []
    i = 0
    counter = [0,0,0]
    while i < df.shape[0]-window:
        rmin = i
        rmax = i+window
        
        loc_arr = emg_array[rmin:rmax]
        # X.append(loc_arr)
        angles_flexion = all_angle_flexion[rmin:rmax]
        angles_radial = all_angle_radial[rmin:rmax]
        diff_flexion = difference(angles_flexion).mean()
        diff_radial = difference(angles_radial).mean()
        
        if(abs(diff_flexion)<angle_thresh and abs(diff_radial)<angle_thresh):
            # y.append([1,0,0])
            segments.append([rmin,rmax,'1'])
            counter[0]=counter[0]+1
        elif(diff_flexion>0 and diff_radial>0):
            y.append([1,0])
            X.append(loc_arr)
            segments.append([rmin,rmax,'r'])
            counter[1]=counter[1]+1
        else:
            y.append([0,1])
            X.append(loc_arr)
            segments.append([rmin,rmax,'b'])
            counter[2]=counter[2]+1
        i = i + interval
            
    X = np.array(X)
    y = np.array(y)

    
    ## Plot Codes
    if plot==True:
        fig_size = (18,8)
        emg_df = pd.DataFrame(np.array(emg_df),columns=emg_labels)
        emg_df.plot(figsize=fig_size,title='EMG',legend=True)
        a = map_it(interval,(0,1000),(0,1))
        for x in segments:
            plt.axvspan(x[0],x[1],facecolor= x[2], alpha=a)
        clear_patch = mpatches.Patch(color='white', label='No Motion')
        blue_patch = mpatches.Patch(color='blue', label='DTM Backward')
        red_patch = mpatches.Patch(color='red', label='DTM Forward')
        plt.legend(handles=[clear_patch,red_patch,blue_patch],loc=2)
        plt.show()
        
        plt.figure(figsize=fig_size)
        plt.plot(all_angle_flexion)
        plt.plot(all_angle_radial)
        plt.title('DTM Angles')
        plt.legend(['Flexion Angle','Radial Angle'],loc=2)
        plt.xlabel('t [2000hz]')
        plt.ylabel('Angle (deg)')
        for x in segments:
            plt.axvspan(x[0],x[1],facecolor= x[2], alpha=a)
        plt.show()
        for i in range(3):
            print(i,'->',counter[i]*100/X.shape[0],'%')
    return X, y

def multiple_prep_data(df_list,window,angle_label,interval,Normalize=False,rms=False,angle_thresh=0.001):
    X_all, y_all = [], []
    for df in tqdm(df_list):
        X, y = prep_data(df,window,angle_label,interval,Normalize,rms,angle_thresh)
        X_all.append(X)
        y_all.append(y)
        
    X_all_stack = list_vstacker(X_all)
    y_all_stack = list_vstacker(y_all)

    return X_all_stack, y_all_stack

def multiple_prep_data_DTM(df_list,window,interval,Normalize=False,rms=False,angle_thresh=0.008):
    X_all, y_all = [], []
    for df in tqdm(df_list):
        X, y = prep_data_DTM(df,window,interval,Normalize,rms,angle_thresh)
        X_all.append(X)
        y_all.append(y)
        
    X_all_stack = list_vstacker(X_all)
    y_all_stack = list_vstacker(y_all)

    return X_all_stack, y_all_stack

def multiple_prep_data_DTM_bin(df_list,window,interval,Normalize=False,rms=False,angle_thresh=0.008):
    X_all, y_all = [], []
    for df in tqdm(df_list):
        X, y = prep_data_DTM_bin(df,window,interval,Normalize,rms,angle_thresh)
        X_all.append(X)
        y_all.append(y)
        
    X_all_stack = list_vstacker(X_all)
    y_all_stack = list_vstacker(y_all)

    return X_all_stack, y_all_stack

def multiple_prep_data_prosup(df_list,window,interval,Normalize=False,rms=False,angle_thresh=0.001):
    X_all, y_all = [], []
    for df in tqdm(df_list):
        X, y = prep_data_prosup(df,window,interval,Normalize,rms,angle_thresh)
        X_all.append(X)
        y_all.append(y)
        
    X_all_stack = list_vstacker(X_all)
    y_all_stack = list_vstacker(y_all)
    
    return X_all_stack, y_all_stack

def multiple_prep_data_prosup_bin(df_list,window,interval,Normalize=False,rms=False,angle_thresh=0.001):
    X_all, y_all = [], []
    for df in tqdm(df_list):
        X, y = prep_data_prosup_bin(df,window,interval,Normalize,rms,angle_thresh)
        X_all.append(X)
        y_all.append(y)
        
    X_all_stack = list_vstacker(X_all)
    y_all_stack = list_vstacker(y_all)
    
    return X_all_stack, y_all_stack
    
def system_sleep():
    os.system("rundll32.exe powrprof.dll,SetSuspendState 0,1,0")

def list_vstacker(data_list):
    main_X = data_list[0]
    data_list = data_list[1:]
    for x in data_list:
        main_X = np.vstack((main_X,x))
    return main_X

def freaq_window(data,Fs=2000):
    spec_list = []
    for i in range(data.shape[1]):
        y = data[:,i]
        spec,freq,line = plt.magnitude_spectrum(y,Fs)
        spec_list.append(spec)
    X = np.array(spec_list[0])
    for x in spec_list[1:]:
        X = np.vstack((X,np.array(x)))
    return np.transpose(X)

def scrambled(orig):
    dest = orig[:]
    shuffle(dest)
    return dest

def prep_data_velocity(df,window,angle_label,interval=0,Normalize=False,rms=False,rms_window=20):
    

    emg_labels = ['EMG1', 'EMG2', 'EMG3', 'EMG4', 'EMG5', 'EMG6','EMG7', 'EMG8'] # , 'EMG9', 'EMG10', 'EMG11', 'EMG12'
    emg_df = df[emg_labels] 

    if(Normalize==True):
        emg_df = pd.DataFrame(norm(emg_df),columns=emg_labels)
        
    if(rms==True):
        emg_df = rms_df(emg_df,rms_window)
        
    emg_array = np.array(emg_df)
    all_angle = np.array(df[angle_label])

    all_angle = pd.Series(all_angle).interpolate().values

    velocity = np.diff(list(all_angle)+[all_angle[-1]])
    velocity = velocity/(1/2000)
    b,a = signal.butter(3, 1,fs=2000)
    velocity = signal.lfilter(b, a,velocity)
    # velocity = filter_array(velocity,order=1,cf=50,fs=2000)
    # velocity = filter_array(velocity,order=1,cf=50,fs=2000)
    # velocity = filter_array(velocity,order=1,cf=50,fs=2000)
    
    
    X = []
    y = []
    i = 0
    while i < df.shape[0]-window:
        rmin = i
        rmax = i+window
        
        loc_arr = emg_array[rmin:rmax]
        X.append(loc_arr)
        angles = all_angle[rmin:rmax]
        diff = difference(angles).mean()
        y.append(velocity[rmax])
        i = i + interval
    # scaler = MinMaxScaler(feature_range=(-1,1))  
    scaler = StandardScaler()       
    X = np.array(X)
    y = np.array(y)
    # y = scaler.fit_transform(y.reshape(y.shape[0],1))
    y = pd.Series(y.reshape((len(y)))).interpolate()
    y = y.values.reshape((len(y),1))
    return X, y

def multiple_prep_data_velocity(df_list,window,angle_label,interval=0,Normalize=False,rms=False):
    X_all, y_all = [], []
    for df in tqdm(df_list):
        X, y = prep_data_velocity(df,window,angle_label,interval,Normalize,rms)
        X_all.append(X)
        y_all = y_all + list(y)
        
    X_all_stack = list_vstacker(X_all)
    y_all_stack = list_vstacker(y_all)

    return X_all_stack, y_all_stack    

def variance(data, ddof=0):
    n = len(data)
    mean = sum(data) / n
    return sum((x - mean) ** 2 for x in data) / (n - ddof)

def filter_df(files_df,order=1,cf=50,fs=2000):
    emg_labels = ['EMG1','EMG2','EMG3','EMG4','EMG5','EMG6','EMG7','EMG8']
    b,a = signal.butter(order, cf,fs=fs)
    for i in range(len(files_df)):
        emg_df = pd.DataFrame(columns=emg_labels)
        for labels in emg_labels:
            emg_df[labels] = signal.lfilter(b, a,files_df[i][labels])
        files_df[i][emg_labels] = np.array(emg_df)
    return files_df

def emg_normalize(df,mean_list):
    cols = df.columns
    for i in range(len(mean_list)):
        df[cols[i]] = df[cols[i]] - mean_list[i]
    return df

def system_shutdown(t = 30):
    txt = 'shutdown /s /f /t '+str(t)
    os.system(txt)

def filter_array(arr,cf=50,order=1,fs=100):
    if len(arr.reshape((-1)).shape) < 2:
        arr = arr.reshape((-1))
    b,a = signal.butter(order, cf,fs=fs)
    arr = signal.lfilter(b, a,arr)
    return arr

def preprocessor_arr(emg_arr):
    fs = 2000
    n = 4
    Fa = 80
    Fb = 800
    cf = np.array([Fa,Fb])
    emg_arr_1 = filter_array(emg_arr,cf=cf,order=n,fs=fs,btype='bandpass')
    emg_arr_2 = abs(emg_arr_1)
    emg_arr_3 = filter_array(emg_arr_2,cf=6,order=4,fs=fs,btype='lowpass')
    return emg_arr_3

def preprocessor_df(df):
    emg_labels = ['EMG1','EMG2','EMG3','EMG4','EMG5','EMG6','EMG7','EMG8', 'EMG9', 'EMG10', 'EMG11', 'EMG12']
    fdf = df.copy()
    for lbs in emg_labels:
        fdf[lbs] = preprocessor_arr(df[lbs].values)
    return fdf