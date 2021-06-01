import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from time import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.patches as mpatches
import os
from random import shuffle
from pickle import dump
from scipy import signal

def rms_df(df,window=200):
    if type(df)!=pd.DataFrame:
        df = pd.DataFrame(df)
    labels = df.columns
    for x in labels:
        rms_vals = [0]*window
        for i in range(len(df[x])-window):
            j = i + window
            rms = np.sqrt(np.mean(df[x][j-window:j]**2))
            rms_vals.append(rms)
        df[x] = np.array(rms_vals)
    return np.array(df)

def rms(array,window=200):
    rms_array = []
    i = 0
    while len(rms_array)<len(array)-window:
        rms = np.sqrt(np.mean(array[i:i+window]**2))
        rms_array.append(rms)
        i=i+1
    return np.array(rms_array)

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
    scaler = StandardScaler(with_mean=True,
                            with_std=True,
                            copy=False).fit(df)
    df = scaler.transform(df)
    dump(scaler, open('standard_scaler.pkl', 'wb'))
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
        emg_df = pd.DataFrame(norm(emg_df),columns=emg_labels)
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
    emg_labels = ['EMG1', 'EMG2', 'EMG3', 'EMG4', 'EMG5', 'EMG6',
           'EMG7', 'EMG8'] # , 'EMG9', 'EMG10', 'EMG11', 'EMG12'
    emg_df = df[emg_labels]
    emg_df = pd.DataFrame(np.array(emg_df),columns=emg_labels)

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
        emg_df = pd.DataFrame(norm(emg_df),columns=emg_labels)
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

def multiple_prep_data_prosup(df_list,window,interval,Normalize=False,rms=False,angle_thresh=0.001):
    X_all, y_all = [], []
    for df in tqdm(df_list):
        X, y = prep_data_prosup(df,window,interval,Normalize,rms,angle_thresh)
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

def prep_data_velocity(df,window,angle_label,interval=0,Normalize=False,rms=False):
    
    emg_labels = ['EMG1', 'EMG2', 'EMG3', 'EMG4', 'EMG5', 'EMG6','EMG7', 'EMG8'] # , 'EMG9', 'EMG10', 'EMG11', 'EMG12'
    emg_df = df[emg_labels]
    

    if(Normalize==True):
        emg_df = pd.DataFrame(norm(emg_df),columns=emg_labels)
        
    if(rms==True):
        emg_df = rms_df(emg_df,window)
        
    emg_array = np.array(emg_df)
    all_angle = np.array(df[angle_label])


    velocity = np.diff(list(all_angle)+[0])
    b,a = signal.butter(1, 1,fs=2000)
    velocity = signal.lfilter(b, a,velocity)
    
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
    y = scaler.fit_transform(y.reshape(y.shape[0],1))
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
    b,a = signal.butter(1, cf,fs=fs)
    for i in range(len(files_df)):
        emg_df = pd.DataFrame(columns=emg_labels)
        for labels in emg_labels:
            emg_df[labels] = signal.lfilter(b, a,files_df[i][labels])
        files_df[i][emg_labels] = np.array(emg_df)
    return files_df