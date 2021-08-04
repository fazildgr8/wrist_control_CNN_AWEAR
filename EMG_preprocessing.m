%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%               EMG Post-processing
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 0. Offset    

emg_new = dlmread(Ip_coord,'\t', 5, 2);

%% 1. Pass Band Filter:
    subplot( 5,1,1)
    plot( emg_new(:,EMGID));   
    title('1. Raw data')  
    Fs=1000;%sampling rate Hz
    n=4;
    Fa=40;
    Fb=450; %500
    Wn=([Fa Fb]*2)/Fs;

    [B,A] = butter(n,Wn);
    emg_new_f1 = filtfilt(B,A,emg_new); 

    subplot( 5,1,2)
    plot(emg_new_f1(:,EMGID));   
    title('2. Filtration')
    %% 2.  Rectification:
    emg_new_f1_rett=abs(emg_new_f1);
    
    subplot( 5,1,3)
    plot(emg_new_f1_rett(:,EMGID));   
     title('3. Rectifiaction')  
    %% 3. Smoothing(Low Pass Filter)
    n=4; % order of filter
    Fc=3;%6; %Winter used 3 Hz

    Wn=(Fc*2)/Fs;    % [Hz] 
    if Wn>1.0
        Wn=0.99; 
    end

    [B,A] = butter(n,Wn);       % Butterworth 4th order filter
    emg_new_f1_rett_f2 = (filtfilt(B,A,emg_new_f1_rett)); 
    
    
        subplot( 5,1,4)
        plot( emg_new_f1_rett_f2(:,EMGID));   
        title('4. Smoothing')  
        
%% 4. Scaling                                                                 
 emg_new_f1_rett_f2=abs(emg_new_f1_rett_f2);%to be sure
emg_export=[];
    for i=1:c
       temp = (emg_new_f1_rett_f2(:,i) - minEmgVal(i))/(maxEmgVal(i)-minEmgVal(i));  
        emg_export=[emg_export temp];
    end
      
    subplot( 5,1,5);
    plot( emg_export(:,EMGID));  
    title('5. Scaling');