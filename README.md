# [CNN-based Controller for Multi-DoF Prosthetic Wrist using sEMG Data during Activities of Daily Living](https://ieeexplore.ieee.org/document/9925506)
#### This is the cumulative project repository of the research paper titled [CNN-based Controller for Multi-DoF Prosthetic Wrist using sEMG Data during Activities of Daily Living](https://ieeexplore.ieee.org/document/9925506) published in IEEE/RAS BioRob2022.
#### This repository consists of all the Data processing pipeline codes, custom data preprocessing library built for this project, and all the time series CNN training Jupyter notebooks using the Data collected within the [AWEAR lab, University at Buffalo](https://www.awearlab.com/).
![FULL_framework](https://github.com/fazildgr8/wrist_control_CNN_AWEAR/assets/24454678/88eabad7-abb9-416a-a60e-97e5688d0e48)
### Abstract:
Controllers based on surface EMG (sEMG) data and pattern recognition are widely investigated methods for prosthetic arms with multiple Degrees of Freedom (DoF). Most of these controllers have been trained with movements that are artificially performed for training the machine learning models. In real-life scenarios, the output of the models could be poor-performed due to various arm postures, duration of the movement, and range of motion. Thus, the suggested work introduces a framework to generate a controller for a multi-DoF prosthetic wrist, trained by raw sEMG data collected during Activities of Daily Living (ADL) tasks. During ADL tasks, a motion capture system is used to label the kinematic data of the subject's wrist motion, which is trained for a deep Convolution Neural Networks (CNN) model. The paper focuses on two major functional wrist movements: Pronation-Supination and Dart throwing movement (DTM). Further, a wrist controller design based on multiple CNN models is proposed, which would directly map the sEMG signals to the joint velocities of the wrist. The prosthetic wrist controller is designed based on the data of eight participants and its performance is evaluated in Root Mean Square Error (RMSE), Mean Absolute Error (MAE), and Pearson Correlation. A novel kinematic approach to calculating the multi-plane DTM angles is also presented. Further, the model proposes a framework for integrating classification and regression that is based on real-world ADL data. The presented work also includes a robust case study on investigating the effect of data from different heights of ADL tasks.
![emg_cnn_output](https://user-images.githubusercontent.com/24454678/131933038-0e5d0c20-93b3-4db0-b94f-7c59553ccc8e.PNG)

## Dataset Generation Pipeline
<img src="https://user-images.githubusercontent.com/24454678/131930116-0697ab66-ebd4-4336-b964-79322d5fd974.PNG" alt="DataPipeline1" width="600"/> <br> The above pipeline is used to process, compute wrist angles and generate labeled datasets using the **Raw data found within the Subjects folder which should be added manually** The Computed data are saved back to the Subjects folder. All the codes and Jupyter notebook associated with the above mentioned pipeline can be found in the files.

- [wrist_data_computations.ipynb](wrist_data_computations.ipynb)
- [wrist_data_computations_modified.ipynb](wrist_data_computations_modified.ipynb)
- [Data_preparation_Library.py](Data_preparation_Library.py)(Main Library that holds all the custom data processing functions)
- [DTM_data_proscessing.ipynb](DTM_data_proscessing.ipynb) 

## Training Data Generation Pipeline
<img src="https://user-images.githubusercontent.com/24454678/131931423-eb5c254b-0b7e-4f63-bfed-0182a0a6d467.PNG" alt="DataPipeline1" width="600"/> <br> The above mentioned pipeline is used to generate Training Data(X,y) for the CNN training. All the codes and Jupyter notebook associated with the above mentioned pipeline can be found in the files.
- [Data_preparation_Library.py](Data_preparation_Library.py) 
- All the Jupyter notebooks that performs CNN training that could be understood by file names.

## CNN Models and Training
All the CNN models for this project was built using Keras library which comes now inbuilt within the Tensorflow 2.5. The project also uses a state of the art CNN architecture named Inception-Time which was forked from the repository [hfawaz/InceptionTime](https://github.com/hfawaz/InceptionTime) and could be understod from the published paper by the author [InceptionTime: Finding AlexNet for Time Series Classification](https://arxiv.org/abs/1909.04939). The IncpetionTime's codes can be found within the InceptionTime folder where model building codes are slightly modified for our purpose but the original architecture is still intact and unmodified with the original.

## Setting up your Machine
This repository's CNN training codes were built with Tensorflow-gpu version 2.5 with Python 3.6+ and Jupyter Notebook setup. It is highly recommended to setup your machine with compatible **Nvidia GPU which supports Cuda** to run all the training with GPU processing power. Further use the **requirements.txt** to install the other required python libraries. 

#### Installing Tensforflow-GPU with Cuda (Windows)
1. Update the Nvidia drivers to the latest version from [Nvidia Drivers](https://www.nvidia.com/Download/index.aspx)
2. Download and install CUDA Toolkit(Version 11.3 supported currently) from [CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive)
3. Download and Extract cuDNN SDK from [cuDNN SDK](https://developer.nvidia.com/cudnn)
4. Follow the Windows setup tutorial from [GPU Windows Setup](https://www.tensorflow.org/install/gpu#windows_setup) and set the Environment variables appropiately.
5. Install Tensorflow-gpu 2.5 using  ``` pip install tensorflow==2.5 ```



