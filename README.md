# Grouped Convolutional Neural Networks  

A simple tensorFlow implementation of the paper, 

[Grouped Convolutional Neural Networks for Multivariate Time Series](https://arxiv.org/abs/1703.09938), Subin Yi, Janghoon Ju, Man-Ki Yoon, and Jaesik Choi, 2017.  

\* Code for the Section 3.1 is included only and the Section 3.2 is not implemented here.
  
  
  
## Dependencies
* tensorflow>=1.0  
* numpy  
* scipy  
* pandas
* matplotlib  
  
  
## Train & Test
You should have your data and cluster index in .csv files.  
The data should be in (_time_step_, _feature_) shape where _time_step_ is the length of the time series data and _feature_ is the number of features.  
The cluster should be in (_feature_) shape where each element has the index to the cluster in which the feature belongs to.


  
## LICENSE
MIT License
  
  
  
## Contacts
Subin Yi (yisubin0202@gmail.com)  
Janghoon Ju (wnwkdgns@gmail.com)  
Jaesik Choi (jaesik@unist.ac.kr) 
