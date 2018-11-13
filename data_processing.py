import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import random


def get_SensorData(files, target_files, read_initial_data = True, min_max_norm = False, target='coke', nc=None):
    column_name = ['timestamp', 'f_0', 'f_1', 'f_2', 'f_3', 'f_4', 'f_5', 'f_6', 'f_7', 'f_8', 'f_9', 'f_10', 'f_11', 'f_12', 'f_13', 'f_14', 'f_15', 'f_16', 'f_17', 'f_18', 'f_19', 'f_20', 'f_21', 'f_22', 'f_23', 'f_24', 'f_25', 'f_26', 'f_27', 'f_28', 'f_29', 'f_30', 'f_31', 'f_32', 'f_33', 'f_34', 'f_35', 'f_36', 'f_37', 'f_38', 'f_39', 'f_40', 'f_41', 'f_42', 'f_43', 'f_44', 'f_45', 'f_46', 'f_47']

    if read_initial_data:  ### Training and test sensor data ###

        data_file = pd.read_csv(files)
        data_file.columns = column_name
        target_file = pd.read_csv(target_files)
        #target_file.columns = target
        
        #### standard normalization ####
        mean = data_file.iloc[:, 1:len(list(data_file))].mean()
        std = data_file.iloc[:, 1:len(list(data_file))].std()
        std.replace(0, 1, inplace=True)
        #print("std", std)
        ################################

        if min_max_norm:
            scaler = MinMaxScaler()
            data_file.iloc[:, 1:len(list(data_file))] = scaler.fit_transform(data_file.iloc[:, 1:len(list(data_file))])
        else:
            data_file.iloc[:, 1:len(list(data_file))] = (data_file.iloc[:, 1:len(list(data_file))] - mean) / std

        data_file = data_file.fillna(0)
        
        data_file.to_csv('normalized_train_data.csv')
        
        mean_y = target_file.iloc[:, 1].mean()
        std_y = target_file.iloc[:, 1].std()
        
        target_file.iloc[:, 1] = (target_file.iloc[:, 1] - mean_y)/std_y
        
        # apply PCA
        
        colnames = ['timestamp']
        
     
        if nc is not None:
            for i in range(nc):
                colnames.append("nc"+str(i))               
            pca, data = my_pca(data_file.iloc[:,1:len(list(data_file))],nc)
            data_file = pd.concat([data_file["timestamp"], data], axis=1)
            data_file.columns = colnames
        else:
            print (1)
        
        
        #print(data_file.head())
        #split into (train+test+validate) and submit datasets
        X = data_file.loc[data_file["timestamp"].isin(target_file["timestamp"])]
        #submit_X = data_file.loc[~data_file["timestamp"].isin(target_file["timestamp"])]
   
    
    return X, target_file, data_file, mean_y, std_y


def my_pca(data, nc = 10):
    pca = PCA(n_components=nc)
    dataX = pd.DataFrame(pca.fit_transform(data))
    return pca, dataX
	
	

def lstm_sampling (data, y=None, timesteps=250):
    # split into samples 
    samples = list()
    targets = list()
    length = timesteps
    n = data.shape[0]
    
    # step over the 5,000 in jumps of 200
    for i in range(n-length):
        # grab from i to i + length
        sample = data[i:i+length]
        samples.append(sample)
        if y is not None:
            target = y[i:i+length]
            targets.append(target)
    
    print("Samples length",len(samples))
    print("Targets length",len(targets))
    
    # convert list of arrays into 2d array
    data = np.stack(samples)
    if y is not None:
        y = np.stack(targets)

    
    return data, y
    
    
    
    
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	"""
	Frame a time series as a supervised learning dataset.
	Arguments:
		data: Sequence of observations as a list or NumPy array.
		n_in: Number of lag observations as input (X).
		n_out: Number of observations as output (y).
		dropnan: Boolean whether or not to drop rows with NaN values.
	Returns:
		Pandas DataFrame of series framed for supervised learning.
	"""
	n_vars = 1 if type(data) is list else data.shape[1]
	df = pd.DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = pd.concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg