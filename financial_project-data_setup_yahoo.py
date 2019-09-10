import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import quandl
import time
import datetime
import urllib3
# import scipy as sci
import scipy.stats as stats
import torch
import torch.nn as nn
import torch.nn.functional as F

from time import localtime, strftime

from multiprocessing import cpu_count
from multiprocessing import Pool
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import pdb
from pdb import set_trace as bp


nasdaq = pd.read_csv('QQQ.csv')
nasdaq

def get_percentage_change_old(data):
    N = len(data)
    final_array = []
    for index in range(N-1):
        final_array.append((data[index + 1] - data[index])/data[index])
    return np.array(final_array, dtype=np.float64)

def get_percentage_change(data, ref_data=None, offset=0):
    N = len(data)
    final_array = []
    if ref_data is not None:
        N = min(len(data), len(ref_data))
        for index in range(N-1 - offset):
            final_array.append((data[index + 1 + offset] - ref_data[index + offset])/ref_data[index + offset])
        return np.array(final_array, dtype=np.float64)
    else:
        for index in range(N-1 - offset):
            final_array.append((data[index + 1 + offset] - data[index + offset])/data[index + offset])
        return np.array(final_array, dtype=np.float64)

nasdaq['Date'] = nasdaq['Date'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d') if str.isdigit(x[0]) else np.NAN )
nasdaq['open_change'] = np.append(0,get_percentage_change(data=nasdaq['Open'].values))
nasdaq['open_change_pct'] = 100 * nasdaq['open_change'].values

nasdaq['next_day_open_change'] = np.append(nasdaq['open_change'].values[1:-1], [0, 0])
nasdaq['next_day_open_change_pct'] = 100 * nasdaq['next_day_open_change']

nasdaq['open_change_wrt_close'] = np.append(0,get_percentage_change(data=nasdaq['Open'].values,
                                                                        ref_data=nasdaq['Close'].values))
nasdaq['open_change_wrt_close_pct'] = 100 * nasdaq['open_change_wrt_close'].values

nasdaq['open_change_wrt_high'] = np.append(0,get_percentage_change(data=nasdaq['Open'].values,
                                                                        ref_data=nasdaq['High'].values))

nasdaq['open_change_wrt_low'] = np.append(0,get_percentage_change(data=nasdaq['Open'].values,
                                                                        ref_data=nasdaq['Low'].values))

nasdaq['open_change_wrt_volume'] = np.append(0,get_percentage_change(data=nasdaq['Open'].values,
                                                                        ref_data=nasdaq['Volume'].values))

nasdaq['next_day_open_change_wrt_close'] = np.append(nasdaq['open_change_wrt_close'].values[1:-1], [0, 0])
nasdaq['next_day_open_change_wrt_close_pct'] = 100 * nasdaq['next_day_open_change_wrt_close'].values

nasdaq['next_day_open_change_wrt_high'] = np.append(nasdaq['open_change_wrt_high'].values[1:-1], [0, 0])
nasdaq['next_day_open_change_wrt_low'] = np.append(nasdaq['open_change_wrt_low'].values[1:-1], [0, 0])
nasdaq['next_day_open_change_wrt_volume'] = np.append(nasdaq['open_change_wrt_volume'].values[1:-1], [0, 0])

nasdaq


nasdaq['close_change_pct'] = 100 * np.append(0,get_percentage_change(data=nasdaq['Close'].values))
nasdaq
nasdaq['close_change'] = np.append(0,get_percentage_change(data=nasdaq['Close'].values))
nasdaq



nasdaq['high_change_pct'] = 100 * np.append(0,get_percentage_change(data=nasdaq['High'].values))
nasdaq['high_change'] = np.append(0,get_percentage_change(data=nasdaq['High'].values))
nasdaq



nasdaq['low_change_pct'] = 100 * np.append(0,get_percentage_change(data=nasdaq['Low'].values))
nasdaq['low_change'] = np.append(0,get_percentage_change(data=nasdaq['Low'].values))
nasdaq


nasdaq['volume_change_pct'] = 100 * np.append(0,get_percentage_change(data=nasdaq['Volume'].values))
nasdaq['volume_change'] = np.append(0,get_percentage_change(data=nasdaq['Volume'].values))
nasdaq


def high_low_range(high, low, ref=None):
    if ref is None:
        return high - low
    else:
        return (high - low)/ ref





nasdaq['high_low_range'] = high_low_range(nasdaq['High'], nasdaq['Low'])
nasdaq['high_low_range_with_ref_open'] = high_low_range(nasdaq['High'], nasdaq['Low'], nasdaq['Open'])
nasdaq['high_low_range_with_ref_open_pct'] = 100 * high_low_range(nasdaq['High'], nasdaq['Low'], nasdaq['Open'])
nasdaq['high_low_range_with_ref_close'] = high_low_range(nasdaq['High'], nasdaq['Low'], nasdaq['Close'])
nasdaq['high_low_range_with_ref_close_pct'] = 100 * high_low_range(nasdaq['High'], nasdaq['Low'], nasdaq['Close'])

nasdaq


def binary_labeling(data, threshold):
    # bp()
    return data > threshold



nasdaq['gt_1'] = binary_labeling(nasdaq['close_change_pct'], 1)
nasdaq['gt_1.5'] = binary_labeling(nasdaq['close_change_pct'], 1.5)
nasdaq['gt_2.5'] = binary_labeling(nasdaq['close_change_pct'], 2.5)

nasdaq


print(nasdaq['gt_1'].sum() / nasdaq.index.size)
print(nasdaq['gt_1.5'].sum() / nasdaq.index.size)
print(nasdaq['gt_2.5'].sum() / nasdaq.index.size)


nasdaq.to_pickle('nasdaq_qqq_yahoo.pckl')

