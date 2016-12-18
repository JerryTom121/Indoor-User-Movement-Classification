'''
Created on Dec 14, 2016
Python 3.5.2
@author: Nidhalios
'''

import itertools
from operator import itemgetter
import random

from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

import models.multiDTW as multiDTW
import numpy as np
import pandas as pd


def frange(a, b, step):
    """
    A generator function for float ranges 
    """
    while a < b:
        yield a
        a += step
        
def readData():
    """
    This function is responsible for loading and matching the data of the RSS
    sequences, the Target data, the Dataset groups and the Paths data. All of the consolidated
    data will be stored in a list of dictionaries 'series'   
    """
    series = []
    
    groupData = pd.read_csv('../data/groups/MovementAAL_DatasetGroup.csv', sep=',')
    #print(groupData.head())
    #print(groupData.info())
    #print("Missing Group Data:",groupData.isnull().values.any())
    pathData = pd.read_csv('../data/groups/MovementAAL_Paths.csv', sep=',')
    #print(pathData.head())
    #print(pathData.info())
    #print("Missing Path Data:",pathData.isnull().values.any())
    pathData = pathData.drop(pathData.columns[[ 2, 3]], axis=1)
    targetData = pd.read_csv('../data/dataset/MovementAAL_target.csv', sep=',')
    #print(targetData.head())
    #print(targetData.info())
    #print("Missing Target Data:",targetData.isnull().values.any())
    
    # Iterate through the 314 sequences CSV files to extract the series, 
    # associate them with the group_id and the corresponding path_id.
    for i in range(1,315):
        df = pd.read_csv('../data/dataset/MovementAAL_RSS_{}.csv'.format(i), sep=',',header=0, 
                     names=['anchor1', 'anchor2', 'anchor3','anchor4'])
        rng = list(frange(0, df.shape[0]*0.125, 0.125))
        df.index = rng
        #print("i : ",i," Missing : ",df.isnull().values.any())
        if(not df.isnull().values.any()):
            series.append({'id':i,
                           'series':df,
                           'group_id':groupData[groupData.sequence_ID==i].iloc[0,1],
                           'path_id':pathData[pathData.sequence_ID==i].iloc[0,1],
                           'target':targetData[targetData.sequence_ID==i].iloc[0,1]
                })
    
    return series

def normalize(data):
    """
    Normalize the array to avoid outliers and stabilize the values
    """
    for dimension in range(len(data[0])):
        dev = np.std(data[:, dimension])
        mean = np.mean(data[:, dimension])
        for i in range(len(data)):
            data[i][dimension] = (data[i][dimension] - mean) / dev
    return data

def vizData(series):
    """
    This function's main goal is visualizing data from 'series' for exploration and 
    analysis purposes
    """
    # Plotting Anchors Time Series Example
    item = series[0]['series']
    plt.figure(figsize=(12, 8))
    plt.subplot(4, 1, 1)
    plt.plot(item['anchor1'],'r')
    plt.ylabel('Anchor1')
    plt.subplot(4, 1, 2)
    plt.plot(item['anchor2'],'g')
    plt.ylabel('Anchor2')
    plt.subplot(4, 1, 3)
    plt.plot(item['anchor3'],'b')
    plt.ylabel('Anchor3')
    plt.subplot(4, 1, 4)
    plt.plot(item['anchor4'],'y')
    plt.ylabel('Anchor4')
    plt.tight_layout()
    plt.show()
    
    # Create a list of colors (from iWantHue)
    colors = ["#7de852",
              "#4fe4e4",
              "#d7e343",
              "#75e7a3",
              "#ecc568",
              "#c2e388"]
    cols1 = ['path_id','count']
    series.sort(key=itemgetter("group_id"))
    for key, group in itertools.groupby(series, lambda item: item["group_id"]): 
        df = pd.DataFrame(columns=cols1)
        for key1, group1 in itertools.groupby(group, lambda item: item["path_id"]):
            df = df.append(pd.DataFrame([[key1,len(list(group1))]], columns=cols1), ignore_index=True) 
        
        # Create a pie chart for each Group
        plt.pie(df['count'],labels=df['path_id'].map(lambda x: "Path {}".format(x)),
                shadow=False,colors=colors,startangle=90,autopct='%1.1f%%',
                )
        plt.axis('equal')
        plt.title('Groupe {}'.format(key))
        # View the plot
        plt.tight_layout()
        plt.show()
    
    # Plot stacked bar charts to see group_ID Vs Target(Changed/Unchanged)
    cols2 = ['group_id','changed','unchanged']
    series.sort(key=itemgetter('group_id'))
    target = itemgetter('target')
    df1 = pd.DataFrame(columns=cols2)
    for key, group in itertools.groupby(series, lambda item: item["group_id"]):
        changed = 0
        unchanged = 0
        for i in group:
            if target(i) == 1: changed +=1 
            else: unchanged +=1
        df1 = df1.append(pd.DataFrame([[key,changed,unchanged]], columns=cols2), ignore_index=True) 
            
    # Create a bar chart for each Group
    plt.bar(df1.group_id, df1.changed , color = '#d7e343', align='center')
    plt.bar(df1.group_id, df1.unchanged, color = '#75e7a3', align='center', bottom = df1.changed)
    plt.xticks(df1.group_id, ['Group1','Group2','Group3'])
    plt.legend(["changed(1)", "unchanged(-1)"])
    plt.title('Group_ID Vs Target')
    # View the plot
    plt.tight_layout()
    plt.show()    
      
    # Plot stacked bar charts to see path_ID Vs Target(Changed/Unchanged)
    cols3 = ['path_id','changed','unchanged']
    series.sort(key=itemgetter('path_id'))
    target = itemgetter('target')
    df2 = pd.DataFrame(columns=cols3)
    for key, group in itertools.groupby(series, lambda item: item["path_id"]):
        changed = 0
        unchanged = 0
        for i in group:
            if target(i) == 1: changed +=1 
            else: unchanged +=1
        df2 = df2.append(pd.DataFrame([[key,changed,unchanged]], columns=cols3), ignore_index=True) 
            
    # Create a bar chart for each Path
    plt.bar(df2.path_id, df2.changed , color = '#d7e343', align='center')
    plt.bar(df2.path_id, df2.unchanged, color = '#75e7a3', align='center', bottom = df2.changed)
    plt.xticks(df2.path_id, ['Path1','Path2','Path3','Path4','Path5','Path6'])
    plt.legend(["changed(1)", "unchanged(-1)"])
    plt.title('Path_ID Vs Target')
    # View the plot
    plt.tight_layout()
    plt.show()    
    
    # Plot Min/Avg/Median/Max Sequence Length for each Group (Bar Chart)
    cols4 = ['group_id','length']
    series.sort(key=itemgetter('group_id'))
    ser = itemgetter('series')
    df3 = pd.DataFrame(columns=cols4)
    for key, group in itertools.groupby(series, lambda item: item["group_id"]):
        for i in group:
            df3 = df3.append(pd.DataFrame([[key,ser(i).shape[0]]], columns=cols4), ignore_index=True) 
    aggregations = {'length':
                        {'min':'min',
                         'avg':'mean',
                         'median': 'median',
                         'max':'max'}
                    }
    stats = df3.groupby(['group_id'],0).agg(aggregations)
    # I chose to plot each Bar one by one in order to have control over their order
    # from lower to higher min<=avg<=median<=max
    ax = plt.subplot(111)
    w = 0.15
    x = np.arange(1,4)
    print(x)
    ax.bar(x-2*w, stats.length['min'], width=w, color = '#d7e343', align='center')
    ax.bar(x-w, stats.length['avg'], width=w, color = '#75e7a3', align='center')
    ax.bar(x, stats.length['median'], width=w, color = '#ecc568', align='center')
    ax.bar(x+w, stats.length['max'], width=w, color = '#4fe4e4', align='center')
    plt.xticks(range(1,4), ['Group1','Group2','Group3'])
    plt.legend(["Min", "Avg", "Median", "Max"],loc=2)
    plt.title('RSS Sequences Length Stats by Group')
    plt.show()
    
    # Plot Min/Avg/Median/Max Sequence Length for each Target Class (Bar Chart)
    cols5 = ['target','length']
    series.sort(key=itemgetter('target'))
    ser = itemgetter('series')
    df4 = pd.DataFrame(columns=cols5)
    for key, group in itertools.groupby(series, lambda item: item["target"]):
        for i in group:
            df4 = df4.append(pd.DataFrame([[key,ser(i).shape[0]]], columns=cols5), ignore_index=True) 
    aggregations = {'length':
                        {'min':'min',
                         'avg':'mean',
                         'median': 'median',
                         'max':'max'}
                    }
    stats = df4.groupby(['target'],0).agg(aggregations)
    
    # I chose to plot each Bar one by one in order to have control over their order
    # from lower to higher min<=avg<=median<=max
    ax = plt.subplot(111)
    w = 0.15
    x = np.arange(1,3)
    ax.bar(x-w, stats.length['min'], width=w, color = '#d7e343', align='center')
    ax.bar(x, stats.length['avg'], width=w, color = '#75e7a3', align='center')
    ax.bar(x+w, stats.length['median'], width=w, color = '#ecc568', align='center')
    ax.bar(x+2*w, stats.length['max'], width=w, color = '#4fe4e4', align='center')
    plt.xticks(range(1,4), ['Unchanged(-1)','Changed(1)'])
    plt.xlabel("Target")
    plt.legend(["Min", "Avg", "Median", "Max"],loc=2)
    plt.title('RSS Sequences Length Stats by Target Class')
    plt.tight_layout()
    plt.show()

def classification(series):
    
    # Mapping table for target classes
    labels = {1:'Changed', -1:'Unchanged'}
    
    # Randomly Sample 70% for Training and 30% for Testing, and Transform them to Tuples
    TRAIN_PERCENTAGE = 70
    t = len(series) * TRAIN_PERCENTAGE // 100
    indicies = random.sample(range(len(series)), t)
    train = [series[i] for i in indicies]
    train_tuples = list(map(lambda x: (x['id'],
                                       int(x['target']),
                                       normalize(np.array(x['series']))),train))
    test = [series[i] for i in range(len(series)) if i not in indicies]
    test_tuples = list(map(lambda x: (x['id'],
                                       int(x['target']),
                                       normalize(np.array(x['series']))),test))
    
    
    # Train a K-nearest neighbor classifier that uses Dynamic Time Warping (DTW) to  
    # evaluate distance between two given Multivariate Time Series 
    KNN_NEIGHBORS = 2
    results = []
    for i in range(len(test_tuples)):
        print('{}/{}...'.format(i+1, len(test_tuples)))
        id,pred,proba = multiDTW.predict(test_tuples[i], train_tuples, KNN_NEIGHBORS)
        results.append([id,pred,proba])
    
    # Display Classification metrics like precision, recall and support to have an idea on 
    # the k-NN model prediction performance 
    tar_results = [row[1] for row in results]
    tar_test = [x[1] for x in test_tuples]
    print(classification_report(tar_results,tar_test, target_names=[l for l in labels.values()]))
    conf_mat = confusion_matrix(tar_results,tar_test)
    
    # Plot the classification confusion matrix 
    plt.figure(figsize=(5,5))
    plt.imshow(np.array(conf_mat), cmap=plt.get_cmap('summer'), interpolation='nearest')
    for i, row in enumerate(conf_mat):
        for j, c in enumerate(row):
            plt.text(j-.2, i+.1, c, fontsize=14)

    plt.title('Confusion Matrix')
    _ = plt.xticks(range(2), [l for l in labels.values()], rotation=90)
    _ = plt.yticks(range(2), [l for l in labels.values()])
    plt.tight_layout()
    plt.show()
    
def main():
    
    series = readData()
    #vizData(series)
    classification(series)
    
    
if __name__ == '__main__': main()