'''
Created on Dec 17, 2016
Class imported from : https://github.com/christopher-dG/multiDimensional-DTW/blob/master/multiDTW.py
Author: Chris de Graaf

My Modifications:
 - removed file logging
 - removed 1-NN code
 - adapted data structures
 - 
'''

import numpy as np

def gen_dist_matrix(a, b):

    # computes the distance matrix for two k-dimensional matrices
    # parameters: a, b = k-dimensional matrices 
    # return: dist = distance matrix for a and b

    dist = np.ndarray((len(a), len(b)))
    dist.fill(float('inf'))
    for i in range(len(dist)): # for each row
        for j in range(len(dist[i])): # for each column
            # sum the absolute differences of each dimension
            dimSum = 0
            for k in range(len(a[i])):
                dimSum += abs(a[i][k] - b[j][k])                
                dist[i][j] = dimSum
    return dist

def gen_cost(distance):

    # generates the matrix of minimum cost to reach any point in the graph
    # parameter: distance = distance matrix
    # return: cost = cost matrix (same dimensions as distance)

    cost = np.zeros(distance.shape)
    cost[0][0] = distance[0][0]
    # the easy rows
    for i in range(1, len(cost[0])):
        cost[0][i] = distance[0][i] + cost[0][i-1]
    for i in range(1, len(cost)):
        cost[i][0] = distance[i][0] + cost[i-1][0]
    # fill the rest of the matrix
    for i in range(1, len(cost)):
        for j in range(1, len(cost[i])):
            cost[i][j] = min(cost[i-1][j-1], cost[i-1][j], cost[i][j-1]) + distance[i][j]
    return cost

def gen_path(cost):

    # finds the lowest-cost path from corner to corner
    # parameter: cost = cost matrix from gen_cost
    # return: path = shortest path in the form of a list of coordinates

    path = [[len(cost) - 1, len(cost[0]) - 1]]
    i = len(cost) - 1
    j = len(cost[0]) - 1
    while i>0 and j>0:
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            if cost[i-1][j] == min(cost[i-1][j-1], cost[i-1][j], cost[i][j-1]):
                i -= 1
            elif cost[i][j-1] == min(cost[i-1][j-1], cost[i-1][j], cost[i][j-1]):
                j -= 1
            else:
                i -= 1
                j -= 1
        path.append([i, j])
    if path[len(path) - 1] != [0, 0]:
        path.append([0, 0])
    return path

def gen_total_cost(path, distance):

    # finds the total cost of the shortest path
    # parameters: path = shortest path from gen_path, distance = distance matrix from gen_dist_matrix
    # return: cost = total cost to follow path

    cost = 0
    for [y, x] in path:
        cost = cost + distance[y][x]
    return cost

def compare(a, b):
    
    # compare two time series, low cost = higher similarity
    # parameters: a, b: two m X n arrays representing different time series
    # return: total_cost = measure of similarity for the inputs, lower is better

    distance = gen_dist_matrix(a, b)
    cost = gen_cost(distance)
    path = gen_path(cost)
    total_cost = gen_total_cost(path, distance)
    return total_cost

def predict(test, train_tuples, K):
    scores = []
    for f in train_tuples:
        if f is not test:
            scores.append((f[0], compare(test[2], f[2]), f[1])) # (id, score, target)
    scores = sorted(scores, key=lambda x: x[1]) # sort in ascending order of scores
    KNN_predictions = scores[:K] # take the K lowest scores
    changed, unchanged, KNN_proba = 0, 0, 0.0
    for p in KNN_predictions:
        if p[2] == 1:
            changed += 1
        else:
            unchanged += 1
    if changed > unchanged:
        KNN_prediction = 1
        KNN_proba = changed/K
    else:
        KNN_prediction = -1
        KNN_proba = unchanged/K
    
    return test[0], int(KNN_prediction), float(KNN_proba)
