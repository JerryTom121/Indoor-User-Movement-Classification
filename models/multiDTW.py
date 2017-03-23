'''
Created on Dec 16, 2016
Python 3.5.2
@author: Nidhalios
'''

from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score


def k_fold_generation(X, K, randomise=False):
    """
	Generates K (training, validation) pairs from the items in X.
	Each pair is a partition of X, where validation is an iterable
	of length len(X)/K. So each training iterable is of length (K-1)*len(X)/K.
	If randomise is true, a copy of X is shuffled before partitioning,
	otherwise its order is preserved in training and validation.
	"""
    if randomise:
        from random import shuffle;
        X = list(X);
        shuffle(X)
    for k in iter(range(K)):
        training = [x for i, x in enumerate(X) if i % K != k]
        validation = [x for i, x in enumerate(X) if i % K == k]
        yield training, validation

def predict(test, train_tuples, K):
    scores = []
    print(len(test[2]))
    for f in train_tuples:
        if f is not test:
            distance, path = fastdtw(test[2], f[2], dist=euclidean)
            scores.append((f[0], distance, f[1]))  # (id, score, target)
    scores = sorted(scores, key=lambda x: x[1])  # sort in ascending order of scores
    KNN_predictions = scores[:K]  # take the K lowest scores
    changed, unchanged, KNN_proba = 0, 0, 0.0
    for p in KNN_predictions:
        if p[2] == 1:
            changed += 1
        else:
            unchanged += 1
    if changed > unchanged:
        KNN_prediction = 1
        KNN_proba = changed / K
    else:
        KNN_prediction = -1
        KNN_proba = unchanged / K

    return test[0], int(KNN_prediction), float(KNN_proba)


def cross_validation(data, K=5):
    j = 1
    acc_tab = []
    prec_tab = []
    roc_auc_tab = []
    for training, validation in k_fold_generation(data, K=K):
        # Train a K-nearest neighbor classifier that uses Dynamic Time Warping (DTW) to
        # evaluate distance between two given Multivariate Time Series
        KNN_NEIGHBORS = 2
        results = []
        print('Round : {}'.format(j))
        for i in range(len(validation)):
            print('{}/{}...'.format(i + 1, len(validation)))
            id, pred, proba = predict(validation[i], training, KNN_NEIGHBORS)
            results.append([id, pred, proba])
        acc_tab.append(accuracy_score([x[1] for x in validation], [row[1] for row in results]))
        prec_tab.append(precision_score([x[1] for x in validation], [row[1] for row in results]))
        roc_auc_tab.append(roc_auc_score([x[1] for x in validation], [row[2] for row in results]))
        j += 1

    print('KFold Cross Validation Metrics :')
    print('Accuracy Score : '.format(sum(acc_tab) / len(acc_tab)))
    print('Precision Score : '.format(sum(prec_tab) / len(prec_tab)))
    print('ROC AUC Score : '.format(sum(roc_auc_tab) / len(roc_auc_tab)))
