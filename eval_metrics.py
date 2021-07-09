import operator
import numpy as np
from sklearn.model_selection import KFold
from scipy import interpolate
import pandas as pd
import math
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score

def one_hot_transfer(label,class_num=7):
    return np.eye(class_num)[label]

def metric_for_Exp(gt,pred,class_num=7):
    # compute_acc
    acc = accuracy_score(gt,pred)
    # compute_F1
    gt = one_hot_transfer(gt,class_num)
    pred = one_hot_transfer(pred,class_num)
    F1 = []
    for i in range(class_num):
        gt_ = gt[:,i]
        pred_ = pred[:,i]
        F1.append(f1_score(gt_.flatten(), pred_))
    F1_mean = np.mean(F1)
    return F1_mean,acc,F1

def metric_for_AU(gt,pred,class_num=12):
    #compute_F1,acc
    F1 = []
    gt = np.array(gt)
    pred = np.array(pred)
    cate_acc = np.sum((np.array(pred>0,dtype=np.float))==gt)/(gt.shape[0]*gt.shape[1])
    # print(pred.shape)
    for type in range(class_num):
        gt_ = gt[:, type]
        pred_ = pred[:, type]
        new_pred = ((pred_ >= 0.5) * 1).flatten()
        F1.append(f1_score(gt_.flatten(), new_pred))

    F1_mean = np.mean(F1)

    #compute total acc
    counts = gt.shape[0]
    accs = 0
    for i in range(counts):
        pred_label = ((pred[i,:] >= 0.5) * 1).flatten()
        gg = gt[i].flatten()
        j = 0
        for k in range(12):
            if int(gg[k]) == int(pred_label[k]):
                    j+=1
        if j==12:
            accs+=1
    acc = 1.0*accs/counts
    return F1_mean,acc,F1,cate_acc

def metric_for_AU_mlce(gt,pred,class_num=12):
    #compute_F1,acc
    F1 = []
    gt = np.array(gt)
    pred = np.array(pred)
    cate_acc = np.sum((np.array(pred>0,dtype=np.float))==gt)/(gt.shape[0]*gt.shape[1])
    
    for type in range(class_num):
        gt_ = gt[:, type]
        pred_ = pred[:, type]
        new_pred = ((pred_ >= 0.) * 1).flatten()
        F1.append(f1_score(gt_.flatten(), new_pred))
    F1_mean = np.mean(F1)

    #compute total acc
    counts = gt.shape[0]
    accs = 0
    for i in range(counts):
        pred_label = ((pred[i,:] >= 0.) * 1).flatten()
        gg = gt[i].flatten()
        j = 0
        for k in range(12):
            if int(gg[k]) == int(pred_label[k]):
                    j+=1
        if j==12:
            accs+=1
    acc = 1.0*accs/counts

    return F1_mean,acc,F1,cate_acc

def CCC_score(x, y):
    vx = x - np.mean(x)
    vy = y - np.mean(y)
    rho = np.sum(vx * vy) / (np.sqrt(np.sum(vx**2)) * np.sqrt(np.sum(vy**2)))
    x_m = np.mean(x)
    y_m = np.mean(y)
    x_s = np.std(x)
    y_s = np.std(y)
    ccc = 2*rho*x_s*y_s/(x_s**2 + y_s**2 + (x_m - y_m)**2)
    return ccc

def metric_for_VA(gt_V,gt_A,pred_V,pred_A):
    ccc_V,ccc_A = CCC_score(gt_V,pred_V),CCC_score(gt_A,pred_A)
    return ccc_V,ccc_A

def f1_score_max_for_AU_one_class(gt, pred, thresh,type=0):
    gt = gt[:,type]
    pred = pred[:,type]
    P = []
    R = []
    ACC = []
    F1 = []
    for i in thresh:
        new_pred = ((pred >= i) * 1).flatten()
        P.append(precision_score(gt.flatten(), new_pred))
        R.append(recall_score(gt.flatten(), new_pred))
        ACC.append(accuracy_score(gt.flatten(), new_pred))
        F1.append(f1_score(gt.flatten(), new_pred))

    F1_MAX = max(F1)
    if F1_MAX < 0 or math.isnan(F1_MAX):
        F1_MAX = 0
        F1_THRESH = 0
        accuracy = 0
    else:
        idx_thresh = np.argmax(F1)
        F1_THRESH = thresh[idx_thresh]
        accuracy = ACC[idx_thresh]
    return F1,F1_MAX,F1_THRESH,accuracy

def f1_score_max(gt, pred, thresh,c=12):
    F1_s = []
    F1_t = []
    ACC = []
    from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
    new_pred = ((pred >= 0.5) * 1)[:,1]
    for i in range(c):
        F1, F1_MAX, F1_THRESH,acc = f1_score_max_for_one_class(gt,pred,thresh,i)
        F1_s.append(F1_MAX)
        F1_t.append(F1_THRESH)
        ACC.append(acc)
    return F1_s,F1_t,ACC
