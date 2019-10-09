from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import numpy as np
import matplotlib.pyplot as plt
import sys

# import matplotlib.pyplot as plt



def HTER(label, score, thred=0.5, EERtype ="hter"):
    scores = []
    FAR_SUM = []
    FRR_SUM = []
    TPR_SUM = []
    roc_EER = []
    for i in range(0, len(label)):
        tmp = []
        tmp1 = score[i]
        tmp2 = label[i]
        tmp.append(tmp1)
        tmp.append(tmp2)
        scores.append(tmp)
    #print score
    scores = sorted(scores);  # min->max
    #print scores
    sort_score = np.matrix(scores);
    #print sort_score
    minIndex = sys.maxsize;
    minDis = sys.maxsize;
    minTh = sys.maxsize;
    eer = sys.maxsize;
    alltrue = sort_score.sum(axis=0)[0,1];
    allfalse = len(scores) - alltrue;
    fa = allfalse;
    miss = 0;
    #print sort_score
    #print alltrue
    for i in range(0, len(scores)):
        # min -> max
        if sort_score[i, 1] == 1:
            miss += 1;
        else:
            fa -= 1;

        FAR=float(fa)/allfalse;
        FRR=float(miss)/alltrue;
        TPR=1-FRR
        FAR_SUM.append(FAR)
        FRR_SUM.append(FRR)
        TPR_SUM.append(TPR)
        if FAR == 0.1:
            TPR_r = TPR
            #print "when FAR = 0.1, TPR = %f"%TPR_r

        if abs(FAR - FRR) < minDis:
            minDis = abs(FAR - FRR)
            eer = min(FAR,FRR);
            minIndex = i;
            minTh = sort_score[i, 0];
    roc_auc = auc(FAR_SUM, TPR_SUM)
    #print score
    #print sort_score[:,0]
    cords = list(zip(FAR_SUM, FRR_SUM, sort_score[:,0]))
    ht = []
    ht.append(FAR_SUM)
    ht.append(FRR_SUM)
    ht = np.array(ht)
    ind = np.argmin(np.mean(ht,axis=0))
    # print (ind)
    # print (ht.shape)
    hter_min = (ht[0,ind] + ht[1,ind])/2.0
    # print (ht[:,ind])
    #print cords
    for item in cords:
        item_fpr, item_fnr, item_thd = item
        roc_EER.append(abs(item_thd - thred))
    eer_index = np.argmin(roc_EER)
    eer_fpr, eer_fnr, thd = cords[eer_index]
    hter = (eer_fpr + eer_fnr)/2
    # print (eer_fpr,eer_fnr,thd)
    print (EERtype + " " + 'HTER is :%f %%' % (hter*100))
    print (EERtype + " " + 'FAR is :%f' % eer_fpr)
    print (EERtype + " " + 'FRR is :%f' % eer_fnr)
    return hter

def EER(label, score, EERtype="eer"):
    scores = []
    FAR_SUM = []
    FRR_SUM = []
    TPR_SUM = []
    for i in range(0, len(label)):
        tmp = []
        tmp1 = score[i]
        tmp2 = label[i]
        tmp.append(tmp1)
        tmp.append(tmp2)
        scores.append(tmp)

    scores = sorted(scores);  # min->max
    sort_score = np.matrix(scores);
    minIndex = sys.maxsize;
    minDis = sys.maxsize;
    minTh = sys.maxsize;
    eer = sys.maxsize;
    alltrue = sort_score.sum(axis=0)[0,1];
    allfalse = len(scores) - alltrue;
    fa = allfalse;
    miss = 0;
    #print sort_score
    #print alltrue
    for i in range(0, len(scores)):
        # min -> max
        if sort_score[i, 1] == 1:
            miss += 1;
        else:
            fa -= 1;

        FAR=float(fa)/allfalse;
        FRR=float(miss)/alltrue;
        TPR=1-FRR
        FAR_SUM.append(FAR)
        FRR_SUM.append(FRR)
        TPR_SUM.append(TPR)
       
        if abs(FAR - FRR) < minDis:
            minDis = abs(FAR - FRR)
            eer = min(FAR,FRR)
            minTh = sort_score[i, 0]
    roc_auc = auc(FAR_SUM, TPR_SUM)

    #plt.plot(FAR_SUM, TPR_SUM, lw=1, label='ROC(area = %f)'%(roc_auc))
    #plt.plot(FAR_SUM, TPR_SUM, lw=1)
    #plt.plot([0, 1], [1, 0], '--', color=(0.6, 0.6, 0.6), label='Luck')
    #plt.savefig("test_result")
    #plt.show()

    #plt.plot(FAR_SUM, FRR_SUM, lw=1)
    #plt.plot([0, 1], [1, 0], '--', color=(0.6, 0.6, 0.6), label='Luck')
    #plt.savefig("test_result2")
    #plt.show()
    print (EERtype + " " + 'EER is :%f %%' % (eer*100))
    print (EERtype + " " + 'AUC is :%f' % roc_auc)
    print (EERtype + " " + 'thd is :%f' % minTh)

    return roc_auc, eer, minTh
