import os
import sys
import scipy.io as sio
import numpy as np
import pickle as pkl
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd

MATPATH = 'AoAmat'
PKLPATH = 'AoApkl'
RESPATH = 'AoAres'
SHAPE = ['Square','ULA']
TYPE = ['GAmp','UAmp']
NUM = [2,3,4]
SNR = ['5Pow']
dc = {'GAmp':'Data',
      'UAmp':'Data'}
    
def getData():
    for sp in SHAPE:
        for tp in TYPE:
            for nm in NUM:
                for sr in SNR:
                    matPath = '{}/{}_{}_{}_{}'.format(MATPATH,sp,tp,nm,sr)
                    pklPath = '{}/{}_{}_{}_{}.pkl'.format(PKLPATH,sp,tp,nm,sr)
                    if not os.path.exists(pklPath) and os.path.exists(matPath):
                        X = []
                        y = []
                        print(matPath)
                        for i,_ in enumerate(range(-85,86,1)):
                            pathname = matPath + '/{}'.format(_) + '/'
                            files = os.listdir(pathname)
                            for j in range(len(files)):
                                # print(pathname+files[j])
                                data = sio.loadmat(pathname+files[j])
                                X.append(data[dc[tp]])
                                y.append(i)

                        X = np.array(X)
                        y = np.array(y)

                        with open(pklPath,'wb') as f:
                            pkl.dump((X,y),f)

def testData(filename,tUsed = None):
    pklLoad = '{}/{}.pkl'.format(PKLPATH,filename)
    with open(pklLoad,'rb') as f:
            X,y = pkl.load(f)
    model = svm.SVC(gamma='auto')
    # model = MLPClassifier()
    if tUsed is None:
        tUsed = X.shape[2]
    pklSave = '{}/{}_{}.pkl'.format(RESPATH,filename,tUsed)
    if not os.path.exists(pklSave):
        print('X.shape:{}'.format(X.shape))
        
        X = X[:,:,:tUsed]
        # X = transformT(X,t1)
        X = X.reshape(X.shape[0],-1)    
        # # X = np.concatenate((X[:,200:400] - X[:,:200],X[:,400:] - X[:,200:400]),1)
        # # X = np.sum(X,2)
        X = StandardScaler().fit_transform(X)
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.2,random_state=31,shuffle=True)
        model.fit(X_train,y_train)
        y_pred = model.predict(X_test)

        with open(pklSave,'wb') as f:
            pkl.dump((y_pred,y_test),f)

def evalueData(filename):
    pklLoad = '{}/{}.pkl'.format(RESPATH,filename)
    with open(pklLoad,'rb') as f:
        y_pred,y_test = pkl.load(f)
    # plt.figure()
    # plt.imshow(confusion_matrix(y_pred,y_test))
    # plt.show()
    return np.mean(np.abs(y_pred - y_test) <= 8)


def transformT(X,t1):
    X = X[:,:t1,:]
    return X

import sklearn


def generateFrame():
    rec = {'Shape':[],'Type':[],'Num':[],'S/N':[],'Time':[],'Acc(-8 to +8)':[]}
    
    for file in os.listdir(RESPATH):
        filename,_ = file.split('.')
        sp,tp,nm,s2n,ti = filename.split('_') 
        acc = evalueData(filename)
        rec['Shape'].append(sp)
        rec['Type'].append(tp)
        rec['Num'].append(nm)
        rec['S/N'].append(s2n)
        rec['Time'].append(ti)
        rec['Acc(-8 to +8)'].append('{:.2%}'.format(acc))
    df = pd.DataFrame(rec) 
    df = df.sort_values(by=['Shape','Num','Time'])
    
    mdData = (csv2md(df.to_csv()))
    print(mdData)


def csv2md(csvData):
    ret = ''
    lines = csvData.split('\n')
    for i,line in enumerate(lines):
        if len(line) == 0 :
            break
        ret += ' | '
        words = line.split(',')
        for word in words:
            ret += word
            ret += ' | '
        ret += '\n'
        if i == 0:
            ret += ' | '
            for word in words:
                ret += '-' * max(1,len(word))
                ret += ' | '
            ret += '\n'

    return ret


if __name__ == '__main__':
    # getData()
    # n = 4

    # testData('Square_GAmp_4_5Pow')
    # tobetest = ['Square_UAmp_4_5Pow','ULA_UAmp_2_5Pow','ULA_UAmp_3_5Pow','ULA_UAmp_4_5Pow']
    # ts = [1,3,5]
    # for fn in tobetest:
    #     for t in ts:
    #         testData(fn,t)
    # evalueData('ULA_UAmp_4_5Pow',1)
    # print(sklearn.__version__)
    generateFrame()