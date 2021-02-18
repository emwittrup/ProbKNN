import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


# Based on prop within labels
def probKnn(Xtrain, Ytrain, Xtest, k, d = None):
    if d == None: 
        d = len(Xtrain.columns)
        
    #todo, pick columns randomly if d is specified
    output = pd.Series(0, index = Xtest.index)
    
    #Extract all classes
    classes = Ytrain.unique()

    for r_idx, row in Xtest.iterrows():
        z_mat_all = pd.Series()
        for cls in classes:
            #Isolate only training data for this class
            lab_mat = Xtrain.loc[Ytrain==cls].iloc[:,0:d]
            #Create data frame of zscores for each element relative to it's column for data within this class
            z_mat = pd.DataFrame(stats.zscore(lab_mat), index=lab_mat.index, columns=lab_mat.columns)
            #Calculate z scores of all cols in row of test data
            z_test = (row.iloc[0:d] - lab_mat.mean())/ lab_mat.std()
            #Create data frame of z_test that is the same size as training data for this class
            z_test_df = pd.DataFrame(np.repeat([z_test.values], len(lab_mat), axis = 0), index=lab_mat.index, columns = lab_mat.columns)
            #Subtract this row's zscores from this class' training data's zscores
            z_test_mat = z_mat - z_test_df
            #Take absolute sum of all differences in zscores - smaller = closer match
            z_test_mat = z_test_mat.abs().sum(axis=1)
  
            #Repeat for all classes
            z_mat_all = z_mat_all.append(z_test_mat)
        #Pick k smallest numbers from summed difference in zscores
        z_nn = z_mat_all.nsmallest(k).index
        #Take the mode of classes for k closest points
        mode, _ = stats.mode(Ytrain[z_nn].values)
        output.loc[r_idx] = mode[0]
    return output

def getPerformanceMetrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    tnr = tn / (tn+fp)
    tpr = tp / (tp + fn)
    acc = np.sum(y_pred == y_true)/len(y_pred)
    ppv = tp / (tp + fp)

    print(f"Accuracy: {acc}")
    print(f"Specificity: (TNR): {tnr}")
    print(f"Sensitivity: (TPR): {tpr}")
    print(f"Precision: (PPV): {ppv}")
