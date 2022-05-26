import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import tree
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler, NearMiss
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler, NearMiss
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_text
from sklearn.metrics import classification_report
from collections import Counter
from sklearn import svm
import math
import joblib
import pickle
import os

constant_random_state = 42

def apply_linear_regression(x_train,y_train,x_test,y_test,file_name) :

    logreg = LogisticRegression(max_iter = 3000 ,solver='lbfgs')

    logreg.fit(x_train,y_train)
    print(logreg.coef_)
    y_pred=logreg.predict(x_test)
    logregCf = confusion_matrix(y_test, y_pred)

    fn = "MemoizedTrainResults/LR_" + file_name
    fn = fn.replace("csv","joblib")
    joblib.dump(logreg, fn)
    
    

    return [accuracy_score(y_test,y_pred),logregCf]

def apply_discession_tree(x_train,y_train,x_test,y_test,file_name) :
    dec_tree = DecisionTreeClassifier(
        random_state = constant_random_state ,max_features = int(math.log2(int(64/2))))

    dec_tree.fit(x_train,y_train)
    y_pred = dec_tree.predict(x_test)
    decTreeCf = confusion_matrix(y_test, y_pred)

    fn = "MemoizedTrainResults/DT_" + file_name
    fn = fn.replace("csv","joblib")

    joblib.dump(dec_tree, fn)

    
    return [accuracy_score(y_test,y_pred),decTreeCf]

def apply_svm_linear_kernel(x_train,y_train,x_test,y_test,file_name) :

    clf = svm.SVC()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    svmCf= confusion_matrix(y_test, y_pred)

    fn = "MemoizedTrainResults/SVM_" + file_name
    fn = fn.replace("csv","joblib")
    joblib.dump(clf, fn)
    

    return [(metrics.accuracy_score(y_test, y_pred)),svmCf]


cleaned_data_list = os.listdir("cleanedData")

mx_linear_before_normalization = 0.0
mx_linear_after_normalization = 0.0

mx_discession_tree_before_normalization = 0.0
mx_discession_tree_after_normalization = 0.0

mx_svm_before_normalization = 0.0
mx_svm_after_normalization = 0.0

i = 1
for file_name in cleaned_data_list :
    if file_name == "cleaned_data.csv" :
        continue

    df = pd.read_csv("cleanedData/"+file_name)
    df['class'] = pd.to_numeric(df['class'])

    X = df.drop('class',axis=1)    
    Y = df['class']

    ss = StandardScaler()
    df = pd.DataFrame(ss.fit_transform(df),columns = df.columns)
    
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3 , random_state = constant_random_state)

    ros = RandomOverSampler(random_state=0)
    X_train_ros, y_train_ros= ros.fit_resample(x_train, y_train)
    # print(sorted(Counter(y_train_ros).items()))

    rf = RandomForestClassifier()
    ros_model = rf.fit(X_train_ros, y_train_ros)
    ros_prediction = ros_model.predict(x_test)

    smote = SMOTE(random_state=constant_random_state)
    # print(classification_report(y_test, ros_prediction))
    X_train_smote, Y_train_smote= smote.fit_resample(x_train, y_train)

    

    x_train = X_train_smote
    y_train = Y_train_smote

    linear_regression_accuracy = apply_linear_regression(x_train,y_train,x_test,y_test,file_name)[0]
    discession_tree_accuracy = apply_discession_tree(x_train,y_train,x_test,y_test,file_name)[0]
    svm_linear_kernel_accuracy = apply_svm_linear_kernel(x_train,y_train,x_test,y_test,file_name)[0]

    print(file_name)
    print("logistic regression : ",linear_regression_accuracy)
    print("DecisionTreeClassifier : ",discession_tree_accuracy)
    print("SVM kernel linear : ",svm_linear_kernel_accuracy)
    print('\n')

    if (i == 1) :
        mx_linear_before_normalization = linear_regression_accuracy
        mx_discession_tree_before_normalization = discession_tree_accuracy
        mx_svm_before_normalization = svm_linear_kernel_accuracy
    else :
        mx_linear_after_normalization = linear_regression_accuracy
        mx_discession_tree_after_normalization = discession_tree_accuracy
        mx_svm_after_normalization = svm_linear_kernel_accuracy

    
    i += 1

print("done classifing")
print("max logistic before normalization : ",mx_linear_before_normalization)
print("max logistic after normalization : ",mx_linear_after_normalization)
print('\n')
print("max discession tree before normalization : ",mx_discession_tree_before_normalization)
print("max discession tree after normalization : ",mx_discession_tree_after_normalization)
print('\n')
print("max svm before normalization : ",mx_svm_before_normalization)
print("max svm after normalization : ",mx_svm_after_normalization)



results = str(mx_linear_before_normalization) + ','+str(mx_linear_after_normalization) + '\n'
results += str(mx_discession_tree_before_normalization) +','+ str(mx_discession_tree_after_normalization) + '\n'
results += str(mx_svm_before_normalization) + ','+str(mx_svm_after_normalization) + '\n'
file = open("results.txt",'w')
file.write(results)
file.close()

input()