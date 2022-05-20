import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
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
from sklearn.metrics import classification_report
from collections import Counter
from sklearn import svm
import joblib
import pickle
import os

constant_random_state = 42

def apply_linear_regression(x_train,y_train,x_test,y_test,file_name) :

    logreg = LogisticRegression(max_iter = 3000)

    logreg.fit(x_train,y_train)
    fn = "MemoizedTrainResults/LR_" + file_name
    fn = fn.replace("csv","joblib")
    joblib.dump(logreg, fn)
    
    y_pred=logreg.predict(x_test)

    return logreg.score(x_train,y_train)

def apply_discession_tree(x_train,y_train,x_test,y_test,file_name) :
    clf_entropy = DecisionTreeClassifier(
        criterion = "entropy",
        random_state = constant_random_state ,
        max_depth = 3,
        min_samples_leaf = 5)

    clf_entropy.fit(x_train,y_train)
    fn = "MemoizedTrainResults/DT_" + file_name
    fn = fn.replace("csv","joblib")

    joblib.dump(clf_entropy, fn)

    y_pred_en = clf_entropy.predict(x_test)
    return accuracy_score(y_test,y_pred_en)

def apply_svm_linear_kernel(x_train,y_train,x_test,y_test,file_name) :

    clf = svm.SVC(kernel='linear')
    clf.fit(x_train, y_train)
    fn = "MemoizedTrainResults/SVM_" + file_name
    fn = fn.replace("csv","joblib")
    joblib.dump(clf, fn)
    y_pred = clf.predict(x_test)

    return (metrics.accuracy_score(y_test, y_pred))


cleaned_data_list = os.listdir("cleanedData")
for file_name in cleaned_data_list :
    if file_name == "cleaned_data.csv" :
        continue

    df = pd.read_csv("cleanedData/"+file_name)

    Y = df['class']
    X = df.drop('class',axis=1)

    for i in range(len(Y)) :
        Y[i] = int(Y[i])

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

    print(file_name)
    print("Linear regression : ",apply_linear_regression(x_train,y_train,x_test,y_test,file_name))
    print("DecisionTreeClassifier : ",apply_discession_tree(x_train,y_train,x_test,y_test,file_name))
    print("SVM kernel linear : ",apply_svm_linear_kernel(x_train,y_train,x_test,y_test,file_name))
    print('\n')

input()