import pandas as pd
import sklearn as sklearn
import os
constant_random_state = 42

def apply_linear_regression(x_train,y_train,x_test,y_test) :

    logreg = LogisticRegression(max_iter = 3000)

    logreg.fit(x_train,y_train)
    y_pred=logreg.predict(x_test)

    return logreg.score(x_train,y_train)

def apply_discession_tree(x_train,y_train,x_test,y_test) :
    clf_entropy = DecisionTreeClassifier(
        criterion = "entropy",
        random_state = constant_random_state ,
        max_depth = 3,
        min_samples_leaf = 5)

    clf_entropy.fit(x_train,y_train)

    y_pred_en = clf_entropy.predict(x_test)
    return accuracy_score(y_test,y_pred_en)

def apply_svm_linear_kernel(x_train,y_train,x_test,y_test) :

    clf = svm.SVC(kernel='linear')
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    return (metrics.accuracy_score(y_test, y_pred))


cleaned_data_list = os.listdir("cleanedData")
for file_name in cleaned_data_list :
    if file_name == "cleaned_data.csv" :
        continue

    df = pd.read_csv("cleanedData/"+file_name)

    ss = StandardScaler()
    df = pd.DataFrame(ss.fit_transform(df),columns = df.columns)

    X = df.drop('class',axis=1)
    Y = df['class']

    for i in range(len(Y)) :
        Y[i] = int(Y[i])

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3 , random_state = constant_random_state)


    print(file_name)
    print("Linear regression : ",apply_linear_regression(x_train,y_train,x_test,y_test))
    print("DecisionTreeClassifier : ",apply_discession_tree(x_train,y_train,x_test,y_test))
    print("SVM kernel linear : ",apply_svm_linear_kernel(x_train,y_train,x_test,y_test))
    print('\n')
