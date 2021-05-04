# -*- coding: utf-8 -*-
#Data Loading and manipulation
import numpy as np
import pandas as pd
#Plotting
import matplotlib.pyplot as plt
#Data Preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split 
#Machine Learning Models
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
#Evaluation
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import itertools
#%%
#Loading dataset into pandas dataframe and printer first 5 rows
cc_data = pd.read_csv("data-master//credit-card//creditcard.csv")
print (cc_data.head())
#%%
#counting and printing number of transactions, number of fraudulents transactions, number of non-fraudulent transactions
#and last calculating the percentage of fraudulent transactions.
num_total = len(cc_data)
num_fraud = len(cc_data[cc_data.Class == 1])
num_not_fraud = len(cc_data[cc_data.Class == 0])
percent_fraud = round(num_fraud/num_not_fraud*100, 2)
print("Total number of transactions:", num_total)
print("\n Number of fraudulent transactions:", num_fraud)
print("\n Number of non-fraudulent transactions:", num_not_fraud)
print("\n Fraudulent transactions make up", percent_fraud, "percent")
#%%
#we plot the different features to see degree of variance
fig = plt.figure(figsize = (20, 15))
ax = fig.subplots(6, 5, sharex=True)

for x in range(0, 6): #using 2 for-in loops, we can create a grid using x and y to determine position of z column feature values
    for y in range(0, 5):
        if x == 0:
            z = x+1*y+1 #the first row of the plot grid. +1 to start at 1 instead of 0, since we dont want to plot time.
        elif x > 0:
            z = 1+x*len(range(0, 5))+y #every new row counts the counts the range of the row and times it by x (the vertical position)
            if z >= 30:
                break #only want to show column 1-29
        ax[x, y].plot(cc_data[cc_data.columns[z]]) #x is vertical position, y is horizontal position and z is the numerical determinater of a column
        ax[x, y].set_title(cc_data.columns[z])
plt.savefig('feature_grid.png') #save as .png will be in appendixes
plt.show()
   #%% A different way of plotting the plotgrid. The for-in loops above essentially does this automatically     
#ax[0, 0].plot(dataframe.V1)
#ax[0, 0].set_title('V1')
#ax[0, 1].plot(dataframe.V2)
#ax[0, 1].set_title('V2')
#ax[0, 2].plot(dataframe.V3)
#ax[0, 2].set_title('V3')
#ax[0, 3].plot(dataframe.V4)
#ax[0, 3].set_title('V4')
#ax[0, 4].plot(dataframe.V5)
#ax[0, 4].set_title('V5')
#ax[1, 0].plot(dataframe.V6)
#ax[1, 0].set_title('V6')
#ax[1, 1].plot(dataframe.V7)
#ax[1, 1].set_title('V7')
#ax[1, 2].plot(dataframe.V8)
#ax[1, 2].set_title('V8')
#ax[1, 3].plot(dataframe.V9)
#ax[1, 3].set_title('V9')
#ax[1, 4].plot(dataframe.V10)
#ax[1, 4].set_title('V10')
#ax[2, 0].plot(dataframe.V11)
#ax[2, 0].set_title('V11')
#ax[2, 1].plot(dataframe.V12)
#ax[2, 1].set_title('V12')
#ax[2, 2].plot(dataframe.V13)
#ax[2, 2].set_title('V13')
#ax[2, 3].plot(dataframe.V14)
#ax[2, 3].set_title('V14')
#ax[2, 4].plot(dataframe.V15)
#ax[2, 4].set_title('V15')
#ax[3, 0].plot(dataframe.V16)
#ax[3, 0].set_title('V16')
#ax[3, 1].plot(dataframe.V17)
#ax[3, 1].set_title('V17')
#ax[3, 2].plot(dataframe.V18)
#ax[3, 2].set_title('V18')
#ax[3, 3].plot(dataframe.V19)
#ax[3, 3].set_title('V19')
#ax[3, 4].plot(dataframe.V20)
#ax[3, 4].set_title('V20')
#ax[4, 0].plot(dataframe.V21)
#ax[4, 0].set_title('V21')
#ax[4, 1].plot(dataframe.V22)
#ax[4, 1].set_title('V22')
#ax[4, 2].plot(dataframe.V23)
#ax[4, 2].set_title('V23')
#ax[4, 3].plot(dataframe.V24)
#ax[4, 3].set_title('V24')
#ax[4, 4].plot(dataframe.V25)
#ax[4, 4].set_title('V25')
#ax[5, 0].plot(dataframe.V26)
#ax[5, 0].set_title('V26')
#ax[5, 1].plot(dataframe.V27)
#ax[5, 1].set_title('V27')
#ax[5, 2].plot(dataframe.V28)
#ax[5, 2].set_title('V28')
#ax[5, 3].plot(dataframe.Amount)
#ax[5, 3].set_title('Amount')
#%%
#as the feature 'Amount' has high variance, we decide to normalize these values using StandardScaler
sc = StandardScaler()
feat_amount = cc_data['Amount'].values
cc_data['Amount'] = sc.fit_transform(feat_amount.reshape(-1, 1))
print (cc_data['Amount'].head)
#%%
#We split the data into target: 'Class', which we want to predict, and the rest of the relevant features
#dropping 'time' column, as it is just a linear increment of time and deemed unimportant
featurenames = cc_data.iloc[:, 1:30].columns
targetname = cc_data.iloc[:1, 30: ].columns
features = cc_data[featurenames].values
target = cc_data[targetname].values
print("These are our features: \n", featurenames, "\n And our feature values: \n", features)
print("This is our target: \n", targetname, "\n And our target values: \n", target)
#Run train_test_split to prepare data for ML models
xtrain, xtest, ytrain, ytest = train_test_split(features, target, train_size=0.25, random_state=1)
#%% The processing time of the code in this cell is very long.
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
#we prepare our different ML models, fitting the data and running predictions
#Some models use specific adjustable parameters. I run loops for iteration of the parameters to find parameters with highest accuracy.
#Decision Tree
criteria = ['gini', 'entropy']
for crit in criteria:
    for depth in range(2, 10):
        dtree = DecisionTreeClassifier(max_depth = depth, criterion = crit)
        dtree.fit(xtrain, ytrain)
        dtree_pred = dtree.predict(xtest)
        print("Decision Tree Accuracy Score with paremeters Criterion = ",crit,"\n Max Depth = ",depth,":{}".format(accuracy_score(ytest, dtree_pred)),"\n")
        print("Decision Tree F1 Score with parameters Criterion = ",crit,"\n Max Depth = ",depth,":{}".format(f1_score(ytest, dtree_pred)),"\n")
#K-nearest Neighbors (this takes a while)
r_ytrain = np.ravel(ytrain)
for n in range(3, 10, 2):
    knn = KNeighborsClassifier(n_neighbors = n)
    knn.fit(xtrain, r_ytrain)
    knn_pred = knn.predict(xtest)
    print("K-Nearest Neighbors Accuracy Score with parameter n_neighbors = ",n,":{}".format(accuracy_score(ytest, knn_pred)),"\n")
    print("K-Nearest Neighbors F1 Score with parameter n_neighbors = ",n,":{}".format(f1_score(ytest, knn_pred)),"\n")
    
#Logistic Regression
lr = LogisticRegression()
lr.fit(xtrain, r_ytrain)
lr_pred = lr.predict(xtest)
print("Logistic Regression Accuracy Score: {}".format(accuracy_score(ytest, lr_pred)),"\n")
print("Logistic Regression F1 Score: {}".format(f1_score(ytest, lr_pred)),"\n")
#LinearSVC
lsvc = LinearSVC()
lsvc.fit(xtrain, r_ytrain)
lsvc_pred = lsvc.predict(xtest)
print("LinearSVC Accuracy Score: {}".format(accuracy_score(ytest, lsvc_pred)),"\n")
print("LinearSVC F1 Score: {}".format(f1_score(ytest, lsvc_pred)),"\n")
#Random Forest Tree
for d in range(2, 10):
    rft = RandomForestClassifier(max_depth = d)
    rft.fit(xtrain, r_ytrain)
    rft_pred = rft.predict(xtest)
    print("Random Forest Tree Accuracy Score with parameter Max Depth =",d,":{}".format(accuracy_score(ytest, rft_pred)),"\n")
    print("Random Forest Tree F1 Score with parameter Max Depth = ",d,":{}".format(f1_score(ytest, rft_pred)),"\n")


#%% Remaking the models using best parameters to be used in model comparison
# DecisionTree with highest accuracy has parameters max_depth = 4, criterion = entropy
b_dtree = DecisionTreeClassifier(max_depth = 4, criterion = 'entropy')
b_dtree.fit(xtrain, ytrain)
b_dtree_pred = b_dtree.predict(xtest)
#K-Nearest Neighbors with highest accuracy has parameter n_neighbors = 5
b_knn = KNeighborsClassifier(n_neighbors = 5)
b_knn.fit(xtrain, r_ytrain)
b_knn_pred = b_knn.predict(xtest)
#RandomForestClassifier with highest acurract has paremeter max_depth = 9
b_rft = RandomForestClassifier(max_depth = 9)
b_rft.fit(xtrain, r_ytrain)
b_rft_pred = b_knn.predict(xtest)
#%% The processing time in this cell is very long.
#Accuracy Score Comparison Barplot
#Create series of data consisting of accuracy scores, now with best parameters
plot_data = {'Decision Tree': accuracy_score(ytest, b_dtree_pred), 'K-Nearest Neighbors': accuracy_score(ytest, b_knn_pred), 'Logistic Regression': accuracy_score(ytest, lr_pred), 'LinearSVC': accuracy_score(ytest, lsvc_pred), 'Random Forest': accuracy_score(ytest, b_rft_pred)}
#Create a categorical barplot, use series to get categories (x-axis) and values (y-axis)
cat = list(plot_data.keys())
val = list(plot_data.values())
fig1, ax1 = plt.subplots(figsize=(10, 10))
ax1.bar(cat, val)
ax1.set_ylim([0.999, 0.9995]) #custom limiter to be able to see differences between accuracy scores of different models
fig1.suptitle('Accuracy Score')
for i, v in enumerate(val):
    ax1.text(i-0.5, v+0.000005, str(v), va='center') #Text above each bar showing the value. i and v are using for position of text (correct x and y axis, adjustment for readability)
plt.savefig('Accuracy_score.png')
plt.show()
#%% The processing time in this cell is very long.
#F1 Score Comparison Barplot. Same as cell above, but with F1 scores instead.
plot_data0 = {'Decision Tree': f1_score(ytest, b_dtree_pred), 'K-Nearest Neighbors': f1_score(ytest, b_knn_pred), 'Logistic Regression': f1_score(ytest, lr_pred), 'LinearSVC': f1_score(ytest, lsvc_pred), 'Random Forest': f1_score(ytest, b_rft_pred)}
cat0 = list(plot_data0.keys())
val0 = list(plot_data0.values())
fig0, ax0 = plt.subplots(figsize=(10, 10))
ax0.bar(cat0, val0)
ax0.set_ylim([0.65, 0.85])
fig0.suptitle('F1 Score')
for i0, v0 in enumerate(val0):
    ax0.text(i0-0.5, v0+0.005, str(v0), va='center')
plt.savefig('F1_score.png')
plt.show()
#%% The processing time in this cell is fairly long.
#Confusion Matrix
def plot_cm(cm, classes, title, cmap=plt.cm.Reds): #function for plotting of the confusion matrix.
    title = title,"Confusion Matrix" #title of model from which function is called + confusion model
    plt.imshow(cm, interpolation = 'nearest', cmap = cmap) #show plot as image without interpolating pixels
    plt.title(title) #set title of plot based on line 223
    plt.colorbar() #colorbar on rightside of image showing numerical values of color gradient
    inc = np.arange(len(classes)) #number of tick-increments i.e. height and width of matrix. classes is passed when calling function.
    plt.xticks(inc, classes, rotation = 45) #creating x axis based on lenght of classes passed when function called. Rotated to be horizontal
    plt.yticks(inc, classes) #Creating y axis same way.
    t = cm.max()/2. #variable for when color (red in this case) is halfway to max
    for ik, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, ik, format(cm[ik, j]), horizontalalignment = 'center', color = 'white' if cm[ik, j] > t else 'black') #loop to determine position and color text using itertools and t-variable
    plt.tight_layout()
    plt.ylabel('Actual Value')
    plt.xlabel('Model Predictions')
#Creating the matrix for each model
dtree_matrix = confusion_matrix(ytest, b_dtree_pred, labels = [0, 1]) 
knn_matrix = confusion_matrix(ytest, b_knn_pred, labels = [0, 1])
lr_matrix = confusion_matrix(ytest, lr_pred, labels = [0,1])
lsvc_matrix = confusion_matrix(ytest, lsvc_pred, labels = [0, 1])
rft_matrix = confusion_matrix(ytest, b_rft_pred, labels = [0, 1])
plt.rcParams['figure.figsize'] = (6,6)
#calling plotting function passing model matrixes created above, classes and a title.
dtree_cm = plot_cm(dtree_matrix, classes = ['Not Fraud(0)', 'Fraud(1)'], title = 'Decision Tree')
plt.savefig('dtree_cm.png')
plt.show()
knn_cm = plot_cm(knn_matrix, classes = ['Not Fraud(0)', 'Fraud(1)'], title = 'K-Nearest Neighbors')
plt.savefig('knn_cm.png')
plt.show()
lr_cm = plot_cm(lr_matrix, classes = ['Not Fraud(0)', 'Fraud(1)'], title = 'Logistic Regression')
plt.savefig('lr_cm.png')
plt.show()
lsvc_cm = plot_cm(lsvc_matrix, classes = ['Not Fraud(0)', 'Fraud(1)'], title = 'LinearSVC')
plt.savefig('lsvc.png')
plt.show()
rft_cm = plot_cm(rft_matrix, classes = ['Not Fraud(0)', 'Fraud(1)'], title = 'Random Forest')
plt.savefig('rft_cm.png')
plt.show()