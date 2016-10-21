## Always sync this file with the following command from your terminal:
## "cp /Users/csaikia/Dropbox/Monodeep/SML_ass_2/hw2/classifier_2.py /Users/csaikia/SML_assignment/Machine_Learning_questions/KNN_HOT_OR_NOT/KNN_classifier.py"
from __future__ import division
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt
import time
import copy

start_time = time.time()

## Save all the information from the faces.mat file
data=sio.loadmat('faces.mat')
testlabels=data['testlabels']
testlabels=testlabels[:,0]
trainlabels=data['trainlabels']
trainlabels=trainlabels[:,0]
testdata=data['testdata']
traindata=data['traindata']

## Learning function outputs traing label and test label
trainlabels_learner=[]
testlabels_learner=[]
cosine_labels_test=[]
cosine_labels_train=[]
cosine_dist=[]

def cos_dist_terow_trrow(data_row,train_row):
    denom=sqrt(np.sum(np.square(data_row)) * np.sum(np.square(train_row)))
    dist= (1- (np.dot(data_row,np.transpose(train_row))/denom))
    return dist

# Write function for pre-computed distances between all pairs of points
def cosine_distance(data_row):
    del cosine_dist[:]
    i=0
    for train_row in traindata:
        cos_dist_rc=cos_dist_terow_trrow(data_row,train_row)
        cosine_dist.append(cos_dist_rc)
        i=i+1

# Write function to find cosine distance, my k-NN learner function
#def learner_function(arr,type_of_data,k):
def learner_function(type_of_data):
    print("In learner")
    del cosine_dist[:]
    j=0
    if type_of_data=="test":
        data_matrix=testdata.copy()
    elif type_of_data=="train":
        data_matrix=traindata.copy()
    for one_row in data_matrix:
        cosine_distance(one_row)
        if type_of_data=="test":
            cosine_labels_test.append(copy.deepcopy(cosine_dist))
        elif type_of_data=="train":
            cosine_labels_train.append(copy.deepcopy(cosine_dist))
        j=j+1

def sort_store_index(row):
    return np.argsort(row)

def k_nn_algorithm(type_of_data,k):
    del trainlabels_learner[:]
    del testlabels_learner[:]
    print "In k_nn_algorithm"
    if type_of_data == "test":
        labels_to_compare=cosine_labels_test
    elif type_of_data == "train":
        labels_to_compare=cosine_labels_train
    i=0
    for row in labels_to_compare:
        ### Write a function to sort the array and return index of the array in traindata/testdata
        nb=sort_store_index(row)
        neighbours=nb[0:k]
        #time.sleep(5)
        count_1=0
        count_2=0
        ## Find votes of my neighbours

        for y in range(len(neighbours)):
            index = neighbours[y]
            if trainlabels[index] == 1:
                count_1=count_1+1
            else:
                count_2=count_2+1

        #time.sleep(5)
        if count_2==count_1:
            count_2=0
            count_1=0
            for y in range(len(neighbours)-1):
                index = neighbours[y]
                if trainlabels[index] == 1:
                    count_1 = count_1 + 1
                else:
                    count_2 = count_2 + 1

        if count_2>count_1:
            labels_from_knn=2
        elif count_2<count_1:
            labels_from_knn=1

        if type_of_data == "train":
            trainlabels_learner.append(copy.deepcopy(labels_from_knn))
        elif type_of_data == "test":
            testlabels_learner.append(copy.deepcopy(labels_from_knn))
        i=i+1

#Write function to calculate error: Error would be: (Total wrong predictions*100)/ Total predictions
def calculate_error(arr1,arr2):
    count=0
    for i in range(len(arr1)):
        if arr1[i] != arr2[i]:
            count += 1

    return (count*100/len(arr1))

## Find training/test error for a specific value of k
## type_of_data can be either training data or test data
def data_error(type_of_data,k):
    if type_of_data == "train":
        #trainlabels_learner=learner_function(traindata,type_of_data,k)
        error_percentage=calculate_error(trainlabels_learner,trainlabels)
    elif type_of_data == "test":
        #learner_function(testdata,type_of_data,k)
        error_percentage=calculate_error(testlabels_learner,testlabels)
    return error_percentage

## data_error function
def plot_points(arr):
    for i,j in zip(k_arr,arr):
        ax.annotate('%s)' % j, xy=(i, j), xytext=(30, 0), textcoords='offset points')
        ax.annotate('(%s,' % i, xy=(i, j))

## Main
#k_arr=[3]
k_arr=[1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
test_error_arr=[]
train_error_arr=[]

learner_function("train")
learner_function("test")

for x in range(len(k_arr)):
    k=k_arr[x]
    print "Running k-NN algorithm for k=%s" %(k)
    k_nn_algorithm("train", k)
    print "Saving training data accuracy for k=%s" %(k)
    train_error=data_error("train",k)
    train_error_arr.append(train_error)
    k_nn_algorithm("test", k)
    print "Saving test data accuracy for k=%s" %(k)
    test_error=data_error("test",k)
    test_error_arr.append(test_error)

print
print
print("Printing training error percentages for increasing k value:")
print(train_error_arr)
print("Printing test error percentages for increasing k value:")
print(test_error_arr)
print

# Find k which shows minimum training error
k_min_train_error= k_arr[train_error_arr.index(min(train_error_arr))]
print "Value of k for minimum training error is %s" %(k_min_train_error)

## Find k which shows minimum test error
k_min_test_error=k_arr[test_error_arr.index(min(test_error_arr))]
print "Value of k for minimum test error is %s" %(k_min_test_error)

if k_min_train_error != k_min_test_error:
    print("The two values of k are different for minimum training error and test error")
else:
    print("The value of k is same for minimum training error and test error")

## Plot training data accuracy for each element in k_arr
fig=plt.figure()
#ax=fig.add_subplot(111)
plt.title("Error vs value of 'k'")
plt.xlabel("Value of k")
plt.ylabel("Error in Percentage")
plt.plot(k_arr, train_error_arr, c = 'k',  lw = 3., ls='--', label='Train error')
#plot_points(train_error_arr)
plt.plot(k_arr, test_error_arr, c = 'k',  lw = 3., label='Test error')
#plot_points(test_error_arr)
plt.legend(loc='center left', bbox_to_anchor=(0.5, 0.5))
plt.show()

print("--- %s seconds ---" % (time.time() - start_time))