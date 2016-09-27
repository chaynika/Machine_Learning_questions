## Always sync this file with the following command from your terminal:
## "cp /Users/csaikia/Dropbox/Monodeep/SML_ass_2/hw2/* /Users/csaikia/SML_assignment/Machine_Learning_questions/KNN_HOT_OR_NOT/"

import scipy.io as sio
import matplotlib.pyplot as plt

## Save all the information from the faces.mat file
data=sio.loadmat('faces.mat')
testlabels=data['testlabels']
testlabels=testlabels[:,0]
trainlabels=data['trainlabels']
trainlabels=trainlabels[:,0]
testdata=data['testdata']
traindata=data['traindata']
evaldata=data['evaldata']

## Learning function outputs traing label and test label
trainlabels_learner=[]
testlabels_learner=[]

# Write function for pre-computed distances between all pairs of points
def euclidean_distance():

# Write function to find Euclidian distance, my k-NN learner function
def learner_function(arr,type_of_data,k):

# Write function to calculate error: Error would be: (Total wrong predictions*100)/ Total predictions
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
        trainlabels_learner=learner_function(arr,type_of_data,k)
        error_percentage=calculate_error(trainlabels_learner,trainlabels)
    elif type_of_data == "test":
        testlabels_learner=learner_function(arr,type_of_data,k)
        error_percentage=calculate_error(testlabels_learner,testlabels)

    return error_percentage

## data_error function
def plot_points(arr):
    for i,j in zip(k_arr,arr):
        ax.annotate('%s)' % j, xy=(i, j), xytext=(30, 0), textcoords='offset points')
        ax.annotate('(%s,' % i, xy=(i, j))

## Main function

k_arr=[1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
test_error_arr=[]
train_error_arr=[]

for x in range(len(k_arr)):
    k=k_arr[x]
    print "Running k-NN algorithm for k=%s" %(k)
    k_nn_algorithm(k)
    print "Saving training data accuracy for k=%s" %(k)
    train_error=data_error("train",k)
    train_error_arr.append(train_error)
    print "Saving test data accuracy for k=%s" %(k)
    test_error=data_error("test",k)
    test_error_arr.append(test_error)

## Plot training data accuracy for each element in k_arr
fig=plt.figure()
ax=fig.add_subplot(111)
plt.title("Error vs value of 'k'")
plt.xlabel("Value of k")
plt.ylabel("Error in Percentage")
plt.plot(k_arr, train_error_arr, 'r--')
plot_points(train_error_arr)
plt.plot(k_arr, test_error_arr, 'b')
plot_points(test_error_arr)
plt.show()

## Find k which shows minimum training error
k_min_train_error= k_arr[train_error_arr.index(min(train_error_arr))]
print "Value of k for minimum training error is %s" %(k_min_train_error)

## Find k which shows minimum test error
k_min_test_error=k_arr[test_error_arr.index(min(test_error_arr))]
print "Value of k for minimum test error is %s" %(k_min_test_error)

if k_min_train_error != k_min_test_error:
    print "The two values of k are different for minimum training error and test error"
else:
    print "The value of k is same for minimum training error and test error"

