from __future__ import division
import numpy as np
import random
import matplotlib.pyplot as plt


# Save the data into an array
data_matrix = np.genfromtxt('breast_cancer_data.csv', delimiter=',')

### Remove rows with ? data
rows_to_del=[]
for i in range(len(data_matrix)):
    if 1 in np.isnan(data_matrix[i]).astype(int):
        rows_to_del.append(i)

rows_to_del.sort(reverse=True)
for row in rows_to_del:
    data_matrix = np.delete(data_matrix, row, 0)

# Remove first column from array

data_matrix = data_matrix[:, 1:]
features = len(data_matrix[1, :]) - 1
# Replace last column of every row to +1 for label 2, -1 for label 4

for i in range(len(data_matrix)):
    if data_matrix[i, -1] == 2:
        data_matrix[i, -1] = 1
    elif data_matrix[i, -1] == 4:
        data_matrix[i, -1] = -1

# Need to make a def function which will return a matrix (no of features * no of discrete values each feature can take)
# which has information about P(Xi=x|Y=k)

def likelihood(feature_no, value, class_var,training_data,count_for_prior):
   # print 'For feature no %s whose discrete value is %s for class output %s' % (feature_no, value, class_var)
    count = 0
    for i in range(len(training_data)):
        if (training_data[i, feature_no] == value) & (data_matrix[i, -1] == class_var):
            count += 1


    if class_var == 1:
        lklhd = (count + 1) / (count_for_prior + (1 * features))
    else:
        lklhd = (count + 1) / (len(training_data) - count_for_prior + (1 * features))
    #print 'Likelihood is %s' %(lklhd)
    return lklhd

# Now for every test example, we need to predict the output. We find the probability of every test example assuming
# class output to be 1. If it is less than 0.5, then output class is -1, else it is 1.

def p_x(arr,class_var,training_data,count_for_prior):
    ans=1
    for i in range(len(arr)):
        ans=ans*likelihood(i,arr[i],class_var,training_data,count_for_prior)
    return ans

def nb_classifier(arr,prior_y_1,prior_y_0,training_data,count_for_prior):
    p_x_given_1=p_x(arr[:-1],1,training_data,count_for_prior)
 #   print "DEBUG START"
   # print p_x_given_1
    p_x_given_0=p_x(arr[:-1],-1,training_data,count_for_prior)
    # print p_x_given_0
 #   print "DEBUG END"
    p_y_given_x=prior_y_1*p_x_given_1/((prior_y_1*p_x_given_1)+(prior_y_0*p_x_given_0))

  #  print 'p_y_given_x is %s' %(p_y_given_x)
    return p_y_given_x

per_of_training_data = [.01 ,.02 ,.03 ,.125 ,.625 ,1]


def setup(arr,flag):

    # Define two variables training_data and test_data
    if flag == 1:
        np.random.shuffle(data_matrix)

    # training_idx = numpy.random.randint(data_matrix.shape[0], size=80)
    # test_idx = numpy.random.randint(data_matrix.shape[0], size=20)
    # training_data, test_data = data_matrix[training_idx, :], data_matrix[test_idx, :]
    length = len(data_matrix)
    training_matrix_len = (length * 2 / 3)
    training_data = data_matrix[:training_matrix_len, :]
    #np.random.shuffle(training_data)
    test_data = data_matrix[-(training_matrix_len) / 2:, :]

    prediction_arr=[]
    accuracy_arr=[]
    training_data_size_arr=[]
    for x in arr:
        #print x
        correct_prediction = 0
        wrong_prediction = 0

        td = training_data[:training_matrix_len*x,:]


        #features = len(test_data[1, :]) - 1
        # print "------------>>>", training_data[0]
        # Calculate prior of Y
        count_for_prior = 0
        for i in range(len(td)):
            if td[i, -1] == 1:
                count_for_prior = count_for_prior + 1

        prior_y_1 = (count_for_prior + 1) / (len(td) + 1 * 2)  # 1 is added to numerator for Laplace smoothing
        prior_y_0 = (1 - prior_y_1)


        for i in range(len(test_data)):
            # print 'For test_data %s' %(i)
            prediction=nb_classifier(test_data[i],prior_y_1,prior_y_0,td,count_for_prior)
            prediction_arr.append(prediction)
            if prediction > 0.5:
                output=1
            else:
                output=-1

            if output == test_data[i,-1]:
                # print("Prediction matches")
                correct_prediction+=1
            else:
                # print("Prediction wrong")
                wrong_prediction+=1

        print 'Correct prediction: %s, Wrong prediction: %s' %(correct_prediction,wrong_prediction)

        accuracy=(correct_prediction/(correct_prediction+wrong_prediction) )* 100
        len_of_training_data = len(td)
        print 'Y-axis: %s, X-axis: %s' % (accuracy, len_of_training_data)
        training_data_size_arr.append(len_of_training_data)
        accuracy_arr.append(accuracy)

    prediction_avg=np.mean(prediction_arr)
    print "Average prediction is: %s" %(prediction_avg*100)

    print accuracy_arr
    print training_data_size_arr
    if flag == 1:
        plt.title("Plot against random shuffled data set and random fractions")
    else:
        plt.title("Plot against given data and given random fractions")

    plt.xlabel("Training data size")
    plt.ylabel("Accuracy in %")
    plt.plot(training_data_size_arr,accuracy_arr)
    plt.show()

#setup(per_of_training_data,0)

print "========================================================================================"

print "For second part of question 5.2"
random.seed(9)
per_of_training_data_2=[random.random() for x in range(1,6)]
per_of_training_data_2=np.sort(per_of_training_data_2)
print per_of_training_data_2
setup(per_of_training_data_2,1)


