from __future__ import division
import sys
import numpy as np

# Save the data into an array
data_matrix = np.genfromtxt('breast_cancer_data.csv', delimiter=',')

### Remove rows with ? data
rows_to_del = []
for i in range(len(data_matrix)):
    if 1 in np.isnan(data_matrix[i]).astype(int):
        rows_to_del.append(i)

rows_to_del.sort(reverse=True)
for row in rows_to_del:
    data_matrix = np.delete(data_matrix, row, 0)

# Remove first column from array

data_matrix = data_matrix[:, 1:]

# Replace last column of every row to +1 for label 2, -1 for label 4

for i in range(len(data_matrix)):
    if data_matrix[i, -1] == 2:
        data_matrix[i, -1] = 1
    elif data_matrix[i, -1] == 4:
        data_matrix[i, -1] = -1

# Define two variables training_data and test_data
length = len(data_matrix)
training_matrix_len = (length * 2 / 3)
training_data = data_matrix[:training_matrix_len, :]
test_data = data_matrix[-(training_matrix_len) / 2:, :]

features = len(test_data[1, :]) - 1

#Initialize the weight vector w
w=np.zeros(features+1)

# Append 1 column to training_data

a=np.ones((len(training_data),1))
training_data=np.hstack((a, training_data))

count=0
#threshold=1e-5
diff=sys.maxint
learning_rate=0.001
print training_data[:,:-1].shape
print w.shape
## P(Y=1|X) = exp(sum(wi*xi))/(1+exp(sum(wi*xi))


while (count<1000):
    #print "Count"
    num=np.exp(training_data[:,:-1].dot(w))
    p_y_1_given_x=num/(1+num)
    print p_y_1_given_x
    #w=w+learning_rate*(np.sum(np.dot((training_data[:,-1] - p_y_1_given_x),training_data[:,:-1]),axis=1))
    #p = ((np.exp(np.dot(features, w)) / (1 + (np.exp(np.dot(features, w))))))
    var1=training_data[:,-1] - p_y_1_given_x
    var2=var1.dot(training_data[:,:-1])
    w=w+(learning_rate*var2)
    #w = w + learning_rate * (np.transpose(np.dot((training_data[:,-1] - p_y_1_given_x), training_data[:,:-1]), axis=0))
    #print p_y_1_given_x
   # print count
   # print w
    print w
    #print count
    count=count+1
    print "----------------"





