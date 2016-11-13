<<<<<<< HEAD
#!/usr/bin/env python
##################################################
#     Copyright (c) 2016 by Chaynika Saikia      #
##################################################
#
=======
##################################################
#     Copyright (c) 2016 by Chaynika Saikia      #
##################################################
>>>>>>> 43fccf9901c0ecafb1f5cec23cbd634ef0d6b065
import numpy as np
import random
import matplotlib.pyplot as plt
import math
from collections import Counter

data=np.genfromtxt('seeds_dataset.txt')
output_labels=data[:,-1]

k_arr=[int(k) for k in range(2,11)]
features=len(data[0])
dataset_size=len(data)
max_iterations=10000

# Randomly allocate cluster points function

test_label=np.zeros(len(data)).astype(int)

def allocate_cluster_centers(k):
    data[:, -1] = 0
    #print("In allocate_cluster_centers")
    centres_index=random.sample(range(1, len(data)), k)
    centre_arr=[]
    for centre in centres_index:
        #print(data[centre,:-1])
        centre_arr.append(data[centre,:-1])
    #print(centre_arr)
    return centre_arr

def classify(centre_arr,k):
    #print("In classify")
    for l in range(len(data)):
        classify_arr=[]
        for i in range(k):
            #print("Computing between -->", k, centre_arr[i],data[l,:-1])
            classify_arr.append(np.linalg.norm(centre_arr[i]-data[l,:-1]))
        data[l,-1]= np.argmin(classify_arr)
    #print(data[:,-1])
    return data[:,-1]

def recentre(k):
    #print ("In recentre")
    centre_arr=[]
    for i in range(k):
        centre=sum(data[data[:,-1] == i][:,0:(len(data[0])-1)])/len(data[data[:,-1] == i][:,0:(len(data[0])-1)])
        centre_arr.append(centre)
    return centre_arr

def largest_cluster():
    return max(Counter(data[:,-1])).astype(int)

def split_cluster():
    largest_cluster_index=largest_cluster()
    indices=[]
    for i in range(len(data)):
        if data[i,-1].astype(int) == largest_cluster_index:
            indices.append(i)
    new_centre=data[random.choice(indices),:-1]
    return new_centre

def compute_min_objective(centres):
    #print("In compute_min_objective")
    ans=0
    for i in range(len(data)):
        ans=ans+math.pow(np.linalg.norm(centres[data[i,-1].astype(int)]-data[i,:-1]),2)
    return ans

def k_means_algo(k):
    print("In k_means_algo for -->",k)
    current_centres=allocate_cluster_centers(k)
    current_class_labels=data[:,-1]
    iteration=0
    while iteration<max_iterations :
        prev_class_labels=current_class_labels
        current_class_labels=classify(current_centres,k)
        prev_centres=current_centres
        current_centres=recentre(k)
        counters = Counter(data[:, -1])
        if 0 in counters.values():
            j = counters.keys()[counters.values().index(0)]
            current_centres[j]=split_cluster()
        if (np.array_equal(prev_class_labels,current_class_labels)) and (np.array_equal(current_centres, prev_centres)) :
            print("For convergence----> iteration", iteration)
            print("For convergence----> labels ", current_class_labels)
            print("For convergence----> centres ",current_centres)
            return compute_min_objective(current_centres)
        iteration=iteration+1
    return compute_min_objective(current_centres)

def plot_points(arr):
    for i,j in zip(k_arr,arr):
        ax.annotate('%s)' % j, xy=(i, j), xytext=(30, 0), textcoords='offset points')
        ax.annotate('(%s,' % i, xy=(i, j))

##########
#  MAIN  #
##########
objective_function=[]
for k in k_arr:
    computed_cost=k_means_algo(k)
    print("Computed cost is  --->", computed_cost)
    objective_function.append(computed_cost)

fig=plt.figure()
ax=fig.add_subplot(111)
plt.title("Objective function vs value of 'k'")
plt.xlabel("Value of k")
plt.ylabel("Objective function")
plt.plot(k_arr, objective_function)
plot_points(objective_function)
plt.show()