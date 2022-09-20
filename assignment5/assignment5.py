import pandas as pd
import operator
import numpy as np  
from sklearn.neighbors import KNeighborsClassifier	
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score
import matplotlib.pyplot as plt 
import math
def calDist(d1, d2, length):
    distance = 0
    for x in range(length):
        distance += np.square(d1[x] - d2[x])
       
    return np.sqrt(distance)


def knn(data,test,k):
    dist = {}
    length = test.shape[1]
    # print(length)

    for i in range(len(data)):    
        cal_dist = calDist(test,data.iloc[i],length)
        dist[i] = cal_dist[0]
    
    sort_d = sorted(dist.items(),key=operator.itemgetter(1))

    nn = []
    for i in range(k):
        nn.append(sort_d[i][0])

    class_list = data['Species'].unique()
    class_dict = {}
    for i in range(len(class_list)):
        class_dict[class_list[i]] = 0

    # print(class_dict)
    for i in range(len(nn)):
        response = data.iloc[nn[i]][-1]
 
        if response in class_dict:
            class_dict[response] += 1
        else:
            class_dict[response] = 1  

    sortedVotes = sorted(class_dict.items(), key=operator.itemgetter(1), reverse=True)
    print(sortedVotes)
    return (sortedVotes[0][0],nn)

def in_built_knn(data):
    x=data.iloc[:,:4]
    y=data['Species']
    neigh=KNeighborsClassifier(n_neighbors=4)
    neigh.fit(data.iloc[:,:4],data["Species"])
    

def calculate_likelihood_gaussian(df, feat_name, feat_val, Y, label):
    feat = list(df.columns)
    df = df[df[Y]==label]
    mean, std = df[feat_name].mean(), df[feat_name].std()
    p_x_given_y = (1 / (np.sqrt(2 * np.pi) * std)) *  np.exp(-((feat_val-mean)**2 / (2 * std**2 )))
    return p_x_given_y

def naive_bayes_gaussian(df, X, Y):
    features = list(df.columns)[:-1]
    # calculate prior
    classes = sorted(list(df[Y].unique()))
    prior = []
    for i in classes:
        prior.append(len(df[df[Y]==i])/len(df))

    Y_pred = []
    for x in X:
        # calculate likelihood
        labels = sorted(list(df[Y].unique()))
        likelihood = [1]*len(labels)
        for j in range(len(labels)):
            for i in range(len(features)):
                likelihood[j] *= calculate_likelihood_gaussian(df, features[i], x[i], Y, labels[j])

        # calculate posterior probability
        post_prob = [1]*len(labels)
        for j in range(len(labels)):
            post_prob[j] = likelihood[j] * prior[j]

        Y_pred.append(np.argmax(post_prob))

    return np.array(Y_pred) 		

def main():
	file_name="D:/College/BTech/SEM 7/Data Mining/DataSet/breast-cancer.csv"
	data = pd.read_csv(file_name)
	print(data)
	train, test = train_test_split(data, test_size=.2, random_state=41)
	print(test,train)
	X_test = test.iloc[:,:-1].values
	Y_test = test.iloc[:,-1].values
	Y_pred = naive_bayes_gaussian(train, X=X_test, Y="diagnosis")
	print(confusion_matrix(Y_test, Y_pred))
	print(f1_score(Y_test, Y_pred))

	# testSet = [[2.4, 3.6, 4.4]]
	# k = int(input())
	# test = pd.DataFrame(testSet)
	# nn = knn(data, test,k)
	# print(nn)

	

    # print(type(data))


if __name__ == "__main__":
    main()


# upload data
# split into test and train
# 
