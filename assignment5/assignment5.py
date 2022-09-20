import pandas as pd
import operator
import numpy as np  
from sklearn.neighbors import KNeighborsClassifier	
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.preprocessing import MinMaxScaler
import sklearn.linear_model
from sklearn import preprocessing
from sklearn import metrics
import itertools
import matplotlib.pyplot as plt 
from tkinter import *
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

np.set_printoptions(threshold=np.inf)

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



def Sigmoid(Z):
    return 1/(1+np.exp(-Z))

def Relu(Z):
    return np.maximum(0,Z)

def dRelu2(dZ, Z):    
    dZ[Z <= 0] = 0    
    return dZ

def dRelu(x):
    x[x<=0] = 0
    x[x>0] = 1
    return x

def dSigmoid(Z):
    s = 1/(1+np.exp(-Z))
    dZ = s * (1-s)
    return dZ

class dlnet:
    def __init__(self, x, y):
        self.debug = 0;
        self.X=x
        self.Y=y
        self.Yh=np.zeros((1,self.Y.shape[1])) 
        self.L=2
        self.dims = [9, 15, 1] 
        self.param = {}
        self.ch = {}
        self.grad = {}
        self.loss = []
        self.lr=0.003
        self.sam = self.Y.shape[1]
        self.threshold=0.5
        
    def nInit(self):    
        np.random.seed(1)
        self.param['W1'] = np.random.randn(self.dims[1], self.dims[0]) / np.sqrt(self.dims[0]) 
        self.param['b1'] = np.zeros((self.dims[1], 1))        
        self.param['W2'] = np.random.randn(self.dims[2], self.dims[1]) / np.sqrt(self.dims[1]) 
        self.param['b2'] = np.zeros((self.dims[2], 1))                
        return 

    def forward(self):    
        Z1 = self.param['W1'].dot(self.X) + self.param['b1'] 
        A1 = Relu(Z1)
        self.ch['Z1'],self.ch['A1']=Z1,A1
        
        Z2 = self.param['W2'].dot(A1) + self.param['b2']  
        A2 = Sigmoid(Z2)
        self.ch['Z2'],self.ch['A2']=Z2,A2

        self.Yh=A2
        loss=self.nloss(A2)
        return self.Yh, loss

    def nloss(self,Yh):
        loss = (1./self.sam) * (-np.dot(self.Y,np.log(Yh).T) - np.dot(1-self.Y, np.log(1-Yh).T))    
        return loss

    def backward(self):
        dLoss_Yh = - (np.divide(self.Y, self.Yh ) - np.divide(1 - self.Y, 1 - self.Yh))    
        
        dLoss_Z2 = dLoss_Yh * dSigmoid(self.ch['Z2'])    
        dLoss_A1 = np.dot(self.param["W2"].T,dLoss_Z2)
        dLoss_W2 = 1./self.ch['A1'].shape[1] * np.dot(dLoss_Z2,self.ch['A1'].T)
        dLoss_b2 = 1./self.ch['A1'].shape[1] * np.dot(dLoss_Z2, np.ones([dLoss_Z2.shape[1],1])) 
                            
        dLoss_Z1 = dLoss_A1 * dRelu(self.ch['Z1'])        
        dLoss_A0 = np.dot(self.param["W1"].T,dLoss_Z1)
        dLoss_W1 = 1./self.X.shape[1] * np.dot(dLoss_Z1,self.X.T)
        dLoss_b1 = 1./self.X.shape[1] * np.dot(dLoss_Z1, np.ones([dLoss_Z1.shape[1],1]))  
        
        self.param["W1"] = self.param["W1"] - self.lr * dLoss_W1
        self.param["b1"] = self.param["b1"] - self.lr * dLoss_b1
        self.param["W2"] = self.param["W2"] - self.lr * dLoss_W2
        self.param["b2"] = self.param["b2"] - self.lr * dLoss_b2
        
        return


    def pred(self,x, y):  
        self.X=x
        self.Y=y
        comp = np.zeros((1,x.shape[1]))
        pred, loss= self.forward()    
    
        for i in range(0, pred.shape[1]):
            if pred[0,i] > self.threshold: comp[0,i] = 1
            else: comp[0,i] = 0
    
        print("Acc: " + str(np.sum((comp == y)/x.shape[1])))
        
        return comp
    
    def gd(self,X, Y,root, iter = 3000):
        np.random.seed(1)                         
    
        self.nInit()
    
        for i in range(0, iter):
            Yh, loss=self.forward()
            self.backward()
            # Label(root,text="wait : "+str(100 - (100*i/iter)),fg='red',font=("Helvetica",12)).place(x=600,y=10)
        
            if i % 500 == 0:
                print ("Cost after iteration %i: %f" %(i, loss))
                print(100*i/iter)

                self.loss.append(loss)
        fig, ax = plt.subplots(figsize=(3.5, 3.5)) 
        # Label(root,text="Done "+str(0),fg='red',font=("Helvetica",12)).place(x=600,y=10)

        plt.plot(np.squeeze(self.loss))
        plt.ylabel('Loss')
        plt.xlabel('Iter')
        plt.title("Lr =" + str(self.lr))
        plt2 = FigureCanvasTkAgg(fig, root) 
        plt2.get_tk_widget().place(x=680,y=130)

        return 
        
def plotCf(a,b,t,root,xaxis):
    cf =confusion_matrix(a,b)
    fig, ax = plt.subplots(figsize=(3, 3)) 
    plt.imshow(cf,cmap=plt.cm.Blues,interpolation='nearest')
    plt.colorbar()
    plt.title(t)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    tick_marks = np.arange(len(set(a))) # length of classes
    class_labels = ['0','1']
    plt.xticks(tick_marks,class_labels)
    plt.yticks(tick_marks,class_labels)
    thresh = cf.max() / 2.
    for i,j in itertools.product(range(cf.shape[0]),range(cf.shape[1])):
        plt.text(j,i,format(cf[i,j],'d'),horizontalalignment='center',color='white' if cf[i,j] >thresh else 'black')
    # plt.show()
    plt1 = FigureCanvasTkAgg(fig, root) 
    plt1.get_tk_widget().place(x=xaxis,y=130)

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

	

def ann(filename,root):
    df = pd.read_csv(filename)
    # df = pd.read_csv('D:/College/BTech/SEM 7/Data Mining/DataSet/iris.csv')
    # class_list = df['Species'].unique()
    # class_dict = {}
    # print(df['Species'].unique())
    # for i in range(len(class_list)):
    #     class_dict[class_list[i]] = i+1
    # df['Species'] = [class_dict[item] for item in df['Species']]
    # print(df)
    # print(df.shape[1])
    df.columns = range(df.shape[1])
    # print(df)
    df = df[~df[6].isin(['?'])]
    # print(df)
    df = df.astype(float)
    # print(df)
    df.iloc[:,10].replace(2, 0,inplace=True)
    df.iloc[:,10].replace(4, 1,inplace=True)

    df.head(3)
    scaled_df=df
    names = df.columns[0:10]
    scaler = MinMaxScaler() 
    scaled_df = scaler.fit_transform(df.iloc[:,0:10]) 
    scaled_df = pd.DataFrame(scaled_df, columns=names)
    x=scaled_df.iloc[0:500,1:10].values.transpose()
    y=df.iloc[0:500,10:].values.transpose()

    xval=scaled_df.iloc[501:683,1:10].values.transpose()
    yval=df.iloc[501:683,10:].values.transpose()

    print(df.shape, x.shape, y.shape, xval.shape, yval.shape)

    nn = dlnet(x,y)
    nn.lr=0.07
    nn.dims = [9, 15, 1]
    nn.gd(x, y,root, iter = 30000)
    pred_train = nn.pred(x, y)
    pred_test = nn.pred(xval, yval)
    Label(root,text="Accuracy: "+str(np.sum((pred_test == yval)/xval.shape[1])),fg='red',font=("Helvetica",12)).place(x=10,y=30)


    nn.threshold=0.5

    nn.X,nn.Y=x, y 
    target=np.around(np.squeeze(y), decimals=0).astype(np.int)
    predicted=np.around(np.squeeze(nn.pred(x,y)), decimals=0).astype(np.int)
    plotCf(target,predicted,'Cf Training Set',root,0)

    nn.X,nn.Y=xval, yval 
    target=np.around(np.squeeze(yval), decimals=0).astype(np.int)
    predicted=np.around(np.squeeze(nn.pred(xval,yval)), decimals=0).astype(np.int)
    plotCf(target,predicted,'Cf Validation Set',root,320)
    nn.X,nn.Y=xval, yval 
    yvalh, loss = nn.forward()
    print("\ny",np.around(yval[:,0:50,], decimals=0).astype(np.int))       
    print("\nyh",np.around(yvalh[:,0:50,], decimals=0).astype(np.int),"\n")     
# if __name__ == "__main__":
    # main()
    # ann()


# upload data
# split into test and train
# 
