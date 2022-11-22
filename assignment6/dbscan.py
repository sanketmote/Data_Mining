import numpy.random as random
from numpy.core.fromnumeric import *
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from tkinter import *
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg



def calDist(X1, X2):
    sum = 0
    for x1, x2 in zip(X1, X2):
        sum += (x1 - x2) ** 2
    return sum ** 0.5
    return (((X1[0]-X2[0])**2)+(X1[1]-X2[1])**2)**0.5

def getNeibor(data, dataSet, e):
    res = []
    for i in range(len(dataSet)):
        if calDist(data, dataSet[i]) < e:
            res.append(i)
    return res

def DBSCAN(dataSet, e, minPts):
    coreObjs = {}
    C = {}
    for i in range(len(dataSet)):
        neibor = getNeibor(dataSet[i], dataSet, e)
        if len(neibor) >= minPts:
            coreObjs[i] = neibor
    oldCoreObjs = coreObjs.copy()
    # st.write(oldCoreObjs)
    # CoreObjs set of COres points
    k = 0
    notAccess = list(range(len(dataSet)))

    # his will check the relation of core point with each other
    while len(coreObjs) > 0:
        OldNotAccess = []
        OldNotAccess.extend(notAccess)
        cores = coreObjs.keys()
        randNum = random.randint(0, len(cores))
        cores = list(cores)
        core = cores[randNum]
        queue = []
        queue.append(core)
        notAccess.remove(core)
        while len(queue) > 0:
            q = queue[0]
            del queue[0]
            if q in oldCoreObjs.keys():
                delte = [val for val in oldCoreObjs[q]
                            if val in notAccess]
                queue.extend(delte)
                notAccess = [
                    val for val in notAccess if val not in delte]
        k += 1
        C[k] = [val for val in OldNotAccess if val not in notAccess]
        for x in C[k]:
            if x in coreObjs.keys():
                del coreObjs[x]
    # st.write(C)
    print(C)
    return C

def draw(C, dataSet,attribute1,attribute2,root):
    color = ['r', 'y', 'g', 'b', 'c', 'k', 'm']
    vis = set()
    for i in C.keys():
        X = []
        Y = []
        datas = C[i]
        for k in datas:
            vis.add(k)
        for j in range(len(datas)):
            X.append(dataSet[datas[j]][0])
            Y.append(dataSet[datas[j]][1])
        plt.scatter(X, Y, marker='o',
                    color=color[i % len(color)], label=i)
    vis = list(vis)
    unvis1 = []
    unvis2 = []
    for i in range(len(dataSet)):
        if i not in vis:
            unvis1.append(dataSet[i][0])
            unvis2.append(dataSet[i][1])
    fig, ax = plt.subplots(figsize=(6, 6)) 
    plt.xlabel(attribute1)
    plt.ylabel(attribute2)
    plt.scatter(unvis1, unvis2, marker='o', color='black')
    plt.legend(loc='lower right')
    plt1 = FigureCanvasTkAgg(fig, root) 
    plt1.get_tk_widget().place(x=30,y=130)
    # plt.show()
    # st.pyplot()

def main_dbscan(data,root):
    cols = []

    for i in data.columns[:-1]:
        cols.append(i)
    # atr1, atr2 = st.columns(2)
    Clickattribute1 = StringVar(root)
    Clickattribute1.set("Select Attribute 1")
    dropCols = OptionMenu(root, Clickattribute1,*cols)
    dropCols.place(x=30,y=90)
    Clickattribute2 = StringVar(root)
    Clickattribute2.set("Select Attribute 2")
    dropCols1 = OptionMenu(root, Clickattribute2,*cols)
    dropCols1.place(x=190,y=90)

    dataset = []
    arr1 = []
    arr2 = []

    label=Label(root, text="Insert value for eps", font=("Helvetica",12))
    label.place(x=30,y=30)
    entry= Entry(root, width= 40)
    entry.focus_set()
    entry.place(x=230,y=30)
    
    label1=Label(root, text="Insert mimimum number of points in cluster", font=("Helvetica",12))
    label1.place(x=30,y=60)
    entry1= Entry(root, width= 40)
    entry1.focus_set()
    entry1.place(x=230,y=60)
    
    
    def helper():
        attribute1 = Clickattribute1.get()
        attribute2 = Clickattribute2.get()

        for i in range(len(data)):
            arr1.append(data.loc[i, attribute1])
        for i in range(len(data)):
            arr2.append(data.loc[i, attribute2])
        for i in range(len(arr1)):
            tmp = []
            tmp.append(arr1[i])
            tmp.append(arr2[i])
            dataset.append(tmp)
        r = entry.get()
        mnp = entry1.get()
        C = DBSCAN(dataset, r, mnp)
        draw(C, dataset,attribute1,attribute2)    
    
    Button(root,text="Measure",command= lambda:helper()).place(x=30,y=130)