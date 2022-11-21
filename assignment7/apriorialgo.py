import pandas as pd
from itertools import combinations
from itertools import permutations

def APRIORI_MY(data, min_support=0.04,  max_length = 4):
   
    support = {} 
    L = list(data.columns)
    # print(L)
    for i in range(1, max_length+1):
        c = set(combinations(L,i))
        L =set()     
  
        for j in list(c):

            sup = data.loc[:,j].product(axis=1).sum()/len(data.index)
            if sup > min_support:
                support[j] = sup
                
                L = list(set(L) | set(j))
        
    result = pd.DataFrame(list(support.items()), columns = ["Items", "Support"])
    print("Support")
    return(result)

def ASSOCIATION_RULE_MY(df, min_threshold=0.5):
    
    support = pd.Series(df.Support.values, index=df.Items).to_dict()
    data = []
    L= df.Items.values
    
    p = list(permutations(L, 2))
    
    for i in p:
        if set(i[0]).issubset(i[1]):
            conf = support[i[1]]/support[i[0]]
            if conf > min_threshold:
                j = i[1][not i[1].index(i[0][0])]
                lift = support[i[1]]/(support[i[0]]* support[(j,)])
                leverage = support[i[1]] - (support[i[0]]* support[(j,)])
                convection = (1 - support[(j,)])/(1- conf)
                data.append([i[0], (j,), support[i[0]], support[(j,)], support[i[1]], conf, lift, leverage, convection])

    result = pd.DataFrame(data, columns = ["antecedents", "consequents", "antecedent support", "consequent support","support", "confidence", "Lift", "Leverage", "Convection"])
    print("confidence")
    return(result)

def main_housevote(filename,threshold,support,un_col_list):
    bread = pd.read_csv(filename)
    # bread = bread.drop_duplicates()
    bread = bread.drop('Class Name', axis=1)
    print(bread)
    cols = bread.columns.values
    print(cols)
    newDataSet = []
    # st.write(len(df_rows))
    i, cnt = 0, 0
    # for row in bread:
    #     # print(row)
    #     i += 1
    #     if '?' in row:
    #         continue
    #     else:
    #         lst = []
    #         cnt += 1
    #         for k in range(1, len(row)):
    #             if row[k] == 'y':
    #                 lst.append(cols[k])
    #         newDataSet.append(lst)
    # print(newDataSet)


def main_bakery(filename,threshold,support,un_col_list):
    bread = pd.read_csv(filename)
    bread = bread.drop_duplicates()
    bread = bread.drop('DateTime', axis=1)
    
    transaction = pd.crosstab(index= bread['TransactionNo'], columns= bread['Items'])
    print(transaction)  

    my_freq_itemset = APRIORI_MY(transaction, threshold, 3)
    my_freq_itemset.sort_values(by = 'Support', ascending = False)

    print(my_freq_itemset)

    my_rule = ASSOCIATION_RULE_MY(my_freq_itemset, support)
    print(my_rule)


if __name__ == '__main__':
    # main_bakery("D:/College/BTech/SEM 7/Data Mining/DataSet/Bakery.csv",0.04,0.5,[])
    main_housevote("D:/College/BTech/SEM 7/Data Mining/DataSet/house-votes-84.data.csv",0.04,0.5,[])