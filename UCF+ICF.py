import pandas as pd
import numpy as np
import random
import math
import json
from functools import reduce


def load_file(filename):
    with open(filename, 'r') as f:
        for i, line in enumerate(f):
            if i == 0:  # 去掉文件第一行的title
                continue
            yield line.strip('\r\n')
    print('Load %s success!' % filename)




# 给用户user推荐，前K个相关用户;如果user推荐为空值的时候，利用该user已有的item，进行前N个item相关性最大的推荐
def recommond_items(item_W,rank_u,train_data,user, N=10):
    if user in rank_u.keys():
        # 1.针对拿到的user抽出给他推荐的K个最相关的用户和相关度
        user_k = rank_u[user]# dict
        train_user_k = {}
        # 2. 拿到所有在train_data中所有相关user的items和value值
        for user_i,item_value in train_data.items():
            if user_i not in user_k.keys():
                continue
            train_user_k.setdefault(user_i,item_value)
        # 3. 对所有的items进行打分（相互之间不存在的打0）
        items_score = {}
        for user1,user_sim in user_k.items():
            for user2,item_value in train_user_k.items():
                if user1 == user2:
                    for items,values in item_value.items():
                        items_score.setdefault(items,0)
                        items_score[items] += user_sim * int(values)
        # 4 返回前N个items
        recommond_its = dict(sorted(items_score.items(),key=lambda x:x[1],reverse=True)[0:N])
        return recommond_its
    # 如果不存在：利用item相关性进行前N个推荐
    else:
        print("{} has no relationship with other users!".format(user))
        recommond_its = {}
        if train_data[user]:
            user_items = train_data[user]

            tes = train_data["45911"]
            # tes2 = item_W["128169"]
            # 在Item_W里面将所有的items包含的相关性的itms及其关联的进行排序
            itms = {}
            for i,score in user_items.items():
                if i in item_W.keys():
                    for ii,value in item_W[i].items():
                        itms.setdefault(ii,0)
                        itms[ii] += value*int(score)
                else:
                    print("{} in {} has no item in item_W!".format(i,user))
            recommond_its = dict(sorted(itms.items(),key=lambda x:x[1],reverse=True)[0:N])
        else:
            print("We have no data in {} users !".format(user))
        return recommond_its



def rank_users(W,trust_data,dataSet,K = 3):
    rank = dict()
    for u,related_users in W.items():
        sort_dict = dict()
        if u in trust_data.keys():
            for v,vs in trust_data.items():
                if u == v:
                    for us,value in related_users.items():
                        if us in vs.keys():
                            sort_dict[us] = value
        else:
            sort_dict = related_users
        if len(sort_dict)>K:
            # 对字典中的related_users 进行排序（倒序）,选取前K个users
            revers_users = sorted(sort_dict.items(), key=lambda x: x[1], reverse=True)[0:K]
            for n in revers_users:
                rank.setdefault(u,{})
                rank[u][n[0]] = n[1]
        else:
            rank[u] = sort_dict
    return rank



def ItemSimilarity(train):
    # 建立物品-物品的共现矩阵
    C = dict()  # 物品-物品的共现矩阵
    N = dict()  # 物品被多少个不同用户购买
    for user,item_value in train.items():
        for item in item_value.keys():
            N.setdefault(item,0)
            N[item] += 1
            C.setdefault(item,{})
            for item_2 in item_value.keys():
                if item == item_2:
                    continue
                C[item].setdefault(item_2,0)
                # 下面公式能消除热爱购物的人的对物品相似度的影响
                C[item][item_2] += 1/math.log(1+len(item_value.keys()))
    # 构建item_item 相似度矩阵
    W= {}
    for item,item_value in C.items():
        for item_2,value in C[item].items():
            item_sim = value/(math.sqrt(N[item]*N[item_2]))
            W.setdefault(item,{})
            W[item][item_2] = item_sim
    return W



def UserSimilarity(train):
    # build inverse table for item_users
    item_users = dict()
    for u, items in train.items():
        for i in items.keys():
            if i not in item_users:
                item_users[i] = set()
            item_users[i].add(u)
    # calculate co-rated items between users
    C = dict()
    N = dict()
    for i, users in item_users.items():
        for u in users:
            N.setdefault(u, 0)
            N[u] += 1
            for v in users:
                if u == v:
                    continue
                C.setdefault(u,{})
                C[u].setdefault(v,0)
                # 这里能利用下面公式对用户相似度消除热门物品的影响，这很重要
                C[u][v] += 1/math.log(1+len(users))
    # calculate finial similarity matrix W
    W = dict()
    for u, related_users in C.items():
        W.setdefault(u, {})
        for v, cuv in related_users.items():
            W[u][v] = cuv / math.sqrt(N[u] * N[v])
    return W

def precision_result(recomm_data,test_data):
    precision = []
    all_len = []
    all_len2 = []
    for user,rec_items in recomm_data.items():
        for user2, items_value in test_data.items():
           if user == user2:
               if not rec_items or not items_value:
                   continue
               count = 0
               all_len.append(len(rec_items.keys()))
               for item,relarity in rec_items.items():
                    if item in items_value.keys():
                        count += 1
               precision.append(count)
               all_len2.append(len(items_value.keys()))
    aa = reduce(lambda x,y:x+y,precision)
    bb = reduce(lambda x,y:x+y,all_len)
    cc = reduce(lambda x,y:x+y,all_len2)
    precision = aa/bb
    recall = aa/cc
    f1_score = 2*(precision*recall)/(precision+recall)
    print("Precision is {}".format(precision))
    print("Recall is {}".format(recall))
    print("F1-score is {}".format(f1_score))
    return precision,recall,f1_score
# 这种验证方式按照书上的公式来的


def read_dict(filepath):
    f = open(filepath,"r")
    return json.load(f)

def write_json(data,filepath):
    file_dict = json.dumps(data)
    f = open(filepath, "w")
    f.write(file_dict)




if __name__ == '__main__':
    # 变成字典形式
    dataSet = {}
    filename = "original_data/ratings_data.txt"
    for line in load_file(filename):
        user, item, rating, = line.split(' ')
        dataSet.setdefault(user, {})
        dataSet[user][item] = rating

    """划分数据集"""
    train_data, test_data = train_test_split(dataSet)

    trust_file = load_file("original_data/trust_data.txt")
    trust_data = dict()
    for line2 in trust_file:
        try:
            user1,user2,trust_value = filter(None,line2.split(" ")) # 这里直接用str.split(" ")无法完全拆开
            trust_data.setdefault(user1,{})
            trust_data[user1][user2] = trust_value
        except:
            print("{} and {} is problem!".format(user1,user2))

    """Item相关性矩阵保存为item_W,User相关性矩阵保存为W"""
    item_W = ItemSimilarity(train_data)
    user_W = UserSimilarity(train_data)
    write_json(item_W,"generate_data/Item_W.txt")
    write_json(user_W,"generate_data/User_W.txt")

    """读取json格式的User_W和Item_W"""
    # user_W = read_dict("generate_data/User_W.txt")
    # item_W = read_dict("generate_data/Item_W.txt")


    """ user1 相关前topK个相关度最大的userK"""
    rank_u = rank_users(user_W,trust_data,train_data)
    write_json(rank_u,"generate_data/rank_u.txt")

    # rank_u = read_dict("generate_data/rank_u.txt")


    "给出一个用户，推荐前1个物品"
    user = '22605'
    recmm_its = recommond_items(item_W,rank_u,train_data,user)
    print(recmm_its)

    """实验模型准确性"""
    # 取前1000个user出来
    # user_sample_trust = random.sample(test_data.keys(), 1000) #list
    # test_items_dict = {}
    # for user,items_value in test_data.items():
    #     if user not in user_sample_trust:
    #         continue
    #     test_items_dict.setdefault(user,items_value)
    # recmm_item_dict = {}
    # for user in user_sample_trust:
    #     re_items = recommond_items(item_W,rank_u,train_data,user)
    #     recmm_item_dict.setdefault(user,re_items)

    """全部进行测试"""
    recmm_item_dict = {}
    fg_item = []
    for user,items_value in test_data.items():
        recommond_i = recommond_items(item_W,rank_u,train_data,user)
        recmm_item_dict.setdefault(user,recommond_i)
        for item,value in recommond_i.items():
            fg_item.append(item)
    precision, recall, f1_score = precision_result(recmm_item_dict, test_data)

    # 覆盖率
    Item_all = len(item_W.keys())
    recomm_all = len(list(set(recmm_item_dict)))
    coverage = recomm_all/Item_all
    print("Coverage is {}".format(coverage))











