import pandas as pd
import numpy as np
import random
import math
import json
from functools import reduce
from sklearn.model_selection import train_test_split
class Collaborative_Filter():
    def read_dict(self,filepath):
        f = open(filepath, "r")
        return json.load(f)

    def write_json(self,data, filepath):
        file_dict = json.dumps(data)
        f = open(filepath, "w")
        f.write(file_dict)

    def load_file(self,filename):
        with open(filename, 'r') as f:
            for i, line in enumerate(f):
                if i == 0:  # 去掉文件第一行的title
                    continue
                yield line.strip('\r\n')
        print('Load %s success!' % filename)




    def train_test_split(self):
        self.train_data = {}
        self.test_data = {}
        for user, item_value in self.dataSet.items():
            self.train_data.setdefault(user, {})
            self.test_data.setdefault(user, {})
            if len(item_value.keys()) > 1:
                key_len = len(item_value.keys())
                train_num = int((1 - self.test_size) * key_len)
                test_num = key_len - train_num
                samples_train = random.sample(item_value.keys(), train_num)
                for item, value in item_value.items():
                    if item in samples_train:
                        self.train_data[user][item] = value
                    else:
                        self.test_data[user][item] = value
            else:
                self.train_data[user] = item_value
                self.test_data[user] = {}
        return self.train_data, self.test_data

    def ItemSimilarity(self):
        # 建立物品-物品的共现矩阵
        C = dict()  # 物品-物品的共现矩阵
        N = dict()  # 物品被多少个不同用户购买
        for user, item_value in self.train_data.items():
            for item in item_value.keys():
                N.setdefault(item, 0)
                N[item] += 1
                C.setdefault(item, {})
                for item_2 in item_value.keys():
                    if item == item_2:
                        continue
                    C[item].setdefault(item_2, 0)
                    C[item][item_2] += 1/math.log(1+len(item_value.keys()))
        # 构建item_item 相似度矩阵
        self.item_W = {}
        for item, item_value in C.items():
            for item_2, value in C[item].items():
                item_sim = value / (math.sqrt(N[item] * N[item_2]))
                self.item_W.setdefault(item, {})
                self.item_W[item][item_2] = item_sim
        return self.item_W

    def UserSimilarity(self):
        # build inverse table for item_users
        item_users = dict()
        for u, items in self.train_data.items():
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
                    C.setdefault(u, {})
                    C[u].setdefault(v, 0)
                    C[u][v] += 1/math.log(1+len(users))
        # calculate finial similarity matrix W
        self.user_W = dict()
        for u, related_users in C.items():
            self.user_W.setdefault(u, {})
            for v, cuv in related_users.items():
                self.user_W[u][v] = cuv / math.sqrt(N[u] * N[v])
        return self.user_W

    def rank_users(self):
        self.rank_u = dict()
        for u, related_users in self.user_W.items():
            sort_dict = dict()
            if u in self.trust_data.keys():
                for v, vs in self.trust_data.items():
                    if u == v:
                        for us, value in related_users.items():
                            if us in vs.keys():
                                sort_dict[us] = value
            else:
                sort_dict = related_users
            if len(sort_dict) > self.K:
                # 对字典中的related_users 进行排序（倒序）,选取前K个users
                revers_users = sorted(sort_dict.items(), key=lambda x: x[1], reverse=True)[0:self.K]
                for n in revers_users:
                    self.rank_u.setdefault(u, {})
                    self.rank_u[u][n[0]] = n[1]
            else:
                self.rank_u[u] = sort_dict
        return self.rank_u

    # 给用户user推荐，前K个相关用户;如果user推荐为空值的时候，利用该user已有的item，进行前N个item相关性最大的推荐
    def recommond_items(self,user):
        if user in self.rank_u.keys():
            # 1.针对拿到的user抽出给他推荐的K个最相关的用户和相关度
            self.user_k = self.rank_u[user]  # dict
            train_user_k = {}
            # 2. 拿到所有在train_data中所有相关user的items和value值
            for user_i, item_value in self.train_data.items():
                if user_i not in self.user_k.keys():
                    continue
                train_user_k.setdefault(user_i, item_value)
            # 3. 对所有的items进行打分（相互之间不存在的打0）
            items_score = {}
            for user1, user_sim in self.user_k.items():
                for user2, item_value in train_user_k.items():
                    if user1 == user2:
                        for items, values in item_value.items():
                            items_score.setdefault(items, 0)
                            items_score[items] += user_sim * int(values)
            # 4 返回前N个items
            recommond_its = dict(sorted(items_score.items(), key=lambda x: x[1], reverse=True)[0:self.N])
            self.recommd_items = recommond_its
            return self.recommd_items
        # 如果不存在：利用item相关性进行前N个推荐
        else:
            print("{} has no relationship with other users!".format(user))
            recommond_its = {}
            if self.train_data[user]:
                user_items = self.train_data[user].keys()
                # 在Item_W里面将所有的items包含的相关性的itms及其关联的进行排序
                itms = {}
                for i in user_items:
                    if i in self.item_W.keys():
                        for ii, value in self.item_W[i].items():
                            itms.setdefault(ii, 0)
                            itms[ii] += value
                    else:
                        print("{} in {} has no item in item_W!".format(i, user))
                recommond_its = dict(sorted(itms.items(), key=lambda x: x[1], reverse=True)[0:self.N])
            else:
                print("We have no data in {} users !".format(user))
            self.recommd_items = recommond_its
            return self.recommd_items


    def precision_result(self,recomm_data,test_data):
        precision = []
        all_len = []
        all_len2 = []
        for user, rec_items in recomm_data.items():
            for user2, items_value in test_data.items():
                if user == user2:
                    if not rec_items or not items_value:
                        continue
                    count = 0
                    all_len.append(len(rec_items.keys()))
                    for item, relarity in rec_items.items():
                        if item in items_value.keys():
                            count += 1
                    precision.append(count)
                    all_len2.append(len(items_value.keys()))
        aa = reduce(lambda x, y: x + y, precision)
        bb = reduce(lambda x, y: x + y, all_len)
        cc = reduce(lambda x, y: x + y, all_len2)
        self.precision = aa / bb
        self.recall = aa / cc
        self.f1_score = 2 * (self.precision * self.recall) / (self.precision + self.recall)
        print("Precision is {}".format(self.precision))
        print("Recall is {}".format(self.recall))
        print("F1-score is {}".format(self.f1_score))
        return self.precision, self.recall, self.f1_score

    def __init__(self):
        self.test_size = 0.2
        self.K = 3
        self.N = 10

        """1.读取数据"""
        self.dataSet = {}
        self.filename = "original_data/ratings_data.txt"
        for line in self.load_file(self.filename):
            user, item, rating, = line.split(' ')
            self.dataSet.setdefault(user, {})
            self.dataSet[user][item] = rating

        trust_file = self.load_file("original_data/trust_data.txt")
        self.trust_data = dict()
        for line2 in trust_file:
            try:
                user1, user2, trust_value = filter(None, line2.split(" "))  # 这里直接用str.split(" ")无法完全拆开
                self.trust_data.setdefault(user1, {})
                self.trust_data[user1][user2] = trust_value
            except:
                print("{} and {} is problem!".format(user1, user2))

        """2.划分数据集"""
        self.train_data, self.test_data = self.train_test_split()
        self.write_json(self.train_data, "train_data.txt")
        self.write_json(self.test_data, "test_data.txt")
        # self.write_json(self.train_data,"generate_data/train_data.txt")

        """3.Item相关性矩阵保存为item_W,User相关性矩阵保存为W"""
        self.item_W = self.ItemSimilarity()

        self.write_json(self.item_W, "Item_W.txt")
        # self.user_W = self.UserSimilarity()
        # # self.write_json(self.user_W, "generate_data/User_W.txt")
        #
        # """4.对用户进行相关性前K排序"""
        # self.rank_u = self.rank_users()
        # self.write_json(self.rank_u,"generate_data/rank_u.txt")


class one_user_recommend():
    def read_dict(self, filepath):
        f = open(filepath, "r")
        return json.load(f)

    def recommond_items(self,user):
        if user in self.rank_u.keys():
            # 1.针对拿到的user抽出给他推荐的K个最相关的用户和相关度
            self.user_k = self.rank_u[user]  # dict
            train_user_k = {}
            # 2. 拿到所有在train_data中所有相关user的items和value值
            for user_i, item_value in self.train_data.items():
                if user_i not in self.user_k.keys():
                    continue
                train_user_k.setdefault(user_i, item_value)
            # 3. 对所有的items进行打分（相互之间不存在的打0）
            items_score = {}
            for user1, user_sim in self.user_k.items():
                for user2, item_value in train_user_k.items():
                    if user1 == user2:
                        for items, values in item_value.items():
                            items_score.setdefault(items, 0)
                            items_score[items] += user_sim * int(values)
            # 4 返回前N个items
            recommond_its = dict(sorted(items_score.items(), key=lambda x: x[1], reverse=True)[0:self.N])
            self.recommd_items = recommond_its
            return self.recommd_items
        # 如果不存在：利用item相关性进行前N个推荐
        else:
            print("{} has no relationship with other users!".format(user))
            recommond_its = {}
            if self.train_data[user]:
                user_items = self.train_data[user].keys()
                # 在Item_W里面将所有的items包含的相关性的itms及其关联的进行排序
                itms = {}
                for i in user_items:
                    if i in self.item_W.keys():
                        for ii, value in self.item_W[i].items():
                            itms.setdefault(ii, 0)
                            itms[ii] += value
                    else:
                        print("{} in {} has no item in item_W!".format(i, user))
                recommond_its = dict(sorted(itms.items(), key=lambda x: x[1], reverse=True)[0:N])
            else:
                print("We have no data in {} users !".format(user))
            self.recommd_items = recommond_its
            return self.recommd_items

    def __init__(self):
        self.test_size = 0.2
        self.K = 3
        self.N = 10
        self.item_W = self.read_dict("generate_data/Item_W.txt")
        self.train_data = self.read_dict("generate_data/train_data.txt")
        self.rank_u = self.read_dict("generate_data/rank_u.txt")


if __name__ == '__main__':
    collabarative_filter = Collaborative_Filter()
    # 给出一个用户，推荐前1个物品
    user = '22605'
    item_recomm_byuser = collabarative_filter.recommond_items(user)
    print("User {} is recommended items for {}".format(user,item_recomm_byuser.keys()))

    """全部test进行测试"""
    test_data = collabarative_filter.test_data
    recmm_item_dict = {}
    fg_item = []
    for user, items_value in test_data.items():
        recommond_i = collabarative_filter.recommond_items( user)
        recmm_item_dict.setdefault(user, recommond_i)
        for item, value in recommond_i.items():
            fg_item.append(item)

    # precision,recall,f1_score
    precision, recall, f1_score = collabarative_filter.precision_result(recmm_item_dict, test_data)

    # 覆盖率
    Item_all = len(item_W.keys())
    recomm_all = len(list(set(recmm_item_dict)))
    coverage = recomm_all / Item_all
    print("Coverage is {}".format(coverage))
