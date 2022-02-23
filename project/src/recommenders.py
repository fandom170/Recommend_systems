import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix, coo_matrix

import matplotlib.pyplot as plt
#%matplotlib inline

from lightfm import LightFM
from lightfm.evaluation import precision_at_k, recall_at_k


from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import bm25_weight, tfidf_weight, ItemItemRecommender



class Recommender:

    def __init__(self, data, weighning = 0):
        self.user_top_purchases = data.group_by(['user_id', 'item_id'])['quantity'].count().reset_index()
        self.user_top_purchases.sort_values('quantity', ascending=False, inplace=True)
        self.user_top_purchases = self.user_top_purchases[self.user_top_purchases['item_id' != 999999]]

        self.general_top_purchases =  data.group_by(['item_id'])['quantity'].count().reset_index()
        self.general_top_purchases.sort_values('quantity', ascending=False, inplace=True)
        self.general_top_purchases = self.general_top_purchases[self.general_top_purchases['item_id' != 999999]]

        self.general_top_purchases = self.general_top_purchases['item_id'].tolist()

        self.user_item_matrix = self.get_ui_matrix(data)

        self.id_to_itemid, \
        self.id_to_user_id, \
        self.itemid_to_id, \
        self.user_id_to_id = self.prepare_data(self.user_item_matrix)

        if weighning == 'b':
            self.user_item_matrix = bm25_weight(self.user_item_matrix.T).T
        elif weighning == 't':
            self.user_item_matrix = tfidf_weight(self.user_item_matrix.T).T

        self.initial_users = list(self.user_id_to_id.keys())

    def train_model(self):
        pass

    def prepare_data(user_item_matrix):
        pass

    def get_ui_matrix(data):
        pass

    def get_recommendations(self, user, model, N=5):
        res = [self.id_to_itemid[rec[0]] for rec in
               model.recommend(userid=userid_to_id[user],
                               user_items=sparse_user_item,  # на вход user-item matrix
                               N=N,
                               filter_already_liked_items=False,
                               filter_items=[itemid_to_id[999999]],
                               recalculate_user=True)]
        return res



    def get_similar_items_recommendation(user, model, id_to_userid, N=5):
        """Рекомендуем товары, похожие на топ-N купленных юзером товаров"""
        similar_users = model.similar_users(user, N=N)
        res = [id_to_userid[sim[0]] for sim in similar_users]
        return res


    def get_similar_users_recommendation(item, model, itemid_to_id, N=5, ):
        """Рекомендуем топ-N товаров, среди купленных похожими юзерами"""
        similar_items = model.similar_items(item, N=N)
        res = [itemid_to_id[sim[0]] for sim in similar_items]
        return res





