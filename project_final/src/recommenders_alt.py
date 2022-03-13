import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from lightfm import LightFM
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import ItemItemRecommender
from implicit.nearest_neighbours import bm25_weight, tfidf_weight


class alt_recommender:
    def __init__(self, data, n=5000, weighting=False, ):
        # Топ покупок по всему датасету
        self.overall_top_purchases = data.groupby('item_id')['quantity'].sum().reset_index()
        # self.popularity.rename(columns={'quantity': 'n_sold'}, inplace=True)

        # determining of top n
        self.top_n = self.overall_top_purchases.sort_values('quantity', ascending=False).head(n).item_id.tolist()
        # self.overall_top_purchases.sort_values('quantity', ascending=False, inplace=True)
        self.overall_top_purchases.loc[~self.overall_top_purchases['item_id'].isin(self.top_n), 'item_id'] = 999999

        # Топ покупок каждого юзера
        self.top_purchases = data.groupby(['user_id', 'item_id'])['quantity'].count().reset_index()
        self.top_purchases.sort_values('quantity', ascending=False, inplace=True)
        self.top_purchases = self.top_purchases[self.top_purchases['item_id'] != 999999]

        # preparing of dictionaries and matrices
        self.user_item_matrix = None
        self.prepare_user_item_matrix(data)

        # index lost after bm 25. It need to be checked
        if weighting:
            pass
            # self.user_item_matrix = bm25_weight(self.user_item_matrix.T).T

        self.id_to_itemid, \
        self.id_to_userid, \
        self.itemid_to_id, \
        self.userid_to_id = self.prepare_dicts(self.user_item_matrix)

        self.model = None
        self.own_recommender = None

    def prepare_user_item_matrix(self, data):
        """Готовит user-item матрицу"""
        user_item_matrix = pd.pivot_table(data,
                                          index='user_id',
                                          columns='item_id',
                                          values='quantity',  # Можно пробовать другие варианты
                                          aggfunc='count',
                                          fill_value=0
                                          )
        user_item_matrix = user_item_matrix.astype(float)
        self.user_item_matrix = user_item_matrix

    def prepare_dicts(self, user_item_matrix):
        userids = user_item_matrix.index.values
        itemids = user_item_matrix.columns.values

        matrix_userids = np.arange(len(userids))
        matrix_itemids = np.arange(len(itemids))

        id_to_itemid = dict(zip(matrix_itemids, itemids))
        id_to_userid = dict(zip(matrix_userids, userids))

        itemid_to_id = dict(zip(itemids, matrix_itemids))
        userid_to_id = dict(zip(userids, matrix_userids))

        return id_to_itemid, id_to_userid, itemid_to_id, userid_to_id

    def train_model(self, factors=15, lambda_reg=0.0001, iterations=15):
        model = AlternatingLeastSquares(factors=factors,
                                        regularization=lambda_reg,
                                        iterations=iterations,
                                        calculate_training_loss=True,
                                        use_gpu=False)

        model.fit(csr_matrix(self.user_item_matrix).T.tocsr(), show_progress=True)
        self.model = model

    def get_recommendations(self, user, N=5):
        res = [self.id_to_itemid[rec[0]] for rec in
               self.model.recommend(userid=self.userid_to_id[user],
                                    user_items=csr_matrix(self.user_item_matrix).tocsr(),  # на вход user-item matrix
                                    N=N,
                                    filter_already_liked_items=False,
                                    filter_items=[self.itemid_to_id[999999]],  # !!!
                                    recalculate_user=True)]
        return res

    def get_own_recommendations(self, user, N=5):
        res = [self.id_to_itemid[rec[0]] for rec in
               self.own_recommender.recommend(userid=self.userid_to_id[user],
                                    user_items=csr_matrix(self.user_item_matrix).tocsr(),  # на вход user-item matrix
                                    N=N,
                                    filter_already_liked_items=False,
                                    filter_items=[self.itemid_to_id[999999]],  # !!!
                                    recalculate_user=True)]
        return res

    def fit_own_recommender(self):
        """Обучает модель, которая рекомендует товары, среди товаров, купленных юзером"""
        own_recommender = ItemItemRecommender(K=1, num_threads=4)
        own_recommender.fit(csr_matrix(self.user_item_matrix).T.tocsr())
        self.own_recommender = own_recommender

    def get_als_recommendations(self, user, N=5):
        """Рекомендации через стардартные библиотеки implicit"""
        # self.update_dict(user_id=user)
        return self.get_recommendations(user, N=N)

    def get_own_recommendations(self, user, N=5):
        """Рекомендуем товары среди тех, которые юзер уже купил"""
        #self._update_dict(user_id=user)
        return self.get_own_recommendations(user, N=N)

    def get_similar_items_recommendation(self, user, N=5):
        """Рекомендуем товары, похожие на топ-N купленных юзером товаров"""
        top_users_purchases = self.top_purchases[self.top_purchases['user_id'] == user].head(N)
        res = top_users_purchases['item_id'].apply(lambda x: self._get_similar_item(x)).tolist()
        res = self._extend_with_top_popular(res, N=N)

        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return res

    def _get_similar_item(self, item_id):
        """Находит товар, похожий на item_id"""
        recs = self.model.similar_items(self.itemid_to_id[item_id], N=2)  # Товар похож на себя -> рекомендуем 2 товара
        top_rec = recs[1][0]  # И берем второй (не товар из аргумента метода)
        return self.id_to_itemid[top_rec]

    def _extend_with_top_popular(self, recommendations, N=5):
        """Если кол-во рекоммендаций < N, то дополняем их топ-популярными"""

        if len(recommendations) < N:
            recommendations.extend(self.overall_top_purchases[:N])
            recommendations = recommendations[:N]

        return recommendations

    def get_similar_users_recommendation(self, user, N=5):
        """Рекомендуем топ-N товаров, среди купленных похожими юзерами"""

        res = []

        # Находим топ-N похожих пользователей
        similar_users = self.model.similar_users(self.userid_to_id[user], N=N + 1)
        similar_users = [rec[0] for rec in similar_users]
        similar_users = similar_users[1:]  # удалим юзера из запроса

        for user in similar_users:
            res.extend(self.get_own_recommendations(user, N=1))

        res = self._extend_with_top_popular(res, N=N)

        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return res







    def get_predictions(self, data):
        pass

    def update_dict(self, user_id):
        """Если появился новыю user / item, то нужно обновить словари"""

        if user_id not in self.userid_to_id.keys():
            max_id = max(list(self.userid_to_id.values()))
            max_id += 1

            self.userid_to_id.update({user_id: max_id})
            self.id_to_userid.update({max_id: user_id})


'''

        self.user_item_matrix = self._prepare_matrix(data)  # pd.DataFrame
        self.id_to_itemid, self.id_to_userid, \
        self.itemid_to_id, self.userid_to_id = self._prepare_dicts(self.user_item_matrix)



        self.model = self.fit(self.user_item_matrix)


    def _get_recommendations(self, user, model, N=5):
        """Рекомендации через стардартные библиотеки implicit"""

        self._update_dict(user_id=user)
        res = [self.id_to_itemid[rec[0]] for rec in model.recommend(userid=self.userid_to_id[user],
                                                                    user_items=csr_matrix(
                                                                        self.user_item_matrix).tocsr(),
                                                                    N=N,
                                                                    filter_already_liked_items=False,
                                                                    filter_items=[self.itemid_to_id[999999]],
                                                                    recalculate_user=True)]

        res = self._extend_with_top_popular(res, N=N)

        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return res



    def get_similar_items_recommendation(self, user, N=5):
        """Рекомендуем товары, похожие на топ-N купленных юзером товаров"""

        top_users_purchases = self.top_purchases[self.top_purchases['user_id'] == user].head(N)

        res = top_users_purchases['item_id'].apply(lambda x: self._get_similar_item(x)).tolist()
        res = self._extend_with_top_popular(res, N=N)

        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return res


'''
