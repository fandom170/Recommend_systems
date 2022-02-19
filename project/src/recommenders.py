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