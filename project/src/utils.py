def prefilter_items(data, item_data):
    # Уберем самые популярные товары (их и так купят)
    popularity = data.groupby('item_id')['user_id'].nunique().reset_index() / data['user_id'].nunique()
    popularity.rename(columns={'user_id': 'share_unique_users'}, inplace=True)

    top_popular = popularity[popularity['share_unique_users'] > 0.5].item_id.tolist()
    data = data[~data['item_id'].isin(top_popular)]

    # Уберем самые НЕ популярные товары (их и так НЕ купят)
    top_notpopular = popularity[popularity['share_unique_users'] < 0.01].item_id.tolist()
    data = data[~data['item_id'].isin(top_notpopular)]

    # Уберем товары, которые не продавались за последние 12 месяцев
    less_sold = data[data['week_no'] > 52].item_id.tolist()
    data = data[~data['item_id'].isin(less_sold)]

    # Уберем не интересные для рекоммендаций категории (department)
    # selected 'vide rental category
    not_interested = item_data[item_data['department'] == 'VIDEO RENTAL']
    data = data[~data['item_id'].isin(not_interested)]

    # Уберем слишком дешевые товары (на них не заработаем). 1 покупка из рассылок стоит 60 руб.
    too_cheap = data[data['sales_value'] < 0.6].item_id.tolist()
    data = data[~data['item_id'].isin(too_cheap)]

    # Уберем слишком дорогие товары (e.g. sales value > 500)
    too_expensive = data[data['sales_value'] > 500].item_id.tolist()
    data = data[~data['item_id'].isin(too_expensive)]

    #Also possible to remove goods by manufacturer or weight etc.

    return data


def postfilter_items(user_id, recommednations):
    pass