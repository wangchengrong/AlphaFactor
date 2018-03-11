import pandas as pd
import matplotlib.pyplot as plt

import apsign
import apdset

from apdset.utils.cache import store_df, load_df


def load_group_factor_price(index_symbol, factor_name, start, end):

    universe = apdset.get_index_com(index_symbol)['code']

    # 1. 获取分组信息
    cache_indu_cls_file = '{}_indu_cls_{}.pkl'.format(index_symbol, factor_name)
    indu_cls = load_df(cache_indu_cls_file)
    if indu_cls is None:
        indu_cls = apdset.get_indu_cls(universe)
        store_df(indu_cls, 'hs300_indu_cls.pkl')

    group = indu_cls[['code', 'c_code']].set_index('code')['c_code'].to_dict()
    group_labels = indu_cls[['code', 'c_name']].set_index('code')['c_name'].to_dict()

    # 2. 获取因子信息
    cache_factor_file = '{}_factor_{}_{}_{}.pkl'.format(index_symbol,
                                                        factor_name,
                                                        start, end)
    factors = load_df(cache_factor_file)
    if factors is None:
        factors = apdset.get_multi_stock_factors(universe, 'rsi', start=start, end=end)
        store_df(factors, cache_factor_file)
    factors['date'] = pd.to_datetime(factors['date'])

    factors.rename(columns={'code': 'asset'}, inplace=True)
    factors = factors.sort_values(by='date').set_index(['date', 'asset']).iloc[:, 0]

    # 3. 获取价格数据
    cache_price_file = '{}_price_{}_{}.pkl'.format(index_symbol,
                                                   start, end)
    prices = load_df(cache_price_file)
    if prices is None:
        prices = apdset.get_stock_hist(universe, start=start, end=end)[['date', 'code', 'close']]
        store_df(prices, cache_price_file)
    prices['close'] = prices['close'].fillna(method='ffill').fillna(0)
    prices['date'] = pd.to_datetime(prices['date'])

    prices.rename(columns={'code': 'asset', 'close': 'price'}, inplace=True)
    prices = prices.pivot(index='date', columns='asset', values='price')

    return prices, factors, group, group_labels


def main():
    prices, factors, group, group_labels = load_group_factor_price('hs300', 'rsi', start='2016-01-01', end='2018-01-01')
    factor_data = apsign.utils.get_clean_factor_and_forward_returns(factor=factors, prices=prices, quantiles=5)
    apsign.tears.create_returns_tear_sheet(factor_data)
    plt.savefig('/Users/polo/apsign.png')

if __name__ == '__main__':
    main()
