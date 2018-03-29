import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime, timedelta

import alphalens
import tushare as ts

import apdset as ads
import AlphaFactor as aps

cache_path = os.path.expanduser('~/.cache/')


def calendar(start, end, is_open=False):
    cache_file = cache_path + 'calendar_{start}_{end}_{open}.csv'.format(start=start,
                                                                         end=end,
                                                                         open=int(is_open))
    if os.path.exists(cache_file):
        return pd.read_csv(cache_file, index_col=0, squeeze=True, names=['date'])

    trade_dates = ads.calendar(start=start, end=end, is_open=is_open)
    trade_dates.to_csv(cache_file)

    return trade_dates


def get_stock_hist():
    pass


def get_hs300_component():

    cache_file = cache_path + 'hs300_component_{date}.csv'.format(date=datetime.now().strftime('%Y%m%d'))
    if os.path.exists(cache_file):
        return pd.read_csv(cache_file, dtype={'code': str}, index_col=0)

    hs300_component_df = ts.get_hs300s()
    hs300_component_df.to_csv(cache_file)

    return hs300_component_df


def get_industry_classified():

    cache_file = cache_path + 'industry_class_{date}.csv'.format(date=datetime.now().strftime('%Y%m%d'))
    if os.path.exists(cache_file):
        return pd.read_csv(cache_file, dtype={'code': str}, index_col=0)

    industry_cls_df = ts.get_industry_classified(standard='sw')
    industry_cls_df.to_csv(cache_file)

    return industry_cls_df


def get_area_classified(force=False):

    cache_file = cache_path + 'area_class_{date}.csv'.format(date=datetime.now().strftime('%Y%m%d'))
    if force is False and os.path.exists(cache_file):
        return pd.read_csv(cache_file, dtype={'code': str}, index_col=0)

    area_cls_df = ts.get_area_classified()

    area_names = area_cls_df['area'].unique()
    area_codes = range(1, 1 + len(area_cls_df['area'].unique()))

    area = pd.DataFrame(np.full((len(area_names), 2), np.nan), columns=['area', 'area_code'])
    area['area'] = area_names
    area['area_code'] = area_codes

    area_cls_df = pd.merge(area_cls_df, area, on='area')
    area_cls_df.to_csv(cache_file)

    return area_cls_df


def get_classified(classified='hs300'):

    if classified == 'hs300':
        return get_hs300_component()['code']
    else:
        return


def gen_group(codes, start=None, end=None, is_dict=False):
    trade_dates = calendar(start=start, end=end, is_open=True)

    area_cls_df = get_area_classified(force=False)

    hs300_area_cls_df = area_cls_df[area_cls_df['code'].isin(codes)]

    if is_dict:
        group = hs300_area_cls_df[['code', 'area_code']].set_index('code')['area_code'].to_dict()
    else:
        group = pd.Series(np.full(len(codes) * len(trade_dates), np.nan),
                          index=pd.MultiIndex.from_product([trade_dates, hs300_area_cls_df['code']]))
        group.name = 'group_code'
        group.index.levels[0].name = 'date'
        group.index.levels[1].name = 'code'

        group = group.reset_index()
        group.drop(columns='group_code', inplace=True)
        group = pd.merge(group, hs300_area_cls_df[['code', 'area_code']], on='code')
        group = group.sort_values(by='date').set_index(['date', 'code'])['area_code']

    group_labels = hs300_area_cls_df[['area_code', 'area']].set_index('area_code')['area'].to_dict()

    return group, group_labels


def gen_prices_and_factor(codes, start, end):

    early_start = (datetime.strptime(start, '%Y-%m-%d') - timedelta(days=100)).strftime('%Y-%m-%d')
    delay_end = (datetime.strptime(end, '%Y-%m-%d') + timedelta(days=100)).strftime('%Y-%m-%d')

    cache_file = cache_path + 'hist_{start}_{end}.csv'.format(start=early_start, end=delay_end)

    trade_dates = calendar(start=early_start, end=delay_end, is_open=True)

    if os.path.exists(cache_file):
        hist_data = pd.read_csv(cache_file,
                                dtype={'code': str},
                                index_col=0)
    else:
        hist_data = pd.DataFrame()
        for code in codes:
            data = ts.get_k_data(code, start=early_start, end=delay_end)
            if data.empty:
                continue

            data = pd.DataFrame(data.set_index('date'), index=trade_dates).fillna(method='ffill')
            data['code'] = code
            data['ma60'] = data['close'].rolling(window=60).mean()

            hist_data = hist_data.append(data)

        hist_data.to_csv(cache_file)

    hist_data.index = pd.to_datetime(hist_data.index)

    prices = hist_data.loc[start:delay_end][
        ['code', 'close']].reset_index().pivot(index='date', columns='code', values='close')

    factors = hist_data.loc[start:end][
        ['code', 'ma60']].sort_index().reset_index().set_index(['date', 'code'])

    return prices, factors


def main():
    start = '2016-01-01'
    end = '2017-12-31'

    com_codes = get_classified('hs300')
    group, group_labels = gen_group(com_codes,
                                    start=start,
                                    end=end,
                                    is_dict=True)

    prices, factors = gen_prices_and_factor(com_codes,
                                            start=start,
                                            end=end)
    # factor_data = alphalens.utils.get_clean_factor_and_forward_returns(factors,
    #                                                                    prices=prices,
    #                                                                    quantiles=5,
    #                                                                    groupby=group,
    #                                                                    groupby_labels=group_labels)
    # print(factor_data)

    factor_data = aps.utils.get_clean_factor_and_forward_returns(factors,
                                                                 prices=prices,
                                                                 quantiles=5,
                                                                 groupby=group,
                                                                 groupby_labels=group_labels)

    aps.tears.create_event_returns_tear_sheet(factor_data, prices=prices, by_group=True)
    # alphalens.tears.create_event_returns_tear_sheet(factor_data, prices=prices)

    plt.savefig('/Users/polo/event_returns.png')


if __name__ == '__main__':
    main()

