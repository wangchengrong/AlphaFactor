import matplotlib.pyplot as plt

import AlphaFactor
import FinData


def main():
    factor_name = 'revs60'
    cache_factor_price_group_file = 'cache_factor_price_{}.pkl'.format(factor_name)
    data = FinData.cache.load_df(cache_factor_price_group_file)
    if data is None:
        data = FinData.gen_factor_price_group('hs300', factor_name,
                                             start='2016-01-01', end='2018-01-01')
        FinData.cache.store_df(data, cache_factor_price_group_file)

    factors, prices, groups, group_labels = data

    factors = factors.groupby(level=0).apply(AlphaFactor.utils.winsorize)
    factors = factors.groupby(level=0).apply(AlphaFactor.utils.standardize)

    factor_data = AlphaFactor.utils.get_clean_factor_and_forward_returns(factor=factors, prices=prices, quantiles=5)
    AlphaFactor.tears.create_returns_tear_sheet(factor_data)
    plt.savefig('/Users/polo/AlphaFactor/{}.png'.format(factor_name))

if __name__ == '__main__':
    main()
