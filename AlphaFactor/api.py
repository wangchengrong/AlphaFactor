import pandas as pd
from scipy import stats

import FinCal

from . import utils
from . import plotting
from . import performance as perf


def plot_factor_weighted_returns(factor_data, long_short=True, gf=None):
    """
    计算因子加权累计收益

    Parameters
    ----------
    factor_data: pd.DataFrame - MultiIndex
        以date和asset为索引的DataFrame，数据涉及因子、不同周期收益、因子分位和资产分组。
    long_short: bool
        是否多空方式计算组合收益，如果是，根据因子距离均值决定多空的方向与权重。
    gf: utils.GridFigure
        用于图表布局

    Returns
    -------
    gf: utils.GridFigure
    """
    factor_data = factor_data.copy()

    factor_returns = perf.factor_returns(factor_data, long_short=long_short)

    if gf is None:
        gf = utils.GridFigure(len(factor_returns.columns), 1)

    for p in factor_returns:
        plotting.plot_cumulative_returns(factor_returns[p],
                                         period=p,
                                         ax=gf.next_row())

    return gf


def plot_factor_mean_returns_by_quantile(factor_data, demeaned=True, gf=None):
    """
    绘制因子分位收益柱状图

    Parameters
    ----------
    factor_data: pd.DataFrame
        以date和asset为索引的DataFrame，数据涉及因子、不同周期收益、因子分位和资产分组。
    demeaned: bool
        是否对收益中心化处理，计算超额收益
    gf: utils.GridFigure
        用于图表布局

    Returns
    -------
    gf: utils.GridFigure
    """

    mean_ret_quantile, std_quantile = perf.mean_return_by_quantile(factor_data,
                                                                   demeaned=demeaned)
    mean_compret_quantile = mean_ret_quantile.apply(utils.rate_of_return, axis=0)

    mean_ret_quantile_daily, std_quantile_daily = perf.mean_return_by_quantile(factor_data,
                                                                               by_date=True,
                                                                               demeaned=demeaned)
    mean_compret_quantile_daily = mean_ret_quantile_daily.apply(utils.rate_of_return, axis=0)

    if gf is None:
        gf = utils.GridFigure(len(mean_compret_quantile.columns) + 2, cols=1)

    plotting.plot_quantile_returns_bar(mean_compret_quantile, by_group=False,
                                       ax=gf.next_row())
    plotting.plot_quantile_returns_violin(mean_compret_quantile_daily,
                                          ylim_percentiles=(1, 99),
                                          ax=gf.next_row())

    for p in mean_compret_quantile_daily:
        plotting.plot_cumulative_returns_by_quantile(mean_ret_quantile_daily[p],
                                                     period=p,
                                                     ax=gf.next_row())

    return gf


def print_returns_summary_table_by_quantile(factor_data, bm_prices=None):
    """
    输出分位收益相关信息

    Parameters
    ----------
    factor_data: pd.DataFrame - MultiIndex
        以date和asset为索引的DataFrame，数据涉及因子、不同周期收益、因子分位和资产分组。
    bm_prices: pd.Series or int
        比较基准价格序列
    """

    mean_ret_quantile_daily, std_quantile_daily = perf.mean_return_by_quantile(factor_data,
                                                                               by_date=True)
    mean_compret_quantile_daily = mean_ret_quantile_daily.apply(utils.rate_of_return, axis=0)

    quantiles = mean_compret_quantile_daily.index.get_level_values('factor_quantile').unique().values
    table = pd.DataFrame(
        index=pd.MultiIndex.from_product([['win_ratio', 'information_ratio'], quantiles]),
        columns=mean_compret_quantile_daily.columns
    )
    table.index.names = ['name', 'quantile']

    if bm_prices is None:
        bm_ret_daily = pd.DataFrame(0, index=mean_compret_quantile_daily.index.get_level_values('date').unique(),
                                    columns=mean_compret_quantile_daily.columns)
    else:
        bm_ret = pd.DataFrame()
        for p in mean_compret_quantile_daily:
            bm_ret[p] = utils.forward_returns(prices=bm_prices, period=p)
        bm_ret_daily = bm_ret.apply(utils.rate_of_return, axis=0)
        bm_ret_daily = pd.DataFrame(bm_ret_daily,
                                    index=mean_compret_quantile_daily.index.get_level_values('date').unique(),
                                    columns=mean_compret_quantile_daily.columns)

    for q in quantiles:
        mean_ret_daily = mean_ret_quantile_daily.loc[q]
        mean_compret_daily = mean_compret_quantile_daily.loc[q]

        win_ratio = FinCal.win_ratio(mean_ret_daily, bm_ret_daily)
        table.loc['win_ratio', q] = win_ratio

        information_ratio = pd.Series(index=mean_ret_quantile_daily.columns)
        for p in mean_compret_daily:
            information_ratio[p] = FinCal.information_ratio(mean_compret_daily[p], bm_ret_daily[p])

        table.loc['information_ratio', q] = information_ratio

    utils.print_table(table)


def plot_factor_mean_returns_by_ff(factor_data, gf=None):
    """
    绘制采用FF打分因子计算因子收益的折线图

    factor_data: pd.DataFrame - MultiIndex
        以date和asset为索引的DataFrame，数据涉及因子、不同周期收益、因子分位和资产分组。
    gf: utils.GridFigure
        用于图表布局

    Returns
    -------
    gf: utils.GridFigure
    """
    *_, mean_ret_ff_daily = perf.mean_returns_by_ff(factor_data)

    if gf is None:
        gf = utils.GridFigure(rows=3, cols=1)

    for p in mean_ret_ff_daily:
        plotting.plot_cumulative_returns_by_ff(mean_ret_ff_daily[p],
                                               period=p,
                                               ax=gf.next_row())

    return gf


def print_returns_summary_table_by_ff(factor_data):
    """
    输出采用FF打分因子回报的相关信息

    Parameters
    ----------
    factor_data: pd.DataFrame - MultiIndex
        以date和asset为索引的DataFrame，数据涉及因子、不同周期收益、因子分位和资产分组。
    """

    low_ret_ff_daily, high_ret_ff_daily, mean_ret_ff_daily = perf.mean_returns_by_ff(factor_data)

    win_ratio = FinCal.win_ratio(mean_ret_ff_daily, 0)
    win_ratio.name = 'win_ratio'

    low_compret_ff_daily = low_ret_ff_daily.apply(utils.rate_of_return, axis=0)
    high_compret_ff_daily = high_ret_ff_daily.apply(utils.rate_of_return, axis=0)
    mean_compret_ff_daily = mean_ret_ff_daily.apply(utils.rate_of_return, axis=0)

    information_ratio = pd.Series(index=high_compret_ff_daily.columns)
    for p in high_compret_ff_daily:
        information_ratio[p] = FinCal.information_ratio(high_compret_ff_daily[p],
                                                        low_compret_ff_daily[p])
    information_ratio.name = 'information_ratio'

    max_drawdown = pd.Series(index=high_compret_ff_daily.columns)
    for p in high_compret_ff_daily:
        max_drawdown[p] = -FinCal.max_drawdown(mean_compret_ff_daily[p])
    max_drawdown.name = 'max_drawdown'

    table = pd.concat([win_ratio, information_ratio, max_drawdown], axis=1).T
    table.index.name = 'name'

    utils.print_table(table)


def plot_factor_mean_returns_by_quantile_and_group(factor_data, demeaned=True, gf=None):
    """
    分组绘制分位收益图

    Parameters
    ----------
    factor_data: pd.DataFrame
        以date和asset为索引的DataFrame，数据涉及因子、不同周期收益、因子分位和资产分组。
    demeaned: bool
        是否对收益做中心化处理
    gf: utils.GridFigure
        用途绘图布局

    Returns
    -------
    gf: utils.GridFigure
    """
    mean_return_quantile_group, mean_return_quantile_group_std_err = \
        perf.mean_return_by_quantile(factor_data,
                                     by_date=False,
                                     by_group=True,
                                     demeaned=demeaned)

    mean_compret_quantile_group = mean_return_quantile_group.apply(utils.rate_of_return, axis=0)

    num_groups = len(mean_return_quantile_group.index.get_level_values('group').unique())
    if gf is None:
        vertical_sections = 1 + (((num_groups - 1) // 2) + 1)
        gf = utils.GridFigure(rows=vertical_sections, cols=2)

    ax_quantile_returns_bar_by_group = [gf.next_cell() for _ in range(num_groups)]

    plotting.plot_quantile_returns_bar(mean_compret_quantile_group,
                                       by_group=True,
                                       ylim_percentiles=(5, 95),
                                       ax=ax_quantile_returns_bar_by_group)

    return gf


def print_ic_table(factor_data):
    """
    打印IC相关统计信息

    Parameters
    -----------
    factor_data: pd.DataFrame - MultiIndex
        以date和asset为索引的DataFrame，数据涉及因子、不同周期收益、因子分位和资产分组。
    """
    ic = perf.factor_information_coefficient(factor_data)
    plotting.plot_information_table(ic)


def plot_ic_ts(factor_data, gf=None):
    """
    绘制IC时间序列图

    Parameters
    ----------
    factor_data: pd.DataFrame - MultiIndex
        以date和asset为索引的DataFrame，数据涉及因子、不同周期收益、因子分位和资产分组。
    gf: utils.GridFigure
        用于绘图布局

    Returns
    -------
    gf: utils.GridFigure
    """
    ic = perf.factor_information_coefficient(factor_data)

    fr_cols = len(ic.columns)
    if gf is None:
        gf = utils.GridFigure(rows=fr_cols, cols=1)

    ax_ic_ts = [gf.next_row() for _ in range(fr_cols)]
    plotting.plot_ic_ts(ic, ax_ic_ts)

    return gf


def plot_ic_distribution(factor_data, theoretical_dist=stats.norm, gf=None):
    """
    绘制与IC分布相关的统计图

    factor_data: pd.DataFrame - MultiIndex
        以date和asset为索引的DataFrame，数据涉及因子、不同周期收益、因子分位和资产分组。
    theoretical_dist: rv_continuous
        某连续分布类型，默认为正态分布
    gf: utils.GridFigure
        用于绘图布局

    Returns
    -------
    gf: utils.GridFigure
    """
    ic = perf.factor_information_coefficient(factor_data)

    fr_cols = len(ic.columns)
    if gf is None:
        gf = utils.GridFigure(rows=fr_cols, cols=2)

    ax_ic_hqq = [gf.next_cell() for _ in range(fr_cols * 2)]

    plotting.plot_ic_hist(ic, ax=ax_ic_hqq[::2])
    plotting.plot_ic_qq(ic, theoretical_dist, ax=ax_ic_hqq[1::2])

    return gf


def plot_ic_heatmap(factor_data, gf=None):
    """
    IC月度热力图

    Parameters
    ----------
    factor_data: pd.DataFrame - MultiIndex
        以date和asset为索引的DataFrame，数据涉及因子、不同周期收益、因
        子分位和资产分组。
    gf: utils.GridFigure
        用于绘图布局

    Returns
    -------
    gf: utils.GridFigure
    """
    mean_monthly_ic = perf.mean_information_coefficient(factor_data, by_time='M')

    fr_cols = len(mean_monthly_ic.columns)
    if gf is None:
        gf = utils.GridFigure(rows=fr_cols, cols=2)

    ax_monthly_ic_heatmap = [gf.next_cell() for _ in range(fr_cols)]
    plotting.plot_monthly_ic_heatmap(mean_monthly_ic, ax=ax_monthly_ic_heatmap)

    return gf


def plot_ic_bar_by_group(factor_data, gf=None):
    """
    绘制IC分组柱状图

    Parameters
    ----------
    factor_data: pd.DataFrame - MultiIndex
        以date和asset为索引的DataFrame，数据涉及因子、不同周期收益、因
        子分位和资产分组。
    gf: utils.GridFigure
        用于绘图布局

    Returns
    -------
    gf: utils.GridFigure
    """
    if gf is None:
        gf = utils.GridFigure(rows=1, cols=1)

    mean_group_ic = perf.mean_information_coefficient(factor_data,
                                                      by_group=True)
    plotting.plot_ic_by_group(mean_group_ic, ax=gf.next_row())

    return gf


def print_turnover_table(factor_data):
    """
    打印换手率相关信息

    Parameters
    ----------
    factor_data: pd.DataFrame - MultiIndex
        以date和asset为索引的DataFrame，数据涉及因子、不同周期收益、因
        子分位和资产分组。
    """
    turnover_periods = utils.get_forward_returns_columns(factor_data.columns)
    quantile_factor = factor_data['factor_quantile']

    quantile_turnover = \
        {p: pd.concat([perf.quantile_turnover(quantile_factor, q, p)
                       for q in range(1, int(quantile_factor.max()) + 1)],
                      axis=1) for p in turnover_periods}

    autocorrelation = pd.concat(
        [perf.factor_rank_autocorrelation(factor_data, period) for period in
         turnover_periods], axis=1)

    plotting.plot_turnover_table(autocorrelation, quantile_turnover)


def plot_top_bottom_quantile_turnover(factor_data, gf=None):
    """
    绘制高低分位的换手率在时间序列的变化曲线

    Parameters
    ----------
    factor_data: pd.DataFrame - MultiIndex
        以date和asset为索引的DataFrame，数据包括因子暴露、不同周期收益、因子分位和资产分组。
    gf: utils.GridFigure
        用于绘图布局

    Returns
    -------
    gf: utils.GridFigure
    """
    pass


def plot_factor_rank_auto_correlation(factor_data, gf=None):
    """
    绘制因子暴露在时间前后的自相关性

    Parameters
    ----------
    factor_data: pd.DataFrame - MultiIndex
        以date和asset为索引的DataFrame，数据包括因子暴露、不同周期收益、因子分位和资产分组。
    gf: utils.GridFigure
        用于绘图布局

    Returns
    -------
    gf: utils.GridFigure
    """

