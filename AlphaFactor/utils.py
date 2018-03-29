import numpy as np
import pandas as pd

from IPython.display import display


class NonMatchingTimezoneError(Exception):
    pass


class MaxLossExceededError(Exception):
    pass


def rethrow(exception, additional_message):
    """
    Re-raise the last exception that was active in the current scope
    without losing the stacktrace but adding an additional message.
    This is hacky because it has to be compatible with both 2/3
    """

    e = exception
    m = additional_message
    if not e.args:
        e.args = (m,)
    else:
        e.args = (e.args[0] + m, ) + e.args[1:]

    raise e


def non_unique_bin_edges_error(func):
    """
    Give user a more informative error in case it is not possible
    to properly calculate quantiles on the input dataframe (factor).
    """

    message = """

    An error occurred while computing bins/quantiles on the input provided.
    This usually happens when the input contains too many identifical
    values and they span more than on quantile. The quantile are choosen
    to have the same number of records each, but the same value cannot span
    multiple quantiles. Possible workarounds are:
    1 - Decrease the number of quantiles
    2 - Specify a custom quantiles range, eg. [0, .50, .75, 1.] to get unequal
        number of records per quantile
    3 - Use 'bins' option instead of 'quantiles', 'bins' chooses the
        buckets to be evenly spaced according to the values themselves, while
        'quantiles' forces the buckets to have the same number of records.
    4 - for factors with discrete values use the 'bins' option with custom
        ranges and create a range for each discrete value
    Please see utils.get_clean_factor_and_forward_returns documentation for
    full documentation of 'bins' and 'quantiles' options.
"""

    def dec(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ValueError as e:
            if 'Bin edges must be unique' in str(e):
                rethrow(e, message)

    return dec


@non_unique_bin_edges_error
def quantize_factor(factor_data,
                    quantiles=5,
                    bins=None,
                    by_group=False):
    """
    计算每期因子值分位数

    Parameters
    ----------
    factor_data: pd.DataFrame - MultiIndex
        以date和asset为索引的DataFrame，数据包含因子值、不同周期未来收益
        以及asset所属分组。
    quantiles: int or sequence[float]
        分位数设置，如果为int，表示分位均分，如为list，按给定的分位数分位，
        如[0, 0.10, .50, 0.90, 1]
    bins: int or sequence[float]
        类似于quantiles，区别是按数值等分
    by_group: bool
        如为True，按组计算每个因子所属分位

    Returns
    -------
    factor_quantile: pd.Series
        以date和asset为索引的Series，值为因子分位数
    """

    def quantile_calc(x, _quantiles, _bins):
        if _quantiles is not None and _bins is None:
            return pd.qcut(x, _quantiles, labels=False) + 1
        elif _bins is not None and _quantiles is None:
            return pd.cut(x, _bins, labels=False)
        raise ValueError('Either quantiles or bins should be provided')

    grouper = [factor_data.index.get_level_values('date')]
    if by_group:
        grouper.append('group')

    factor_quantile = factor_data.groupby(grouper)['factor'].apply(quantile_calc, quantiles, bins)
    factor_quantile.name = 'factor_quantile'

    return factor_quantile.dropna()


def compute_forward_returns(prices,
                            periods=(1, 5, 10),
                            filter_zscore=None):
    """
    计算每个资产的N周期未来收益率

    Parameters
    ----------
    prices: pd.DataFrame
        价格数据，用于计算资产未来收益率，数据时间长度必须大于periods中的最大周期
    periods: sequence[int]
        未来收益率的计算周期
    filter_zscore: int or float, optional
        过滤异常收益，如果收益率减均值大于标准差的filter_zscore倍，则认为该值为异常值，
        设置为NaN。

    Returns
    -------
    forward_returns: pd.DataFrame - MultiIndex
        以date和asset为索引的DataFrame，值为未来收益率，每个列名为未来窗口大小，
        如'1D'、'5D'、'10D'等等。

        示例如下：
        -----------------------------------------------
                    |             | 1D  | 5D  | 10D
        -----------------------------------------------
            date    | asset
        -----------------------------------------------
                    | 000001.XSHE | 0.09|-0.01|-0.079
                    -----------------------------------
                    | 000002.XSHE | 0.02| 0.06| 0.020
                    -----------------------------------
        2014-01-01  | 600000.XSHG | 0.03| 0.09| 0.036
                    -----------------------------------
                    | 002415.XSHE |-0.02|-0.06|-0.029
                    -----------------------------------
                    | 600104.XSHG |-0.03| 0.05|-0.009
        -----------------------------------------------

    """

    forward_returns = pd.DataFrame(index=pd.MultiIndex.from_product(
        [prices.index, prices.columns], names=['date', 'asset']))

    for period in periods:
        delta = prices.pct_change(period).shift(-period)

        if filter_zscore is not None:
            mask = abs(delta - delta.mean()) > (filter_zscore * delta.std())
            delta[mask] = np.nan

        forward_returns[period] = delta.stack()

    print(forward_returns)
    forward_returns.index = forward_returns.index.rename(['date', 'asset'])

    return forward_returns


def demean_forward_returns(factor_data, grouper=None):
    """
    Convert forward returns to returns relative to mean
    period wise all-universe or group returns.
    group-wise normalization incorporates are assumption of a
    group neutral portfolio constraint and thus allows the
    factor to be evaluated across groups.

    Parameters
    ----------
    factor_data: pd.DataFrame - MultiIndex
        Forward returns in indexed by date and asset.
        Separate column for each forward return window.
    grouper: list
        If True, demean according to group.

    Returns
    -------
    adjusted_forward_returns: pd.DataFrame - MultiIndex
        DataFrame of the same format as the input, but with each
        security's returns normalize by group.
    """

    factor_data = factor_data.copy()

    if not grouper:
        grouper = factor_data.index.get_level_values('date')

    cols = get_forward_returns_columns(factor_data.columns)
    factor_data[cols] = factor_data.groupby(
        grouper)[cols].transform(lambda x: x - x.mean())

    return factor_data


def print_table(table, name=None, fmt=None):
    """
    Pretty print a pandas DataFrame.

    Uses HTML output if running inside Jupyter Notebook, otherwise
    formatted text output.

    Parameters
    ----------
    table : pd.Series or pd.DataFrame
        Table to pretty-print.
    name : str, optional
        Table name to display in upper left corner.
    fmt : str, optional
        Formatter to use for displaying table elements.
        E.g. '{0:.2f}%' for displaying 100 as '100.00%'.
        Restores original setting after displaying.
    """
    if isinstance(table, pd.Series):
        table = pd.DataFrame(table)

    if isinstance(table, pd.DataFrame):
        table.columns.name = name

    prev_option = pd.get_option('display.float_format')
    if fmt is not None:
        pd.set_option('display.float_format', lambda x: fmt.format(x))

    display(table)

    if fmt is not None:
        pd.set_option('display.float_format', prev_option)


def get_clean_factor_and_forward_returns(factor,
                                         prices,
                                         groupby=None,
                                         by_group=False,
                                         quantiles=5,
                                         bins=None,
                                         periods=(1, 5, 10),
                                         filter_zscore=20,
                                         groupby_labels=None):
    """
    将因子factor、价格prices、分组group等数据格式化，生成新的DataFrame，DataFrame多重索引
    为date与asset。通过get_clean_factor_and_forward_returns整理后的数据即可使用AlphaFactor进行
    分析。如数据已经符合AlphaFactor因子分参数要求，无需使用get_clean_factor_and_forward_return
    亦可。

    Parameters
    ----------
    factor: pd.Series - MultiIndex
        因子数据，以date(level)和asset(level1)为索引
        示例如下:
            -----------------------------------
                date    | asset       |
            -----------------------------------
                        | 000001.XSHE | 0.5
                        -----------------------
                        | 000002.XSHE | -1.1
                        -----------------------
            2014-01-01  | 600000.XSHG | 1.7
                        -----------------------
                        | 002415.XSHE | -0.1
                        -----------------------
                        | 600104.XSHG | 2.7
            -----------------------------------
    prices: pd.DataFrame
        价格数据，以date为索引，以assets为每列列名。确保价格数据的索引date无误，以
        防止用到未来数据。价格需覆盖因子起止时间向后推迟max(periods)交易日的时间窗
        口。举例说明，假设交易所无假日，如因子为20150101-20160101为数据，periods为
        [1, 5, 10]，则价格数据需至少覆盖20150111-20160111。

        示例如下：
            -----------------------------------------------------------------------------------
                        | 000001.XSHE | 000002.XSHE | 600000.XSHG | 002415.XSHE | 600104.XSHG
            -----------------------------------------------------------------------------------
               Date     |             |               |           |             |
            -----------------------------------------------------------------------------------
            2014-01-01  | 605.12      | 24.58         | 11.72     | 54.43       |  37.14
            -----------------------------------------------------------------------------------
            2014-01-02  | 604.35      | 22.23         | 12.21     | 52.78       |  33.63
            -----------------------------------------------------------------------------------
            2014-01-03  | 607.94      | 21.68         |  14.36    | 53.94       |  29.37
            -----------------------------------------------------------------------------------
    groupby: pd.Series - MultiIndex or dict
        分组数据，支持两种格式数据：以date和asset为多重索引的series，值为分组类型
        或asset与分组类型的dict映射，dict表示分组情况在整个分析测试期间不做改变。

        series分组示例：
            -----------------------------------
                date    | asset       |
            -----------------------------------
                        | 000001.XSHE | 000001
                        -----------------------
                        | 000002.XSHE | 000201
                        -----------------------
            2015-01-01  | 600000.XSHG | 000203
                        -----------------------
                        | 002415.XSHE | 000001
                        -----------------------
                        | 600104.XSHG | 000203
            -----------------------------------
            ...         | ...         | ...
            -----------------------------------
                        | 000001.XSHE | 000001
                        -----------------------
                        | 000002.XSHE | 000001
                        -----------------------
            2016-01-01  | 600000.XSHG | 000203
                        -----------------------
                        | 002415.XSHE | 000001
                        -----------------------
                        | 600104.XSHG | 000203
            -----------------------------------
        dict分组示例：
            {
                "000001.XSHE": '000001',
                "000002.XSHE": '000201',
                "600000.XSHE": '000203',
                "002415.XSHE": '000001',
                "600104.XSHE": '000203'
            }
    by_group: bool
        分组分析，如果True，对因子按组分析
    quantiles: int or sequence[float]
        分位quantile设置，如果设置为int，表示分位均分，如果设置list，不必均分，按指定list
        进行分位。
        quantiles与bins不能全为None。
    bins: int or sequence[int]
        数值bins切分，按数值进行切分，如果设置为int，表示数值均分，如果设置list，不均均分，
        按指定list进行数值切分。
    periods: sequence[int]
        用于因子与未来收益的周期，或理解为调仓周期，默认为[1, 5, 10]，即分析因子对未来1日
        收益、5日收益、10日收益的影响。
    filter_zscore: int or float, optional
        如果未来收益大于指定的标准偏差，设置未来收益为nan.
    groupby_labels: dict
        分组标签信息，例如
        {
            '000001': '消费',
            '000002': '能源'
            ...
        }

    Returns
    -------
    merged_data: pd.DataFrame - MultiIndex
        DataFrame结构数据，符合AlphaFactor分析函数参数要求，以date(level 0)和asset(level 1)
        为索引。其包含因子值、不同未来周期的收益、分组信息以及因子所属分位信息。

        示例如下:
           -------------------------------------------------------------------
                      |             | 1D  | 5D  | 10D  |factor|group|factor_quantile
           -------------------------------------------------------------------
               date   | asset       |     |     |      |      |     |
           -------------------------------------------------------------------
                      | 000001.XSHE | 0.09|-0.01|-0.079|  0.5 |  G1 |      3
                      --------------------------------------------------------
                      | 000002.XSHE | 0.02| 0.06| 0.020| -1.1 |  G2 |      5
                      --------------------------------------------------------
           2014-01-01 | 600000.XSHG | 0.03| 0.09| 0.036|  1.7 |  G2 |      1
                      --------------------------------------------------------
                      | 002415.XSHE |-0.02|-0.06|-0.029| -0.1 |  G3 |      5
                      --------------------------------------------------------
                      | 600104.XSHG |-0.03| 0.05|-0.009|  2.7 |  G1 |      2
           -------------------------------------------------------------------
    """

    if factor.index.levels[0].tz != prices.index.tz:
        raise NonMatchingTimezoneError("The timezone of 'factor' is not the "
                                       "same of the timezone of 'prices'. See "
                                       "the pandas methods tz_localize and "
                                       "tz_convert.")

    periods = sorted(periods)

    factor = factor.copy()
    factor.index = factor.index.rename(['date', 'asset'])

    merged_data = compute_forward_returns(prices, periods, filter_zscore)
    merged_data['factor'] = factor

    if groupby is not None:
        if isinstance(groupby, dict):
            diff = set(factor.index.get_level_values('asset')) - set(groupby.keys())
            if len(diff) > 0:
                raise KeyError(
                    'Assets {} not in group mapping'.format(
                        list(diff)))

            ss = pd.Series(groupby)
            groupby = pd.Series(index=factor.index,
                                data=ss[factor.index.get_level_values(
                                    'asset')].values)

        if groupby_labels is not None:
            diff = set(groupby.values) - set(groupby_labels.keys())
            if len(diff) > 0:
                raise KeyError('groups {} not passed group names'.format(
                    list(diff)))

            sn = pd.Series(groupby_labels)
            groupby = pd.Series(index=factor.index,
                                data=sn[groupby.values].values)

        merged_data['group'] = groupby.astype('category')

    merged_data = merged_data.dropna()

    merged_data['factor_quantile'] = quantize_factor(merged_data,
                                                     quantiles,
                                                     bins,
                                                     by_group)

    merged_data = merged_data.dropna()

    return merged_data


def common_start_returns(factor,
                         prices,
                         before,
                         after,
                         cumulative=False,
                         mean_by_date=False,
                         demean=None):
    """
    A date and equity pair is extracted from each index row in the factor
    dataframe and for each of these pairs a return series is built starting
    from 'before' the date and ending 'after' the date specified in the pair.
    All those returns series are then aligned to a common index (-before to
    after) and returned as a single DataFrame

    Parameters
    ----------
    factor: pd.DataFrame
        DataFrame with at least date and equity as index, the columns are
        irrelevant.
    prices: pd.DataFrame
        A wide form Pandas DataFrame indexed by date with assets
        in the columns. Pricing data should span the factor
        analysis time period plus/minus an additional buffer window
        corresponding to after/before period parameters.
    before: int
        How many returns to load before factor date.
    after: int
        How many returns to load after factor date.
    cumulative: bool, optional
        Return cumulative returns
    mean_by_date: bool, optional
        If True, compute mean returns for each date and return that
        instead of a return series for each asset.
    demean: pd.DataFrame, optional
        DataFrame with at least date and equity as index, the columns are
        irrelevant. For each date a list of equities is extracted from 'demean'
        index and used as universe to compute demeaned mean returns (long short
        portfolio)

    Returns
    -------
    aligned_returns: pd.DataFrame
        DataFrame containing returns series for each factor aligned to the same
        index: -before to after.
    """

    if cumulative:
        returns = prices
    else:
        returns = prices.pct_change(axis=0)

    all_returns = []

    for timestamp, df in factor.groupby(level='date'):

        equities = df.index.get_level_values('asset')

        try:
            day_zero_index = returns.index.get_loc(timestamp)
        except KeyError:
            continue

        starting_index = max(day_zero_index - before, 0)
        ending_index = min(day_zero_index + after + 1, len(returns.index))

        equities_slice = set(equities)
        if demean is not None:
            demean_equities = demean.loc[timestamp].index.get_level_values('asset')
            equities_slice |= set(demean_equities)

        series = returns.loc[returns.index[starting_index:ending_index],
                             equities_slice]
        series.index = range(starting_index - day_zero_index,
                             ending_index - day_zero_index)

        if cumulative:
            series = (series / series.loc[0, :]) - 1

        if demean is not None:
            mean = series.loc[:, demean_equities].mean(axis=1)
            series = series.loc[:, equities]
            series = series.sub(mean, axis=0)

        if mean_by_date:
            series = series.mean(axis=1)

        all_returns.append(series)

    return pd.concat(all_returns, axis=1)


def cumulative_returns(returns, period):
    """
    Builds cumulative returns from N-periods returns.

    When 'period' N is greater than 1 the cumulative returns plot is computed
    building and averaging the cumulative returns of N interleaved portfolios
    (started at subsequent periods 1,2,3,...,N) each one rebalancing every N
    periods.

    Parameters
    ----------
    returns: pd.Series
        pd.Series containing N-periods returns
    period: integer
        Period for which the returns are computed

    Returns
    -------
    pd.Series
        Cumulative returns series
    """

    returns = returns.fillna(0)

    if period == 1:
        return returns.add(1).cumprod()

    # build N portfolios from the single returns Series

    def split_portfolio(ret, period):
        return pd.DataFrame(np.diag(ret))

    sub_portfolios = returns.groupby(np.arange(len(returns.index)) // period,
                                     axis=0).apply(split_portfolio, period)
    sub_portfolios.index = returns.index

    def rate_of_returns(ret, period_):
        return ((np.nansum(ret) + 1) ** (1. / period_)) - 1

    sub_portfolios = sub_portfolios.rolling(
        window=period,
        min_periods=1
    ).apply(rate_of_returns, args=(period,))

    sub_portfolios = sub_portfolios.add(1).cumprod()

    return sub_portfolios.mean(axis=1)


def rate_of_return(period_ret):
    """
    1-period Growth Rate: the average rate of 1-period returns
    """

    return period_ret.add(1).pow(1 / period_ret.name).sub(1)


def std_conversation(period_std):
    """
    1-period standard deviation (or standard error) approximation

    Parameters
    ----------
    period_std: pd.DataFrame
        DataFrame containing standard deviation or standard error values
        with column headings representing the return period.

    Returns
    -------
    pd.DataFrame
        DataFrame in same format as input but with one-period
        standard deviation/ error values.
    """

    period_len = period_std.name
    return period_std / np.sqrt(period_len)


def get_forward_returns_columns(columns):
    return columns[columns.astype('str').str.isdigit()]


def winsorize(factor_data, win_type='norm_dist', n_draw=5, p_value=0.05):
    """
    因子去极值方法

    Parameters
    ----------
    factor_data: dict or pd.Series
        原始因子数据
    win_type: str
        去极值方法，有以下两种方式
        norm_dist   正态分布去极值，大于3标准差的值被视为异常
        quantile    分位数去极值
    n_draw: int
        正态分布去极值迭代次数，默认为5次，只有当win_type为'norm_dist'时有效
    p_value: float
        分位数去极值的指定分位数，默认为0.05，只有当win_type为'quantile'时有效

    Returns
    -------
    winsorize_factor_data: pd.Series
        去极值后的因子数据
    """

    factor_data = factor_data.copy()
    factor_data = factor_data if isinstance(factor_data, pd.Series) else pd.Series(factor_data)

    if win_type == 'norm_dist':
        for i in range(n_draw):
            mean = factor_data.mean()
            std = factor_data.std(ddof=1)
            factor_data[factor_data - mean <= - 3 * std] = mean - 3 * std
            factor_data[factor_data - mean >= 3 * std] = mean + 3 * std
    elif win_type == 'quantile':
        up = 1 - p_value / 2
        down = p_value / 2

        down_value = factor_data.quantile(down)
        factor_data[factor_data < down_value] = down_value

        up_value = factor_data.quantile(up)
        factor_data[factor_data > up_value] = up_value

    return factor_data


def standardize(factor_data):
    """
    因子标准化函数

    Parameters
    ----------
    factor_data: pd.Series or dict
        原始因子数据

    Returns
    -------
    standardize_factor_data: pd.Series
        标准化后的数据
    """

    factor_data = factor_data.copy()
    factor_data = factor_data if isinstance(factor_data, pd.Series) else pd.Series(factor_data)

    return (factor_data - factor_data.mean()) / factor_data.std(ddof=1)


def neutral(factor_data):
    """
    因子中性化函数

    Parameters
    ----------
    factor_data: pd.Series or dict
        原始因子数据

    Returns
    -------
    neutral_factor_data: pd.Series
        中性化后的因子
    """
    factor_data = factor_data.copy()

    return factor_data
