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
    Computes period wise factor quantiles

    Parameters
    ----------
    factor_data: pd.DataFrame - MultiIndex
        A MultiIndex DataFrame indexed by data (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for
        each period, the factor quantile/bin that factor value belongs to, and
        (optionally) the group the assets belongs to.
    quantiles: int or sequence[float]
        Number of equal-sized quantile buckets to use in factor bucketing.
        Alternately sequence of quantiles, allowing non-equal-sized buckets
        e.g. [0., .10, .5, .90, 1.] or [.05, .5, .95]
        Only one of 'quantiles' or 'bins' can be not-none
    bins: int or sequence[float]
        Number of equal-with (valuewise) bins to use in factor bucketing.
        Alternately sequence of bin edges allowing for non-uniform bin width
        e.g. [-4, -2, -0.5, 0, 10]
        Only one of 'quantiles' or 'bins' can be not-None
    by_group: bool
        If True, compute quantile buckets separately for each group.
    no_raise: bool, optional
        If True, no exceptions are thrown and the values for which the
        exception would have been thrown are set to np.NaN

    Returns
    -------
    factor_quantile: pd.Series
        Factor quantiles indexed by date and asset.
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


def compute_forward_returns(factor_idx,
                            prices,
                            periods=(1, 5, 10),
                            filter_zscore=None):
    """
    Finds the N period forward returns (as percent change) for each asset
    provided.

    Parameters
    ----------
    factor_idx: pd.DatetimeIndex
        The factor datetimes for which we are computing the forward returns
    prices: pd.DataFrame
        Pricing data to use in forward price calculation.
        Assets as columns, dates as index. Pricing data must
        span the factor analysis time period plus an additional buffer window
        that is greater than the maximum number of expected periods
        in the forward returns calculations.
    periods: sequence[int]
        periods to compute forward returns on.
    filter_zscore: int or float, optional
        Set forward returns greater than X standard deviations
        from the mean to nan. Set it to 'None' to avoid filtering.
        Caution: this outlier filtering incorporates lookahead bias.

    Returns
    -------
    forward_returns: pd.DataFrame - MultiIndex
        Forward returns in indexed by date and asset.
        Separate column for each forward return window.
    """

    factor_idx = factor_idx.intersection(prices.index)

    forward_returns = pd.DataFrame(index=pd.MultiIndex.from_product(
        [factor_idx, prices.columns], names=['date', 'asset']))

    custom_calendar = False

    for period in periods:
        #
        # build forward returns
        #
        delta = prices.pct_change(period).shift(-period)

        if filter_zscore is not None:
            mask = abs(delta - delta.mean()) > (filter_zscore * delta.std())
            delta[mask] = np.nan

        forward_returns[period] = delta.stack()

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
    Formats the factor data, pricing data, and group mappings into a DataFrame
    that contains aligned MultiIndex indices of timestamp and asset. The
    returned data will be formatted to be suitable for Apfactor functions.

    It is safe to skip a call to this function and still make use of Apfactor
    functionalities as long as the factor data conforms the format returned
    from get_clean_data_and_forward_returns and documented here.

    Parameters
    ----------
    factor: pd.Series - MultiIndex
        A MultiIndex Series indexed by a timestamp (level 0) and asset
        (level 1), containing the values for a single alpha factor.
        For example:
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
        A wide form Pandas DataFrame indexed by timestamp with assets
        in the columns. It is important to pass the
        correcting pricing data in depending on what time of period your
        signal was generated so to void lookahead bias, or
        delayed calculations. Pricing data must span the factor
        analysis time period plus an additional buffer window
        that is greater than the maximum number of the expected periods
        in the forward returns calculations.
        'Price' must contains at least an entry for each timestamp/asset
        combination in 'factor'. This entry must be the asset price
        at the time the asset factor value is computed and it will be
        considered the buy price for that asset at that timestamp.
        'Prices' must also contain entries for timestamps following each
        timestamp/asset combination in 'factor', as many more timestamps
        as the maximum value in 'periods'. The asset price after 'period'
        timestamps will be considered the sell price for that asset when
        computing 'period' forward returns.
        For example:
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
        Either A MultiIndex Series indexed by date and asset,
        containing the period wise group codes for each asset, or
        a dict of asset to group mappings. If a dict is passed,
        it is assumed that group mappings are unchanged for the
        entire time period of the passed factor data.
    by_group: bool
        If True, compute quantile buckets separately for each group.
    quantiles: int or sequence[float]
        Number of equal-sized quantile buckets to use in factor bucketing.
        Alternative sequence of quantiles, allowing non-equal-sized buckets
        eg. [0., .10, .5, .9, 1.] or [.05, .5, .9]
        Only one of 'quantiles' or 'bins' can be not-None
    bins: int or sequence[int]
        Number of equal-with (value wise) bins to use in factor bucketing
        Alternately sequence of bin edges allowing for non-uniform bin width
        e.g. [-4, -2, -0.5, 0, 10]
        Chooses the buckets to be evenly spaced according to the values
        themselves. Useful when the factor contains discrete values.
    periods: sequence[int]
        periods to compute forward returns on.
    filter_zscore: int or float, optional
        Sets forward returns greater than X standard deviations
        from the the mean to nan. Set it to 'None' to avoid filtering.
        Caution: this outlier filtering incorporates lookahead bias.
    groupby_labels: dict
        A dictionary keyed by group code with values corresponding
        to the display name for each group.

    Returns
    -------
    merged_data: pd.DataFrame - MultiIndex
        A MultiIndex Series indexed by date (level 0) and asset (level 1),
        containing the values for a single alpha factor, forward returns for
        each period, the factor quantile/bin that factor value belongs to, and
        (optionally) the group the assets belongs to.
        - forward returns column names follow  the format accepted by
          pd.Timedelta (e.g. '1D', '30m', '3h15m', '1D1h', etc)
        - 'date' index freq property (merged_data.index.level[0].freq) will be
          set to Calendar day or business day (pandas DateOffset) depending on
          what was inferred from the input data. This is currently used only in
          cumulative returns computation but it can be later set to any
          pd.DateOffset (e.g. US trading calendar) to increase the accuracy
          of the results
        For example:
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
    factor_dateindex = factor.index.get_level_values('date').unique()

    merged_data = compute_forward_returns(factor_dateindex, prices, periods,
                                          filter_zscore)
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
