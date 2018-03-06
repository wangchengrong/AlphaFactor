=======
API介绍
=======


本篇将介绍apfactor常用API。

.. note::

    演示数据来源于apdset，这是个人整理的一些易于使用的演示数据python库(暂不完整)。如获取平安银行某个时间的每日收益 returns = apdset.get_stock_returns('0000001', start='2013-01-01', end='2017-12-31')，某个时段的价格信息 prices = apdset.get_stock_prices('000001', start='2013-01-01', end='2017-12-31')。

工具API
----

utils.gets
------------

累计收益率或价格转化为每日收益率

:参数:

    * cum_returns_or_prices: pd.Series, np.ndarray or pd.DataFrame, 可选
        标的累计收益率或价格

    * one_or_zero: str, 可选
        初始净值，one表示1、zero表示0，默认为zero

:返回:

    * returns: pd.Series[float], np.ndarray[float] or pd.DataFrame[float]
        返回标的的累计年化收益

:示例:

通过平安银行每日价格获取每日收益。

::

    In [3]: aprical.calc_returns(prices)
    Out[3]: array([ 0.        ,  0.01929792, -0.01839163, ..., -0.02708638,
                   -0.00601956,  0.00681302])

annual_return
-------------

年化收益，采用复利累计方式计算

:参数:

    * returns: pd.Series, np.ndarray, or pd.DataFrame, 可选
        周期收益率序列
    * period: str, optional
        指定周期，用于决定年化因子大小。如果指定annualization，该值忽略。默认'daily'为244，'weekly'为52，'monthly'为12，
    * annualization: int, optional
        年化因子，如244、52等。如指定该值，period将被忽略

:返回:
    * annual_return: float
        年化收益

:示例:

计算平安银行2013年1月1日至2017年12月31的年化收益。

::

    In [2]: aprical.annual_return(returns)
    Out[2]: 0.19661396487274785


cum_returns
-----------

由指定周期的收益率计算累计收益率

:参数:

    * returns: pd.Series, np.ndarray, or pd.DataFrame
        周期收益率

    * starting_value: float, 可选
        累计收益率或净值的初始值

:返回:
    cum_returns: pd.Series, np.ndarray, or pd.DataFrame
        累计收益率序列

:示例:

计算平安银行2013年01月01日至2017年12月31日期间累计收益率序列。

::

    In [4]: aprical.cum_returns(returns)
    Out[4]: date
            2013-01-04    0.000000
            2013-01-07    0.019298
            2013-01-08    0.000551
            2013-01-09   -0.008271
            2013-01-10   -0.007535
            2013-01-11   -0.028304
                            ...
            2017-12-22    1.484837
            2017-12-25    1.435214
            2017-12-26    1.510568
            2017-12-27    1.442566
            2017-12-28    1.427863
            2017-12-29    1.444404


alpha
-----

阿尔法，CAPM模型表达式中的残余项，表示策略持有组合的收益中与市场整体无关的部分，是策略选股能力的度量。当策略所选股票的总体表现优于市场时，阿尔法为正，反之为负。

:参数:

    * returns: pd.Series
        指定周期收益率序列

    * bm_returns: pd.Series
        指定周期基准收益率，通常为市场指数

    * risk_free: float, 可选
        指定周期无风险收益率

:返回:

    * alpha: float
        阿尔法

:示例:

计算平安银行2013年01月01日至2017年12月31日期间的阿尔法

::

    In [5]: aprical.alpha(returns, bm_returns)
    Out[5]: 0.10712917133035547

.. note::

    bm_returns为基准收益序列，可通过 apdset.get_stock_returns('hs300', start='2013-01-01', end='2017-12-31') 获取。


beta
----

贝塔，CAPM模型中市场基准组合项的系数，表示策略组合收益对市场收益波动的敏感程度。

:参数:

    * returns: pd.Series
        指定周期收益率序列

    * bm_returns: pd.Series
        指定周期基准收益率序列

    * risk_free: float, optional
        指定周期无风险收益率

:返回:

    * beta: float
        贝塔

:示例:

计算平安银行2013年01月01日至2017年12月31日的贝塔。

::

    In [6]: aprical.beta(returns, bm_returns)
    Out[6]: 1.097511291861336


sharpe_ratio
------------

夏普比率，衡量策略相对于无风险组合的表现，是策略所获得风险溢价的度量，即策略额外承担一单位的风险，可以获得多少单位的风险补偿。

:参数:

    * returns: pd.Series, np.ndarray, or pd.DataFrame
        指定周期收益率序列

    * risk_free: float, 可选
        指定周期无风险收益，默认为0.0

    * period: str, 可选
        详情见 `annual_return`_ 介绍

    * annualization: int, 可选
        详情见 `annual_return`_ 介绍

:返回:

    * sharpe_ratio: float
        夏普比率

:示例:

计算平安银行2013年01月01日至2017年12月31日的夏普比率。

::

    In [7]: aprical.sharpe_ratio(returns)
    Out[7]: 0.6809710556604535


max_drawdown
------------

最大回撤，表示任意交易日向后推算，策略总收益走到最低点时收益率回撤幅度的最大值。最大回撤是评价策略极端风险管理能力的重要指标。

:参数:

    * returns: pd.Series or np.ndarray
        周期收益率序列

:返回:

    * max_drawdown: float
        最大回撤

:示例:

计算平安银行2013年01月01日至2017年12月31日的夏普比率。

::

    In [8]: aprical.max_drawdown(returns)
    Out[8]: -0.4528814528814529

最大回撤为0.4528814528814529。


omega_ratio
-----------

omega ratio是2002年提出的，新的业绩衡量指标，考虑了收益的整个分布情况，包含了所有高阶矩信息。

:参数:

    * returns: pd.Series
        指定周期收益率序列

    * risk_free: float，可选
        指定周期无风险收益率，默认为0

    * required_return: float，可选
        收益率阀值，投资者能接受的最低日收益率

    * annualization: int, 可选
        年化因子，一般收益率周期设定，如'daily'为244，'weekly'为52，'monthly'为12

:返回:

    * omega_ratio: float

:示例:

计算平安银行2013年01月01日至2017年12月31日的omega ratio。

::

    In [9]: aprical.omega_ratio(returns)
    Out[9]: 1.1443133991591286

omega ratio为1.1443133991591286，大于1

calmar_ratio
------------

Calmar比率(Calmar Ratio) 描述收益和最大回撤之间的关系。计算方式为年化收益率与历史最大回撤的比率。Calmar比率越大，基金的业绩表现越好。反之，基金的业绩表现越差。

:参数:

    * returns: pd.Series
        指定周期收益率序列

    * period: str, 可选
        详情见 `annual_return`_ 介绍，默认为'daily'

    * annualization: int, 可选
        详情见 `annual_return`_ 介绍

:返回:

    calmar_ratio: float

:示例:

计算平安银行2013年01月01日至2017年12月31日的omega ratio。

::

    In [10]: aprical.calmar_ratio(returns)
    Out[10]: 0.4341400241096071

sortino_ratio
-------------

索提诺比率，衡量投资组合相对表现的指标，与夏普比率类似，不同的是区分了波动的好坏。因此计算波动率是计算的不是标准差，而是下行标准差。因为投资组合的上涨符合投资者的期望。

:参数:

    * returns: pd.Series
        指定周期收益率序列

    * required_return: float
        指定周期投资者最低接收的收益率，默认为0.0

    * period: str, 可选
        详情见 `annual_return`_ 介绍，默认为'daily'

    * annualization: int, 可选
        详情见 `annual_return`_ 介绍

    * _downside_risk: float, 可选
        下行风险，如未提供，通过returns计算而得

:返回:

    * sortino_ratio: float
        索提诺比率

:示例:

计算平安银行2013年01月01日至2017年12月31日的索提诺比率。

::

    In [11]: aprical.sortino_ratio(returns)
    Out[11]: 1.0574420445426946

downside_risk
-------------

相比波动率，下行波动率对收益向下和向上波动做了区分，并认为只有收益向下波动才意味着风险。实际计算中可使用基准组合收益为目标收益，作为向上波动和向下波动的判断标准。

:参数:

    * returns: pd.Series, pd.DataFrame, np.ndarray

:返回:

:示例:

.. _annual_returns: http://aprical.readthedocs.io/en/latest/api.html#annual-return
