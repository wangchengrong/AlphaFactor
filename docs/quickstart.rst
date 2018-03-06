============
快速开始
============

安装
-------

直接通过pip命令::

   $ pip install aprical

测试数据由另外一个包apdset提供，主要提供一些便于使用的数据接口。

安装如下::

    $ pip install apdset

至此，安装完成！

使用
-------

场景：计算平安银行(000001)从2014年01月01日至2017-12-31日历史表现评价指标，比较基准沪深300。


数据
~~~~

首先，导入相关包::

    import apdset
    import aprical

获取组合和基准的历史表现，即平安银行和沪深300的历史收益::

    returns = apdset.get_stock_returns('0000001', start='2014-01-01', end='2017-12-31')
    bm_returns = apdset.get_stock_returns('hs300', start='2014-01-01', end='2017-12-31')

下面案例演示！

案例
~~~~

:计算年化收益:
::

     In[1] : aprical.annual_return(returns)
     Out[1]: 0.19267589510150018

:计算alpha:
::

    In[2] : aprical.alpha(returns, bm_returns)

    Out[2]: 0.0579482074583966

:计算最大回撤:
::

    In[3] : aprical.max_drawdown(returns)

    Out[3]: -0.45288145288145276

就是这么简单，更多API，查看API章节。
