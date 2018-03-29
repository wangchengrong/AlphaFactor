import pandas as pd
import AlphaFactor


ss = pd.Series(range(20000), index=['%06.f.XSHE' % i for i in range(20000)])
ss.iloc[-1] = 1000000
ss.iloc[-2] = 100000

# 去极值
print(AlphaFactor.utils.winsorize(ss, win_type='quantile'))

# 标准化
ss_winsorize = AlphaFactor.utils.winsorize(ss)
print(AlphaFactor.utils.standardize(ss_winsorize))
