import pandas as pd

import alphalens


factor_data = pd.read_pickle('factor_data.pkl')

alphalens.tears.create_returns_tear_sheet(factor_data)
