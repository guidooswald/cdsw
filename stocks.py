#!pip install plotly

from pandas_datareader import data as web
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.finance import candlestick_ohlc
import numpy as np
import StringIO

# Read data from Yahoo stock API using [pandas](http://bit.ly/HHNQqn).

start = datetime.datetime(2014, 12, 1)
end = datetime.datetime(2017, 12, 31)
stocks = web.DataReader(['HDP','CLDR'], 'google', start, end)
print stocks
stocks['Open'].plot(title="Hadoop vendors")

# This line is necessary for the plot to appear in CDSW
%matplotlib inline
# Control the default size of figures in this CDSW
%pylab inline

stocks["50d"] = np.round(stocks["Close"].rolling(window = 50, center = False).mean(), 2)
stocks["200d"] = np.round(stocks["Close"].rolling(window = 200, center = False).mean(), 2)
 
pandas_candlestick_ohlc(stocks.loc['2017-01-04':'2017-06-07-19',:], otherseries = ["20d", "50d", "200d"])
