%pyspark
# Setup
# -----

#!pip install plotly

from pandas_datareader import data as web
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.finance import candlestick_ohlc#import plotly
#import plotly.plotly as py
#import plotly.graph_objs as go
import numpy as np
import StringIO

# plot something...
plot(rand(20), mfc='g', mec='r', ms=40, mew=4, ls='--', lw=3)

# evenly sampled time at 200ms intervals
t = np.arange(0., 5., 0.2)
print t

# red dashes, blue squares and green triangles
plt.plot(t, t, 'r--', t, t**2, 'bs', t, t**3, 'g^')

#some curves
def f(t):
    return np.exp(-t) * np.cos(2*np.pi*t)

t1 = np.arange(0.0, 5.0, 0.1)
t2 = np.arange(0.0, 5.0, 0.02)

plt.plot(t1, f(t1), 'bo', t2, f(t2), 'k')

plt.plot(t2, np.cos(2*np.pi*t2), 'r--')


# Fixing random state for reproducibility
np.random.seed(19680801)

mu, sigma = 100, 15
x = mu + sigma * np.random.randn(10000)

# the histogram of the data
n, bins, patches = plt.hist(x, 50, normed=1, facecolor='g', alpha=0.75)
#plt.xlabel('Smarts')
#plt.ylabel('Probability')
#plt.title('Histogram of IQ')
#plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
#plt.axis([40, 160, 0, 0.03])
#plt.grid(True)

def show(p):
    img = StringIO.StringIO()
    p.savefig(img, format='svg')
    img.seek(0)
    print "%html <div style='width:600px'>" + img.buf + "</div>"

# Example data
people = ('Tom', 'Dick', 'Harry', 'Slim', 'Jim')
y_pos = np.arange(len(people))
performance = 3 + 10 * np.random.rand(len(people))
error = np.random.rand(len(people))

plt.barh(y_pos, performance, xerr=error, align='center', alpha=0.4)
#plt.yticks(y_pos, people)
#plt.xlabel('Performance')
#plt.title('How fast do you want to go today?')

#show(plt)
