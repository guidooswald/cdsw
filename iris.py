!pip3 install seaborn

# code example from https://www.kaggle.com/ashokdavas/iris-data-analysis-pandas-numpy

# First, we'll import pandas, a data processing and CSV file I/O library
import pandas as pd

# We'll also import seaborn, a Python graphing library
import warnings # current version of seaborn generates a bunch of warnings that we'll ignore
warnings.filterwarnings("ignore")
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="white", color_codes=True)
import numpy as np

# Next, we'll load the Iris flower dataset, which is in the "../input/" directory
iris = pd.read_csv("input/iris.csv") # the iris dataset is now a Pandas DataFrame

# Let's see what's in the iris data - Jupyter notebooks print the result of the last thing you do
iris.head()

# Plot a histogram of the sepal length
iris['sepal.length'].hist(bins=30)

# univariate plots to see individual distribution
sns.distplot(a=iris["sepal.length"],rug=True) #kde=true & hist=true by default

# A final seaborn plot useful for looking at univariate relations is the kdeplot,
# which creates and visualizes a kernel density estimate of the underlying feature
# not univariate in complete sense. 3 different plots for 3 species values
sns.FacetGrid(iris, hue="variety", size=6) \
    .map(sns.kdeplot, "sepal.length") \
    .add_legend()

# you can see that all below combinations are providing a good distribution of "Species"
sns.factorplot(x="sepal.length",y="sepal.width",data=iris,hue="variety", size=10)
sns.factorplot(x="sepal.length",y="petal.length",data=iris,hue="variety", size=10)
sns.factorplot(x="petal.width",y="sepal.width",data=iris,hue="variety", size=10)
sns.factorplot(x="petal.length",y="petal.width",data=iris,hue="variety", size=10)

# let you easily view both a joint distribution and its marginals at once.
# A seaborn jointplot shows bivariate scatterplots and univariate histograms in the same figure
# Can't provide hue in joint plot
sns.jointplot(x="sepal.length", y="sepal.width", data=iris,size=5,kind="scatter") #scatter is default kind

# A clear picture of distribution can be seen with pairplot. Pairplot displays distribution of data
# according to every combination.
# In pair plot, members except diagonals are joint plot
sns.pairplot(iris,hue="variety",diag_kind="kde")
