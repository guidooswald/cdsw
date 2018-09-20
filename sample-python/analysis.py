# Setup
# -----

import pandas as pd
import seaborn as sns

# Basic Data Manipulation
# -----------------------
#
# Use the seaborn tips dataset to generate a best fitting linear regression line

tips = sns.load_dataset("tips")
sns.set(font="DejaVu Sans")
sns.jointplot("total_bill", "tip", tips, kind='reg').fig.suptitle("Tips Regression", y=1.01)

# Examine the difference between smokers and non smokers
sns.lmplot("total_bill", "tip", tips, col="smoker").fig.suptitle("Tips Regression - categorized by smoker", y=1.05)

# Explore the dataframe in grid output
pd.options.display.html.table_schema = True
tips.head()

# Using IPython's Rich Display System
# -----------------------------------
#
# IPython has a [rich display system](bit.ly/HHPOac) for
# interactive widgets.

from IPython.display import IFrame
from IPython.core.display import display

# Define a google maps function.
def gmaps(query):
  url = "https://maps.google.com/maps?q={0}&output=embed".format(query)
  display(IFrame(url, '700px', '450px'))

gmaps("Golden Gate Bridge")

# Worker Engines
# -----------------
#
# You can launch worker engines to distribute your work across a cluster.
# Uncomment the following to launch two workers with 2 cpu cores and 0.5GB
# memory each.

# import cdsw
# workers = cdsw.launch_workers(n=2, cpu=0.2, memory=0.5, code="print 'Hello from a CDSW Worker'")
