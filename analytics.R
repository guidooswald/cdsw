# Setup
# -----
#
# The CDSW library includes
# a number of helper functions you can use from within R sessions.

library('cdsw')

# [ggplot2](http://ggplot2.org/) is a great way to make pretty graphs.

library('ggplot2')

# Load Data
# ---------
#
# Download and load Boston housing price data.

system('wget -nc https://raw.github.com/vincentarelbundock/Rdatasets/master/csv/MASS/Boston.csv')
boston <- read.csv('Boston.csv')
head(boston)
summary(boston$price)

# Explore
# -------

qplot(boston$medv, main="Boston Median Housing Price")
qplot(boston$crim, boston$medv,
  main="Median Housing Prices vs. Crime",
  xlab="Crime", ylab="Median Housing Price (thousands)")
qplot(boston$age, boston$medv,
    main="Median Housing Prices vs. Building Age",
    xlab="Building Age", ylab="Median Housing Price (thousands)") +
  geom_smooth(method = "loess")


# Model
# -----
#
# Regress median housing value on crime and building age.

fit <- lm(medv ~ crim + age, data=boston)
summary(fit)

# Worker Engines
# -----------------
#
# Worker engines run on remote containers and can be networked together
# to run distributed jobs.  Using workers, you can easily scale your analysis
# 100s of cores.  All workers share the same project filesystem, have
# access to a virtual private network, and multiplex their output into
# a single master engine for easy debugging.
#
# The following command launches two workers and runs some code remotely:

# workers <- launch.workers(n=2, cpu=0.2, memory=0.5, code="print('Hello from a CDSW Worker')")

# You can stop workers with `stop.workers()`.  All workers will automatically
# be cleaned up when you stop the session as well (top-right session drop down).
