#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os, sys
if "google.colab" in sys.modules:
    data_path = "https://s3.amazonaws.com/bebi103.caltech.edu/data/"
else:
    data_path = "../data/"
    
import pandas as pd
import numpy as np
import scipy.stats
import bebi103
import iqplot
import scipy.stats as st
import bokeh.io
import warnings
import tqdm
import numba
bokeh.io.output_notebook()


# Kaiwen mainly worked on this problem and Rongrong reviewed and commented on the solution. We make sure people understand what is written below.

# ## a)
# 
# Read the data frame.

# In[2]:


df = pd.read_csv(os.path.join("https://s3.amazonaws.com/bebi103.caltech.edu/data/gardner_mt_catastrophe_only_tubulin.csv"), comment='#')
# taking a look
df.head()


# According to the comment, the values are the time before catastrophe in seconds, and the column names are the concentration of tubulin. Do some data wrangling to clean up the data frame.

# In[3]:


df_7 = df["7 uM"].values
df_9 = df["9 uM"].values
df_10 = df["10 uM"].values
df_12 = df["12 uM"].values
df_14 = df["14 uM"].values


# Create a new clean data frame.

# In[4]:


# naming
name_7 = np.array(["7 uM"]*len(df["7 uM"]))
name_9 = np.array(["9 uM"]*len(df["9 uM"]))
name_10 = np.array(["10 uM"]*len(df["10 uM"]))
name_12 = np.array(["12 uM"]*len(df["12 uM"]))
name_14 = np.array(["14 uM"]*len(df["14 uM"]))

data = np.concatenate((df_7,df_9,df_10,df_12,df_14))
name = np.concatenate((name_7,name_9,name_10,name_12,name_14))

# data frame
df_clean = pd.DataFrame({"concentration": name,"time": data})

#check
df_clean


# Drop NAN rows, according to https://hackersandslackers.com/pandas-dataframe-drop/#:~:text=If%20you're%20looking%20to,method%20is%20specifically%20for%20this.&text=Technically%20you%20could%20run%20df,rows%20where%20are%20completely%20empty.

# In[5]:


df_clean.dropna(
    axis=0,
    how='any',
    thresh=None,
    subset=None,
    inplace=True
)

df_clean


# Now we have a clean data frame! Let's proceed to EDA.

# In[6]:


p = iqplot.ecdf(
    data=df_clean,
    q="time",
    cats="concentration",
    style="staircase",
)

bokeh.io.show(p)


# There is a weak trend that if the concentration of tubulin is higher, the time it takes before catastrophe is longer.

# ## b)

# Perform parameter estimate for the Gamma distriubtion model with the dataset with 12 uM first. Define functions for calculating the log likelihood and MLE for the model.
# 
# ### Gamma model

# In[7]:


def log_like_iid_gamma(params, n):
    """Log likelihood for i.i.d. Gamma measurements, parametrized
    by alpha, beta."""
    alpha, beta = params

    if alpha <= 0 or beta <= 0:
        return -np.inf

    return np.sum(st.gamma.logpdf(n, alpha, scale=1/beta))


def mle_iid_gamma(n):
    """Perform maximum likelihood estimates for parameters for i.i.d.
    Gamma measurements, parametrized by alpha, beta"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        res = scipy.optimize.minimize(
            fun=lambda params, n: -log_like_iid_gamma(params, n),
            x0=np.array([3, 3]),
            args=(n,),
            method='Powell'
        )

    if res.success:
        return res.x
    else:
        raise RuntimeError('Convergence failed with message', res.message)


# Slice the data where the tubulin concentration is 12 uM.

# In[8]:


df_clean_12 = df_clean.loc[df_clean["concentration"]=="12 uM"]
df_clean_12


# Then we can perform MLE.

# In[9]:


df_mle_12 = pd.DataFrame(columns=['parameter', 'mle'])

# Calculate the mle and record in the new data frame
mle_12 = mle_iid_gamma(df_clean_12["time"].values)

sub_df = pd.DataFrame({'parameter': ['alpha', 'beta'], 'mle': mle_12})
df_mle_12 = pd.concat([df_mle_12, sub_df])
df_mle_12 = df_mle_12.reset_index()
df_mle_12

df_mle_12.drop(['index'], axis = 1)


# We can go on to calculate the confidence interval for the estimates.

# In[10]:


rg = np.random.default_rng(3252)

def draw_parametric_bs_reps_mle(
    mle_fun, gen_fun, data, args=(), size=1, progress_bar=False
):
    """Draw parametric bootstrap replicates of maximum likelihood estimator.

    Parameters
    ----------
    mle_fun : function
        Function with call signature mle_fun(data, *args) that computes
        a MLE for the parameters
    gen_fun : function
        Function to randomly draw a new data set out of the model
        distribution parametrized by the MLE. Must have call
        signature `gen_fun(*params, size)`.
    data : one-dimemsional Numpy array
        Array of measurements
    args : tuple, default ()
        Arguments to be passed to `mle_fun()`.
    size : int, default 1
        Number of bootstrap replicates to draw.
    progress_bar : bool, default False
        Whether or not to display progress bar.

    Returns
    -------
    output : numpy array
        Bootstrap replicates of MLEs.
    """
    params = mle_fun(data, *args)

    if progress_bar:
        iterator = tqdm.tqdm(range(size))
    else:
        iterator = range(size)

    return np.array(
        [mle_fun(gen_fun(*params, size=len(data), *args)) for _ in iterator]
    )


# Perform a parametric random number sampling based on the MLE parameters from the Gamma distribution.

# In[11]:


def time_gamma(alpha, beta, size):
    return rg.gamma(alpha, 1/beta, size=size)


# In[12]:


bs_reps_parametric = draw_parametric_bs_reps_mle(
    mle_iid_gamma,
    time_gamma,
    df_clean_12["time"].values,
    args=(),
    size=10000,
    progress_bar=True,
)


# Now we can calculate the confidence interval!

# In[13]:


np.percentile(bs_reps_parametric, [2.5, 97.5], axis=0)


# We have finished the part for the Gamma model. Let's proceed to the successive Poisson processes (with different rates) model. 
# 
# ### Successive Poisson processes model

# In this computation, we set the value of $\beta_1$ and $\beta_2 - \beta_1$ as parameters, in which the latter is given a name $\beta_d$. As described above, for computation convenience, we require that $\beta_1 < \beta_2$, therefore, we have $\beta_d > 0$.
# 
# Also, to avoid the calculation difficulty in computing $log(\mathrm e^{x_1} - \mathrm e^{x_2})$, we make it as $log(\mathrm e^{x_2}) + log(\mathrm e^{x_1 - x_2} - 1)$.
# 
# Now we construct the functions for making MLE of the parameters in the model.

# In[14]:


def log_like_iid_betas(params, n):
    """Log likelihood for i.i.d. measurements, parametrized
    by beta_1, beta_d."""
    beta_1, beta_d = params

    if beta_1 <= 0 or beta_d <= 0:
        return -np.inf
    
    log_non_exp_diff = (np.log(beta_1) + np.log(beta_d + beta_1) - np.log(beta_d))*len(n)
    
    return np.sum(-((beta_1 + beta_d)*n) + np.log(np.exp(beta_d*n) - 1)) + log_non_exp_diff


# In[15]:


def mle_iid_betas(n):
    """Perform maximum likelihood estimates for parameters for i.i.d.
     measurements, parametrized by beta_1, beta_d"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        res = scipy.optimize.minimize(
            fun=lambda params, n: -log_like_iid_betas(params, n),
            x0=np.array([0.004, 0.004]),
            args=(n,),
            method='Nelder-Mead' #according to Dr. Justin Bois' hint in Ed platform
        )

    if res.success:
        return res.x
    else:
        raise RuntimeError('Convergence failed with message', res.message)


# Calculating the MLE for microtubules in a tubulin concentration of 12 uM:

# In[16]:


betas_12 = mle_iid_betas(df_clean_12["time"].values)
betas_12


# The two values are beta1 and betad, and let's back calculate beta2.
# 
# beta1 = 0.00525313784 
# beta2 = 0.00525313784+0.00000160206380 = 0.005254739904
# 
# Due to the non-identifiability, the beta1 and beta2 can swap. So it would be either beta1 = 0.005253, beta2 = 0.005254, or beta1 = 0.005254, beta2 = 0.005253.

# Let's proceed to get the confidence interval. Create the functions to generate bootstrap replicates.

# In[17]:


def draw_bs_sample(data):
    """Draw a bootstrap sample from a 1D data set."""
    return rg.choice(data, size=len(data))


def draw_bs_reps_mle(mle_fun, data, args=(), size=1, progress_bar=False):
    """Draw nonparametric bootstrap replicates of maximum likelihood estimator."""
    if progress_bar:
        iterator = tqdm.tqdm(range(size))
    else:
        iterator = range(size)

    return np.array([mle_fun(draw_bs_sample(data), *args) for _ in iterator])


# Compute confidence interval.

# In[18]:


bs_reps = draw_bs_reps_mle(
    mle_iid_betas, df_clean_12["time"].values, size=100, progress_bar=True,
)


# The confidence intervals for beta1 and betad are (in columns):

# In[19]:


conf_int = np.percentile(bs_reps, [2.5, 97.5], axis=0)

conf_int


# The confidence interval for beta2 is therefore:

# In[20]:


conf_int_beta_2 = np.percentile(np.sum(bs_reps, axis = 1), [2.5, 97.5], axis=0)
conf_int_beta_2


# In summary,
# 
# CI beta 1 = [0.00506, 0.00547]
# CI beta 2 = [0.00506, 0.00549]
# 
# However, it should be noted that the positions of $\beta_1$ and $\beta_2$ are interchangeable as we discussed above. So actually we are calculating the confidence intervals of min{$\beta_1, \beta_2$} and max{$\beta_1, \beta_2$}.
# 
# The CI of both models look reasonable though.

# Now that we have MLE of the parameters for the gamma distribution and the successive Poisson model, we can go on to do model assessment. We will start from graphical assessment and go into Akaike's test. Let's test the gamma distribution first.

# ### Graphical assessment, gamma distribution.
# 
# Set up the parameters, according to https://numpy.org/doc/stable/reference/random/generated/numpy.random.gamma.html

# In[21]:


shape = mle_12[0]
scale = 1/mle_12[1]


# Sample from a gamma distribution with the MLE of parameters, with the size equal to the 12 uM data points, and sample with 100 repeats.

# In[22]:


rg = np.random.default_rng()

#shape = k = alpha scale = theta = 1/beta. 
single_samples = np.array(
    [rg.gamma(shape, scale, size=len(df["12 uM"])) for _ in range(100)]
)


# Now we can do graphical assessment, most code are from https://bebi103a.github.io/lessons/27/implementation_of_graphical_model_selection.html
# 
# First we will try Q-Q plot, it plots the simulated data against the original data set (both sorted from lowest to highest). If the two agree strongly with each other, the predicted traces (blue shade) will be aligning well with the diagonal.

# In[23]:


p = bebi103.viz.qqplot(
    data=df_clean_12["time"].values,
    samples=single_samples,
    x_axis_label="time before catastrophe (s)",
    y_axis_label="time before catastrophe (s)",
)

bokeh.io.show(p)


# As the prediction varies a bit from the diagonal, the model is not good enough to describe the data when the time is higher than 1000, but may fit better for the scenario where the time before catastrophe is below 1000.
# 
# Let's try predictive ECDFs too, for a more direct insight. Use bebi103 package for predicted ECDF (overlaying the empirical distribution with the ECDF generated from the sampled data).

# In[24]:


# Use discrete=True for discrete data
p1 = bebi103.viz.predictive_ecdf(
    samples=single_samples, data=df_clean_12["time"].values, discrete=True, x_axis_label="time before catastrophe (s)"
)

bokeh.io.show(p1)


# Most dots fall into the 95% confidence interval! If we take a closer look by plotting the differences between the ECDFs:

# In[25]:


p = bebi103.viz.predictive_ecdf(
    samples=single_samples, data=df_clean_12["time"].values, diff="ecdf", discrete=True, x_axis_label="time before catastrophe (s)"
)

bokeh.io.show(p)


# Actually quite a few points falls out of the 65% confidence interval of the predictive ECDF, but still most points are in the 95% confidence interval. Let's see how the other model does.

# ### Graphical assessment: successive Poisson
# Get the parameters set up.

# In[26]:


rg =  np.random.default_rng()

#try with beta 1 = 5, beta 2 = 5, the ratio = 1
beta1=betas_12[0]
beta2=betas_12[0]+betas_12[1]

t = np.zeros(len(df["12 uM"]))


# As the two events are independent with each other, the time t before anything happens will be the sum that each event happens

# In[27]:


for i in range(len(df["12 uM"])):
    t1 = rg.exponential(1/beta1) #t1 = time before event 1 happens
    t2 = rg.exponential(1/beta2) #t2 = time before event 2 happen
    t[i] = t1+t2 
    #the t1+t2 will be the total waiting time before the final event can happen.


# Define a function so that we can generate random numbers from the distribution parameterized by the MLE outcome.

# In[28]:


def rg_betas(beta1, beta2, size = 0):
    """random number sampling from successive Poisson process"""
    
    rg =  np.random.default_rng()
    t = np.zeros(size)
    
    for i in range(size):
        t1 = rg.exponential(1/beta1) #t1 = time before event 1 happens
        t2 = rg.exponential(1/beta2) #t2 = time before event 2 happen
        t[i] = t1+t2
    return t


# Sample data points.

# In[29]:


single_samples_betas = np.array(
    [rg_betas(beta1, beta2, size=len(df["12 uM"])) for _ in range(100)]
)


# Q-Q plot.

# In[30]:


p = bebi103.viz.qqplot(
    data=df_clean_12["time"].values,
    samples=single_samples_betas,
    x_axis_label="time before catastrophe (s)",
    y_axis_label="time before catastrophe (s)",
)

bokeh.io.show(p)


# It seems that the successive poisson model fits better? Check predictive ECDF too.

# In[31]:


# Use discrete=True for discrete data
p1 = bebi103.viz.predictive_ecdf(
    samples=single_samples_betas, data=df_clean_12["time"].values, discrete=True, x_axis_label="time before catastrophe (s)"
)

bokeh.io.show(p1)


# The deviation here is much more obvious comparing to the gamma distribution. This is also evident from the difference plot.

# In[32]:


p = bebi103.viz.predictive_ecdf(
    samples=single_samples_betas, data=df_clean_12["time"].values, diff="ecdf", discrete=True, x_axis_label="time before catastrophe (s)"
)

bokeh.io.show(p)


# We can see this is way worse than the gamma model: a lot more points fall out of the 95% confidence interval.

# For a final judgement, let's try Akaike information criterion:

# ### AIC
# First get the MLE parameters of the Gamma model regarding the 12 uM condition.

# In[33]:


df_mle_12 = pd.DataFrame(columns=['parameter', 'mle'])

# Calculate the mle and record in the new data frame
mle_12 = mle_iid_gamma(df_clean_12["time"].values)

sub_df = pd.DataFrame({'parameter': ['alpha', 'beta'], 'mle': mle_12})
df_mle_12 = pd.concat([df_mle_12, sub_df])
df_mle_12 = df_mle_12.reset_index()
df_mle_12

df_mle_12.drop(['index'], axis = 1)


# Calculate the log likelihood.

# In[34]:


ell_gamma = log_like_iid_gamma(df_mle_12["mle"].values, df_clean_12["time"].values)
ell_gamma


# Now try for the successive Poisson.

# In[35]:


beta1=betas_12[0]
beta2=betas_12[0]+betas_12[1]
param_b = np.array([beta1, beta2])
param_b


# Calculate the log likelihood.

# In[36]:


ell_betas = log_like_iid_betas(param_b, df_clean_12["time"].values)
ell_betas


# The log likelihood is a lot different from each other. Both models have 2 parameters. The AIC for each is therefore:

# In[37]:


AIC_gamma = -2 * (ell_gamma - 2)
AIC_betas = -2 * (ell_betas - 2)

AIC_max = max(AIC_gamma,AIC_betas)


# We go on to calculate the weight_gamma.

# In[38]:


numerator = np.exp(-(AIC_gamma - AIC_max)/2)
denominator = numerator + np.exp(-(AIC_betas - AIC_max)/2)
w_gamma = numerator / denominator
w_gamma


# It seems that the weight has a very strong preference towards the gamma model, which goes along with the graphical assessment. We just go with the gamma model.

# ## c)
# 
# Let's go with the gamma model!
# 
# Run everything again on 12 uM condition.

# In[39]:


df_clean_12 = df_clean.loc[df_clean["concentration"]=="12 uM"]
df_mle_12 = pd.DataFrame(columns=['parameter', 'mle'])

# Calculate the mle and record in the new data frame
mle_12 = mle_iid_gamma(df_clean_12["time"].values)

sub_df = pd.DataFrame({'parameter': ['alpha', 'beta'], 'mle': mle_12})
df_mle_12 = pd.concat([df_mle_12, sub_df])
df_mle_12 = df_mle_12.reset_index()
df_mle_12

df_mle_12 = df_mle_12.drop(['index'], axis = 1)

#CI
bs_reps_parametric = draw_parametric_bs_reps_mle(
    mle_iid_gamma,
    time_gamma,
    df_clean_12["time"].values,
    args=(),
    size=10000,
    progress_bar=True,
)

CI_12 = np.percentile(bs_reps_parametric, [2.5, 97.5], axis=0)

print(df_mle_12)
print(CI_12)


# A summary chart:

# In[40]:


df_mle_12["CI"] = ([CI_12[0][0], CI_12[1][0]], [CI_12[0][1],CI_12[1][1]])
df_mle_12


# 7 uM condition.

# In[41]:


df_clean_7 = df_clean.loc[df_clean["concentration"]=="7 uM"]
df_mle_7 = pd.DataFrame(columns=['parameter', 'mle'])

# Calculate the mle and record in the new data frame
mle_7 = mle_iid_gamma(df_clean_7["time"].values)

sub_df = pd.DataFrame({'parameter': ['alpha', 'beta'], 'mle': mle_7})
df_mle_7 = pd.concat([df_mle_7, sub_df])
df_mle_7 = df_mle_7.reset_index()
df_mle_7

df_mle_7 = df_mle_7.drop(['index'], axis = 1)

#CI
bs_reps_parametric = draw_parametric_bs_reps_mle(
    mle_iid_gamma,
    time_gamma,
    df_clean_7["time"].values,
    args=(),
    size=10000,
    progress_bar=True,
)

CI_7 = np.percentile(bs_reps_parametric, [2.5, 97.5], axis=0)


# In[42]:


df_mle_7["CI"] = ([CI_7[0][0], CI_7[1][0]], [CI_7[0][1],CI_7[1][1]])

df_mle_7


# 9 uM condition.

# In[43]:


df_clean_9 = df_clean.loc[df_clean["concentration"]=="9 uM"]
df_mle_9 = pd.DataFrame(columns=['parameter', 'mle'])

# Calculate the mle and record in the new data frame
mle_9 = mle_iid_gamma(df_clean_9["time"].values)

sub_df = pd.DataFrame({'parameter': ['alpha', 'beta'], 'mle': mle_9})
df_mle_9 = pd.concat([df_mle_9, sub_df])
df_mle_9 = df_mle_9.reset_index()
df_mle_9

df_mle_9 = df_mle_9.drop(['index'], axis = 1)

#CI
bs_reps_parametric = draw_parametric_bs_reps_mle(
    mle_iid_gamma,
    time_gamma,
    df_clean_9["time"].values,
    args=(),
    size=10000,
    progress_bar=True,
)

CI_9 = np.percentile(bs_reps_parametric, [2.5, 99.5], axis=0)

df_mle_9["CI"] = ([CI_9[0][0], CI_9[1][0]], [CI_9[0][1],CI_9[1][1]])

df_mle_9


# 10 uM condition.

# In[44]:


df_clean_10 = df_clean.loc[df_clean["concentration"]=="10 uM"]
df_mle_10 = pd.DataFrame(columns=['parameter', 'mle'])

# Calculate the mle and record in the new data frame
mle_10 = mle_iid_gamma(df_clean_10["time"].values)

sub_df = pd.DataFrame({'parameter': ['alpha', 'beta'], 'mle': mle_10})
df_mle_10 = pd.concat([df_mle_10, sub_df])
df_mle_10 = df_mle_10.reset_index()
df_mle_10

df_mle_10 = df_mle_10.drop(['index'], axis = 1)

#CI
bs_reps_parametric = draw_parametric_bs_reps_mle(
    mle_iid_gamma,
    time_gamma,
    df_clean_10["time"].values,
    args=(),
    size=10000,
    progress_bar=True,
)


# In[45]:


CI_10 = np.percentile(bs_reps_parametric, [2.5, 99.5], axis=0)

df_mle_10["CI"] = ([CI_10[0][0], CI_10[1][0]], [CI_10[0][1],CI_10[1][1]])

df_mle_10


# In[46]:


df_clean_14 = df_clean.loc[df_clean["concentration"]=="14 uM"]
df_mle_14 = pd.DataFrame(columns=['parameter', 'mle'])

# Calculate the mle and record in the new data frame
mle_14 = mle_iid_gamma(df_clean_14["time"].values)

sub_df = pd.DataFrame({'parameter': ['alpha', 'beta'], 'mle': mle_14})
df_mle_14 = pd.concat([df_mle_14, sub_df])
df_mle_14 = df_mle_14.reset_index()
df_mle_14

df_mle_14 = df_mle_14.drop(['index'], axis = 1)

#CI
bs_reps_parametric = draw_parametric_bs_reps_mle(
    mle_iid_gamma,
    time_gamma,
    df_clean_14["time"].values,
    args=(),
    size=10000,
    progress_bar=True,
)


# In[47]:


CI_14 = np.percentile(bs_reps_parametric, [2.5, 99.5], axis=0)

df_mle_14["CI"] = ([CI_14[0][0], CI_14[1][0]], [ CI_14[0][1],CI_14[1][1]])

df_mle_14


# Let's create a summary plot:

# In[48]:


summaries_a = [
    dict(label="7 uM", estimate=df_mle_7["mle"][0], conf_int=df_mle_7["CI"][0]),
    dict(label="9 uM", estimate=df_mle_9["mle"][0], conf_int=df_mle_9["CI"][0]),
    dict(label="10 uM", estimate=df_mle_10["mle"][0], conf_int=df_mle_10["CI"][0]),
    dict(label="12 uM", estimate=df_mle_12["mle"][0], conf_int=df_mle_12["CI"][0]),
    dict(label="14 uM", estimate=df_mle_14["mle"][0], conf_int=df_mle_14["CI"][0]),
]

p1 = bebi103.viz.confints(summaries_a, x_axis_label="alpha", frame_height=150)

summaries_b = [
    dict(label="7 uM", estimate=df_mle_7["mle"][1], conf_int=df_mle_7["CI"][1]),
    dict(label="9 uM", estimate=df_mle_9["mle"][1], conf_int=df_mle_9["CI"][1]),
    dict(label="10 uM", estimate=df_mle_10["mle"][1], conf_int=df_mle_10["CI"][1]),
    dict(label="12 uM", estimate=df_mle_12["mle"][1], conf_int=df_mle_12["CI"][1]),
    dict(label="14 uM", estimate=df_mle_14["mle"][1], conf_int=df_mle_14["CI"][1]),]

p1 = bebi103.viz.confints(summaries_a, x_axis_label="alpha", frame_height=150)

p2 = bebi103.viz.confints(summaries_b, x_axis_label="beta", frame_height=150)

bokeh.io.show(bokeh.layouts.gridplot([p1, p2], ncols=1))


# And a summary chart!

# In[49]:


df_mle_7["concentration"] = np.array(["7 uM","7 uM"])
df_mle_9["concentration"] = np.array(["9 uM","9 uM"])
df_mle_10["concentration"] = np.array(["10 uM","10 uM"])
df_mle_12["concentration"] = np.array(["12 uM","12 uM"])
df_mle_14["concentration"] = np.array(["14 uM","14 uM"])


# In[50]:


df_mle_all = pd.concat([df_mle_7, df_mle_9, df_mle_10, df_mle_12, df_mle_14])
df_mle_all


# In[51]:


bokeh.io.show(bokeh.layouts.gridplot([p1, p2], ncols=1))


# We can refer to the summary chart and plot to get conclusions.

# Looking at alpha, the numbers of arrivals it takes to start a catastrophe is higher in a higher tubulin concentration, probably meaning the chemical reactions it takes to achieve catastrophe is different in low tubulin concentration and high concentration, or as the microtubules can polymerize faster, the catastrophe becomes harder and it requires more reactions to triggger the catastrophe.
# 
# For beta, it seems that the rate of arrivals of the Poissosn process (the rate that one of the reactions leading to catastrophy happens) becomes faster from tubulin concentration at 7 uM - 10 uM, and then decreases at higher concentrations. It may indicate that there is an optimal tubulin concentration for the catastrophe to happen, and in high tubulin concentration (12 & 14 uM), the reaction leading to catastrphe may become slower, which may indicate that tubulin is antagonizing the catastrophe reactions and/or competing for the same substrate that is necessary for the catastrophe reactions to happen.

# In[52]:


get_ipython().run_line_magic('load_ext', 'watermark')
get_ipython().run_line_magic('watermark', '-v -p numpy,pandas,numba,bokeh,iqplot,jupyterlab,tqdm')

