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
import scipy.stats as st
import scipy.optimize
import tqdm
import warnings
import bebi103
import iqplot
import bokeh.io
import numba
bokeh.io.output_notebook()


# Rongrong mainly worked on this problem and Kaiwen read through and comment on the solution. People fully understand what is written below.

# ### a)

# In[2]:


# Read the file:
df = pd.read_csv(os.path.join("https://s3.amazonaws.com/bebi103.caltech.edu/data/gardner_time_to_catastrophe_dic_tidy.csv"))
# Take a look:
df.head()


# Calculate the likelihood function and do the mle, codes mostly adopted from https://bebi103a.github.io/lessons/21/mle_confidence_intervals.html

# In[3]:


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


# Make a new Dataframe.

# In[4]:


df_mle = pd.DataFrame(columns=['Labeled', 'parameter', 'mle'])

# Slice the data based on labeled or not
df_labeled = df.loc[df["labeled"] == True]
df_unlabeled = df.loc[df["labeled"] == False]

# Calculate the mle and record in the new data frame
for i in [df_labeled, df_unlabeled]:
    mle = mle_iid_gamma(i["time to catastrophe (s)"].values)
    sub_df = pd.DataFrame({'Labeled': [i["labeled"].values[0]]*2, 'parameter': ['alpha', 'beta'], 'mle': mle})
    df_mle = pd.concat([df_mle, sub_df])

df_mle = df_mle.reset_index()


# Take a look.

# In[5]:


df_mle


# Drop the index column. And here is the **MLE of the parameters**:

# In[6]:


df_mle.drop(['index'], axis = 1)


# Now we try to get the confidence intervals. As we already proved before, the time to catastrophe is Gamma distribution, we will use the parametric Bootstrap to get the confidence intervals.

# In[7]:


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


# Define the function to randomly draw a new data set out of the model distribution parametrized by the MLE.

# In[8]:


def time_gamma(alpha, beta, size):
    return rg.gamma(alpha, 1/beta, size=size)


# Now we can calculate the confidence intervals.

# In[9]:


bs_reps_parametric = draw_parametric_bs_reps_mle(
    mle_iid_gamma,
    time_gamma,
    df_labeled["time to catastrophe (s)"].values,
    args=(),
    size=10000,
    progress_bar=True,
)


# In[10]:


np.percentile(bs_reps_parametric, [2.5, 97.5], axis=0)


# So we get the 95% confidence intervals for $\alpha$ and $\beta$ for the labeled microtubule's time to catastrophe, which are [2.04, 2.93] and [0.005, 0.007], respectively. Now let's take a look at the unlabeled microtubules:

# In[11]:


bs_reps_parametric_un = draw_parametric_bs_reps_mle(
    mle_iid_gamma,
    time_gamma,
    df_unlabeled["time to catastrophe (s)"].values,
    args=(),
    size=10000,
    progress_bar=True,
)


# In[12]:


np.percentile(bs_reps_parametric_un, [2.5, 97.5], axis=0)


# So we get the 95% confidence intervals for $\alpha$ and $\beta$ for the unlabeled microtubule's time to catastrophe, which are [1.70, 2.90] and [0.004, 0.007], respectively. 

# ### b)

# As we already proved in homework 4.2, when $\beta_1 = \beta_2$, the distribution of the catastrophe time satisfies the story of Gamma distribution, and we've already done the mle in question a). So here we will focus on the condition that $\beta_1 \neq \beta_2$.
# 
# We notice that in this question, the positions of $\beta_1$ and $\beta_2$ are interchangeable, so for calculation convenience, let's assume $\beta_2 > \beta_1$.
# 
# First we write down the log likelihood function:
# 
# \begin{align}
# L(\beta_1, \beta_2; t) = f(t; \beta_1, \beta_2) = \prod_{i=1}^n (\frac{\beta_1 \beta_2}{\beta_2 - \beta_1} (\mathrm e^{-\beta_1 t_i} - \mathrm e^{-\beta_2 t_i}))
# \end{align}
# 
# \begin{align}
# \ell(\beta_1, \beta_2; t) = log L(\beta_1, \beta_2; t) = n log \beta_1 + n log \beta_2 - n log (\beta_2 - \beta_1) + \sum_{i=1}^n log (\mathrm e^{-\beta_1 t_i} - \mathrm e^{-\beta_2 t_i})
# \end{align}
# 
# \begin{align}
# \frac{\partial \ell}{\partial \beta_1} &= \frac{n}{\beta_1} + \frac{n}{\beta_2 - \beta_1} - \sum_{i=1}^n \frac{t_i}{1 - \mathrm e^{(\beta_1 - \beta_2)t_i}}
# \end{align}
# 
# \begin{align}
# \frac{\partial \ell}{\partial \beta_2} &= \frac{n}{\beta_2} - \frac{n}{\beta_2 - \beta_1} + \sum_{i=1}^n \frac{t_i \mathrm e^{(\beta_1 - \beta_2)t_i}}{1 - \mathrm e^{(\beta_1 - \beta_2)t_i}}
# \end{align}
# 
# Let $\frac{\partial \ell}{\partial \beta_1} = 0$ and $\frac{\partial \ell}{\partial \beta_2} = 0$, and by adding (3) and (4) together, we will have:
# 
# \begin{align}
# \frac{1}{\beta_1} + \frac{1}{\beta_2} &= \frac{\sum_{i=1}^n t_i}{n}
# \end{align}
# 
# However, even we have got this constraint, it's still hard to not move away from the infinity in computation, which leads to the `RuntimeError`. So actually here we can first do a 'moment approximation', which is potentially helpful in getting a sense of the parameters.
# 
# \begin{align}
# \mu_1(t) = \frac{\sum_{i=1}^n t_i}{n} \\
# \mu_2(t) = \frac{\sum_{i=1}^n t_i^2}{n}
# \end{align}
# 
# According to the p.d.f, we can write down the theoretical moments of $\mu_1(t)$ and $\mu_2(t)$:
# 
# \begin{align}
# \mu_{1theo}(t) &= \int_0^{\infty} t f(t) dt \\
# &= \frac{\beta_1 \beta_2}{\beta_2 - \beta_1} \int_0^{\infty} t (\mathrm e^{-\beta_1 t} - \mathrm e^{-\beta_2 t}) dt \\
# &= \frac{\beta_1 \beta_2}{\beta_2 - \beta_1} (-\frac{t + \frac{1}{\beta_1}}{\beta_1} \mathrm e^{-\beta_1 t} + \frac{t + \frac{1}{\beta_2}}{\beta_2} \mathrm e^{-\beta_2 t}) |_{0}^{\infty} \\
# &= \frac{1}{\beta_1} + \frac{1}{\beta_2}
# \end{align}
# 
# By equating the theoretical $\mu_1$ and the $\mu_1$ from the sample, we have: 
# 
# \begin{align}
# \frac{1}{\beta_1} + \frac{1}{\beta_2} &= \frac{\sum_{i=1}^n t_i}{n}
# \end{align}
# 
# which is exactly the same from what we got from the MLE.
# 
# Then we go to calculate the theoretical value of $\mu_2(t)$:
# 
# \begin{align}
# \mu_{2theo}(t) &= \int_0^{\infty} t^2 f(t) dt \\
# &= \frac{\beta_1 \beta_2}{\beta_2 - \beta_1} \int_0^{\infty} t^2 (\mathrm e^{-\beta_1 t} - \mathrm e^{-\beta_2 t}) dt \\
# &= \frac{\beta_1 \beta_2}{\beta_2 - \beta_1} (-\frac{t^2 + \frac{2t}{\beta_1} + \frac{2}{\beta_1^2}}{\beta_1} + \frac{t^2 + \frac{2t}{\beta_2} + \frac{2}{\beta_2^2}}{\beta_2}) |_{0}^{\infty} \\
# &= \frac{2(\beta_1^2 + \beta_1 \beta_2 + \beta_2^2)}{\beta_1^2 \beta_2^2}
# \end{align}
# 
# By equating the theoretical $\mu_2$ and the $\mu_2$ from the sample, we have:
# 
# \begin{align}
# Var(t) = \mu_2(t) - (\mu_1(t))^2 \\
# Var_{theo}(t) = \mu_{2theo}(t) - (\mu_{1theo}(t))^2 &= \frac{1}{\beta_1^2} + \frac{1}{\beta_2^2}
# \end{align}
# 
# Now let's use the moment approximation to calculate the value of $\beta_1$ and $\beta_2$:

# In[13]:


a_ul = df_unlabeled["time to catastrophe (s)"].mean()
b_ul = df_unlabeled["time to catastrophe (s)"].var()

delta = 2*b_ul - (a_ul**2)

beta_1_u_1 = 2 / (a_ul + delta**0.5)
beta_1_u_2 = 2 / (a_ul - delta**0.5)

print('beta_1_unlabeled:', beta_1_u_1, beta_1_u_2)


# According to the symmetric property of the equations, we finally got the moment approximation of $\beta_1$ and $\beta_2$ for the unlabeled microtubule's time to catastrophe, which is (0.0037, 0.0072) or (0.0072, 0.0037). Actually they are quite close to the value of $\beta$ that we calculate in the question a) by using Gamma distribution. However, it seems that this equating moment strategy fails in giving an approximation of $\beta_1$ and $\beta_2$ of the labeled microtubule's time to catastrophe as the equations don't have a real number solution.

# Back to the problem, in this computation, we set the value of $\beta_1$ and $\beta_2 - \beta_1$ as parameters, in which the latter is given a name $\beta_d$. As described above, for computation convenience, we require that $\beta_1 < \beta_2$, therefore, we have $\beta_d > 0$.
# 
# Also, to avoid the calculation difficulty in computing $log(\mathrm e^{x_1} - \mathrm e^{x_2})$, we make it as $log(\mathrm e^{x_2}) + log(\mathrm e^{x_1 - x_2} - 1)$.
# 
# Besides, we also optimize the initial conditions as we have already figured out some of them based on Gamma distribution and moment equating approximation.

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
            x0=np.array([0.004, 0.001]),
            args=(n,),
            method='Nelder-Mead' #according to Dr. Justin Bois' hint in Ed platform
        )

    if res.success:
        return res.x
    else:
        raise RuntimeError('Convergence failed with message', res.message)


# In[16]:


# Calculating the mle for labeled microtubules:
betas_labeled = mle_iid_betas(df_labeled["time to catastrophe (s)"].values)
betas_labeled


# Therefore, the mle for labeled microtubule's catastrophe time is either (0.004536, 0.004541) or (0.004541, 0.004536).

# In[17]:


# Calculating the mle for unlabeled microtubules:
betas_unlabeled = mle_iid_betas(df_unlabeled["time to catastrophe (s)"].values)
betas_unlabeled


# Therefore, the mle for unlabeled microtubule's catastrophe time is either (0.003869, 0.006499) or (0.006499, 0.003869). Interestingly but not too surprisingly, this approximation is actually similar with what we calculate using the Moment Approximation method.

# ### c)

# Here we will do non-parametric Bootstrap to get the confidence intervals for $\beta_1$ and $\beta_2$. Actually the confidence interval here we get the confidence intervals for $\beta_1$ and $\beta_d$.

# In[18]:


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


# In[19]:


bs_reps = draw_bs_reps_mle(
    mle_iid_betas, df_labeled["time to catastrophe (s)"].values, size=10000, progress_bar=True,
)


# In[20]:


conf_int = np.percentile(bs_reps, [2.5, 97.5], axis=0)

conf_int


# In[21]:


conf_int_beta_2 = np.percentile(np.sum(bs_reps, axis = 1), [2.5, 97.5], axis=0)
conf_int_beta_2


# Therefore, the 95% confidence intervals of labeled microtubule's catastrophe time's $\beta_1$ and $\beta_2$ are [0.003427, 0.004953] and [0.004176, 0.006551], respectively. However, it should be noted that the positions of $\beta_1$ and $\beta_2$ are interchangeable as we discussed above. So actually we are calculating the confidence intervals of min{$\beta_1, \beta_2$} and max{$\beta_1, \beta_2$}.

# In[22]:


bs_reps_un = draw_bs_reps_mle(
    mle_iid_betas, df_unlabeled["time to catastrophe (s)"].values, size=10000, progress_bar=True,
)


# In[23]:


conf_int_un = np.percentile(bs_reps_un, [2.5, 97.5], axis=0)

conf_int_un


# In[24]:


conf_int_un_beta_2 = np.percentile(np.sum(bs_reps_un, axis = 1), [2.5, 97.5], axis=0)
conf_int_un_beta_2


# Therefore, the 95% confidence intervals of unlabeled microtubule's catastrophe time's $\beta_1$ and $\beta_2$ are [0.002832, 0.005540] and [0.004440, 0.01152], respectively. However, it should be noted that the positions of $\beta_1$ and $\beta_2$ are interchangeable as we discussed above. So actually we are calculating the confidence intervals of min{$\beta_1, \beta_2$} and max{$\beta_1, \beta_2$}.

# In[25]:


get_ipython().run_line_magic('load_ext', 'watermark')
get_ipython().run_line_magic('watermark', '-v -p numpy,pandas,numba,bokeh,iqplot,jupyterlab,tqdm')

