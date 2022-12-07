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
import bokeh.io
import numba
bokeh.io.output_notebook()


# This problem is mainly solved by Rongrong. Kaiwen reviewed it and leave suggestions and necessary edits. Each person makes sure the content below are understood.

# #### a)

# Read the document:

# In[2]:


df = pd.read_csv(os.path.join("https://s3.amazonaws.com/bebi103.caltech.edu/data/gardner_time_to_catastrophe_dic_tidy.csv"))
# taking a look
df.head()


# Slice out the labeled microtubule from the dataframe and take a look.

# In[3]:


df_labeled = df.loc[df["labeled"] == True]
df_labeled.head()


# Slice out the unlabeled microtubule from the dataframe and take a look.

# In[4]:


df_unlabeled = df.loc[df["labeled"] == False]
df_unlabeled.head()


# We can use the `iqplot.ecdf()` function to directly plot the confidence interval: 

# In[5]:


p = iqplot.ecdf(data = df, q = "time to catastrophe (s)", cats = "labeled", 
                style = "staircase", conf_int = True)
bokeh.io.show(p)


# On the other hand, we can also use the Bootstrap method to generate the confidence intervals:

# In[6]:


rg = np.random.default_rng() 
#Adopted from https://bebi103a.github.io/lessons/15/intro_bootstrap.html

# Set up Numpy arrays for convenience
microtubule_labeled = df.loc[df["labeled"] == True, "time to catastrophe (s)"].values
microtubule_unlabeled = df.loc[df["labeled"] == False, "time to catastrophe (s)"].values

# ECDF values for plotting
labeled_ecdf = np.arange(1, len(microtubule_labeled)+1) / len(microtubule_labeled)
unlabeled_ecdf = np.arange(1, len(microtubule_unlabeled)+1) / len(microtubule_unlabeled)

# Make 100 bootstrap samples and plot them
for _ in range(100):
    bs_labeled = rg.choice(microtubule_labeled, size=len(microtubule_labeled))
    bs_unlabeled = rg.choice(microtubule_unlabeled, size=len(microtubule_unlabeled))

    # Add semitransparent ECDFs to the plot
    p.circle(np.sort(bs_labeled), labeled_ecdf, color=bokeh.palettes.Category10_3[0], alpha=0.02)
    p.circle(np.sort(bs_unlabeled), unlabeled_ecdf, color=bokeh.palettes.Category10_3[1], alpha=0.02)

bokeh.io.show(p)


# By just looking at the plot, I think they two could be identically distributed. 

# #### b)

# Use numba to do sampling, adopted from https://bebi103a.github.io/lessons/15/intro_bootstrap.html.
# 

# In[7]:


@numba.njit
def draw_bs_sample(data):
    """Draw a bootstrap sample from a 1D data set."""
    return np.random.choice(data, size=len(data))


# In[8]:


# Use numba to draw boostrap replicates of the mean from 1D data set.
@numba.njit
def draw_bs_reps_mean(data, size=1):
    """Draw boostrap replicates of the mean from 1D data set."""
    out = np.empty(size)
    for i in range(size):
        out[i] = np.mean(draw_bs_sample(data))
    return out


# In[9]:


# Get the bootstrap replicates
bs_reps_mean_labeled = draw_bs_reps_mean(microtubule_labeled, size=10000)
bs_reps_mean_unlabeled = draw_bs_reps_mean(microtubule_unlabeled, size=10000)


# In[10]:


# 95% confidence intervals
mean_labeled_conf_int = np.percentile(bs_reps_mean_labeled, [2.5, 97.5])
mean_unlabeled_conf_int = np.percentile(bs_reps_mean_unlabeled, [2.5, 97.5])

print("""
Mean microtubule catastrophe time 95% conf int labeled (s):   [{0:.2f}, {1:.2f}]
Mean microtubule catastrophe time 95% conf int unlabeled (s): [{2:.2f}, {3:.2f}]
""".format(*(tuple(mean_labeled_conf_int) + tuple(mean_unlabeled_conf_int))))


# It seems that the unlabeled microtubule catastrophe time is more spreading than the labeled microtubule catastrophe time.

# #### c)

# In this part, we use both the permutation hypothesis test and the Bootstrap hypothesis test to calculate the p-value.
# 
# #### Permutation hypothesis test
# 
# Define the function for computing plug-in estimate for the bivariate correlation coefficient, adopted from https://bebi103a.github.io/lessons/17/hacker_nhst.html
# 

# Generate the function for drawing permutation samples and array of permutation replicates.

# In[12]:


@numba.njit
def draw_perm_sample(x, y):
    """Generate a permutation sample."""
    concat_data = np.concatenate((x, y))
    np.random.shuffle(concat_data)

    return concat_data[:len(x)], concat_data[len(x):]

def draw_perm_reps_diff_mean(x, y, size=1):
    """Generate array of permuation replicates."""
    out = np.empty(size)
    for i in range(size):
        x_perm, y_perm = draw_perm_sample(x, y)
        out[i] = np.mean(x_perm) - np.mean(y_perm)

    return out


# Compute test statistic for original data set.

# In[33]:


diff_mean = np.mean(microtubule_unlabeled)- np.mean(microtubule_labeled)

# Draw replicates
perm_reps = draw_perm_reps_diff_mean(microtubule_unlabeled, microtubule_labeled, size=10000000)

# Compute p-value
p_val = np.sum(perm_reps >= diff_mean) / len(perm_reps)

print('p-value =', p_val)


# #### Bootstrap hypothesis test

# In[14]:


# Shift data sets
total_mean = np.mean(np.concatenate((microtubule_labeled, microtubule_unlabeled)))
microtubule_labeled_shift = microtubule_labeled - np.mean(microtubule_labeled) + total_mean
microtubule_unlabeled_shift = microtubule_unlabeled - np.mean(microtubule_unlabeled) + total_mean

# Plot the ECDFs
df_shift = pd.DataFrame(
    data={
        "treatment": ["labeled"] * len(microtubule_labeled_shift)
        + ["unlabeled"] * len(microtubule_unlabeled_shift),
        "time to catastrophe (s)": np.concatenate((microtubule_labeled_shift, microtubule_unlabeled_shift)),
    }
)
p = iqplot.ecdf(df_shift, q="time to catastrophe (s)", cats="treatment")

bokeh.io.show(p)


# In[15]:


# Plot the original ECDFs
p = iqplot.ecdf(
    df,
    q="time to catastrophe (s)",
    cats="labeled",
    x_axis_label="time to catastrophe (s)",
)

# Draw bootstrap samples out of these shifted distributions and plot them together with the original ECDFs.
for _ in range(100):
    labeled_rep = draw_bs_sample(microtubule_labeled_shift)
    unlabeled_rep = draw_bs_sample(microtubule_unlabeled_shift)
    df_rep = pd.DataFrame(
        data={
            "treatment": ["Labeled"] * len(labeled_rep) + ["Unlabeled"] * len(unlabeled_rep),
            "time to catastrophe (s)": np.concatenate((labeled_rep, unlabeled_rep)),
        }
    )

    p = iqplot.ecdf(
        df_rep,
        q="time to catastrophe (s)",
        cats="treatment",
        p=p,
        marker_kwargs=dict(alpha=0.02),
    )

bokeh.io.show(p)


# From the plot we can see that the blue-orange haze-like bootstrap samples totally mixed up with the original ECDFs.

# In[34]:


@numba.njit
def draw_bs_reps_diff_mean(x, y, size=1):
    """
    Generate bootstrap replicates with difference of means
    as the test statistic.
    """
    out = np.empty(size)
    for i in range(size):
        out[i] = np.mean(draw_bs_sample(x)) - np.mean(draw_bs_sample(y))

    return out


# Generate samples (10 million again)
bs_reps = draw_bs_reps_diff_mean(microtubule_unlabeled_shift, microtubule_labeled_shift, size=10000000)

# Compute p-value
p_val = np.sum(bs_reps >= diff_mean) / len(bs_reps)

print("p-value =", p_val)


# The p-value that we calculate using Boostrap hypothesis test is similar as using the permutation test.
# 
# Therefore, we can come to the conclusion that we cannot deny the null hypothesis that the labeled and unlabeled microtubules' time to catastrophe don't have significant differences.

# #### d)

# First we calculate the theoretically-approximated mean of the labeled microtubules' time to catastrophe.

# In[17]:


theoretical_mean_labeled = df_labeled["time to catastrophe (s)"].mean()
theoretical_mean_labeled


# Then we calculate the theoretically-approximated $\sigma^2$ of the labeled microtubules' time to catastrophe.

# In[18]:


sigma2_labeled_total = 0
for i in range(len(df_labeled)):
    sigma2_labeled_total = sigma2_labeled_total + 
    ((df_labeled["time to catastrophe (s)"][i] - theoretical_mean_labeled)**2)

sigma2_labeled = sigma2_labeled_total / len(df_labeled) / (len(df_labeled) - 1)
sigma2_labeled


# And then we can calculate the confidence intervals using the parameters $\mu$ and $\sigma^2$.

# In[19]:


scipy.stats.norm.interval(0.95, loc=theoretical_mean_labeled, scale=(sigma2_labeled)**0.5)


# Then we do the same thing for the unlabeled microtubules.

# In[20]:


theoretical_mean_unlabeled = df_unlabeled["time to catastrophe (s)"].mean()
theoretical_mean_unlabeled


# In[21]:


sigma2_unlabeled_total = 0
for i in range(len(df_unlabeled)):
    sigma2_unlabeled_total = sigma2_unlabeled_total + ((df_unlabeled["time to catastrophe (s)"][i + 211] - theoretical_mean_labeled)**2)

sigma2_unlabeled = sigma2_unlabeled_total / len(df_unlabeled) / (len(df_unlabeled) - 1)
sigma2_unlabeled


# In[22]:


scipy.stats.norm.interval(0.95, loc=theoretical_mean_unlabeled, scale=(sigma2_unlabeled)**0.5)


# In[23]:


print("Mean microtubule catastrophe time 95% conf int labeled (s):", 
      scipy.stats.norm.interval(0.95, loc=theoretical_mean_labeled, 
                                scale=(sigma2_labeled)**0.5))
print("Mean microtubule catastrophe time 95% conf int unlabeled (s):", 
      scipy.stats.norm.interval(0.95, loc=theoretical_mean_unlabeled, 
                                scale=(sigma2_unlabeled)**0.5))


# These quite matches what we got in part b).

# #### e)

# In[24]:


def ecdf(x, data):
    """calculating the the ECDF(x) = fraction of data points ≤ x."""
    
    # Convert the array to a list and sort the list
    datap = data.tolist()
    datap.sort()
    x_position = datap.index(x)
    y = (x_position + 1) / len(datap)
    return y


# #### f)

# In[25]:


# Calculate the value of epsilon of the labeled microtubules
epsilon_labeled = (1/2/len(df_labeled)*np.log(2/0.05)) ** 0.5
epsilon_labeled


# In[26]:


# Plot the ecdf of labeled microtubules and the upper boundary and the lower boundary
p_labeled = iqplot.ecdf(df_labeled, q="time to catastrophe (s)", cats="labeled")

for i in range(len(df_labeled)):
    lower_bound_labeled = max(0, ecdf(df_labeled["time to catastrophe (s)"][i], 
                                      df_labeled["time to catastrophe (s)"]) - epsilon_labeled)
    upper_bound_labeled = min(1, ecdf(df_labeled["time to catastrophe (s)"][i], 
                                      df_labeled["time to catastrophe (s)"]) + epsilon_labeled)
    p_labeled.circle(df_labeled["time to catastrophe (s)"][i], lower_bound_labeled, 
                     color="#d01c8b", alpha=0.3)
    p_labeled.circle(df_labeled["time to catastrophe (s)"][i], upper_bound_labeled, 
                     color="#4dac26", alpha=0.3)


bokeh.io.show(p_labeled)


# In[27]:


# Calculate the value of epsilon of the unlabeled microtubules
epsilon_unlabeled = (1/2/len(df_unlabeled)*np.log(2/0.05)) ** 0.5
epsilon_unlabeled


# In[28]:


# Plot the ecdf of unlabeled microtubules and the upper boundary and the lower boundary
p_unlabeled = iqplot.ecdf(df_unlabeled, q="time to catastrophe (s)", cats="labeled")

for i in range(len(df_unlabeled)):
    lower_bound_unlabeled = max(0, ecdf(df_unlabeled["time to catastrophe (s)"][i + 211], 
                                        df_unlabeled["time to catastrophe (s)"]) - epsilon_unlabeled)
    upper_bound_unlabeled = min(1, ecdf(df_unlabeled["time to catastrophe (s)"][i + 211], 
                                        df_unlabeled["time to catastrophe (s)"]) + epsilon_unlabeled)
    p_unlabeled.circle(df_unlabeled["time to catastrophe (s)"][i + 211], 
                       lower_bound_unlabeled, color="#f1b6da", alpha=0.3)
    p_unlabeled.circle(df_unlabeled["time to catastrophe (s)"][i + 211], 
                       upper_bound_unlabeled, color="#b8e186", alpha=0.3)

bokeh.io.show(p_unlabeled)


# In[29]:


# Plot everything altogether
p_total = iqplot.ecdf(df, q="time to catastrophe (s)", cats="labeled")

for i in range(len(df_labeled)):
    lower_bound_labeled = max(0, ecdf(df_labeled["time to catastrophe (s)"][i], 
                                      df_labeled["time to catastrophe (s)"]) - epsilon_labeled)
    upper_bound_labeled = min(1, ecdf(df_labeled["time to catastrophe (s)"][i], 
                                      df_labeled["time to catastrophe (s)"]) + epsilon_labeled)
    p_total.circle(df_labeled["time to catastrophe (s)"][i], lower_bound_labeled, 
                   color="#d01c8b", alpha=0.3)
    p_total.circle(df_labeled["time to catastrophe (s)"][i], upper_bound_labeled, 
                   color="#4dac26", alpha=0.3)
    
for i in range(len(df_unlabeled)):
    lower_bound_unlabeled = max(0, ecdf(df_unlabeled["time to catastrophe (s)"][i + 211], 
                                        df_unlabeled["time to catastrophe (s)"]) - epsilon_unlabeled)
    upper_bound_unlabeled = min(1, ecdf(df_unlabeled["time to catastrophe (s)"][i + 211], 
                                        df_unlabeled["time to catastrophe (s)"]) + epsilon_unlabeled)
    p_total.circle(df_unlabeled["time to catastrophe (s)"][i + 211], lower_bound_unlabeled, 
                   color="#f1b6da", alpha=0.3)
    p_total.circle(df_unlabeled["time to catastrophe (s)"][i + 211], upper_bound_unlabeled, 
                   color="#b8e186", alpha=0.3)

bokeh.io.show(p_total)


# From the plot, it seems that there are some differences between the upper boundary of the labeled and the unlabeled microtubules' time to catastrophe ecdfs. However, it's very hard to tell the differences between the lower boundary of the labeled and the unlabled microtubules' time to catastrophe ecdfs.

# In[30]:


get_ipython().run_line_magic('load_ext', 'watermark')
get_ipython().run_line_magic('watermark', '-v -p numpy,pandas,numba,bokeh,iqplot,jupyterlab')

