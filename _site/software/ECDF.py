#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import scipy.special
import pandas as pd
import subprocess

import os, sys
if "google.colab" in sys.modules:
    data_path = "https://s3.amazonaws.com/bebi103.caltech.edu/data/"
else:
    data_path = "../data/"

# Our main plotting package (must have explicit import of submodules)
import bokeh.io
import bokeh.plotting

# Enable viewing Bokeh plots in the notebook
bokeh.io.output_notebook()


# <div class="alert alert-info"> Great job with this problem! I love the thorough documentation and testing, which are great habits to develop. Your code and graphs look pretty good as well. Going forward though, I would recommend working on the problem together as much as possible, since having someone to brainstorm with can be very valuable in this class. 
#     34/35. Graded by Grace Liu </div>

# Rongrong and Kaiwen independently completed this problem, and Kaiwen integrated some of his code and suggestions into Rongrong's version.

# #### a)
# 
# Try to use a function to discribe the ECDF.

# In[3]:


def ecdfvals(data):
    """calculating the the ECDF(x) = fraction of data points ≤ x."""
    
    # Convert the array to a list and sort the list
    datap = data.tolist()
    datap.sort()
    
    # x will be the sorted list, showing x values from low to high.
    x = datap
    
    # create an empty list for y
    y = []
    
    # Calculate y, where y(x) = the fraction of data points ≤ x.
    for i in range(0, len(datap)):
        i = i + 1
        y.append(i/len(datap))
        
    return x, y


# Test this out by a dummy array:

# In[4]:


my_ar = np.array([1.1, 3.2, 2, 4])
ecdfvals(my_ar)


# This function succesfully sorted out the list, and produced the fraction of data points ≤ x.

# <div class="alert alert-info">Great job with this function. I really like that you guys documented it so well and also tested it on a dummy - that's a great practice in scientific computing. 15/15</div>

# #### b)

# Import the dataset from online, name as cts.

# In[5]:


cts = os.path.join("https://s3.amazonaws.com/bebi103.caltech.edu/data/gardner_time_to_catastrophe_dic_tidy.csv")


# <div class="alert alert-info">Since you already defined the data_path above, here you can write 'os.path.join(data_path, "gardner_time_to_catastrophe_dic_tidy.csv")'
#     </div>

# Read the file as a data frame.

# In[6]:


df = pd.read_csv(cts, header=[0])

# Take a look
df


# Slice the dataframe to be Pandas series, as input to the function.

# In[7]:


df_T = df.loc[df["labeled"] == True, :]
arr_df_T = df_T["time to catastrophe (s)"]
df_F = df.loc[df["labeled"] == False, :]
arr_df_F = df_F["time to catastrophe (s)"]


# Take a look at one of the arrays:

# In[8]:


arr_df_F


# It sorted out the unlabeled groups, as validated by comparing to the tail of the data frame. Setting up the Plot:

# In[9]:


p = bokeh.plotting.figure(
    width=400,
    height=300,
    x_axis_label="Catastrophe time(s)",
    y_axis_label="Cumulative distribution",
)


# Plot the scatter plot of the catastrophe time and its ECDF:

# In[10]:


p.circle(
    x=ecdfvals(arr_df_T)[0], #the [0] selects first output (x) from the function.
    y=ecdfvals(arr_df_T)[1], #the [1] selects second output (y) from the function.
    color="#a6bddb",
    legend_label="Labeled",
)

p.circle(
    x=ecdfvals(arr_df_F)[0],
    y=ecdfvals(arr_df_F)[1],
    color="#1c9099",
    legend_label="Unlabeled",
)

p.legend.location = "right"
p.legend.click_policy = "hide"

bokeh.io.show(p)


# <div class="alert alert-info">Great job with the plot! You have very informative axis labels and a legend, I would also include a title for the figure in the future. Your documentation here is again very good, and your conclusion is correct as well. Just as a note, I would recommend calling the function ecdfvals and defining the x's and y's before you do the plotting. If you do it within p.circle, it means that you have to run the function twice per series, while you should only have to run it once. It's not a huge problem right now because the functions are fast, but you will encounter functions that take several minutes (or more) to run in the future, and you always want to do fewer computations if possible. 19/20
#     </div>

# By eye test, there seems to be some differences between the two curves, but not deviating big enough. Therefore, it's more likely that the labeled group does not have different catastrophe time from the unlabeled group.

# In[ ]:




