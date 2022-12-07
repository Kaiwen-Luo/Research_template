#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import iqplot

import bokeh.io
bokeh.io.output_notebook()


# Kaiwen mainly worked on a) and b) and Rongrong mainly worked on c) and d). Then we checked for each other's outcome and discuss. Kaiwen also overlay the analytical CDF and ECDF and complete part d) with the Gamma distribution after discussion with Rongrong.

# ### a)
# 
# The story for Gamma distribution: the time length to wait before getting n arrivals of one Poisson process. This can be used to describe a process that can only be completed after multiple steps, with each step happening at the same rate.
# 
# In the problem's setting, as the catastrophe can not be well-characterized by the exponential distribution, it may not be a one-step process (i.e. the single arrival of a Poisson event may not be sufficient to trigger the catastrophe. Therefore, it may take multiple independent steps before the catastrophe can happen. If each step happens with the same rate, the waiting time before n steps happens (i.e. how long it takes before catastrophe happens) can be described by the Gamma distribution.

# ### b)

# Set up the parameters:

# In[2]:


rg =  np.random.default_rng()

#try with beta 1 = 5, beta 2 = 5, the ratio = 1
beta1 = 5
beta2 = 5


# Generate the list of time before waiting for plotting.

# In[3]:


t = np.zeros(150)


# Now loop over for 150 times to simulate the 150 experiments.

# In[4]:


for i in range(150):
    t1 = rg.exponential(1/beta1) #t1 = time before event 1 happens
    t2 = rg.exponential(1/beta2) #t2 = time before event 2 happen
    t[i] = t1+t2 
    #the t1+t2 will be the total waiting time before the final event can happen.


# If the $\beta_1/\beta_2$ = 10:

# In[5]:


beta1 = 50
beta2 = 5

#Generate a second list of time before waiting for plotting.
t_II = np.zeros(150)


# Loop over for 150 times:

# In[6]:


for i in range(150):
    t1 = rg.exponential(1/beta1) 
    t2 = rg.exponential(1/beta2) 
    t_II[i] = t1+t2


# If the $\beta_1/\beta_2$ = 100:

# In[7]:


beta1 = 500
beta2 = 5

#The third array
t_III = np.zeros(150)

for i in range(150):
    t1 = rg.exponential(1/beta1) 
    t2 = rg.exponential(1/beta2) 
    t_III[i] = t1+t2


# If the $\beta_1/\beta_2$ = 0.1:

# In[8]:


beta1 = 0.5
beta2 = 5

#The fourth array
t_IV = np.zeros(150)

for i in range(150):
    t1 = rg.exponential(1/beta1) 
    t2 = rg.exponential(1/beta2) 
    t_IV[i] = t1+t2


# If the $\beta_1/\beta_2$ = 0.01:

# In[9]:


#If the beta1/beta2 = 0.01
beta1 = 0.05
beta2 = 5

#The fifth array
t_V = np.zeros(150)

for i in range(150):
    t1 = rg.exponential(1/beta1) 
    t2 = rg.exponential(1/beta2) 
    t_V[i] = t1+t2


# Combine the columns into one data frame.

# In[10]:


t_collect = np.concatenate((t, t_II, t_III, t_IV, t_V))


# Now create lists for the names of different groups.

# In[11]:


#from https://stackoverflow.com/questions/5891410/numpy-array-initialization-fill-with-identical-values
t_n = np.empty(150); t_n.fill("1")
t_n_II = np.empty(150); t_n_II.fill("10")
t_n_III = np.empty(150); t_n_III.fill("100")
t_n_IV = np.empty(150); t_n_IV.fill("0.1")
t_n_V = np.empty(150); t_n_V.fill("0.01")

t_n_collect = np.concatenate((t_n, t_n_II, t_n_III, t_n_IV, t_n_V))


# Generate a data frame with both time and $\beta_1/\beta_2$.

# In[12]:


df=pd.DataFrame({"time": t_collect,
                 "beta1/beta2":t_n_collect})
#take a look
df.head()


# Plot the ECDF in units of $\beta_1^{-1}$.

# In[13]:


p = iqplot.ecdf(
    data=df,
    q="time",
    cats="beta1/beta2",
    style="staircase",
)

bokeh.io.show(p)


# From the plot we can see when we keep the beta2 unchanged while if we have higher beta1, it will help reduce the waiting time before observing a catastrophe. It's reasonable as if the first step happens faster, the whole process can speed up.

# What if we keep beta 1 a constant and vary beta 2?

# If the $\beta_1/\beta_2$ = 1:

# In[14]:


beta1 = 5
beta2 = 5

#The third array
t = np.zeros(150)

for i in range(150):
    t1 = rg.exponential(1/beta1) 
    t2 = rg.exponential(1/beta2) 
    t[i] = t1+t2


# If the $\beta_1/\beta_2$ = 10:

# In[15]:


#If the beta1/beta2 = 10
beta1 = 5
beta2 = 0.5

#The third array
t_II = np.zeros(150)

for i in range(150):
    t1 = rg.exponential(1/beta1) 
    t2 = rg.exponential(1/beta2) 
    t_II[i] = t1+t2


# If the $\beta_1/\beta_2$ = 100:

# In[16]:


#If the beta1/beta2 = 100
beta1 = 5
beta2 = 0.05

#The third array
t_II = np.zeros(150)

for i in range(150):
    t1 = rg.exponential(1/beta1) 
    t2 = rg.exponential(1/beta2) 
    t_III[i] = t1+t2


# If the $\beta_1/\beta_2$ = 0.1:

# In[17]:


#If the beta1/beta2 = 0.1
beta1 = 5
beta2 = 50

#The third array
t_IV = np.zeros(150)

for i in range(150):
    t1 = rg.exponential(1/beta1) 
    t2 = rg.exponential(1/beta2) 
    t_IV[i] = t1+t2


# If the $\beta_1/\beta_2$ = 0.01:

# In[18]:


#If the beta1/beta2 = 0.01
beta1 = 5
beta2 = 500

#The third array
t_V = np.zeros(150)

for i in range(150):
    t1 = rg.exponential(1/beta1) 
    t2 = rg.exponential(1/beta2) 
    t_V[i] = t1+t2


# Similarly, create a list of names:

# In[19]:


t_collect = np.concatenate((t, t_II, t_III, t_IV, t_V))
#from https://stackoverflow.com/questions/5891410/numpy-array-initialization-fill-with-identical-values
t_n = np.empty(150); t_n.fill("1")
t_n_II = np.empty(150); t_n_II.fill("10")
t_n_III = np.empty(150); t_n_III.fill("100")
t_n_IV = np.empty(150); t_n_IV.fill("0.1")
t_n_V = np.empty(150); t_n_V.fill("0.01")

t_n_collect = np.concatenate((t_n, t_n_II, t_n_III, t_n_IV, t_n_V))


# And create the final data frame:

# In[20]:


df=pd.DataFrame({"time": t_collect,
                 "beta1/beta2":t_n_collect})

#Take a look
df.head()


# Make a plot:

# In[21]:


p = iqplot.ecdf(
    data=df,
    q="time",
    cats="beta1/beta2",
    style="staircase",
)

bokeh.io.show(p)


# Now if we keep beta1 fixed while varying beta2, we can have faster catastrophe if we have a bigger beta2 (where the ratio is lower), which also makes sense as both steps contribute to the final outcome.

# ### c)
# 
# Let's make a simple model of microtubule catastrope by assigning three states $(A, B, C)$, or say, viewing it as two irreversible chemical reactions:
# 
# \begin{align}
# A \xrightarrow{\text{$\beta_1$}} B \xrightarrow{\text{$\beta_2$}} C
# \end{align}

# And we can make the assumption that at $t = 0$, the system only contains $A$ and the catastrophe is triggered. Assume that the initial concentration of $A$ is $[A_0]$. Meanwhile, we define the function $H(t)$. $H(t)$ is defined as the probability of one molecule $A$ completely converting to one molecule $C$ at time $t$. Therefore, it is equal to the c.d.f of the total waiting time of $A$ converting to $C$, as the defination of the waiting time c.d.f is the proportion of molecules that have been converted during the time course $t$. So we have: $\frac{d H(t)}{d t} = h(t)$, in which $h(t)$ represents the p.d.f of the total waiting time.

# Now we can write down the ODEs for $A, B, C$:
# 
# \begin{align}
# \frac{dA}{dt} &= - \beta_1 A \\
# \frac{dB}{dt} &= \beta_1 A - \beta_2 B \\
# \frac{dC}{dt} &= \beta_2 B
# \end{align}
# 
# By simply applying the **General Solution First Order Linear Non-Homogeneous Differential Equations**, we have:
# 
# \begin{align}
# A &= [A_0] \mathrm {e}^{-\beta_1 t} \\
# B &= \begin{cases}
#       \frac{[A_0] \beta_1}{\beta_2 - \beta_1} (\mathrm {e}^{-\beta_1 t} - \mathrm {e}^{-\beta_2 t}) & \text{if $\beta_1 \neq \beta_2$}\\
#       [A_0] \beta_1 t \mathrm {e}^{-\beta_2 t} & \text{if $\beta_1 = \beta_2$}
#     \end{cases} 
# \end{align}
# 
# Now we have:
# 
# \begin{align}
# \frac{dC}{dt} &= \begin{cases}
#       \frac{[A_0] \beta_1 \beta_2}{\beta_2 - \beta_1} (\mathrm {e}^{-\beta_1 t} - \mathrm {e}^{-\beta_2 t}) & \text{if $\beta_1 \neq \beta_2$}\\
#       [A_0] \beta_1 \beta_2 t \mathrm {e}^{-\beta_2 t} & \text{if $\beta_1 = \beta_2$}
#     \end{cases} 
# \end{align}
# 
# If we view the reactions as a whole, we can come to this:
# 
# \begin{align}
# C &= [A_0] H(t)
# \end{align}
# 
# We do the derivative and we have:
# 
# \begin{align}
# \frac{dC}{dt} &= [A_0] h(t)
# \end{align}
# 
# We can solve the $h(t)$ now!
# 
# \begin{align}
# h(t) &= \frac{\beta_1 \beta_2}{\beta_2 - \beta_1} (\mathrm {e}^{-\beta_1 t} - \mathrm {e}^{-\beta_2 t}) & \text{if $\beta_1 \neq \beta_2$}
# \end{align}
# 
# That's the $f(t; \beta_1, \beta_2)$ that we are trying to get!
# 
# And just by simply doing the integral, we have:
# \begin{align}
# F(t; \beta_1, \beta_2) &= \int_{0}^{t} \mathrm{d}t' \, \frac{\beta_1 \beta_2}{\beta_2 - \beta_1} (\mathrm {e}^{-\beta_1 t'} - \mathrm {e}^{-\beta_2 t'}) \\
#                        &= \frac{\beta_1 \beta_2}{\beta_2-\beta_1}\left[\frac{1}{\beta_1}\left(1-\mathrm{e}^{- \beta_1 t}\right)- \frac{1}{\beta_2}\left(1-\mathrm{e}^{-\beta_2 t}\right)\right].
# \end{align}
# 
# **Note:**
# 
# The General Solution First Order Linear Non-Homogeneous Differential Equations is given below.
# 
# \begin{align}
# \frac{d y}{d t} + p(t)y &= f(t) \\
# y(t) &= \mathrm{e}^{-\int p(t) \mathrm{d}t } [\int \mathrm{e}^{\int p(t) \mathrm{d}t} f(t) \mathrm{d}t \ + C] & \text{(C is a constant)}
# \end{align}

# We go on to overlay the CDF with one ECDF with one parameter set from b).

# In[22]:


beta1 = 50
beta2 = 5

t_II = np.zeros(150)

for i in range(150):
    t1 = rg.exponential(1/beta1) 
    t2 = rg.exponential(1/beta2) 
    t_II[i] = t1+t2


# We want to sort the time so that the time points are in order.

# In[23]:


t_II = np.sort(t_II)


# We want to plot the ECDF for this one, but not using the iqplot, so that we can overlay with the CDF easily. Recall taht we created a ecdf function back in hw1.3.

# In[24]:


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


# Define the time (x1) and accumulative probability (y1) in ECDF.

# In[25]:


x1 = ecdfvals(t_II)[0]
y1 = ecdfvals(t_II)[1]


# Now we calculate the CDF prediction of possibility at each time point we generated.

# In[26]:


F_CDF = beta1*beta2/(beta2-beta1)*(1/beta1*(1-np.exp(-1*beta1*t_II))-1/beta2*(1-np.exp(-1*beta2*t_II)))


# Set up the plot:

# In[27]:


p = bokeh.plotting.figure(
    width=400,
    height=300,
    x_axis_label="Catastrophe time",
    y_axis_label="Cumulative Probability",
)


# Make the final plot:

# In[28]:


p.circle(
    x=x1,
    y=y1,
    color="#a6bddb",
    legend_label="Sampled: ECDF",
)

p.circle(
    x=x1,
    y=F_CDF,
    color="#1c9099",
    legend_label="Model:CDF",
)

p.legend.location = "right"
p.legend.click_policy = "hide"

bokeh.io.show(p)


# We can see these two are very close to each other. So the CDF and ECDF match each other!

# ### d)
# 
# As what has already been proved in c), if $\beta_1 = \beta_2$, we have $f(t) = \beta^2 t \mathrm{e}^{-\beta t}$.
# 
# Also, note that when the $\beta$s are the same, the condition is under the story of Gamma distribution, as discussed in a). 
# 
# We know that the number of arrival is 2, thus $\alpha = 2$. Meanwhile $\beta = \beta$. Plug in these values into the expression for Gamma distribution:
# 
# 
# \begin{align}
# \frac{1}{(2-1)!} \frac{(\beta t)^{2}}{t} e^{-\beta t}  &= 1 \times \beta^{2} t e^{-\beta t}  \\
# &= \beta^{2} t e^{-\beta t} \\
# \end{align}
# 
# Therefore, we have: $f(t;\beta)=\beta^{2} t e^{-\beta t}$.

# In[29]:


get_ipython().run_line_magic('load_ext', 'watermark')
get_ipython().run_line_magic('watermark', '-v -p pandas,bokeh,iqplot,jupyterlab')

