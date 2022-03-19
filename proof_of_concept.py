import numpy as np
import streamlit as st 
import math
import matplotlib.pyplot as plt
import pandas as pd
from scipy.special import beta
import operator as op
from functools import reduce
from scipy.stats import gamma

st.header('Exploring Poisson-Gamma Model')

st.markdown(""" The following is a simple web app for exploring Poisson-Gamma model for patient recruitment modeling in multicenter clinical trials.
            Let's first get familiar with Poisson and Gamma distributions. Then explore the center occupancy for different 
            inputs of Poisson-Gamma model. """)

st.subheader('Poisson Distribution')

st.markdown(""" The patients arrive at centers according to a Poisson process.
            Poisson distribution gives the probability of arriving k patients over a fixed period given the constant rate of lambda (l).
            Play with the rate parameter l to investigate how the probability density function changes. """)

l = st.slider("""Set the rate parameter, l""", min_value=1, max_value=10, step=1)

t = [i for i in range(20)]

p = [0] * len(t)

for i in range(len(t)):
    p[i] = l**i * np.exp(-l) / math.factorial(i)
    

fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(t, p)
ax.set_xlim([0, 20])
ax.set_ylim([0, 0.4])
plt.rcParams['font.size'] = '15'
plt.rcParams['axes.linewidth'] = 1
ax.set_title('Poisson Distribution')
ax.set_xlabel('k', fontsize=15)
ax.set_ylabel('P(x=k)', fontsize=15)
st.pyplot(fig)

st.subheader('Gamma Distribution')
st.markdown(""" According to Poisson-Gamma model, the prior distribution of rates is a gamma distribution.
            Play with the shape parameter, alpha (a),and rate parameter, beta (b) to investigate how the probability density function changes. """)
            
a = st.slider("""Set the shape parameter, a""", min_value=0.1, max_value=10.0, step=0.1)
b = st.slider("""Set the rate parameter, b""", min_value=0.1, max_value=10.0, step=0.1)

y = [i*0.1 for i in range(100)]

p = [0] * len(y)

#def gamma_f(n):
    #return math.factorial(n-1)

#for i in range(len(y)):
    #p[i] = (b**a/gamma(a))*(y[i]**(a-1))*np.exp(-b*y[i])
    
p = gamma.pdf(y, a, 0, 1/b)
    
fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(y, p)
ax.set_xlim([0, 10])
ax.set_ylim([0, 1])
plt.rcParams['font.size'] = '15'
plt.rcParams['axes.linewidth'] = 1
ax.set_title('Gamma Distribution')
ax.set_xlabel('y', fontsize=15)
ax.set_ylabel('P(y)', fontsize=15)
st.pyplot(fig)

st.subheader("Analysis of Center Occupancy")
st.markdown(""" Assume we plan to recruit n patients by N centers. 
            The patient enrollment rates are gamma distributed with parameters a and b.
            According to Eq. 8, the center occupancy is only a function of n, N and a not b.
            Assuming a mean rate of 1 (m=1), we can conclude that a = 1/Var[rates].
            Based on real world data 1.2<a<4. Let's reconstruct Fig. 1 of Anisimov and Fedorov paper where n = 720 and N=60. """)
            
df = pd.read_csv('comb_data.csv')

#def beta(m, n):
    #return gamma_f(m)*gamma_f(n)/gamma_f(m+n)

n_p = 720
n_c = 60
a = st.slider("""Set the shape parameter, a (i.e. 1/Var[rate])""", min_value=1, max_value=4, step=1)

n_p_var = [i for i in range(1, n_p)]

m = [0] * len(n_p_var)


for j in n_p_var:
    x = int(df[df['j'] == j]['c'].tolist()[0])
    m[j-1] = n_c*x*beta(a+j, a*(n_c-1)+n_p-j)/beta(a, a*(n_c-1))
    
poisson = [0] * int(n_p/n_c*3)

for i in range(len(poisson)):
    l = int(n_p/n_c)
    poisson[i] = (l**i * np.exp(-l) / math.factorial(i)) * n_c
    
fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(n_p_var, m, label='Poisson-Gamma Model with a= '+ str(a))
ax.plot([i for i in range(int(n_p/n_c*3))], poisson, 'r', label='Poisson Model')
ax.set_xlim([0, l*5])
#ax.set_ylim([0, int(n_c/6)])
plt.rcParams['font.size'] = '15'
plt.rcParams['axes.linewidth'] = 1
ax.set_title('Analysis of Center Occupancy ')
ax.set_xlabel('j', fontsize=15)
ax.set_ylabel('Mean number of centers with j patients recruited', fontsize=15)
ax.legend()
st.pyplot(fig)

def ncr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer // denom

st.subheader('Analysis of Recruitment Time')


n_p = 200
n_c = 20
m = st.slider("""Set the mean rate, m (i.e. E[l] = a/b)""", min_value=1.0, max_value=4.0, step=0.25)
a = st.slider("""Set the shape parameter, a (i.e. 1/Var[rate])""", min_value=1.0, max_value=4.0, step=0.25)

A = n_p
B = n_c*m
y = [i*0.1 for i in range(100)]
p_1 = [0] * len(y)    
p_1 = gamma.pdf(y, A, 0, 1/B)

A=a*n_c
B=a/m
p_2 = [0] * len(y)
for i in range(len(y)):
    p_2[i] = (1/beta(n_p,A))*(y[i]**(n_p-1))*(B**A)/((y[i]+B)**(n_p+A))
    
fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(y, p_1)
ax.plot(y, p_2, 'r')
#ax.set_xlim([0, 10])
#ax.set_ylim([0, 1])
plt.rcParams['font.size'] = '15'
plt.rcParams['axes.linewidth'] = 1
ax.set_title('Gamma Distribution')
ax.set_xlabel('y', fontsize=15)
ax.set_ylabel('P(y)', fontsize=15)
st.pyplot(fig)



