import numpy as np
import streamlit as st 
import math
import matplotlib.pyplot as plt

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
            
a = st.slider("""Set the shape parameter, a""", min_value=1, max_value=5, step=1)
b = st.slider("""Set the rate parameter, b""", min_value=1, max_value=5, step=1)

y = [i*0.1 for i in range(100)]

p = [0] * len(y)

def gamma_f(n):
    return math.factorial(n-1)

for i in range(len(y)):
    p[i] = (b**a/gamma_f(a))*(y[i]**(a-1))*np.exp(-b*y[i])
    
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
            Based on real world data 1.2<a<4. Pick n, N and a and investigate the center occupancy. 
            The following plot corresponds to Fig. 1 in Anisimov and Fedorov paper. """)

def beta(m, n):
    return gamma_f(m)*gamma_f(n)/gamma_f(m+n)

n_p = st.number_input('Enter the total number of patients, n', min_value = 1, max_value= 2000, value = 720, step = 1)
n_c = st.number_input('Enter the total number of centers, N', min_value = 1, max_value= 200, value = 60, step = 1)
a = st.slider("""Set the shape parameter, a (i.e. 1/Var[rate])""", min_value=1, max_value=4, step=1)

n_p_var = [i for i in range(1, n_p)]

m = [0] * len(n_p_var)


for j in n_p_var:
    m[j-1] = n_c*math.comb(n_p, j)*beta(a+j, a*(n_c-1)+n_p-j)/beta(a, a*(n_c-1))
    
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



