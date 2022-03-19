import numpy as np
import streamlit as st 
import math
import matplotlib.pyplot as plt
import pandas as pd
from scipy.special import beta
from scipy.stats import gamma

st.header('Exploring Poisson-Gamma Model')

st.markdown(""" The following is a simple web app for exploring Poisson-Gamma model for patient recruitment modeling in multicenter clinical trials.
            Let's first get familiar with Poisson and Poisson-Gamma Models. Then compare the predictions of recruitment time and center occupancy
            between two models.""")
            
st.write('Assume we intend to recruit n patients by N centers over time T.')

st.subheader('Poisson Model')

st.markdown(""" According to Poisson model, the patients arrive at centers according to a Poisson process.
            The rate of arrival of patients is **equal** and **constant** for all centers.
            Poisson distribution gives the probability of arriving k patients over a fixed period given the constant rate of $\lambda$.
            Play with the rate parameter $\lambda$ to investigate how the probability density function changes. """)

l = st.slider("""Set the rate parameter""", min_value=1, max_value=10, step=1)
t = [i for i in range(21)]
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

st.subheader('Poisson-Gamma Model')
st.markdown(""" According to Poisson-Gamma model, the patients arrive at centers according to a Poisson process.
            However, the rates are **random** and **gamma distributed**.""")
            
latext = r''' 
           $$
           \lambda = G(a, b) 
           $$
        '''
st.write(latext)

latext = r''' 
           $$
           E[\lambda] = m = \frac{a}{b} 
           $$
        '''
st.write(latext)

latext = r''' 
           $$
           Var[\lambda] = \sigma^2 = \frac{a}{b^2} 
           $$
        '''
st.write(latext)
            
a = st.slider("""Set the shape parameter, a""", min_value=1.0, max_value=5.0, value=1.0, step=0.1)
b = st.slider("""Set the rate parameter, b""", min_value=1.0, max_value=5.0, value=1.0, step=0.1)

y = [i*0.1 for i in range(100)]   
p = gamma.pdf(y, a, 0, 1/b)
    
fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(y, p)
ax.set_xlim([0, 10])
ax.set_ylim([0, 1])
plt.rcParams['font.size'] = '15'
plt.rcParams['axes.linewidth'] = 1
ax.set_title('Distribution of Rates')
ax.set_xlabel('rate', fontsize=15)
ax.set_ylabel('P(rate)', fontsize=15)
st.pyplot(fig)

st.subheader("Analysis of Center Occupancy")
st.markdown("""According to Poisson model, center occupancy is a function of n and N only. However, according to Poisson-Gamma model (Eq. 8 of 
            Anisimov and Fedorov paper), center occupancy is a function of n, N and a (not b).
            Assuming a fixed mean rate (e.g. m=1), let's investigate the effect of parameter a on center occupancy.
            **Let's assume m=1, n=720 and N=60 to reconstruct Fig. 1 of Anisimov and Fedorov paper.**""")

latext = r''' 
           $$
           \alpha = \frac{m^2}{Var[\lambda]} 
           $$
        '''
st.write(latext)

st.write('So, for m=1:')

latext = r''' 
           $$
           \alpha = \frac{1}{Var[\lambda]} 
           $$
        '''
st.write(latext)
            
df = pd.read_csv('comb_data.csv')

n_p = 720
n_c = 60
a = st.slider("""Set the shape parameter, a""", min_value=1, max_value=4, step=1)

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
ax.set_xlim([0, 50])
#ax.set_ylim([0, int(n_c/6)])
plt.rcParams['font.size'] = '15'
plt.rcParams['axes.linewidth'] = 1
ax.set_title('Center Occupancy ')
ax.set_xlabel('j', fontsize=15)
ax.set_ylabel('Mean number of centers with j patients recruited', fontsize=15)
ax.legend()
st.pyplot(fig)

st.subheader('Analysis of Recruitment Time')

st.markdown("""According to Poisson model, the recruitment time is gamma distributed: T(n, N) = G(n, mN).
            However, according to Poisson-Gamma model, T(n, N) is the superposition of two independent gamma random variables 
            and can be represented as T(n, N)=bG(n, 1)/G(aN, 1). The pdf of T(n, N) follows 
            Eq. 1 of the Anisimov and Fedorov paper.""")
            
st.markdown("""For **n=200** and **N=20**, let's explore how to models predict the recruitment time:""")


n_p = 100
n_c = 20
m = st.slider("""Set the mean rate, m""", min_value=1.0, max_value=4.0, step=0.25)
a = st.slider("""Set the shape parameter, a""", min_value=1.0, max_value=4.0, step=0.25)

A = n_p
B = n_c*m
y = [i*0.1 for i in range(100)]   
p_1 = gamma.pdf(y, A, 0, 1/B)

A=a*n_c
B=a/m
p_2 = [0] * len(y)
for i in range(len(y)):
    p_2[i] = (1/beta(n_p,A))*(y[i]**(n_p-1))*(B**A)/((y[i]+B)**(n_p+A))
    
fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(y, p_1, label='Poisson Model')
ax.plot(y, p_2,'r', label='Poisson-Gamma Model')
ax.set_xlim([0, 10])
#ax.set_ylim([0, 1.2])
plt.rcParams['font.size'] = '15'
plt.rcParams['axes.linewidth'] = 1
ax.set_title('Recruitment Time')
ax.set_xlabel('time', fontsize=15)
ax.set_ylabel('P(time)', fontsize=15)
ax.legend(loc='upper right')
st.pyplot(fig)



