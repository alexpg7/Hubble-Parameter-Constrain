# 🪐Hubble-Parameter-Constrain ![Static Badge](https://img.shields.io/badge/UAB-%23009900?style=for-the-badge)

![Static Badge](https://img.shields.io/badge/Python-white?logo=Python)
![Static Badge](https://img.shields.io/badge/Jupyter%20Notebook-white?logo=Jupyter)
![Static Badge](https://img.shields.io/badge/status-in%20progress-orange)

A method to contrain the Hubble parameter using measured data and Markov chains.

## ❔Context

The Hubble parameter $H_0$ is a parameter that, somehow, tunes the expansion of the universe. The expansion rate of the universe is given by a function $H(z)$ that depends on the observed redshift $z$.

$z$ can be thought as the "distance" where the object we observe (with our telescope) is from us. $z$ is called the **redshift** because it quantifies the _"redness"_ of the light we are observing, which has been changed by the [Doppler Effect](https://en.wikipedia.org/wiki/Doppler_effect). Due to [Hubble's Law](https://en.wikipedia.org/wiki/Hubble%27s_law), redshift $z$ and distance are perfectly correlated.

The main idea is that $H(z)$ can be modelled by what we call a Cosmology (the given matter densities of the universe $\Omega_i$) through the [Friedmann equation](https://en.wikipedia.org/wiki/Friedmann_equations):

```math
\frac{H^2}{H_0^2}=\Omega_Ra^{-4}+\Omega_m a^{-3}+\Omega_k a^{-2}+\Omega_\Lambda
```

Making some simplifications and given that $a^{-1}=1-z$, we can set our model equation to be:

```math
\boxed{H(z)=H_0\sqrt{\Omega_m(1+z)^3+(1-\Omega_m)}}
```

With this model, one can compare the measurements of $H(z)$ and $z$ to infer which combination of $\Omega_m$ and $H_0$ fits best.

## 📑Data Set

The data we are given is [``Hz_BC03_all.dat``](./DataSet/Hz_BC03_all.dat), which gives us the measurements of $H(z)$ at each redshift $z$, with some error bars on $H(z)$. So, we first import the necessary libraries:

```python
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.optimize import curve_fit
from scipy.stats import norm
from statistics import NormalDist
import plotly.express as px
from scipy.stats import gaussian_kde
```
and load the dataset into a variable:
```python
# read the data
z, h, herr = np.loadtxt('DataSet/Hz_BC03_all.dat',unpack=True)
```
The dataset contains columns of $z$ (``z``), $H(z)$ (``h``) and $\Delta H(z)$ (error bars ``herr``).

### 🔥First test

To see some examples, we can try to plot the dataset with some random-generated curves of our model.

```python
# plot the data
plt.errorbar(z, h, yerr = herr, color = 'grey', fmt = 'o', label = 'Measured data')
x = np.linspace(0, 2, 100)
curves = []
Omegas = np.linspace(0.25, 0.75, 3)
H00 = np.linspace(25, 100, 3)

# sample some curves
def H(aH, H0, Omega):
    k = np.array([])
    for i in range(0, len(aH)):
        k = np.append(k, H0 * np.sqrt(Omega * (1 + aH[i]) ** 3 + (1 - Omega)))
    return k

for l in H00:
    for j in Omegas:
        curves.append(H(x, l, j))
        plt.plot(x, H(x, l, j), label = r'$\Omega_m=${} , $H_0=${}'.format(j,l))

        
plt.xlabel(r'$z$')
plt.ylabel(r'$H(z)$')
plt.legend(loc = 'best', fontsize = 7)
plt.show()
```
![til](./Figures/First_test)

As it can be seen, some curves like $\Omega_m=0.5$ and $H_0=62.5$ fit pretty well; but we do not have yet a way to quantify how well do they fit and how correlated they may be if we change one of them.

To do a further study, we need Bayesian statistics and Markov chain Montecarlo (MCMC).

## ⛓️Markov chain Montecarlo (MCMC)

