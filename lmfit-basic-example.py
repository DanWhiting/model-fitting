from __future__ import division

import matplotlib.pyplot as plt
import numpy as np
import lmfit
from scipy.optimize import curve_fit

def model_func(x,x0,a):
	return np.sin(a*x+x0)

def objective_func(pars,x,obs,err):
	parvals = pars.valuesdict()
	x0 = parvals['x0']
	a = parvals['a']
	ex = model_func(x,x0,a)
	out = (obs-ex)/err
	return out

def fit_function(params, x=None, dat1=None, dat2=None):
    ''' an example of how to write an objective function to fit multiple data sets with shared parameters '''
    model1 = params['offset'] + x * params['slope1']
    model2 = params['offset'] + x * params['slope2']

    resid1 = dat1 - model1
    resid2 = dat2 - model2
    return numpy.concatenate((resid1, resid2))

N = 10
x = np.linspace(0,2*np.pi,N)
y = np.sin(x)+(np.random.random(N)-0.5)/5
errs = np.random.random(N)*0.1
plt.errorbar(x,y,errs,ls='')

########## curve fit #############

res = curve_fit(model_func,x,y,sigma=errs,absolute_sigma=True)
print(res[0])
print(np.diag(res[1])**.5)
xfit = np.linspace(0,2*np.pi,1000)
fit = model_func(xfit,*res[0])
plt.plot(xfit,fit)

########### lmfit ############

p = lmfit.Parameters()
#          (Name, Value, Vary, Min, Max, Expr)
p.add_many(('x0', 0, True, None, None, None),
           ('a', 1, True, None, None, None))

minner = lmfit.Minimizer(objective_func,p,(x,y,errs))
result = minner.minimize()

# calculate final result
a = result.params['a'].value
x0 = result.params['x0'].value
fit = model_func(xfit,x0,a)
plt.plot(xfit,fit)

# write error report
lmfit.report_fit(result)

########### plotting ############

plt.xlim(-1,7)
plt.ylim(-1.5,1.5)
plt.show()
