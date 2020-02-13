from scipy.optimize import minimize
import numpy as np

def multifit(f,x,y,p1,p2):
	''' f - function to fit, x - list of x data sets, y - list of y data sets, p1 - list of initial values for parameters shared across data sets, p2 - list of initial values for the variable parameter '''
	p0 = np.concatenate([p1,p2])
	result = minimize(cost_func,p0,(f,x,y),callback=callback,options={'disp':True})
	return result.x,result.fun

def callback(x):
	print x

def cost_func(p,f,x,y):
	cost = 0
	for i in range(len(x)):
		cost = cost + sum((f(x[i],p[:-len(x)],p[-len(x)+i]) - y[i])**2)
	return cost

if __name__ == "__main__":
	import matplotlib.pyplot as plt

	def lor_func(x,p,c):
		a,b,d = p
		return a/((x-c)**2+b**2)

	def generate_test_data():
		list_ydata=[]
		list_xdata=[]
		for c in range(0,100,20):
			num_points=c+100
			xdata=np.linspace(0,num_points/2, num_points)
			ydata=5.1/((xdata-c)**2+2.1**2)+0.05*((0.5*np.random.rand(num_points))*np.exp(2*np.random.rand(num_points)**2))
			list_ydata.append(ydata)
			list_xdata.append(xdata)
		return list_xdata, list_ydata

	x,y = generate_test_data()
	fitparams = multifit(lor_func,x,y,[6.3,2.3,4.6e5],range(0,100,20))
	print fitparams
	for i in range(len(x)):
		plt.plot(x[i],y[i],marker='o',ls='')
		plt.plot(x[i],lor_func(x[i],fitparams[:3],fitparams[3+i]))
	plt.draw(); raw_input()
