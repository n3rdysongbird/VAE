# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 13:30:59 2018

@author: gonza
"""

###################
#Pre-Processing:
y_2 = pd.DataFrame(columns=np.array(list(data.columns.values))[0:380])
for index, row in data.iterrows():
    x_wavelengths = np.array(list(data.columns.values))
    y=row
    y_spl = UnivariateSpline(x_wavelengths,y,s=0,k=3)
    
    #%pylab inline
    #plt.semilogy(x_wavelengths,y,'ro',label = 'data')
    x_range = np.linspace(x_wavelengths[0],x_wavelengths[-1],381)
    #plt.semilogy(x_range,y_spl(x_range))
    #plt.show()
    
    y_spl_2d = y_spl.derivative(n=1)
    #plt.plot(x_range,y_spl_2d(x_range))
    #plt.show()
    
    coeffs = y_spl_2d.get_coeffs()
    knots = y_spl_2d.get_knots() #same as x_wavelengths
    residual = y_spl_2d.get_residual()
    
    y_der = coeffs*y[0:380]
    
    #plot(knots,y_der)
    
    #y_der.reshape(1,-1) # for one single sample
    y_norm = pd.DataFrame(skl.preprocessing.normalize(y_der.values.reshape(1,-1), norm='l2', axis=1, copy=False, return_norm=False),columns=np.array(list(data.columns.values))[0:380])
    
    #plot(knots,y_norm.T)
    y_2 = y_2.append(y_norm)

#mean center
y_mc = y_2 - y_2.mean()
y_mc.isnull().values.any()
######################