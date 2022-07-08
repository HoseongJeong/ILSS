import numpy

def protectedDiv(left,right):
    return numpy.where(abs(right)<10**(-10**2), 0.0 , numpy.divide(left,right))

def minus(left,right):
    return numpy.subtract(left,right)

def log(left):
    return numpy.where(abs(left)<10**(-10**2), 0.0, numpy.log10(abs(left)))

def exp(left):
    return numpy.where((10<left) , 0.0, numpy.exp(left))
