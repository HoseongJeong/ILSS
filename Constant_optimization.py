import random
import numpy
import copy
from scipy.optimize import curve_fit
import sys
from operator import attrgetter
from Base import *
from Evaluation import *
import Module_wide_function 
import Module_wide_variable as MWV


#######################################################constant optimization#######################################################
def error(expr,*parameter):
  
    expr=copy.deepcopy(expr)
    
    constant=search_constant(expr)
    variable=search_variable(expr)
    variable_dictionary=['input[{}]'.format(i) for i in range(len(MWV.pset.arguments))]
    variable_dictionary=dict(zip(MWV.pset.arguments,variable_dictionary))
    
    replaced_argument_list=[]
    replaced_argument_list.append('input')
    
    if len(constant[0])>0:
        for i in range(len(constant[0])):
            replaced_argument_list.append('parameter{}'.format(i))
            expr[constant[0][i]].value='parameter{}'.format(i)
            expr[constant[0][i]].name='parameter{}'.format(i)
    argument_list=copy.deepcopy(MWV.pset.arguments)
    
    if len(MWV.pset.arguments) > 0:
        # This section is a stripped version of the lambdify
        # function of SymPy 0.6.6.
        for i in range(len(variable[0])):
            expr[variable[0][i]].value=variable_dictionary[variable[1][i]]
            expr[variable[0][i]].name=variable_dictionary[variable[1][i]]
            
    args = ",".join(arg for arg in replaced_argument_list)
    code = str(expr)
    code=code.replace("'","")
    code = "lambda {args}: {code}".format(args=args, code=code)
    try:
        return eval(code, MWV.pset.context,{})
    except :
        print("Failed : call of error function ")
 
def levenberg_marquardt(individual,x_data,y_data) :
    tree=copy.deepcopy(individual)
    func=error(tree)
    
    x=x_data.values
    x=x.T
    y=y_data.values
    
    parameter=numpy.array(search_constant(tree)[1])
    # parameter=numpy.array([1 for i in range(len(parameter))])
    if len(parameter)==0:
        return tree
    else :
        #★★★★★★Caution, If USER-DEFINE-FUNCTION is not proper, LM is not properly works,
        #★★★★★★However, because of "try~ exception code", You may not be aware of these problems.
        #★★★★★★Therfore, If the run is too fast, it is recommended to clear the try excption and run it to look for errors.
        try : 
            popt,pcov = curve_fit(func, x, y, p0=parameter,maxfev=500)
            constant_list=search_constant(tree)
            for i in range(len(constant_list[0])):
                tree[constant_list[0][i]].value=popt[i]
                tree[constant_list[0][i]].name=popt[i]
                
            #★★★★★★ This is code for check curve_fit_run
            # print("Work"+str(tree) )
            return tree
        except:
            # print("Didn't Work"+str(tree) )
            return tree

def levenberg_marquardt_for_semantic_clustering(individual,terminal,Noise_data) :
    tree=copy.deepcopy(individual)
    func=error(tree)
    
    #Noise data is row(data)xcol(feature)
    x=Noise_data
    y=output_list_generator(terminal,x)
    #It is transformed for LM
    x=x.T
    
    parameter=search_constant(tree)[1]
    if len(parameter)==0:
        return tree
    else:
        
        #Caution, If USER-DEFINE-FUNCTION is not proper, LM is not properly works,
        #However, because of "try~ exception code", You may not be aware of these problems.
        #Therfore, If the run is too fast, it is recommended to clear the try excption and run it to look for errors.
        try : 
            popt,pcov = curve_fit(func, x, y, p0=parameter,maxfev=500)
            constant_list=search_constant(tree)
            for i in range(len(constant_list[0])):
                tree[constant_list[0][i]].value=popt[i]
                tree[constant_list[0][i]].name=popt[i]
            return tree
        except:
            return tree
