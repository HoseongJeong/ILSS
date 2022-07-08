import sys
import Module_wide_variable as MWV

def compile(expr):
    """Compile the expression *expr*.

    :param expr: Expression to compile. It can either be a PrimitiveTree,
                 a string of Python code or any object that when
                 converted into string produced a valid Python code
                 expression.
    :param pset: Primitive set against which the expression is compile.
    :returns: a function if the primitive set has 1 or more arguments,
              or return the results produced by evaluating the tree.
    """
    code = str(expr)
    if len(MWV.pset.arguments) > 0:
        # This section is a stripped version of the lambdify
        # function of SymPy 0.6.6.
        args = ",".join(arg for arg in MWV.pset.arguments)
        code = "lambda {args}: {code}".format(args=args, code=code)
    try:
        return eval(code, MWV.pset.context, {})
    except MemoryError:
        _, _, traceback = sys.exc_info()
        raise MemoryError("DEAP : Error in tree evaluation :"
                            " Python cannot evaluate a tree higher than 90. "
                            "To avoid this problem, you should use bloat control on your "
                            "operators. See the DEAP documentation for more information. "
                            "DEAP will now abort.").with_traceback(traceback)


def evaluation_MAE(individual,x_data,y_data):
    func = compile(individual)
    MAE=[abs(func(*x_data.iloc[i].values)-y_data.iloc[i]) for i in range(0,len(x_data))]
    return sum(MAE)/ len(x_data)

def evaluation_MAE_test(individual,x_data,y_data):
    func = compile(individual)
    MAE=[abs(func(*x_data.iloc[i].values)-y_data.iloc[i]) for i in range(0,len(x_data))]
    return sum(MAE)/ len(x_data)

def evaluation_MSE(individual,x_data,y_data):
    MWV.Fitness_count=MWV.Fitness_count+1
    func = compile(individual)
    MSE=[(func(*x_data.iloc[i].values)-y_data.iloc[i])**2 for i in range(0,len(x_data))]
    return sum(MSE)/ len(x_data)

def evaluation_MSE_test(individual,x_data,y_data):
    func = compile(individual)
    MSE=[(func(*x_data.iloc[i].values)-y_data.iloc[i])**2 for i in range(0,len(x_data))]
    return sum(MSE)/ len(x_data)

        
def output_generator(individual,Data):
    # Transform the tree expression in a callable function
    func = compile(individual)
    output_generator=func(*Data)
    return output_generator 

def output_list_generator(individual,Data):
    # Data must be numpy array, and columns are feature
    func = compile(individual)
    Result=[func(*Data[i]) for i in range(0,len(Data))]
    return Result

####################################################### mutation #######################################################
def similarity(individual1,individual2,Noise_data):
    # 이 알고리즘은 fitness 가 높은 유전자를 선택하므로, fitness=1-RMSE가 되어야 함.
    # Transform the tree expression in a callable function
    #Example:  similarity(population[0],population[1],10,-20,120)
    #min_hundred_scale 는 %단위의 최소한계를 의미, -20%=20
    #max_hundred_scale 는 %단위의 최대한계를 의미, 120%=120
    
    func1 = compile(individual1)
    func2 = compile(individual2)

    # Evaluate the mean squared error between the expression
    # and the real function : x**4 + x**3 + x**2 + x
    
    expect1=[abs(func1(*Noise_data[i]))  for i in range(0,len(Noise_data))]
    distance=[abs(func1(*Noise_data[i])-func2(*Noise_data[i]))  for i in range(0,len(Noise_data))]
    Distance_average=sum(distance)/len(Noise_data)
    expect1_mean=sum(expect1)/len(Noise_data)
    if expect1_mean==0:
        Similarity_distance=10^10
    else :
        Similarity_distance=Distance_average/expect1_mean
        
    return Similarity_distance    
        