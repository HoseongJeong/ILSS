from Base import * 
import time
Time_start=time.time()      
Time=time.time()

############################################################################################ILSS############################################################################################
##  Population Parameter
Population_size=1
Population=[]                

##  Library Parameter
The_Number_of_Noise=int(100)
Min_noise=float(0)
Max_noise=float(1)
The_number_of_random_tree=int(1000)
Random_tree_height_min=int(1)
Random_tree_height_max=int(6)
Clustering_distance='euclidean' 
Library_decimal=3

#Evaluation list            
Fitness=[]
Fitness_test=[]
MAE_train=[]
MAE_test=[]
Size_list=[]

Mean_MSE_train=float('nan')
Mean_MSE_test=float('nan')
Mean_MAE_train=float('nan')
Mean_MAE_test=float('nan')
Mean_size=float('nan')

## mutation Parameter
Mutation_rate=1
Mutation_tree_min_height=0
Mutation_tree_max_height=4
    

## Local search Parameter
Local_search_rate=1
Patience=2
Local_optimum_count=0
Improvement_limit_count=0
Selection_limit_count=0
Height_limit_count=0
Pruned_count=0

## Save list
History=[]
Semantic_list=[]
Improvement_history=[]
Selected_tree_history=[]
Fitness_history=[]
MSE_test_history=[]
Validation_history=[]
Fitness_count_history=[]
Size_history=[]
Pruned_history=[]

Total_tree_list=[]
Clustering_evaluation_dataframe_list=[]
Semantic_dataframe_list=[]
Representative_dataframe_list=[]
Noise_data=[]
Total_detailed_data_list=[]

HOF=[]

# Termination Condition
Fitness_count=0
Fitness_count_limit=100000
Generation=1
Generation_max=100000


#Primitiveset
pset=None

############################################################################################RDO############################################################################################
# ##  Population Parameter
# Population_size=1000
# Population=[]        
# Height_limit=15  

# ##  Library Parameter
# The_number_of_random_tree=int(1000)
# Random_tree_height_min=int(1)
# Random_tree_height_max=int(6)
# Library_decimal=3

# #Evaluation list            
# Fitness=[]
# Fitness_test=[]
# MAE_train=[]
# MAE_test=[]
# Size_list=[]

# Mean_MSE_train=float('nan')
# Mean_MSE_test=float('nan')
# Mean_MAE_train=float('nan')
# Mean_MAE_test=float('nan')
# Mean_size=float('nan')

# Tournament_size=4

# ## Crossover Parameter
# Crossover_rate=0.9

# ## mutation Parameter
# Mutation_rate=0.9
# Mutation_tree_min_height=0
# Mutation_tree_max_height=4
    

# ## Save list
# History=[]
# Semantic_list=[]
# Improvement_history=[]
# Selected_tree_history=[]
# Fitness_history=[]
# MSE_test_history=[]
# Validation_history=[]
# Fitness_count_history=[]
# Size_history=[]
# Pruned_history=[]

# Total_tree_list=[]
# Semantic_dataframe_list=[]
# Total_detailed_data_list=[]

# HOF=[]

# # Termination Condition
# Fitness_count=0
# Fitness_count_limit=100000
# Generation=1
# Generation_max=100000


# #Primitiveset
# pset=None