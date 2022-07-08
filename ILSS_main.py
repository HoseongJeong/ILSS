import random
import operator
import copy 
import warnings
import pandas
import numpy
from sklearn.model_selection import train_test_split
# import hdbscan
import time
from datetime import datetime
import pickle
import importlib

warnings.filterwarnings(action='ignore') 

try:
    from collections.abc import Sequence
except ImportError:
    from collections import Sequence
    
iteration_data=pandas.read_excel("iteration.xlsx")
Summary=pandas.DataFrame(columns=["Iteration","Iteration_index","Generation","Time","MWV.HOF","HOF_size","HOF_MSE_train","HOF_MSE_test","HOF_MAE_train","HOF_MAE_test","Local_optimum_count","Improvement_limit_count","Selection_limit_count","Height_limit_count","Pruned_count","Fitness_count"])
today=datetime.today().strftime("%Y-%m-%d-%H-%M-%S")
Summary_file_name='Summary'




for iteration in range(0,1):
    for iteration_data_index in range(0,30):
        
        from Base import *
        from Constant_optimization import *
        from Evaluation import *
        from Initialization import *
        from Library_manager import *
        from Operator import *
        from Selection import *
        import Module_wide_variable as MWV
        from Module_wide_function import * 

        
        #######################################################set parameter#######################################################
        ##  File name
        The_name_of_data_set=str(iteration_data.loc[iteration_data_index,"The_name_of_data_set"])
        file_name=str(The_name_of_data_set)+"_ILSS"
        data_train = pandas.read_csv(str(The_name_of_data_set)+"_train.csv")
        data_test = pandas.read_csv(str(The_name_of_data_set)+"_test.csv")        
        x_train=data_train.loc[:,data_train.columns!='target'].dropna(axis=0)
        y_train=data_train.loc[:,'target'].dropna(axis=0)
        x_test=data_test.loc[:,data_test.columns!='target'].dropna(axis=0)
        y_test=data_test.loc[:,'target'].dropna(axis=0)
        x_data=x_train.append(x_test)
        y_data=y_train.append(y_test)
        
        save_interval=1
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.33, random_state=42)
        
        
        ##  parameter
        MWV.The_Number_of_Noise=int(iteration_data.loc[iteration_data_index,"The_Number_of_Noise"])
        MWV.Min_noise=float(iteration_data.loc[iteration_data_index,"Min_noise"])
        MWV.Max_noise=float(iteration_data.loc[iteration_data_index,"Max_noise"])
        MWV.The_number_of_random_tree=int(iteration_data.loc[iteration_data_index,"The_number_of_random_tree"])
        MWV.Random_tree_height_min=int(iteration_data.loc[iteration_data_index,"Random_tree_height_min"])
        MWV.Random_tree_height_max=int(iteration_data.loc[iteration_data_index,"Random_tree_height_max"])   
        MWV.Patience=float(iteration_data.loc[iteration_data_index,"Patience"])
        MWV.Generation_max=int(iteration_data.loc[iteration_data_index,"Generation_max"])
        
        #Record
        MWV.HOF=HallOfFame(1)
        # Table setting
        MWV.History=pandas.DataFrame(columns=["Generation","Time","HOF","HOF_size","Best_size","Mean_size",
                                  "HOF_MSE_train","HOF_MSE_test","HOF_MAE_train","HOF_MAE_test",
                                  "Best_MSE_train","Best_MSE_test","Best_MAE_train","Best_MAE_test",
                                  "Mean_MSE_train","Mean_MSE_test","Mean_MAE_train","Mean_MAE_test",
                                  "Local_optimum_count","Improvement_limit_count","Selection_limit_count","Height_limit_count","Pruned_count","Fitness_count",
                                  "Improvement_history", "Selected_tree_history", "Fitness_history", "MSE_test_history","Validation_history","Fitness_count_history","Size_history","Pruned_history"])
        
            
        ####################################################### Create Noise #######################################################
        MWV.Noise_data=create_random_noise(MWV.The_Number_of_Noise,MWV.Min_noise,MWV.Max_noise,x_data)
           
        ####################################################### Set a primitive set #######################################################
        MWV.pset = PrimitiveSet("MAIN", len(x_data.columns))
        MWV.pset.addPrimitive(operator.add, 2)
        MWV.pset.addPrimitive(minus, 2)
        MWV.pset.addPrimitive(operator.mul, 2)
        MWV.pset.addPrimitive(protectedDiv, 2)
        MWV.pset.addPrimitive(log, 1)
        MWV.pset.addPrimitive(exp, 1)
        MWV.pset.addPrimitive(numpy.sin, 1)
        MWV.pset.addPrimitive(numpy.cos, 1)
        MWV.pset.addEphemeralConstant('constant',lambda: round(random.uniform(-1000, 1000),1))
        
        MWV.pset.add_index=[primitive_index for primitive_index in range(len(MWV.pset.primitives[MWV.pset.ret])) if MWV.pset.primitives[MWV.pset.ret][primitive_index].name=="add"][0]
        MWV.pset.minus_index=[primitive_index for primitive_index in range(len(MWV.pset.primitives[MWV.pset.ret])) if MWV.pset.primitives[MWV.pset.ret][primitive_index].name=="minus"][0]
        MWV.pset.mul_index=[primitive_index for primitive_index in range(len(MWV.pset.primitives[MWV.pset.ret])) if MWV.pset.primitives[MWV.pset.ret][primitive_index].name=="mul"][0]
        MWV.pset.div_index=[primitive_index for primitive_index in range(len(MWV.pset.primitives[MWV.pset.ret])) if MWV.pset.primitives[MWV.pset.ret][primitive_index].name=="protectedDiv"][0]
        MWV.pset.constant_index= [terminal_index for terminal_index in range(len(MWV.pset.terminals[MWV.pset.ret])) if "constant" in str(MWV.pset.terminals[MWV.pset.ret][terminal_index])][0]
        #######################################################Create linear scailing tree#######################################################
        Linear_scaling_tree=PrimitiveTree(generate_linear_scaling_tree(MWV.pset.terminals[MWV.pset.ret][0],MWV.pset))

        
        #######################################################Clustering#######################################################
        semantic_clustering(MWV.Noise_data,MWV.The_number_of_random_tree, MWV.Random_tree_height_max,range(10,31,10),decimal=MWV.Library_decimal)
        #######################################################Make a Population #######################################################
        for i in range(0,MWV.Population_size):
            # # raise Exception("g")
            MWV.Population.append(
                linear_scailing(
                    Linear_scaling_tree,PrimitiveTree(
                        genHalfAndHalf(
                            MWV.pset,1,6))))
        
        
        ####################################################### Start Generation #######################################################
        while MWV.Generation<MWV.Generation_max+1:
            
            #######################################################Evaluation#######################################################
            
            MWV.Fitness=[]
            MWV.Fitness_test=[]
            MWV.MAE_train=[]
            MWV.MAE_test=[]
            MWV.Size_list=[]
            
            
            for i in range(0,len(MWV.Population)):
                MWV.Population[i].fitness=evaluation_MSE(MWV.Population[i],x_train,y_train)
                MWV.Population[i].MAE_train=evaluation_MAE(MWV.Population[i],x_train,y_train)
                MWV.Population[i].fitness_test=evaluation_MSE_test(MWV.Population[i],x_test,y_test)
                MWV.Population[i].MAE_test=evaluation_MAE_test(MWV.Population[i],x_test,y_test)
                
                MWV.Fitness.append(MWV.Population[i].fitness)
                MWV.Fitness_test.append(MWV.Population[i].fitness_test)
                MWV.MAE_train.append(MWV.Population[i].MAE_train)
                MWV.MAE_test.append(MWV.Population[i].MAE_test)
                MWV.Size_list.append(len(MWV.Population[i]))
            
            MWV.Mean_MSE_train=sum(MWV.Fitness)/len(MWV.Fitness)
            MWV.Mean_MSE_test=sum(MWV.Fitness_test)/len(MWV.Fitness_test)
            MWV.Mean_MAE_train=sum(MWV.MAE_train)/len(MWV.MAE_train)
            MWV.Mean_MAE_test=sum(MWV.MAE_test)/len(MWV.MAE_test)
            MWV.Mean_size=sum(MWV.Size_list)/len(MWV.Size_list)
            
            ####################################################### Update #######################################################
            MWV.HOF.update(MWV.Population)
            ####################################################### Record #######################################################
            MWV.Time=time.time()-MWV.Time_start
            
            print("Generation :  {A}, Time : {B}, HOF : {C} , \nHOF_size : {D}, Best_size : {E}, Mean_size : {F}, \nHOF_MSE_train : {G}, HOF_MSE_test : {H}, \nHOF_MAE_train : {I}, HOF_MAE_test : {J},\nBest_MSE_train : {K}, Best_MSE_test : {L}, \nBest_MAE_train : {M}, Best_MAE_train: {N},\nMean_MSE_train: {O}, Mean_MSE_test: {P}, \nMean_MAE_train:{Q}, Mean_MAE_test: {R}, \nFitness_count:{S}".format(
                A=MWV.Generation,
                B=MWV.Time,
                C=MWV.HOF[0],
                D=len(MWV.HOF[0]), 
                E=len(MWV.Population[0]),
                F=MWV.Mean_size,
                G=MWV.HOF[0].fitness, 
                H=MWV.HOF[0].fitness_test, 
                I=MWV.HOF[0].MAE_train, 
                J=MWV.HOF[0].MAE_test, 
                K=MWV.Population[0].fitness, 
                L=MWV.Population[0].fitness_test, 
                M=MWV.Population[0].MAE_train, 
                N=MWV.Population[0].MAE_test, 
                O=MWV.Mean_MSE_train, 
                P=MWV.Mean_MSE_test, 
                Q=MWV.Mean_MAE_train, 
                R=MWV.Mean_MAE_test, 
                S=MWV.Fitness_count))
            print("#####################################################################################################################################################################")

            MWV.History.loc[MWV.Generation]=[MWV.Generation,
                                             MWV.Time,
                                             MWV.HOF[0],
                                             len(MWV.HOF[0]), 
                                             len(MWV.Population[0]),
                                             MWV.Mean_size,MWV.HOF[0].fitness, 
                                             MWV.HOF[0].fitness_test, 
                                             MWV.HOF[0].MAE_train, 
                                             MWV.HOF[0].MAE_test, 
                                             MWV.Population[0].fitness, 
                                             MWV.Population[0].fitness_test, 
                                             MWV.Population[0].MAE_train, 
                                             MWV.Population[0].MAE_test, 
                                             MWV.Mean_MSE_train, 
                                             MWV.Mean_MSE_test, 
                                             MWV.Mean_MAE_train, 
                                             MWV.Mean_MAE_test,
                                             MWV.Local_optimum_count,
                                             MWV.Improvement_limit_count,
                                             MWV.Selection_limit_count, 
                                             MWV.Height_limit_count,
                                             MWV.Pruned_count, 
                                             MWV.Fitness_count, 
                                             MWV.Improvement_history, 
                                             MWV.Selected_tree_history, 
                                             MWV.Fitness_history, 
                                             MWV.MSE_test_history,
                                             MWV.Validation_history,
                                             MWV.Fitness_count_history,
                                             MWV.Size_history,
                                             MWV.Pruned_history]
            
            if MWV.Generation%save_interval==0:
                
                MWV.History.to_csv(file_name+".csv", mode='w', header=True)   
                # MWV.History.to_pickle(file_name+".pkl")  
                
            ####################################################### Termination #######################################################
            if MWV.Fitness_count>MWV.Fitness_count_limit:
                break
            ####################################################### Search #######################################################

            MWV.Generation=MWV.Generation+1
            children_list=[]
            
            ####################################################### mutation #######################################################
            if random.random()<MWV.Mutation_rate :
                Parent=copy.deepcopy(MWV.HOF[0])
                for i in range(0,10000):
                    Random_tree=PrimitiveTree(genHalfAndHalf(MWV.pset,MWV.Mutation_tree_min_height,MWV.Mutation_tree_max_height))
                    if len(search_variable(Random_tree)[0])!=0:
                        break
                Linear_sclaed_tree=linear_scailing(Linear_scaling_tree,Random_tree)
                Temp_tree=[MWV.pset.primitives[MWV.pset.ret][0]]
                Temp_tree.extend(Parent)
                Temp_tree.extend(copy.deepcopy(Linear_sclaed_tree))
                mutated_children=levenberg_marquardt(PrimitiveTree(Temp_tree),x_train,y_train)
            else:
                mutated_children=MWV.Population[0]
            children_list.append(mutated_children)
        
            ####################################################### Local search #######################################################
            
            for i in range(0,len(children_list)):
                if random.random()<MWV.Local_search_rate:
                    best_tree=multiple_local_searches(children_list[i],x_train,y_train,x_test,y_test,x_val,y_val)
                    children_list=[best_tree]
            MWV.Population=children_list

        ####################################################### Record Total data #######################################################
        Summary.loc[iteration+iteration_data_index]=[iteration,
                                                     iteration_data_index,
                                                     MWV.Generation,
                                                     MWV.Time,MWV.HOF[0],
                                                     len(MWV.HOF[0]), 
                                                     MWV.HOF[0].fitness, 
                                                     MWV.HOF[0].fitness_test, 
                                                     MWV.HOF[0].MAE_train, 
                                                     MWV.HOF[0].MAE_test,
                                                     MWV.History["Local_optimum_count"].sum(),
                                                     MWV.History["Improvement_limit_count"].sum(),
                                                     MWV.History["Selection_limit_count"].sum(),
                                                     MWV.History["Height_limit_count"].sum(),
                                                     MWV.History["Pruned_count"].sum(),MWV.Fitness_count]
        Summary.to_csv(Summary_file_name+".csv", mode='w', header=True)  
        
        # MWV.Total_detailed_data_list.append(MWV.Total_tree_list)
        MWV.Total_detailed_data_list.append(MWV.Clustering_evaluation_dataframe_list)
        MWV.Total_detailed_data_list.append(MWV.Semantic_dataframe_list)
        MWV.Total_detailed_data_list.append(MWV.Representative_dataframe_list)
        MWV.Total_detailed_data_list.append(MWV.Semantic_list)
        MWV.Total_detailed_data_list.append(MWV.History)
        # with open(file_name+'data_detailed.pickle', 'wb') as f:
        #     pickle.dump(MWV.Total_detailed_data_list, f, pickle.HIGHEST_PROTOCOL)
        
        ####################################################### Prevent error #######################################################

        importlib.reload(MWV)