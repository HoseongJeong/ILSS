import random
import copy 
import math
import numpy
import pandas
from operator import attrgetter
from Base import *
import Module_wide_variable as MWV
from Evaluation import *
from Constant_optimization import *
from Initialization import *
from Library_manager import *
from Module_wide_function import * 
from scipy.optimize import curve_fit
import sys
####################################################### Crossover #######################################################

def crossover(individual1,individual2):
    individual1=copy.deepcopy(individual1)
    individual2=copy.deepcopy(individual2)
    slice1=individual1.searchSubtree(random.choice(range(0,len(individual1))))
    slice2=individual2.searchSubtree(random.choice(range(0,len(individual2))))
    subtree1=copy.deepcopy(individual1[slice1])
    subtree2=copy.deepcopy(individual2[slice2])
    
    individual1[slice1]=subtree2
    individual2[slice2]=subtree1
            
    return PrimitiveTree(individual1),PrimitiveTree(individual2)

####################################################### Mutation #######################################################
def mutation(individual,min_height,max_height):
    """Randomly select a point in the tree *individual*, then replace the
    subtree at that point as a root by the expression generated using method
    :func:`expr`.

    :param individual: The tree to be mutated.
    :param expr: A function object that can generate an expression when
                 called.
    :returns: A tuple of one tree.
    """
    tree=copy.deepcopy(individual)
    
    index = random.randrange(len(tree))
    slice_1 = tree.searchSubtree(index)
    mutation_tree=PrimitiveTree(genFull(min_height, max_height))
    tree[slice_1] = mutation_tree
    tree.localsearched=False
    return tree

####################################################### RDO #######################################################

def semantic_backpropagation(x_train, y_train, individual, node_index):
    if node_index==0:
        Desired_semantic=[[y_train.values[i]] for i in range (len(y_train.values))]
    else:
        tree=copy.deepcopy(individual)
        path=tree.searchPath(node_index)
        Desired_semantic=[]
        for i in range(0,len(y_train)):
            Parent_node_semantic=[y_train.values[i]]
            Input=list(x_train.values[i])
            for Current_node_index in path[1:]:
                
                #Define parent node index
                if Current_node_index==0:
                    Parent_index="None"
                else :
                    Parent_index=path[path.index(Current_node_index)-1]
                    
                #Define current node position, and oppsite node index
                if Parent_index!="None":
                    Parent_arity=tree[Parent_index].arity
                    Parent_node_name=tree[Parent_index].name
                    if Parent_arity==1:
                        Oppsite_node_index="None"
                        Current_node_position="None"
                    elif Parent_arity==2 :
                        if Parent_index+Parent_arity==Current_node_index:   
                            Oppsite_node_index=Parent_index+Parent_arity-1
                            Current_node_position="Right"
                        else : 
                            Oppsite_node_index=Parent_index+Parent_arity
                            Current_node_position="Left"
                else :
                    Parent_node_name="None"
                    Oppsite_node_index="None"
                    Current_node_position="None"
                    
                #Define Oppsite_node_output
                if Oppsite_node_index!="None":
                    Oppsite_node_output=[output_generator(PrimitiveTree(tree[tree.searchSubtree(Oppsite_node_index)]),Input)]
                else:
                    Oppsite_node_output="None"
                
                Parent_node_semantic=invert(Parent_node_semantic,Oppsite_node_output,Current_node_position,Parent_node_name)

                if numpy.isnan(Parent_node_semantic.sum()) or numpy.isinf(Parent_node_semantic.sum()):
                    break
                else:
                    continue
                    
            Parent_node_semantic=list(Parent_node_semantic)
            Parent_node_semantic=[numpy.nan if numpy.isnan(x) or numpy.isinf(x) else x for x in Parent_node_semantic]
            Parent_node_semantic=list(set(Parent_node_semantic))
            if numpy.isnan(sum(Parent_node_semantic)) and len(Parent_node_semantic)>1:
               Parent_node_semantic= [x for x in Parent_node_semantic if numpy.isnan(x) == False]
            Desired_semantic.append(Parent_node_semantic)
            
    return Desired_semantic

def invert(Parent_node_semantic,Oppsite_node_output,Current_node_position,Parent_node_name):
    Parent_node_semantic=numpy.array(Parent_node_semantic)
    Oppsite_node_output=numpy.array(Oppsite_node_output)
    Desired_semantic=[]
    if Parent_node_name=="add":
        for i in range(0,Parent_node_semantic.shape[0]):
            Desired_semantic.extend(Parent_node_semantic[i]-Oppsite_node_output)
        
    if Parent_node_name=="minus":
        for i in range(0,Parent_node_semantic.shape[0]):
            if Current_node_position=="Left" :
                Desired_semantic.extend(Parent_node_semantic[i]+Oppsite_node_output)
            else :
                Desired_semantic.extend(Oppsite_node_output-Parent_node_semantic[i])
                
    if Parent_node_name=="mul":
        for i in range(0,Parent_node_semantic.shape[0]):
            if Oppsite_node_output!=0 :
                Desired_semantic.extend(Parent_node_semantic[i]/Oppsite_node_output)
            elif Oppsite_node_output==0 and Parent_node_semantic[i]==0 :
                Desired_semantic.extend([float('nan')]) #In this case all value is possible, but it is ambigious for cluster. Therefore, we will set this values as "nan"
            else:
                Desired_semantic.extend([float('inf')])
                
    if Parent_node_name=="protectedDiv":
        for i in range(0,Parent_node_semantic.shape[0]):
            if Current_node_position=="Left" :
                if Oppsite_node_output!=float('inf') or Oppsite_node_output!=float('-inf') :
                    Desired_semantic.extend(Parent_node_semantic[i]*Oppsite_node_output)
                elif (Oppsite_node_output==float('inf') or Oppsite_node_output==float('-inf')) and Parent_node_semantic[i]==0 :
                    Desired_semantic.extend([float('nan')]) #In this case all value is possible, but it is ambigious for cluster. Therefore, we will set this values as "nan"
                else:
                    Desired_semantic.extend([float('inf')])
            else :
                if Oppsite_node_output!=0 :
                    Desired_semantic.extend(Oppsite_node_output/Parent_node_semantic[i])
                elif Oppsite_node_output==0 and Parent_node_semantic[i]==0 :
                    Desired_semantic.extend([float('nan')]) #In this case all value is possible, but it is ambigious for cluster. Therefore, we will set this values as "nan"
                else:
                    Desired_semantic.extend([float('inf')])

    if Parent_node_name=="log":
        for i in range(0,Parent_node_semantic.shape[0]):
            try : 
                Desired_semantic.extend([numpy.power(10,Parent_node_semantic[i])])
                Desired_semantic.extend([-numpy.power(10,Parent_node_semantic[i])])
            except:
                Desired_semantic.extend([float('inf')])

    if Parent_node_name=="exp":
        for i in range(0,Parent_node_semantic.shape[0]):
            try : 
                Desired_semantic.extend([numpy.log(Parent_node_semantic[i])])
            except:
                Desired_semantic.extend([float('inf')])
                
    if Parent_node_name=="sin":
        for i in range(0,Parent_node_semantic.shape[0]):
            if abs(Parent_node_semantic[i])<=1:
                Desired_semantic.extend([numpy.arcsin(Parent_node_semantic[i])-2*math.pi])
                Desired_semantic.extend([numpy.arcsin(Parent_node_semantic[i])])
            else :
                Desired_semantic.extend([float('inf')])
                
    if Parent_node_name=="cos":
        for i in range(0,Parent_node_semantic.shape[0]):
            if abs(Parent_node_semantic[i])<=1:
                Desired_semantic.extend([numpy.arccos(Parent_node_semantic[i])-2*math.pi])
                Desired_semantic.extend([numpy.arccos(Parent_node_semantic[i])])
            else :
                Desired_semantic.extend([float('inf')])


    return numpy.array(Desired_semantic)

def distance(x,y):
    a=(x-y)**2
    return a

def library_search(Desired_semantic, Library, Tree_list,x_data):
    Check_nan_list=[numpy.isnan(Desired_semantic[i]).sum()/len(Desired_semantic[i]) for i in range(0,len(Desired_semantic))]
    if sum(Check_nan_list)/len(Check_nan_list) != 1.0:
        Total_distance_list=[]
        Total_selected_desired_semantic_index_list=[]
        Total_selected_distance_list=[]
        #Find best
        for i in range(len(Library)):
            Library_semantic=Library.loc[Library.index[i],"Data0":].values
            Distance_list=[]
            Selected_desired_semantic_list=[]
            Selected_distance_list=[]
            
            # Select closest desired semantic
            for j in range(len(Library_semantic)):
                Temp_distance=distance(Library_semantic[j],numpy.array(Desired_semantic[j]))
                Distance_list.append(Temp_distance)
                Selected_desired_semantic_list.append(numpy.argmin(Temp_distance))
                Selected_distance_list.append(Temp_distance.min())
            #Save
            Total_distance_list.append(numpy.array(Distance_list))
            Total_selected_desired_semantic_index_list.append(Selected_desired_semantic_list)
            Total_selected_distance_list.append(Selected_distance_list)
        
        #Find best index
        Mean_distance_list=[numpy.array(i).mean() for i in Total_selected_distance_list]
        Min_distance_index=Mean_distance_list.index(min(Mean_distance_list))
        Best_desired_semantic=[Desired_semantic[i][Total_selected_desired_semantic_index_list[Min_distance_index][i]] for i in range(len(Desired_semantic))]
        Best_desired_semantic=pandas.Series(Best_desired_semantic)
        Best_tree=copy.deepcopy(Tree_list[Library.loc[Min_distance_index,"Combination"]])
        Best_tree=levenberg_marquardt(Best_tree,x_data,Best_desired_semantic)
    else:
        Best_tree=random.choice(Tree_list)
    return Best_tree


def RDO(individual,x_train, y_train):
    
    Tree=copy.deepcopy(individual)
    Random_depth=random.randint(0, Tree.height)
    Depth_list=[len(Tree.searchPath(i))-1 for i in range(1,len(Tree))]
    Depth_list.insert(0,0)
    Selected_node=random.choice(numpy.where(numpy.array(Depth_list) == Random_depth)[0].tolist())
    Desired_semantic=semantic_backpropagation(x_train,y_train, Tree, Selected_node)
    Best_tree=library_search(Desired_semantic, MWV.Semantic_dataframe_list[0], MWV.Total_tree_list[0],x_train)
    slice1=Tree.searchSubtree(Selected_node)
    Tree[slice1]=Best_tree
    return PrimitiveTree(Tree)








####################################################### Semantic cluster operator #######################################################

def internal_search_semantic_partial(individual,proportion):
            
    #Initialization
    Neighborhood=[]
    Full_length=len(individual)
    if Full_length*proportion<1 :
        index_list=numpy.random.choice(Full_length,1,replace=False)
    else : 
        index_list=numpy.random.choice(Full_length,int(round(Full_length*proportion,1)),replace=False)
    #Select node
    for index_modify in index_list:
        #Select node
        
        #Create tree
        for Cluster in MWV.Representative_dataframe_list[0].loc[:,"Clusters"].values:
            
            #Representative tree index
            Representative_tree_index=int(MWV.Representative_dataframe_list[0][MWV.Representative_dataframe_list[0]["Clusters"]==Cluster]["Combination"])
            
            #Copy Original and Representative tree
            Original_tree=copy.deepcopy(individual)
            Representative_tree=copy.deepcopy( MWV.Total_tree_list[0][Representative_tree_index])
            #Crossover
            Slice1=Original_tree.searchSubtree(index_modify)
            Slice2=Representative_tree.searchSubtree(0)
            Original_tree[Slice1]=Representative_tree[Slice2]
            
            #Save information
            Original_tree.Node_position=index_modify
            Original_tree.Cluster=Cluster
            Original_tree.Tree_index=Representative_tree_index

            #Append created tree to neighborhood list
            Neighborhood.append(Original_tree)
    return Neighborhood

    
def internal_search_semantic_detail(individual,Node_position,Cluster):

    #Initialization
    Neighborhood=[]
    
    #Find Semantic neighbor tree
    Semantic_neighbor_tree_list_index=list(MWV.Semantic_dataframe_list[0][MWV.Semantic_dataframe_list[0]["Clusters"]==Cluster]["Combination"])
    Semantic_neighbor_tree_list=[MWV.Total_tree_list[0][int(i)] for i in Semantic_neighbor_tree_list_index]
    

    #Create tree
    for i in range(len(Semantic_neighbor_tree_list)):
        
        #Copy Original and Representative tree
        Original_tree=copy.deepcopy(individual)
        Semantic_neighbor_tree=copy.deepcopy(Semantic_neighbor_tree_list[i])
        
        #Crossover
        Slice1=Original_tree.searchSubtree(Node_position)
        Slice2=Semantic_neighbor_tree.searchSubtree(0)
        Original_tree[Slice1]=Semantic_neighbor_tree[Slice2]
        
        #Save information
        Original_tree.Tree_index=Semantic_neighbor_tree_list_index[i]
        Original_tree.Node_position=Node_position
        Original_tree.Cluster=Cluster

        
        Neighborhood.append(Original_tree)
    return Neighborhood



def pruning_list_generator(individual,type_=None):

    
    if type_ is None:
        type_ = MWV.pset.ret
        
    neighborhood=[]
    index_primitive=search_primitive(individual)
    if len(index_primitive)>0:
        del index_primitive[0]
    for index_modify in index_primitive:
        for terminal in range(0,len(MWV.pset.terminals[type_])):
            tree=copy.deepcopy(individual)
            small_tree=PrimitiveTree(generate_terminal_tree(terminal,MWV.pset))
            slice1=tree.searchSubtree(index_modify)
            tree[slice1]=small_tree #it makes error because constant dosen't have arity when it was not a primitive tree, so PrimitiveTree __setitem__ was modified
            tree=PrimitiveTree(tree)
            tree.terminal_position=index_modify
            tree.cluster=None
            tree.state="pruning"
            neighborhood.append(tree) #We makes uncomplete tree to Primitive tree one more, because previous one can't be evaluated
        
    return neighborhood

def rooting_list_generator(individual,type_=None):

    if type_ is None:
        type_ = MWV.pset.ret
        
    neighborhood=[]
    index_primitive=search_primitive(individual)
    if len(index_primitive)>0:
        del index_primitive[0]
    for index_modify in index_primitive:
        tree=copy.deepcopy(individual)
        slice1=tree.searchSubtree(index_modify)
        tree=tree[slice1] #it makes error because constant dosen't have arity when it was not a primitive tree, so PrimitiveTree __setitem__ was modified
        tree=PrimitiveTree(tree)
        tree.terminal_position=index_modify
        tree.cluster=None
        tree.state="rooting"
        neighborhood.append(tree) #We makes uncomplete tree to Primitive tree one more, because previous one can't be evaluated
    
    return neighborhood

def improvement_stopping_condition(Improvement_history,limit):
    if len(Improvement_history)>int(limit)-1:
        return len([i for i in Improvement_history[-int(limit):] if i>=0])==int(limit)
    else:
        return False
    
def semantic_cluster_operator(Original_tree,Node_search_rate,x_train,y_train):
    ########################################### Representative search ###########################################

    Neighborhood=[]
    Neighborhood.extend(internal_search_semantic_partial(Original_tree,Node_search_rate))

    if Neighborhood==[]:
        return Original_tree
    
    else:
        for i in range(0,len(Neighborhood)):
            Neighborhood[i]=levenberg_marquardt(Neighborhood[i],x_train,y_train)
            Neighborhood[i].fitness=evaluation_MSE(Neighborhood[i],x_train,y_train)
                
        Best_neighbor_tree_internal=copy.deepcopy(min(Neighborhood, key=attrgetter("fitness"))) 
        
        
        ########################################### Detailed search ###########################################
        
        Neighborhood=[]
        Neighborhood.extend(internal_search_semantic_detail(Original_tree,Best_neighbor_tree_internal.Node_position,Best_neighbor_tree_internal.Cluster))
    
    
        for i in range(0,len(Neighborhood)):
            Neighborhood[i]=levenberg_marquardt(Neighborhood[i],x_train,y_train)
            Neighborhood[i].fitness=evaluation_MSE(Neighborhood[i],x_train,y_train)
    
        Best_neighbor_tree=copy.deepcopy(min(Neighborhood, key=attrgetter("fitness")))
    
        return Best_neighbor_tree

    
def multiple_local_searches(individual,x_train,y_train,x_test,y_test,x_val,y_val):
    #### Initial parameter setting
    Count=0
    Count_max=10000
    Local_optimum_count=0
    Improvement_limit_count=0
    Selection_limit_count=0
    Height_limit_count=0
    Pruned_count=0
    
    Original_tree=copy.deepcopy(individual)   
    Original_tree.fitness=evaluation_MSE(Original_tree,x_train,y_train)
    Original_tree.fitness_test=evaluation_MSE_test(Original_tree,x_test,y_test)
    Original_tree.fitness_val=evaluation_MSE(Original_tree,x_val,y_val)
    Best_neighbor_tree=copy.deepcopy(Original_tree)
    Best_neighbor_tree.fitness=1
    Best_val_tree=Original_tree
    Best_val_neighborhood=[Original_tree]
        
    
    Improvement_history=[]
    Selected_tree_history=[]
    Similarity_history=[]
    Fitness_history=[-Original_tree.fitness]
    Validation_history=[-Original_tree.fitness_val]
    MSE_test_history=[-Original_tree.fitness_test]
    Fitness_count_history=[MWV.Fitness_count]
    Size_history=[len(Original_tree)]
    Pruned_history=[0]
    
    
    while Count<Count_max+1:
        
        ########################################### Termination condition ###########################################
        if Count!=0:          
            if improvement_stopping_condition(Improvement_history,MWV.Patience) :
                Improvement_limit_count=Improvement_limit_count+1
                print('\033[95m'+"Improvement_Stopping")
                break
            if MWV.Fitness_count>MWV.Fitness_count_limit:
                print('\033[95m'+"Fitness_count_limit")
                break
            
            Original_tree=copy.deepcopy(Best_neighbor_tree)
    
        ########################################### Representative search ###########################################
    
        Neighborhood=[]
        if Count%2==0:
            Neighborhood.extend(internal_search_semantic_partial(Original_tree,0.0))
        else:
            Neighborhood.extend(internal_search_semantic_partial(Original_tree,1.0))
            # Neighborhood.extend(internal_search_semantic_random(Original_tree,pset))
    
        
        if Neighborhood==[]:
            Neighborhood=[Original_tree]
            print("Empty neighbor")
            break
        
        for i in range(0,len(Neighborhood)):
            Neighborhood[i]=levenberg_marquardt(Neighborhood[i],x_train,y_train)
            Neighborhood[i].fitness=evaluation_MSE(Neighborhood[i],x_train,y_train)
                
        Best_neighbor_tree_internal=copy.deepcopy(min(Neighborhood, key=attrgetter("fitness"))) 
        
        
        ########################################### Detailed search ###########################################
        
        Neighborhood=[]
        Neighborhood.extend(internal_search_semantic_detail(Original_tree,Best_neighbor_tree_internal.Node_position,Best_neighbor_tree_internal.Cluster))
    
    
        for i in range(0,len(Neighborhood)):
            Neighborhood[i]=levenberg_marquardt(Neighborhood[i],x_train,y_train)
            Neighborhood[i].fitness=evaluation_MSE(Neighborhood[i],x_train,y_train)
    
        Best_neighbor_tree=copy.deepcopy(min(Neighborhood, key=attrgetter("fitness")))
        ########################################### Elimination search ###########################################
         
        Pruned_count_before=Pruned_count
        
        if Count%2==0:
            Best_pruned_tree=copy.deepcopy(Best_neighbor_tree)
            Count_pruning=0
            
            while Best_pruned_tree.fitness<=Best_neighbor_tree.fitness:
                
                if Count_pruning!=0:
                    print(Best_pruned_tree.state)
                    Pruned_count=Pruned_count+1
                    Best_neighbor_tree=copy.deepcopy(Best_pruned_tree)
                    
                
                Neighborhood=[]
                Neighborhood.extend(pruning_list_generator(Best_neighbor_tree))
                # Neighborhood.extend(rooting_list(Best_neighbor_tree,pset))
                if Neighborhood==[]:
                    break
                Neighborhood_index_list=list(range(0,len(Neighborhood)))    
                
                
                for i in range(0,len(Neighborhood)):
                    Selected_index=random.randrange(len(Neighborhood_index_list))
                    Neighborhood_index=Neighborhood_index_list.pop(Selected_index)
                    
                    Neighborhood[Neighborhood_index]=levenberg_marquardt(Neighborhood[Neighborhood_index],x_train,y_train)
                    Neighborhood[Neighborhood_index].fitness=evaluation_MSE(Neighborhood[Neighborhood_index],x_train,y_train)
                    
                    if Neighborhood[Neighborhood_index].fitness<=Best_neighbor_tree.fitness:
                        break
                    
                Best_pruned_tree=copy.deepcopy(Neighborhood[Neighborhood_index])
                Best_pruned_tree.Cluster=Best_neighbor_tree.Cluster
                Best_pruned_tree.Tree_index=Best_neighbor_tree.Tree_index
                Count_pruning=Count_pruning+1
        
    
        ########################################### Update best validation tree ###########################################
        Best_neighbor_tree.fitness_test=evaluation_MSE_test(Best_neighbor_tree,x_test,y_test)
        Best_neighbor_tree.fitness_val=evaluation_MSE(Best_neighbor_tree,x_val,y_val)
        Improvement_history.append(Best_neighbor_tree.fitness_val-Original_tree.fitness_val)
        
        if Best_val_tree.fitness_val>Best_neighbor_tree.fitness_val:
            Best_val_tree=copy.deepcopy(Best_neighbor_tree)
            Best_val_neighborhood=copy.deepcopy(Neighborhood)
                            
        Selected_tree_history.append([int(Best_neighbor_tree.Cluster),int(Best_neighbor_tree.Tree_index)])
        Fitness_history.append(-1*Best_neighbor_tree.fitness)
        MSE_test_history.append(-1*Best_neighbor_tree.fitness_test)
        Fitness_count_history.append(MWV.Fitness_count)
        Validation_history.append(-1*Best_neighbor_tree.fitness_val)
        Size_history.append(len(Best_neighbor_tree))
        # Pruned_history.append(Pruned_count-Pruned_count_before)
            
        print(Original_tree)
        print(Best_neighbor_tree)
        print("Original_fitness :  {A}, Improved_fitness : {B}\nOriginal_MSE_test : {C}, Improved_MSE_test : {D},\nOriginal_MSE_val : {E}, Improved_MSE_val : {F}".format(A=Original_tree.fitness,B=Best_neighbor_tree.fitness,C=Original_tree.fitness_test,D=Best_neighbor_tree.fitness_test, E=Original_tree.fitness_val, F=Best_neighbor_tree.fitness_val))
        print("Improvement_history : "+str(Improvement_history))
        print("Selected_tree_history : "+str(Selected_tree_history))
        print("Fitness_count : "+str(MWV.Fitness_count))
        print("Size_history : "+str(Size_history))
        update_list([Best_neighbor_tree],80,MWV.Noise_data,decimal=MWV.Library_decimal)
        Count=Count+1
        
    MWV.Best_val_neighborhood=Best_val_neighborhood
    MWV.Local_optimum_count=Local_optimum_count
    MWV.Improvement_limit_count=Improvement_limit_count
    MWV.Selection_limit_count=Selection_limit_count
    MWV.Height_limit_count=Height_limit_count
    MWV.Pruned_count=Pruned_count
    MWV.Improvement_history=Improvement_history
    MWV.Selected_tree_history=Selected_tree_history
    MWV.Fitness_history=Fitness_history
    MWV.MSE_test_history=MSE_test_history
    MWV.Validation_history=Validation_history
    MWV.Fitness_count_history=Fitness_count_history
    MWV.Size_history=Size_history
    MWV.Pruned_history=Pruned_history
            
    return Best_val_tree
  
 


####################################################### For parameter test #######################################################

def semantic_cluster_operator_for_test(Original_tree,Node_search_rate,x_train,y_train):
    ########################################### Representative search ###########################################

    Neighborhood=[]
    Neighborhood.extend(internal_search_semantic_partial(Original_tree,Node_search_rate))

    if Neighborhood==[]:
        return Original_tree
    
    else:
        for i in range(0,len(Neighborhood)):
            Neighborhood[i]=levenberg_marquardt(Neighborhood[i],x_train,y_train)
            Neighborhood[i].fitness=evaluation_MSE(Neighborhood[i],x_train,y_train)
                
        Best_neighbor_tree_internal=copy.deepcopy(min(Neighborhood, key=attrgetter("fitness"))) 
        
        Neighborhood_pre=copy.deepcopy(Neighborhood)
        ########################################### Detailed search ###########################################
        
        Neighborhood=[]
        Neighborhood.extend(internal_search_semantic_detail(Original_tree,Best_neighbor_tree_internal.Node_position,Best_neighbor_tree_internal.Cluster))
    
    
        for i in range(0,len(Neighborhood)):
            Neighborhood[i]=levenberg_marquardt(Neighborhood[i],x_train,y_train)
            Neighborhood[i].fitness=evaluation_MSE(Neighborhood[i],x_train,y_train)
    
        Best_neighbor_tree=copy.deepcopy(min(Neighborhood, key=attrgetter("fitness")))
        
        
        Neighborhood_det=copy.deepcopy(Neighborhood)
        return Best_neighbor_tree,Neighborhood_pre,Neighborhood_det

def exhaustive_serach(Original_tree,x_train,y_train):
    ########################################### Representative search ###########################################
        Neighborhood=[]
        for Node_position in range(0,len(Original_tree)):
            for Cluster in MWV.Representative_dataframe_list[0]["Clusters"].values:
                Neighborhood.extend(internal_search_semantic_detail(Original_tree,Node_position,Cluster))
    
    
        for i in range(0,len(Neighborhood)):
            Neighborhood[i]=levenberg_marquardt(Neighborhood[i],x_train,y_train)
            Neighborhood[i].fitness=evaluation_MSE(Neighborhood[i],x_train,y_train)
    
        Best_neighbor_tree=copy.deepcopy(min(Neighborhood, key=attrgetter("fitness")))
    
        return Best_neighbor_tree, Neighborhood
    