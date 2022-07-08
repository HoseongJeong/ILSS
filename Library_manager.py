import copy 
import pandas
import numpy
from scipy.optimize import curve_fit
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_samples
from sklearn.preprocessing import RobustScaler
from  Base import *
from  Initialization import *
from  Evaluation import *
from Module_wide_function import * 
import Module_wide_variable as MWV

# import hdbscan
def create_random_noise(Number_of_data,min_scale,max_scale,x_data):
    column=x_data.columns
    column_size=len(column)
    Noise=numpy.random.rand(Number_of_data,column_size)*(max_scale-min_scale)+min_scale
    Noise=Noise.tolist()
    scale_max=x_data.max()
    scale_min=x_data.min()
    Noise_data=pandas.DataFrame(data=Noise,columns=column)
    Noise_data=Noise_data*(scale_max-scale_min)+scale_min
    Noise_data=Noise_data.values
    return Noise_data
        
# def reorder(Clustered_data_mean,Clustered_data,Tree_list):
#     Clustered_data_mean=copy.deepcopy(Clustered_data_mean)
#     Clustered_data=copy.deepcopy(Clustered_data)
    
#     for i in range(len(Clustered_data_mean)):
#         Representative_tree=Tree_list[int(list(Clustered_data_mean["Combination"])[i])]
                    
#     Original_cluster_data_mean=list(Clustered_data_mean["Clusters"].values)
#     New_cluster_data_mean=list(range(0,len(Clustered_data_mean)))
#     Convert_cluster_dictionary=dict(zip(Original_cluster_data_mean,New_cluster_data_mean))
#     #for noise cluster
#     Convert_cluster_dictionary[-1]=-1
    
#     Clustered_data_mean["Clusters"]=New_cluster_data_mean
    
#     New_cluster_data=[Convert_cluster_dictionary[int(list(Clustered_data["Clusters"])[i])] for i in range(len(Clustered_data))]

#     Clustered_data["Clusters"]=New_cluster_data
    
#     return Clustered_data_mean,Clustered_data
    


def semantic_clustering(Noise_data,The_number_of_tree_creations, The_limit_height_of_random_tree,
                      Clustering_range, decimal=3, distance='euclidean'):
    if type(Clustering_range)==int:
        Clustering_range=range(Clustering_range,Clustering_range+1,10)
    #Initial parameter setting
    MWV.Clustering_evaluation_dataframe_list=[]
    MWV.Semantic_dataframe_list=[]
    MWV.Representative_dataframe_list=[]
    MWV.Total_tree_list=[]
    Count=0
    
    
    #Add constant tree in Tree_list
    Constant_tree=PrimitiveTree(generate_terminal_tree(-1,MWV.pset))
    Constant_tree[0].value=0.5
    Constant_tree[0].name=0.5
    Tree_list=[Constant_tree]
    
    #Tree creation
    while Count<The_number_of_tree_creations:
        Count=Count+1
        Full_tree=PrimitiveTree(genHalfAndHalf(MWV.pset, 1,The_limit_height_of_random_tree))
        
        for Node in range(len(Full_tree)):
            #Select subtree
            Temp_tree=PrimitiveTree(Full_tree[Full_tree.searchSubtree(Node)])
            Temp_tree_output=output_list_generator(Temp_tree,Noise_data)
            Temp_tree_output_max=max(Temp_tree_output)
            Temp_tree_output_min=min(Temp_tree_output)
            
            # Since the constant has already been entered, the input of the constant is prohibited.
            if Temp_tree_output_max-Temp_tree_output_min!=0:
                mul_parameter=float(1/abs(Temp_tree_output_max-Temp_tree_output_min))
                add_parameter=float(-mul_parameter*(Temp_tree_output_min))
                #Add tree in library, if parameter is not nan or inf
                if numpy.isnan(mul_parameter)==False and numpy.isinf(mul_parameter)==False and numpy.isnan(add_parameter)==False and numpy.isinf(add_parameter)==False:
                    Linear_scaling_tree=PrimitiveTree(generate_linear_scaling_tree(MWV.pset.terminals[MWV.pset.ret][0], MWV.pset))
                    Linear_scaling_tree[2].value=mul_parameter
                    Linear_scaling_tree[2].name=mul_parameter
                    Linear_scaling_tree[4].value=add_parameter
                    Linear_scaling_tree[4].name=add_parameter
                    Tree_list.append(linear_scailing(Linear_scaling_tree,Temp_tree))
                
    #Sort by size    
    Tree_list=sorted(Tree_list, key=len,reverse=False)
    
    #Make dataframe
    Column_name=["Expression","Combination"]
    Columm_data_number=['Data{}'.format(i) for i in range(0,len(Noise_data))]
    Column_name.extend(Columm_data_number)
    Semantic_dataframe=pandas.DataFrame(columns=Column_name)
    Clustering_evaluation_dataframe=pandas.DataFrame()
    
    #Create Library
    for i in range(0,len(Tree_list)):
        Prediction=[str(Tree_list[i]),None]
        Prediction.extend(list(numpy.round(output_list_generator(Tree_list[i],Noise_data),decimal)))
        Semantic_dataframe.loc[i]=Prediction
        
    #Delete Same semantic 
    Semantic_dataframe=Semantic_dataframe.replace([numpy.inf, -numpy.inf], numpy.nan).dropna(subset=Columm_data_number)
    Semantic_dataframe=Semantic_dataframe.drop_duplicates(subset=Columm_data_number)
    Unique_index=Semantic_dataframe.index.tolist()
    Semantic_dataframe.reset_index(drop=True,inplace=True)
    Semantic_dataframe["Combination"]=Semantic_dataframe.index
    Tree_list=[Tree_list[i] for i in Unique_index]

    #Optimize the number of cluster  
    Data = Semantic_dataframe.loc[:,"Data0":]
    for i in Clustering_range:
        try:
            Model=KMedoids(n_clusters=i)
            y_predict = Model.fit_predict(Data)
            cluster_labels = numpy.unique(y_predict)
            n_clusters = cluster_labels.shape[0]
            silhouette_vals = silhouette_samples(Data, y_predict, metric = 'euclidean')
            silhoutte_avg = numpy.mean(silhouette_vals)
            #기록
            Clustering_evaluation_dataframe.loc[i,"C"]=i
            Clustering_evaluation_dataframe.loc[i,"Evaluation"]=silhoutte_avg
            print(i)
        except:
            break
        
    #Select best cluster and reclustering
    C=int(Clustering_evaluation_dataframe[Clustering_evaluation_dataframe["Evaluation"]==Clustering_evaluation_dataframe["Evaluation"].max()]["C"].values[0])
    Model=KMedoids(n_clusters=C)
    Model.fit(Data)
    y_predict = Model.fit_predict(Data)
    Semantic_dataframe['Clusters'] = y_predict
    
    #Reoder column
    Semantic_dataframe_column = ["Expression","Combination","Clusters"]
    Semantic_dataframe_column.extend(Columm_data_number)
    Semantic_dataframe = Semantic_dataframe[Semantic_dataframe_column]
    
    #Find representatives
    Representative_dataframe=Semantic_dataframe.loc[:,"Clusters":].groupby(['Clusters'], as_index=False).mean()
    Representative_combination=[Model.medoid_indices_[i] for i in Representative_dataframe["Clusters"].values]
    Representative_expression=[str(Tree_list[i]) for i in Representative_combination]
    Representative_dataframe.insert(loc=0,column="Expression",value=Representative_expression)
    Representative_dataframe.insert(loc=1,column="Combination",value=Representative_combination)
    
    #reorder
    # Representative_dataframe,Semantic_dataframe=reorder(Representative_dataframe,Semantic_dataframe,Tree_list)

    #Save dataframe
    MWV.Clustering_evaluation_dataframe_list.append(Clustering_evaluation_dataframe)
    MWV.Semantic_dataframe_list.append(Semantic_dataframe)
    MWV.Representative_dataframe_list.append(Representative_dataframe)
    MWV.Total_tree_list.append(Tree_list)
        

def update_list(population,Height_limit,Noise_data,decimal=3):
    print("Update_Tree_list : ")
    
    population=copy.deepcopy(population)
    #Initialize
    Addition_count=0
    Columm_data_number=['Data{}'.format(i) for i in range(0,len(Noise_data))]
    Initial_size_of_semantic_data_frame=len(MWV.Total_tree_list[0])
    Change_record=0
    
    
    for Tree_index in range(len(population)):      
        for Node_index in range(len(population[Tree_index])):
            #Select subtree
            Temp_tree=copy.deepcopy(population[Tree_index])
            Temp_tree=PrimitiveTree(Temp_tree[Temp_tree.searchSubtree(Node_index)])
            if Temp_tree.height<=Height_limit :
                Temp_tree_output=output_list_generator(Temp_tree,Noise_data)
                Temp_tree_output_max=max(Temp_tree_output)
                Temp_tree_output_min=min(Temp_tree_output)
                # Since the constant has already been entered, the input of the constant is prohibited.
                if Temp_tree_output_max-Temp_tree_output_min!=0:
                    mul_parameter=float(1/abs(Temp_tree_output_max-Temp_tree_output_min))
                    add_parameter=float(-mul_parameter*(Temp_tree_output_min))
                    
                    #Add if parameter is not nan or inf
                    if numpy.isnan(mul_parameter)==False and numpy.isinf(mul_parameter)==False and numpy.isnan(add_parameter)==False and numpy.isinf(add_parameter)==False:
                        Linear_scaling_tree=PrimitiveTree(generate_linear_scaling_tree(MWV.pset.terminals[MWV.pset.ret][0], MWV.pset))
                        Linear_scaling_tree[2].value=mul_parameter
                        Linear_scaling_tree[2].name=mul_parameter
                        Linear_scaling_tree[4].value=add_parameter
                        Linear_scaling_tree[4].name=add_parameter

                        #Calculate output_generator of linearscaled tree
                        Temp_tree=linear_scailing(Linear_scaling_tree,Temp_tree)
                        Temp_output=list(numpy.round(output_list_generator(Temp_tree,Noise_data),decimal))
                        
                        #Check nan
                        if  numpy.isnan(Temp_output).any() == False and numpy.isinf(Temp_output).any() == False:
                            # Check duplicated tree
                            Duplicated_dataframe=None
                            Duplicated_dataframe=MWV.Semantic_dataframe_list[0][Columm_data_number].isin(Temp_output).all(1)
                            if sum(list(Duplicated_dataframe))>0:
                                Target_index=int(MWV.Semantic_dataframe_list[0].loc[Duplicated_dataframe,"Combination"].values[0])
                                Original_tree=MWV.Total_tree_list[0][Target_index]
                            else:
                                Target_index=None
                                
                            #Insert new tree
                            if sum(list(Duplicated_dataframe))==0:
                            
                                #Append data
                                Current_index_of_total_tree=Initial_size_of_semantic_data_frame+Addition_count
        
                                MWV.Total_tree_list[0].append(Temp_tree)
                                Temp_semantic_data=[str(Temp_tree),Initial_size_of_semantic_data_frame+Addition_count,"None"]
                                Temp_semantic_data.extend(Temp_output)
                                MWV.Semantic_dataframe_list[0].loc[Current_index_of_total_tree,:]=Temp_semantic_data
                                
                                print (str(Temp_tree) + "was added to library")
                                
                                Addition_count=Addition_count+1
                                Change_record=Change_record+1
                            
                            #Substitute small tree
                            elif sum(list(Duplicated_dataframe))>0 and len(Temp_tree)<len(Original_tree):
                                Temp_semantic_data=[str(Temp_tree),Target_index,"None"]
                                Temp_semantic_data.extend(Temp_output)
                                MWV.Total_tree_list[0][Target_index]=Temp_tree
                                MWV.Semantic_dataframe_list[0].loc[Target_index,:]=Temp_semantic_data
                                print(str(Target_index)+"th tree \n"+str(Original_tree) + "\n was changed to \n"+str(Temp_tree))
                                
                                Change_record=Change_record+1

    #If more than one of tree was inserted, perform clustering
    if Change_record>0:
        Data = MWV.Semantic_dataframe_list[0].loc[:,"Data0":]

        C=int(MWV.Clustering_evaluation_dataframe_list[0][MWV.Clustering_evaluation_dataframe_list[0]["Evaluation"]==MWV.Clustering_evaluation_dataframe_list[0]["Evaluation"].max()]["C"].values[0])
        Model=KMedoids(n_clusters=C)
        Model.fit(Data)
        y_predict = Model.fit_predict(Data)
        MWV.Semantic_dataframe_list[0]['Clusters'] = y_predict
        
        
        #Find representatives
        MWV.Representative_dataframe_list[0]=MWV.Semantic_dataframe_list[0].loc[:,"Clusters":].groupby(['Clusters'], as_index=False).mean()
        Representative_combination=[Model.medoid_indices_[i] for i in MWV.Representative_dataframe_list[0]["Clusters"].values]
        Representative_expression=[str(MWV.Total_tree_list[0][i]) for i in Representative_combination]
        MWV.Representative_dataframe_list[0].insert(loc=0,column="Expression",value=Representative_expression)
        MWV.Representative_dataframe_list[0].insert(loc=1,column="Combination",value=Representative_combination)
        
        #Organize dataframe
        # MWV.Representative_dataframe_list[0],MWV.Semantic_dataframe_list[0]=reorder(MWV.Representative_dataframe_list[0],MWV.Semantic_dataframe_list[0],MWV.Total_tree_list[0])

def semantic_library(Noise_data,The_number_of_tree_creations, The_limit_height_of_random_tree,decimal=3):
    #Initial parameter setting
    MWV.Semantic_dataframe_list=[]
    MWV.Total_tree_list=[]
    Count=0
    
    
    #Add constant tree in Tree_list
    Constant_tree=PrimitiveTree(generate_terminal_tree(-1,MWV.pset))
    Constant_tree[0].value=0.5
    Constant_tree[0].name=0.5
    Tree_list=[Constant_tree]
    
    #Tree creation
    while Count<The_number_of_tree_creations:
        Count=Count+1
        Full_tree=PrimitiveTree(genHalfAndHalf(MWV.pset, 1,The_limit_height_of_random_tree))
        
        for Node in range(len(Full_tree)):
            #Select subtree
            Temp_tree=PrimitiveTree(Full_tree[Full_tree.searchSubtree(Node)])
            Tree_list.append(Temp_tree)
                
    #Sort by size    
    Tree_list=sorted(Tree_list, key=len,reverse=False)
    
    #Make dataframe
    Column_name=["Expression","Combination"]
    Columm_data_number=['Data{}'.format(i) for i in range(0,len(Noise_data))]
    Column_name.extend(Columm_data_number)
    Semantic_dataframe=pandas.DataFrame(columns=Column_name)
    Clustering_evaluation_dataframe=pandas.DataFrame()
    
    #Create Library
    for i in range(0,len(Tree_list)):
        Prediction=[str(Tree_list[i]),None]
        Prediction.extend(list(numpy.round(output_list_generator(Tree_list[i],Noise_data),decimal)))
        Semantic_dataframe.loc[i]=Prediction
        
    #Delete Same semantic 
    Semantic_dataframe=Semantic_dataframe.replace([numpy.inf, -numpy.inf], numpy.nan).dropna(subset=Columm_data_number)
    Semantic_dataframe=Semantic_dataframe.drop_duplicates(subset=Columm_data_number)
    Unique_index=Semantic_dataframe.index.tolist()
    Semantic_dataframe.reset_index(drop=True,inplace=True)
    Semantic_dataframe["Combination"]=Semantic_dataframe.index
    Tree_list=[Tree_list[i] for i in Unique_index]

    #Save dataframe
    MWV.Semantic_dataframe_list.append(Semantic_dataframe)
    MWV.Total_tree_list.append(Tree_list)
        
def update_list_RDO(population,Height_limit,Noise_data,decimal=3):
    print("Update_Tree_list : ")
    
    population=copy.deepcopy(population)
    #Initialize
    Addition_count=0
    Columm_data_number=['Data{}'.format(i) for i in range(0,len(Noise_data))]
    Initial_size_of_semantic_data_frame=len(MWV.Total_tree_list[0])
    Change_record=0
    
    
    for Tree_index in range(len(population)):      
        for Node_index in range(len(population[Tree_index])):
            #Select subtree
            Temp_tree=copy.deepcopy(population[Tree_index])
            Temp_tree=PrimitiveTree(Temp_tree[Temp_tree.searchSubtree(Node_index)])
            if Temp_tree.height<=Height_limit :
                Temp_tree_output=output_list_generator(Temp_tree,Noise_data)
                Temp_tree_output_max=max(Temp_tree_output)
                Temp_tree_output_min=min(Temp_tree_output)
                # Since the constant has already been entered, the input of the constant is prohibited.
                if Temp_tree_output_max-Temp_tree_output_min!=0:
                    mul_parameter=float(1/abs(Temp_tree_output_max-Temp_tree_output_min))
                    add_parameter=float(-mul_parameter*(Temp_tree_output_min))
                    
                    #Add if parameter is not nan or inf
                    if numpy.isnan(mul_parameter)==False and numpy.isinf(mul_parameter)==False and numpy.isnan(add_parameter)==False and numpy.isinf(add_parameter)==False:
                        Linear_scaling_tree=PrimitiveTree(generate_linear_scaling_tree(MWV.pset.terminals[MWV.pset.ret][0], MWV.pset))
                        Linear_scaling_tree[2].value=mul_parameter
                        Linear_scaling_tree[2].name=mul_parameter
                        Linear_scaling_tree[4].value=add_parameter
                        Linear_scaling_tree[4].name=add_parameter

                        #Calculate output_generator of linearscaled tree
                        Temp_tree=linear_scailing(Linear_scaling_tree,Temp_tree)
                        Temp_output=list(numpy.round(output_list_generator(Temp_tree,Noise_data),decimal))
                        
                        #Check nan
                        if  numpy.isnan(Temp_output).any() == False and numpy.isinf(Temp_output).any() == False:
                        
                            # Check duplicated tree
                            Duplicated_dataframe=None
                            Duplicated_dataframe=MWV.Semantic_dataframe_list[0][Columm_data_number].isin(Temp_output).all(1)
                            if sum(list(Duplicated_dataframe))>0:
                                Target_index=int(MWV.Semantic_dataframe_list[0].loc[Duplicated_dataframe,"Combination"].values[0])
                                Original_tree=MWV.Total_tree_list[0][Target_index]
                            else:
                                Target_index=None
                                
                            #Insert new tree
                            if sum(list(Duplicated_dataframe))==0:
                            
                                #Append data
                                Current_index_of_total_tree=Initial_size_of_semantic_data_frame+Addition_count
        
                                MWV.Total_tree_list[0].append(Temp_tree)
                                Temp_semantic_data=[str(Temp_tree),Initial_size_of_semantic_data_frame+Addition_count,"None"]
                                Temp_semantic_data.extend(Temp_output)
                                MWV.Semantic_dataframe_list[0].loc[Current_index_of_total_tree,:]=Temp_semantic_data
                                print (str(Temp_tree) + "was added to library")
    
                                Addition_count=Addition_count+1
                                Change_record=Change_record+1
                            
                            #Substitute small tree
                            elif sum(list(Duplicated_dataframe))>0 and len(Temp_tree)<len(Original_tree):
                                Temp_semantic_data=[str(Temp_tree),Target_index,"None"]
                                Temp_semantic_data.extend(Temp_output)
                                MWV.Total_tree_list[0][Target_index]=Temp_tree
                                MWV.Semantic_dataframe_list[0].loc[Target_index,:]=Temp_semantic_data
                                print(str(Target_index)+"th tree \n"+str(Original_tree) + "\n was changed to \n"+str(Temp_tree))
                                
                                Change_record=Change_record+1
