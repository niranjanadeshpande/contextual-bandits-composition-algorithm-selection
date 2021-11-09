import warnings
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics.pairwise import rbf_kernel, linear_kernel, sigmoid_kernel
from sklearn.svm import SVR
import sklearn.ensemble
import seaborn as sns
import random
import matplotlib.pyplot as plt
import math
from sklearn.metrics import mean_squared_error
import sys
import time

random.seed(0)

warnings.filterwarnings("ignore")

master_frame_edited = pd.read_pickle("thresholded_master_frame.pkl")



def getCost(selected_label, master_frame, index, max_penalty=3.0):
    min_time = 9999999#float('inf')
    min_mem = 9999999#float('inf')
    penalty = 0.0
    frame_max_util = max(master_frame["MCSP Best Path Utility"].max(), master_frame["ACS 0 Best Path Utility"].max(), master_frame["GA 0 Best Path Utility"].max(), master_frame["PSO 0 Best Path Utility"].max())
    frame_min_util = min(master_frame["MCSP Best Path Utility"].min(), master_frame["ACS 0 Best Path Utility"].min(), master_frame["GA 0 Best Path Utility"].min(), master_frame["PSO 0 Best Path Utility"].min())
    frame_max_time = max(master_frame["MCSP Time"].max(), master_frame["ACS 0 Time"].max(), master_frame["GA 0 Time"].max(), master_frame["PSO 0 Time"].max())
    frame_min_time = min(master_frame["MCSP Time"].min(), master_frame["ACS 0 Time"].min(), master_frame["GA 0 Time"].min(), master_frame["PSO 0 Time"].min())
    
    frame_max_mem = max(master_frame["MCSP Memory"].max(), master_frame["ACS 0 Memory"].max(), master_frame["GA 0 Memory"].max(), master_frame["PSO 0 Memory"].max())
    frame_min_mem = min(master_frame["MCSP Memory"].min(), master_frame["ACS 0 Memory"].min(), master_frame["GA 0 Memory"].min(), master_frame["PSO 0 Memory"].min())
    #max_penalty = 1.0
    if selected_label == master_frame["Final Label"].iloc[index]:
        min_time = master_frame["Final Label Time"].iloc[index]
        min_mem = master_frame["Final Label Memory"].iloc[index]
    elif selected_label=="MCSP":
        min_time = master_frame["MCSP Time"].iloc[index]
        min_mem = master_frame["MCSP Memory"].iloc[index]
    elif selected_label=="GA":  
        min_time = master_frame["GA 0 Time"].iloc[index]
        min_mem = master_frame["GA 0 Memory"].iloc[index]
        #penalty = max_penalty 
    elif selected_label=="ACS":
        min_time = master_frame["ACS 0 Time"].iloc[index]
        min_mem = master_frame["ACS 0 Memory"].iloc[index]
        #penalty = max_penalty 
    elif selected_label=="PSO":
        min_time = master_frame["PSO 0 Time"].iloc[index]
        min_mem = master_frame["PSO 0 Memory"].iloc[index]
        #penalty = max_penalty  
     
    if selected_label in ["GA", "ACS", "PSO"]:
        selected_label = selected_label + " 0"
     
    if selected_label not in master_frame["Pure Solution Algorithms"].iloc[index]:
        penalty = max_penalty
    util_norm = (penalty)
    time_norm = (min_time - frame_min_time)/(frame_max_time - frame_min_time)
    mem_norm = (min_mem - frame_min_mem)/(frame_max_mem - frame_min_mem)         
    cost = util_norm + time_norm + mem_norm
    
    
    return cost#, -1.0*(cost_selected - cost_optimal)

   
'''
def getTrainingCosts(label, data):
    costs = []
    for i in range(0, len(data)):
        cost_raw, cost_diff = getCost(label, data.iloc[i])
        costs.append(cost_diff)
    return costs
'''
def getLabel(selected_algo):
    if selected_algo==0:
        return "MCSP"
    elif selected_algo==1:
        return "ACS"
    elif selected_algo==2:
        return "GA"
    elif selected_algo==3:
        return "PSO"
'''
def rmseCalc(predictions, targets):
    differences = predictions - targets
    differences_squared = differences ** 2
    mean_of_differences_squared = differences_squared.mean()
    rmse_val = np.sqrt(mean_of_differences_squared)
    return rmse_val    
'''

def rmseCalc(true, preds):
    #true, pred
    rmse_val = mean_squared_error(true, preds)
    
    return rmse_val 




def greedy(train_data, oracles, online_dataset, epsilon, verification_dataset, algorithms, master_frame, online_indices, verification_indices, y_cost, mode):
    #oracles is a dictionary now
    all_selections = []
    selections = []
    results = []
    rmse = [] 
    prediction_times = []
    retraining_times = []
    cumulative_cost_arr = []
    cumu_cost = 0.0
    prev_test = pd.DataFrame(data=np.zeros(shape=(train_data.shape[0]+online_dataset.shape[0],online_dataset.shape[1])))
    prev_test.iloc[0:len(train_data),:] = train_data 

    count = len(train_data) - 1 
    
    algo_indices = [train_data.index.tolist() for y in algorithms]

    train_indices = train_data.index.tolist()
    
    for i in range(0, len(online_dataset)): 
        count = count + 1
        print("i is:",i)
        
        test = online_dataset.iloc[i]
        if mode==1:
            #if mode is 1, it is full info, otherwise no learning
            online_indices_curr = online_dataset.index[:-i].tolist()
            for i in range(0, len(oracles.keys())):
                key = list(oracles.keys())[i]
                y_cost_curr = master_frame_edited[online_dataset.columns.tolist()].iloc[train_indices+online_indices_curr]
                x_arr_algo = master_frame[online_dataset.columns.tolist()].iloc[train_indices+online_indices_curr]
                retrain_start = time.time()
                oracles[key] = sklearn.ensemble.RandomForestRegressor(n_estimators = 200, max_depth=20, random_state=19950807)
                oracles[key] = oracles[key].fit(x_arr_algo, y_cost_curr)
                retrain_end = time.time()
                retraining_times.append(retrain_end - retrain_start)
            
            selections = []
            
        test_input = test.values.reshape(1, -1)
        preds = {}
        
        prediction_start = time.time()
        for key in oracles.keys():
            preds[key] = oracles[key].predict(test_input)
        prediction_end = time.time()
        prediction_times.append(prediction_end - prediction_start)
        
        preds = np.asarray([list(a_pred) for a_pred in preds.values()])
        selected_algo_idx = np.argmin(preds) 
        
        selected_algo_label = algorithms[selected_algo_idx]

        index = online_indices[i]
        selected_cost = getCost(selected_algo_label, master_frame, index)
        #online_dataset[selected_algo_label + " Cost"].iloc[i]
        ground_truth_cost = master_frame["Ground Truth Cost"].iloc[index]
        #online_dataset["Ground Truth Cost"].iloc[i]
        #getCost(selected_algo_label, online_dataset.iloc[i])
        cumu_cost += selected_cost
        #(selected_algorithm, selected algorithm cost)
        selections.append((selected_algo_label, selected_cost))
        #y_cost.append(selected_cost)
        algo_indices[selected_algo_idx].append(index)
        #y_cost.append(selected_cost)
        
        #print("len(y_cost)", len(y_cost))
        # for list_index in range(0, len(y_cost)):
        #     list_i = y_cost[list_index]
        #     list_i.append(selected_cost)
        #     y_cost[list_index] = list_i
        y_cost[selected_algo_idx].append(selected_cost)
        #print("len(y_cost)", len(y_cost))
        # for list_index in range(0,len(y_cost)):
        #     list_i = y_cost[list_index]
        #     list_i.append(selected_cost)
        #     y_cost[list_index] = list_i
        
        all_selections.append(selected_algo_label)
        
        rmse.append(rmseCalc([ground_truth_cost], [selected_cost]))
        
        results.append(selected_cost)        
        prev_test.iloc[count,:] = test
        
        cumulative_cost_arr.append(cumu_cost)
    
    with open("greedy_prediction_times.txt", "w") as output:
        output.write(str(prediction_times))
        
        
    verification_labels = []
    for j in range(0, len(verification_dataset)):
        datum = verification_dataset.iloc[j].values
        verification_input = datum.reshape(1, -1)
        
        preds = {}
        for key in oracles.keys():
            preds[key] = oracles[key].predict(verification_input)
        
        
        preds = np.asarray([list(a_pred) for a_pred in preds.values()])
        

        selected_algo_idx = np.argmin(preds)
            
        selected_algo_label = algorithms[selected_algo_idx]
        verification_labels.append(selected_algo_label)
        
        index = verification_indices[j]
        selected_cost = getCost(selected_algo_label, master_frame, index)
        ground_truth_cost = master_frame["Ground Truth Cost"].iloc[index]
        rmse.append(rmseCalc([ground_truth_cost], [selected_cost]))
        
    all_selections += verification_labels

    return rmse, verification_labels, cumulative_cost_arr, oracles, all_selections


def e_greedy(train_data, oracles, online_dataset, epsilon, verification_dataset, algorithms, master_frame, online_indices, verification_indices, y_cost):
    #oracles is a dictionary now
    all_selections = []
    selections = []
    results = []
    rmse = [] 
    
    prediction_times = []
    retraining_times = []
    
    cumulative_cost_arr = []
    cumu_cost = 0.0
    prev_test = pd.DataFrame(data=np.zeros(shape=(train_data.shape[0]+online_dataset.shape[0],online_dataset.shape[1])))
    prev_test.iloc[0:len(train_data),:] = train_data 

    count = len(train_data) - 1 
    
    algo_indices = [train_data.index.tolist() for y in algorithms]

    
    for i in range(0, len(online_dataset)):
        count = count + 1

        
        test = online_dataset.iloc[i]
        if i%10 == 0:
            #Retrain every 10 samples
            if i != 0:
                print("i is:", i)
                    
                for item in selections:
                    
                    np_algorithms = np.asarray(algorithms)
                    key_id = np.where(item[0]==np_algorithms)[0][0]
                    algo_keys = list(oracles.keys())
                    key = algo_keys[key_id]
                    #item[1] is observed cost
                    
                    y_cost_curr = y_cost[key_id]
                    x_arr_algo = master_frame[online_dataset.columns.tolist()].iloc[algo_indices[key_id]]

                    retraining_start = time.time()
                    oracles[key] = sklearn.ensemble.RandomForestRegressor(n_estimators = 200, max_depth=20, n_jobs=4, random_state=19950807)
                    oracles[key] = oracles[key].fit(x_arr_algo, y_cost_curr)
                    retraining_end = time.time()
                    retraining_times.append(retraining_end - retraining_start)
                    
            selections = []
            
        test_input = test.values.reshape(1, -1)
        preds = {}
        prediction_start = time.time()
        for key in oracles.keys():
            preds[key] = oracles[key].predict(test_input)
        prediction_end = time.time()
        prediction_times.append(prediction_end - prediction_start)
        
        #epsilon greedy
        q = random.random()
        preds = np.asarray([list(a_pred) for a_pred in preds.values()])
        #preds = np.asarray([list(a_pred) for a_pred in preds])
        
        if q <= epsilon:
            selected_algo_idx = random.randint(0, len(preds)-1)#np.random.choice(preds)
        else:
            selected_algo_idx = np.argmin(preds) #np.argmax(preds)#
        
        selected_algo_label = algorithms[selected_algo_idx]#getLabel(selected_algo_idx)
        #TODO: Fix this

        index = online_indices[i]
        selected_cost = getCost(selected_algo_label, master_frame, index)
        #online_dataset[selected_algo_label + " Cost"].iloc[i]
        ground_truth_cost = master_frame["Ground Truth Cost"].iloc[index]
        #online_dataset["Ground Truth Cost"].iloc[i]
        #getCost(selected_algo_label, online_dataset.iloc[i])
        cumu_cost += selected_cost
        #(selected_algorithm, selected algorithm cost)
        selections.append((selected_algo_label, selected_cost))
        #y_cost.append(selected_cost)
        algo_indices[selected_algo_idx].append(index)
        #y_cost.append(selected_cost)
        
        #print("len(y_cost)", len(y_cost))
        # for list_index in range(0, len(y_cost)):
        #     list_i = y_cost[list_index]
        #     list_i.append(selected_cost)
        #     y_cost[list_index] = list_i
        y_cost[selected_algo_idx].append(selected_cost)
        #print("len(y_cost)", len(y_cost))
        # for list_index in range(0,len(y_cost)):
        #     list_i = y_cost[list_index]
        #     list_i.append(selected_cost)
        #     y_cost[list_index] = list_i
        
        all_selections.append(selected_algo_label)
        
        rmse.append(rmseCalc([ground_truth_cost], [selected_cost]))
        
        results.append(selected_cost)        
        prev_test.iloc[count,:] = test
        
        cumulative_cost_arr.append(cumu_cost)
    
    with open("e_greedy_prediction_times.txt", "w") as output:
        output.write(str(prediction_times))
        
    with open("e_greedy_retraining_times.txt", "w") as output:
        output.write(str(retraining_times))
        
    verification_labels = []
    for j in range(0, len(verification_dataset)):
        datum = verification_dataset.iloc[j].values
        verification_input = datum.reshape(1, -1)
        
        preds = {}
        for key in oracles.keys():
            preds[key] = oracles[key].predict(verification_input)
        
        
        #epsilon greedy
        q = random.random()
        preds = np.asarray([list(a_pred) for a_pred in preds.values()])
        #preds = np.asarray([list(a_pred) for a_pred in preds])
        

        selected_algo_idx = np.argmin(preds)
            
        selected_algo_label = algorithms[selected_algo_idx]
        verification_labels.append(selected_algo_label)
        
        index = verification_indices[j]
        selected_cost = getCost(selected_algo_label, master_frame, index)
        ground_truth_cost = master_frame["Ground Truth Cost"].iloc[index]
        rmse.append(rmseCalc([ground_truth_cost], [selected_cost]))
        
    all_selections += verification_labels

    return rmse, verification_labels, cumulative_cost_arr, oracles, all_selections

def ucb(train_data, oracles, online_dataset, gamma, verification_dataset, algorithms, master_frame, online_indices, verification_indices, y_cost):
    cumulative_cost_arr = []
    cumu_cost = 0.0
    

    
    
    all_selections = []
    selections = []
    results = []
    rmse = [] 
    
    mcsp_preds_proc = []
    acs_preds_proc = []
    ga_preds_proc = []
    pso_preds_proc = []
    
    selection_counts = np.asarray([1.0, 1.0, 1.0, 1.0]) 
    
    regr_mcsp = oracles["MCSP"]
    regr_acs = oracles["ACS"]
    regr_ga = oracles["GA"]
    regr_pso = oracles["PSO"]
    
    mcsp_val = 0
    acs_val = 0
    ga_val = 0
    pso_val = 0
    
    prev_test = pd.DataFrame(data=np.zeros(shape=(train_data.shape[0]+online_dataset.shape[0],online_dataset.shape[1])))
    prev_test.iloc[0:len(train_data),:] = train_data 
    
    algo_indices = [train_data.index.tolist() for y in algorithms]

    
    count = len(train_data) - 1 
    for i in range(0, len(online_dataset)): 
        count = count + 1
        test = online_dataset.iloc[i]
        if i%10 == 0:
            #print("i is:", i)
            if i != 0:
                for item in selections:
                    #x_arr = prev_test.iloc[0:count].values
                    #x_arr = x_arr.reshape(1,-1)
                    
                    #print("x_arr shape", x_arr.shape)
                    np_algorithms = np.asarray(algorithms)
                    #print("item:", item)
                    key_id = np.where(item[0]==np_algorithms)[0][0]
                    #print("********************************************")
                    print("Key ID:", key_id, "item",item[0])
                    algo_keys = list(oracles.keys())
                    key = algo_keys[key_id]
                    
                    y_cost_curr = y_cost[key_id]
                    x_arr_algo = master_frame[online_dataset.columns.tolist()].iloc[algo_indices[key_id]]

                    oracles[key] = sklearn.ensemble.RandomForestRegressor(n_estimators = 200, max_depth=20, n_jobs=4, random_state=19950807)
                    oracles[key] = oracles[key].fit(x_arr_algo, y_cost_curr)
                    
                   
            selections = []
            
            
        test_input = test.values.reshape(1, -1)
        mcsp_pred = regr_mcsp.predict(test_input)
        acs_pred = regr_acs.predict(test_input)
        ga_pred = regr_ga.predict(test_input)
        pso_pred = regr_pso.predict(test_input)
        
        mcsp_preds_proc.append(mcsp_pred)
        acs_preds_proc.append(acs_pred)
        ga_preds_proc.append(ga_pred)
        pso_preds_proc.append(pso_pred)
        
        
        
        mcsp_val = mcsp_pred + gamma*np.std(np.asarray(mcsp_preds_proc))*selection_counts[0]
        acs_val = acs_pred + gamma*np.std(np.asarray(acs_preds_proc))*selection_counts[1]
        ga_val = ga_pred + gamma*np.std(np.asarray(ga_preds_proc))*selection_counts[2]
        pso_val = pso_pred + gamma*np.std(np.asarray(pso_preds_proc))*selection_counts[3]
    
        
        
        #UCB
        preds = np.asarray([mcsp_val, acs_val, ga_val, pso_val])
        #print("selection:", selection_counts)
        #print("stds:",np.std(np.asarray(mcsp_preds_proc)), np.std(np.asarray(acs_preds_proc)), np.std(np.asarray(ga_preds_proc)), np.std(np.asarray(pso_preds_proc)))
        selected_algo_idx = np.argmin(preds)#np.argmin(preds)
        
        selection_counts[selected_algo_idx] += 1
        
        #print(selected_algo_idx)
        
        selected_algo_label = getLabel(selected_algo_idx)
        
        index = online_indices[i]
        selected_cost = getCost(selected_algo_label, master_frame, index)

        # for list_index in range(0,len(y_cost)):
        #     y_cost[list_index].append(selected_cost)
            #print("List index:", list_index, "length: ",len(y_cost[list_index]))
        #raw_cost, cost_diff = getCost(selected_algo_label, online_dataset.iloc[i])
        
        
        ground_truth_cost = master_frame["Ground Truth Cost"].iloc[index]
        cumu_cost += selected_cost
        selections.append((selected_algo_label, selected_cost))
        
        algo_indices[selected_algo_idx].append(index)

        y_cost[selected_algo_idx].append(selected_cost)
        #selections.append((selected_algo_label, cost_diff))
        
        
        
        
        all_selections.append(selected_algo_label)

        rmse.append(rmseCalc([ground_truth_cost], [selected_cost]))
        
        results.append(selected_cost)        
        prev_test.iloc[count,:] = test
        
        cumulative_cost_arr.append(cumu_cost)
        
        
    verification_labels = []
    for j in range(0, len(verification_dataset)):
        datum = verification_dataset.iloc[j,:].values #1:21
        verification_input = datum.reshape(1, -1)
        mcsp_pred = regr_mcsp.predict(verification_input)
        acs_pred = regr_acs.predict(verification_input)
        ga_pred = regr_ga.predict(verification_input)
        pso_pred = regr_pso.predict(verification_input)
        
        mcsp_preds_proc.append(mcsp_pred)
        acs_preds_proc.append(acs_pred)
        ga_preds_proc.append(ga_pred)
        pso_preds_proc.append(pso_pred)
        

        mcsp_val = mcsp_pred #+ gamma*np.std(np.asarray(mcsp_preds_proc))*selection_counts[0]#*math.sqrt(selection_counts[0])
        acs_val = acs_pred #+ gamma*np.std(np.asarray(acs_preds_proc))*selection_counts[1]#*math.sqrt(selection_counts[1])
        ga_val = ga_pred #+ gamma*np.std(np.asarray(ga_preds_proc))*selection_counts[2]#*math.sqrt(selection_counts[2])
        pso_val = pso_pred #+ gamma*np.std(np.asarray(pso_preds_proc))*selection_counts[3]#*math.sqrt(selection_counts[3])
        #UCB
        preds = np.asarray([mcsp_val, acs_val, ga_val, pso_val])

        selected_algo_idx = np.argmin(preds)#np.argmin(preds)
        
        #selection_counts[selected_algo_idx] += 1
        
        #print(selected_algo_idx)
        
        selected_algo_label = getLabel(selected_algo_idx)
        verification_labels.append(selected_algo_label)
        #TODO: getCost(selected_label, master_frame, index)
        index = verification_indices[j]
        selected_cost = getCost(selected_algo_label, master_frame, index)
        ground_truth_cost = master_frame["Ground Truth Cost"].iloc[index]

        rmse.append(rmseCalc([ground_truth_cost] ,[selected_cost]))
        #raw_cost, cost_diff = getCost(selected_algo_label, online_dataset.iloc[i])
        
        


        #selections.append((selected_algo_label, cost_diff))
        
    all_selections += verification_labels  
    return rmse, verification_labels, cumulative_cost_arr, oracles, all_selections






def getGroundTruthCost(frame):
    frame["Ground Truth Cost"] = np.nan
    
    for i in range(0, len(frame)):
        label = frame["Final Label"].iloc[i]
        cost = getCost(label, master_frame, i)
        #cost = frame[label + " Time"].iloc[i] + frame[label + " Memory"].iloc[i]
        frame["Ground Truth Cost"].iloc[i] = cost
    
    #frame["Ground Truth Cost"] = StandardScaler().fit_transform(frame["Ground Truth Cost"].values)
    return frame

def createBanditsFrame(frame):
    bandits_columns = ["Time Slice", "Abstract Services", "Candidate Services", "Solution Quality", "Constraints", "RT Centroid", "Thrpt Centroid"] 
    
    numbered_columns = ['Mean RT ', 'Variation Coefficient RT ', 'Skew RT ', 'Worst RT ', 'Worst Thrpt ']
    
    max_ab_services = 40
    
    indicator_columns_added = []
    
    for jk in range(0, len(numbered_columns)):
        for ik in range(0, max_ab_services):
            
                key = numbered_columns[jk] + str(ik)
                
                indicator_key = "Indicator " + key
                
                indicator_columns_added.append(indicator_key)
    
    #numbered_columns = [numbered_columns] + [indicator_columns_added]
    bandits_columns += indicator_columns_added 
    
    for j in range(0, len(numbered_columns)):
        for i in range(0, max_ab_services):
            key = numbered_columns[j] + str(i)
            bandits_columns.append(key)
            
                       
    #bandits_columns.append("Final Label")
    
    return_frame = frame[bandits_columns]
    master_columns = frame.columns.tolist()
    
    remove_columns = ["Final Label", "Intermediate Label", "Pure Solution Algorithms", "Pure Time Label", "Pure Memory Label", "Weights"]
    
    
    val_arr = ["Best Path", "Best Path Constraints"]
    algos = ["MCSP","ACS 0", "GA 0", "PSO 0"]
    
    for value in val_arr:
        for algor in algos:
            key = algor + str(" ") + value
            remove_columns.append(key)
    
    
    columns_keep = [x for x in master_columns if x not in remove_columns]

    
    
    master_frame_columns = columns_keep#bandits_columns
    #master_frame_columns.append("Ground Truth Cost")
    
    master_frame_temp = frame[master_frame_columns]
    
    #Removing this because random forests do not need feature scaling
    #frame.loc[:,master_frame_columns] = StandardScaler().fit_transform(master_frame_temp.iloc[:,:])
    
    return return_frame, master_frame

if __name__ == "__main__":

    req_algo_id = int(sys.argv[1])#0
    param_arr = [float(sys.argv[2])]
    #Declare regressors
    algorithms = ["MCSP", "ACS", "GA", "PSO"]
    regressors = {}
    for i in range(0, len(algorithms)):
        regressors[algorithms[i]] = sklearn.ensemble.RandomForestRegressor(n_estimators = 200, max_depth=20, n_jobs=4, random_state=19950807)
    #Read data
    master_frame  = pd.read_pickle("thresholded_master_frame.pkl")
    master_frame = master_frame.fillna(0)
    #add final label costs
    
    master_frame.index = np.arange(0, len(master_frame))
    
    _, master_frame = createBanditsFrame(master_frame)

    master_frame = getGroundTruthCost(master_frame)
    
    
    
    #Make train and test sets
    #Make base, online and verification sets
    #data = train_test_split(df_bandits_base, test_size = 0.9)
    #train_data = data[0]
    #train_data.to_pickle("bandits_train_data_selected_new.pkl")
    train_data = pd.read_pickle("bandits_train_data_selected_new.pkl")



    
    #Training base regressors
    #0th is index
    train_indices = train_data.index#['index']
    #train_data = train_data.drop(["index"], axis=1)
    train_input = train_data.iloc[:,:].values
    cumu_cost_arr = []
    
    keys = regressors.keys()
    #for each regressor for an algorithm, train them
    
    for key in keys:
        current_regressor = regressors[key]
        
        cost_arr = []
        
        for i in range(0, len(train_indices)):
            selected_label = key
            index = train_indices[i]
            cost = getCost(selected_label, master_frame, index)
            cost_arr.append(cost)
        regressors[key] = current_regressor.fit(train_input, cost_arr)
        cumu_cost_arr.append(cost_arr)
        
        
        
        
        
    N = 3
    cumu_cost_arr_csc = [[] for i in range(0,N)]
    bagged_oracles = [[] for i in range(0,N)]   
    for policy_id in range(0, N):
        for key in keys:
            current_regressor = regressors[key]
            
            cost_arr = []
            
            for i in range(0, len(train_indices)):
                selected_label = key
                index = train_indices[i]
                cost = getCost(selected_label, master_frame, index)
                cost_arr.append(cost)
            regressors[key] = current_regressor.fit(train_input, cost_arr)
            cumu_cost_arr_csc[policy_id].append(cost_arr)
        bagged_oracles[policy_id] = regressors.copy()

    #regr_mcsp_preds = regr_mcsp.predict(test_dataset)
    #regr_mcsp.score(test_dataset, test_data_costs)
    
    
    #Make online and verification regressors
    #test_dataset = data[1]
    #df_new = test_dataset.sample(frac=1).reset_index(drop=False)
    #df_new = train_test_split(test_dataset, test_size = 0.1)

    #online_dataset = df_new[0]
    #verification_dataset = df_new[1]
    #online_dataset.to_pickle("bandits_online_dataset_selected_new.pkl")
    #verification_dataset.read_pickle("bandits_verification_dataset_selected_new.pkl")
    online_dataset = pd.read_pickle("bandits_online_dataset_selected_new.pkl")
    verification_dataset = pd.read_pickle("bandits_verification_dataset_selected_new.pkl")
    

    
    try:
        import cPickle as pickle
    except ImportError:  # Python 3.x
        import pickle


    
    online_indices = online_dataset.index#['index']
    verification_indices = verification_dataset.index#['index']
    

    
    oracles = regressors
    
    
    if req_algo_id == 1001:
        
        all_selections = []
        selections = []
        results = []
        rmse = [] 
        
        prediction_times = []
        retraining_times = []
        
        cumulative_cost_arr = []
        cumu_cost_arr_retraining = []
        
        cumu_cost = 0.0

        count = len(train_data) - 1 
        
        full_dataset = pd.concat([train_data, online_dataset], axis=0)
        online_indices = online_dataset.index.tolist()

        train_data_input = train_data.values
        for key in keys:
            current_regressor = regressors[key]
            
            cost_arr = []
            
            for i in range(0, len(train_data.index.tolist())):
                selected_label = key
                index = train_indices[i]
                cost = getCost(selected_label, master_frame, index)
                cost_arr.append(cost)
            regressors[key] = current_regressor.fit(train_data_input, cost_arr)
            cumu_cost_arr_retraining.append(cost_arr)
        oracles = regressors
        np_algorithms = np.asarray(algorithms)

        for key in keys:
            current_regressor = regressors[key]
            key_id = np.where(key==np_algorithms)[0][0]
            cost_arr = []
            
            for i in range(0, len(online_indices)):
                selected_label = key
                index = online_indices[i]
                cost = getCost(selected_label, master_frame, index)
                cost_arr.append(cost)
            cumu_cost_arr_retraining[key_id].extend(cost_arr)
        
        retraining_times = []
        
        
        for i in range( len(train_data), len(full_dataset),10):
            count = count + 1
    
            
            test = full_dataset.iloc[:(i+1)]
            
            
            # if i%10 == 0:
                #Retrain every 10 samples
            if i != len(train_data):
                print("i is:", i)
                
                x_arr_algo = full_dataset[:(i+1)]
                
                
                for key in keys:
                    current_regressor = regressors[key]
                
                    key_id = np.where(key==np_algorithms)[0][0]
                    y_cost_curr = cumu_cost_arr_retraining[key_id][:(i+1)]
                    
                    retraining_start = time.time()
                    oracles[key] = sklearn.ensemble.RandomForestRegressor(n_estimators = 200, max_depth=20, n_jobs=4, random_state=19950807)
                    oracles[key] = oracles[key].fit(x_arr_algo, y_cost_curr)
                    retraining_end = time.time()
                    retraining_times.append(retraining_end - retraining_start)
                    
        with open("e_greedy_retraining_times.txt", "w") as output:
            output.write(str(retraining_times))
        
        
        
    if req_algo_id == 999:
        #no learning
        print("Greedy no learning executing.")
        rmse_greedy_nl, greedy_nl_verification, cumulative_cost_iter, oracles_new, greedy_nl_labels = np.asarray(greedy(train_data,oracles, online_dataset, 0.0, verification_dataset, algorithms, master_frame, online_indices, verification_indices, cumu_cost_arr, 0))
            
        data = {}
        data["rmse"] = rmse_greedy_nl
        data["verification"] = greedy_nl_verification
        data["cumulative_cost_iter"] = cumulative_cost_iter
        data["trained_oracles"] = oracles_new
        data["labels"] = greedy_nl_labels
        with open('nl_greedy_data_selected_corrected.p', 'wb') as fp:
            pickle.dump(data, fp, protocol=pickle.HIGHEST_PROTOCOL)
    
    
    if req_algo_id == 888:
        print("Greedy full info executing.")
        greedy_fi_dataset = pd.concat([train_data, online_dataset], axis=0)
        regressors_fi = regressors
        
        greedy_fi_indices = greedy_fi_dataset.index#['index']
        greedy_fi_input = greedy_fi_dataset.iloc[:,:].values
        cumu_cost_arr_greedy_fi = []
        
        for key in keys:
            current_regressor = regressors_fi[key]
            
            cost_arr = []
            
            for i in range(0, len(greedy_fi_indices)):
                selected_label = key
                index = greedy_fi_indices[i]
                cost = getCost(selected_label, master_frame, index)
                cost_arr.append(cost)
            regressors_fi[key] = current_regressor.fit(greedy_fi_input, cost_arr)
            cumu_cost_arr_greedy_fi.append(cost_arr)
            
        verification_indices = verification_dataset.index.tolist()
        verification_labels = []
        rmse = []
        all_selections = []
        
        #verification_cossts = []
        for j in range(0, len(verification_dataset)):
            datum = verification_dataset.iloc[j].values
            verification_input = datum.reshape(1, -1)
            
            preds = {}
            for key in regressors_fi.keys():
                preds[key] = regressors_fi[key].predict(verification_input)
            
            
            #epsilon greedy
            preds = np.asarray([list(a_pred) for a_pred in preds.values()])
            
    
            selected_algo_idx = np.argmin(preds)
                
            selected_algo_label = algorithms[selected_algo_idx]
            verification_labels.append(selected_algo_label)
            
            index = verification_indices[j]
            selected_cost = getCost(selected_algo_label, master_frame, index)
            ground_truth_cost = master_frame["Ground Truth Cost"].iloc[index]
            rmse.append(rmseCalc([ground_truth_cost], [selected_cost]))
            
        all_selections += verification_labels
        
    
                
        data = {}
        data["rmse"] = rmse#rmse_greedy_fi
        data["verification"] = verification_labels#greedy_fi_verification
        #data["cumulative_cost_iter"] = cumulative_cost_arr#cumulative_cost_iter
        data["trained_oracles"] = oracles#oracles_new
        data["labels"] = all_selections#greedy_fi_labels
        with open('fi_greedy_data_epsilon_selected_corrected.p', 'wb') as fp:
            pickle.dump(data, fp, protocol=pickle.HIGHEST_PROTOCOL)
    
   
    if req_algo_id == 3:
        gamma_arr = param_arr
        param_arr = gamma_arr
        print("UCB running.")
        #rmse_epsilon = [[] for i in epsilon_arr]
        rmse_gamma = [[] for i in gamma_arr]
        
        #cumulative_costs_epsilon = [[] for i in epsilon_arr]
        cumulative_costs_gamma = [[] for i in gamma_arr]
        
        #oracles_arr = [[] for i in epsilon_arr]
        oracles_arr_ucb = [[] for i in gamma_arr]
        
        #eps_greedy_labels_arr = [[] for i in epsilon_arr]
        ucb_labels_arr = [[] for i in gamma_arr]
        for i in range(0,len(param_arr)):#in range(0, 1, 0.1):
            gamma = param_arr[i]
            
            
            
            rmse_ucb_gamma, ucb_verification, cumulative_cost_iter_ucb, oracles_new_ucb, ucb_labels = np.asarray(ucb(train_data, oracles, online_dataset, gamma, verification_dataset, algorithms, master_frame, online_indices, verification_indices, cumu_cost_arr))
            
            
            data = {}
            data["rmse"] = rmse_ucb_gamma
            data["verification"] = ucb_verification
            data["cumulative_cost_iter"] = cumulative_cost_iter_ucb
            data["trained_oracles"] = oracles_new_ucb
            data["labels"] = ucb_labels
            with open('ucb_data_gamma_'+str(gamma)+'_selected_corrected_SOFT_CORRECT_WEIGHTED.p', 'wb') as fp:
                pickle.dump(data, fp, protocol=pickle.HIGHEST_PROTOCOL)
            
            ucb_labels_arr[i].append(ucb_labels)
            
            
            #nrmse_ucb = rmseCalc(master_frame["Ground Truth Cost"].iloc[online_indices.tolist()+verification_indices.tolist()].values, rmse_ucb_gamma)
    
    
            #rmse_gamma[i].append(nrmse_ucb)
            
            #cumulative_costs_gamma[i].append(cumulative_cost_iter_ucb)
            
            #oracles_arr_ucb[i].append(oracles_new_ucb)
            
            
            cumu_cost_arr = []
        
            keys = regressors.keys()
            #for each regressor for an algorithm, train them
            
            for key in keys:
                current_regressor = regressors[key]
                
                cost_arr = []
                
                for i in range(0, len(train_indices)):
                    selected_label = key
                    index = train_indices[i]
                    cost = getCost(selected_label, master_frame, index)
                    cost_arr.append(cost)
                regressors[key] = current_regressor.fit(train_input, cost_arr)
                cumu_cost_arr.append(cost_arr)
    
	#######################################################

	#This is the epsilon greedy code. 
    elif req_algo_id == 4:
        epsilon_arr = param_arr 
        print("Epsilon Greedy running.", epsilon_arr)
        rmse_epsilon = [[] for i in epsilon_arr]
        #rmse_gamma = [[] for i in epsilon_arr]
        
        cumulative_costs_epsilon = [[] for i in epsilon_arr]
        #cumulative_costs_gamma = [[] for i in epsilon_arr]
        
        oracles_arr = [[] for i in epsilon_arr]
        #oracles_arr_ucb = [[] for i in epsilon_arr]
        
        eps_greedy_labels_arr = [[] for i in epsilon_arr]
        #ucb_labels_arr = [[] for i in epsilon_arr]        
        for i in range(0,len(epsilon_arr)):#in range(0, 1, 0.1):
            epsilon = epsilon_arr[i]
            
            rmse_e_greedy, e_greedy_verification, cumulative_cost_iter, oracles_new, eps_greedy_labels = np.asarray(e_greedy(train_data,oracles, online_dataset, epsilon, verification_dataset, algorithms, master_frame, online_indices, verification_indices, cumu_cost_arr))
            
            data = {}
            data["rmse"] = rmse_e_greedy
            data["verification"] = e_greedy_verification
            data["cumulative_cost_iter"] = cumulative_cost_iter
            data["trained_oracles"] = oracles_new
            data["labels"] = eps_greedy_labels
            with open('e_greedy_data_epsilon_'+str(epsilon)+'_selected_corrected_SOFT_CORRECT_WEIGHTED.p', 'wb') as fp:#+'_selected_corrected_NEW.p'
                pickle.dump(data, fp, protocol=pickle.HIGHEST_PROTOCOL)
            
            eps_greedy_labels_arr[i].append(eps_greedy_labels)
            
           
            
            cumu_cost_arr = []
        
            keys = regressors.keys()
            #for each regressor for an algorithm, train them
            
            for key in keys:
                current_regressor = regressors[key]
                
                cost_arr = []
                
                for i in range(0, len(train_indices)):
                    selected_label = key
                    index = train_indices[i]
                    cost = getCost(selected_label, master_frame, index)
                    cost_arr.append(cost)
                regressors[key] = current_regressor.fit(train_input, cost_arr)
                cumu_cost_arr.append(cost_arr)
