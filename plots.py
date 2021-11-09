import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import f1_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns



from matplotlib import rcParams
rcParams['font.family'] = 'Times New Roman'


master_frame = pd.read_pickle("thresholded_master_frame.pkl")

def getCost(selected_label, master_frame, index, max_penalty=3.0):
    min_time = 9999999
    min_mem = 9999999
    penalty = 0.0
    
    frame_max_util = max(master_frame["MCSP Best Path Utility"].max(), master_frame["ACS 0 Best Path Utility"].max(), master_frame["GA 0 Best Path Utility"].max(), master_frame["PSO 0 Best Path Utility"].max())
    frame_min_util = min(master_frame["MCSP Best Path Utility"].min(), master_frame["ACS 0 Best Path Utility"].min(), master_frame["GA 0 Best Path Utility"].min(), master_frame["PSO 0 Best Path Utility"].min())
    
    frame_max_time = max(master_frame["MCSP Time"].max(), master_frame["ACS 0 Time"].max(), master_frame["GA 0 Time"].max(), master_frame["PSO 0 Time"].max())
    frame_min_time = min(master_frame["MCSP Time"].min(), master_frame["ACS 0 Time"].min(), master_frame["GA 0 Time"].min(), master_frame["PSO 0 Time"].min())
    
    frame_max_mem = max(master_frame["MCSP Memory"].max(), master_frame["ACS 0 Memory"].max(), master_frame["GA 0 Memory"].max(), master_frame["PSO 0 Memory"].max())
    frame_min_mem = min(master_frame["MCSP Memory"].min(), master_frame["ACS 0 Memory"].min(), master_frame["GA 0 Memory"].min(), master_frame["PSO 0 Memory"].min())
    # max_penalty = 1.0
    if selected_label == master_frame["Final Label"].iloc[index]:
        min_time = master_frame["Final Label Time"].iloc[index]
        min_mem = master_frame["Final Label Memory"].iloc[index]
    if selected_label=="MCSP":
        min_time = master_frame["MCSP Time"].iloc[index]
        min_mem = master_frame["MCSP Memory"].iloc[index]
    elif selected_label=="GA":  
        min_time = master_frame["GA 0 Time"].iloc[index]
        min_mem = master_frame["GA 0 Memory"].iloc[index]
    elif selected_label=="ACS":
        min_time = master_frame["ACS 0 Time"].iloc[index]
        min_mem = master_frame["ACS 0 Memory"].iloc[index]
    elif selected_label=="PSO":
        min_time = master_frame["PSO 0 Time"].iloc[index]
        min_mem = master_frame["PSO 0 Memory"].iloc[index]
     
    if selected_label in ["GA", "ACS", "PSO"]:
        selected_label = selected_label + " 0"
     
    if selected_label not in master_frame["Pure Solution Algorithms"].iloc[index]:
        penalty =3.0
    util_norm = penalty/max_penalty
    time_norm = (min_time - frame_min_time)/(frame_max_time - frame_min_time)
    mem_norm = (min_mem - frame_min_mem)/(frame_max_mem - frame_min_mem)     
    cost = util_norm + time_norm + mem_norm 
    #Uncomment the next line when generating sbs_vbs_time_and_memory_bandits_cost_curve.pdf
    #cost = time_norm + mem_norm
    
    return cost

def createBanditsFrame(frame):
    bandits_columns = ["Time Slice", "Abstract Services", "Candidate Services", "Solution Quality", "Constraints", "RT Centroid", "Thrpt Centroid"] # "Alloted Time", "Alloted Memory",
    
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

    
    
    master_frame_columns = columns_keep
    
    master_frame_temp = frame[master_frame_columns]
    
    
    return return_frame, master_frame


def getGroundTruthCost(frame):
    frame["Ground Truth Cost"] = np.nan
    
    for i in range(0, len(frame)):
        label = frame["Intermediate Label"].iloc[i]
        cost = frame[label + " Time"].iloc[i] + frame[label + " Memory"].iloc[i]
        frame["Ground Truth Cost"].iloc[i] = cost
    
    return frame


# master_frame = pd.read_pickle("dataframes/PSO_TURU_0_CHANGED_CONSTRAINTS_CHANGED_simplified_3_label_min_time_master_frame_64.pkl")
#pd.read_pickle("thresholded_master_frame.pkl")#
master_frame = master_frame.fillna(0)

    
master_frame.index = np.arange(0, len(master_frame))
    
_, master_frame = createBanditsFrame(master_frame)

master_frame = getGroundTruthCost(master_frame)
    
#train_data = pd.read_pickle("bandits_train_data_selected_new.pkl")#("bandits_train_data.pkl")
online_dataset = pd.read_pickle("bandits_online_dataset_selected_new.pkl")#("bandits_online_dataset.pkl")
verification_dataset = pd.read_pickle("bandits_verification_dataset_selected_new.pkl")#("bandits_verification_dataset.pkl")







with open('nl_greedy_data_selected_corrected.p', 'rb') as fp:
    data = pickle.load(fp)

rmse_greedy_nl = data["rmse"] 
greedy_nl_verification = data["verification"]
cumulative_cost_iter_nl = data["cumulative_cost_iter"]
oracles_new = data["trained_oracles"] 
greedy_nl_labels = data["labels"]

with open('fi_greedy_data_epsilon_selected_corrected.p', 'rb') as fp:
    data = pickle.load(fp)

rmse_greedy_fi = data["rmse"]
greedy_fi_verification = data["verification"]
#data["cumulative_cost_iter"] = cumulative_cost_arr#cumulative_cost_iter
greedy_fi_oracles = data["trained_oracles"]#oracles_new
greedy_fi_labels = data["labels"]



online_indices = online_dataset.index
verification_indices = verification_dataset.index


e_decay_data = {}
ucb_data = {}
e_greedy_data = {}
greedy_data = {}


rmse_ucb = []
rmse_e_greedy = []

cumulative_costs_ucb = []
cumulative_costs_e_greedy = []

cumulative_costs_greedy = []

epsilon_arr = [0.3]
gamma_arr = [0.05]
    
for epsilon in gamma_arr:
    with open('ucb_data_gamma_'+str(epsilon)+'_selected_corrected.p', 'rb') as handle:
        data = pickle.load(handle)
    rmse_ucb_gamma = data["rmse"] 
    ucb_verification = data["verification"] 
    cumulative_cost_iter_ucb = data["cumulative_cost_iter"] 
    oracles_new_ucb = data["trained_oracles"] 
    ucb_labels = data["labels"] 
    
    ucb_data[str(epsilon)] = data
    rmse_ucb.append(rmse_ucb_gamma)
    cumulative_costs_ucb.append(cumulative_cost_iter_ucb)
    

for epsilon in epsilon_arr:  
    if epsilon == 0.3:
        with open('e_greedy_data_epsilon_'+str(epsilon)+'_selected_corrected_NEW.p', 'rb') as handle:
            data = pickle.load(handle)
    else:
        with open('e_greedy_data_epsilon_'+str(epsilon)+'_selected_corrected.p', 'rb') as handle:
            data = pickle.load(handle)
    rmse_e_greedy = data["rmse"] 
    e_greedy_verification = data["verification"]
    cumulative_cost_iter = data["cumulative_cost_iter"] 
    oracles_new = data["trained_oracles"] 
    eps_greedy_labels = data["labels"] 
    
    e_greedy_data[str(epsilon)] = data
    
    rmse_e_greedy.append(rmse_e_greedy)
    cumulative_costs_e_greedy.append(cumulative_cost_iter)



with open('e_greedy_data_epsilon_'+str(0.0)+'_selected_corrected.p', 'rb') as handle:
    greedy_data = pickle.load(handle)
rmse_greedy = greedy_data["rmse"] 
greedy_verification = greedy_data["verification"]
greedy_cumulative_cost_iter = greedy_data["cumulative_cost_iter"] 
greedy_oracles_new = greedy_data["trained_oracles"] 
greedy_labels = greedy_data["labels"] 


rmse_greedy.append(rmse_greedy)
cumulative_costs_greedy.append(greedy_cumulative_cost_iter)




#######################################################################################
#Cost boxplots for greedy fi, greedy nl and greedy learning
#Plots on verification dataset
total_indices = online_dataset.index.tolist() + verification_dataset.index.tolist()

cost_fi = []
cost_nl = []
cost_greedy = []

time_fi = []
time_nl = []
time_greedy = []


mem_fi = []
mem_nl = []
mem_greedy = []

incorr_fi = 0
incorr_nl = 0
incorr_greedy = 0 

sbs_cost = []
vbs_cost = []

greedy_fi_labels
greedy_nl_labels_ver = greedy_nl_labels[-len(verification_indices):]
greedy_labels_ver = greedy_data["labels"][-len(verification_indices):]#e_greedy_data[str(0.0)]["labels"]

for i in range(0, len(verification_indices)):
    idx = verification_indices[i]
    fi_cost = getCost(greedy_fi_labels[i], master_frame, idx)
    nl_cost = getCost(greedy_nl_labels_ver[i], master_frame, idx)
    greedy_cost = getCost(greedy_labels_ver[i], master_frame, idx)
    
    cost_fi.append(round(fi_cost,2))
    cost_nl.append(round(nl_cost,2))
    cost_greedy.append(round(greedy_cost,2))
    
    solutions = master_frame["Pure Solution Algorithms"].iloc[idx]
    
    fi_label = greedy_fi_labels[i]
    nl_label = greedy_nl_labels_ver[i]
    greedy_label = greedy_labels_ver[i]
    
    if fi_label not in ["MCSP"]:
        fi_label = fi_label + " 0"
        
    if nl_label not in ["MCSP"]:
        nl_label = nl_label + " 0"
        
    if greedy_label not in ["MCSP"]:
        greedy_label = greedy_label + " 0"
    
    if fi_label not in solutions:
        incorr_fi += 1

    if nl_label not in solutions:
        incorr_nl += 1
    
    if greedy_label not in solutions:
        incorr_greedy += 1
    
    
    
    curr_time_fi = master_frame[fi_label + " Time"].iloc[idx]
    curr_time_nl = master_frame[nl_label + " Time"].iloc[idx]
    curr_time_greedy = master_frame[greedy_label + " Time"].iloc[idx]
    
    time_fi.append(round(curr_time_fi,2))
    time_nl.append(round(curr_time_nl,2))
    time_greedy.append(round(curr_time_greedy,2))
    
    curr_mem_fi = master_frame[fi_label + " Memory"].iloc[idx]
    curr_mem_nl = master_frame[nl_label + " Memory"].iloc[idx]
    curr_mem_greedy = master_frame[greedy_label + " Memory"].iloc[idx]
    
    mem_fi.append(round(curr_mem_fi,2))
    mem_nl.append(round(curr_mem_nl,2))
    mem_greedy.append(round(curr_mem_greedy,2))
    
    
    sbs_cost.append(getCost("GA 0", master_frame, idx))
    vbs_cost.append(getCost(master_frame["Final Label"].iloc[idx], master_frame, idx))
    

print(greedy_fi_labels == greedy_nl_labels_ver)


import sklearn.metrics as met 

nl_error =  (np.asarray(cost_nl) - np.asarray(sbs_cost))/(np.asarray(vbs_cost)-np.asarray(sbs_cost))
fi_error = (np.asarray(cost_fi) - np.asarray(sbs_cost))/(np.asarray(vbs_cost)-np.asarray(sbs_cost))

nl_error = np.nan_to_num(nl_error).reshape(1,-1)
fi_error = np.nan_to_num(fi_error).reshape(1,-1)


# cost_df = pd.DataFrame(data=[nl_error, fi_error]).T#(data = [cost_nl, cost_fi]).T
# sns.violinplot(data=cost_df)
# sns.swarmplot(data=cost_df,color="black")


sns.set_style("darkgrid")
h = np.arange(0, len(cost_nl))
plt.figure(figsize=(12,7))
#plt.title("Cost Distribution", fontsize=20, y=1.05)
plt.xticks(fontsize=30)

# plt.xticks([0, 1,2],["Greedy NL","Greedy", "Greedy FI"], fontsize=50)
plt.ylabel("Cost",fontsize=30)
plt.xlabel("Composition Tasks",fontsize=30)
plt.xticks()
plt.yticks(fontsize=30)
#plt.ylim(0.0, 5000)
plt.scatter(y=cost_nl, x=h, c='blue',marker=".",s=500,label="greedy-nl")
plt.scatter(y=cost_greedy, x=h, c='purple', marker='x',s=500 ,label="greedy")
plt.scatter(y=cost_fi, x=h, c="green", marker='*',s=500 ,label="greedy-fi")

plt.legend( fontsize=30,loc=1, bbox_to_anchor=(1.0,0.9))
#plt.xticks([0, 1,2],["Greedy NL","Greedy", "Greedy FI"], fontsize=50)
plt.savefig("online_learning_ablation.pdf", bbox_inches='tight')
plt.show()

ind_col = np.arange(0, len(cost_nl))
dataset_temp = pd.DataFrame(data=[ind_col,cost_nl, cost_greedy, cost_fi]).T
dataset_temp.to_csv("greedy_ablation_dataset.csv",sep=",",header=None, index=None)






print("Median Cost FI:",np.median(cost_fi))
print("Median Cost NL:",np.median(cost_nl))
print("Median Cost Greedy:",np.median(cost_greedy))

print("Mean Cost FI:",np.mean(cost_fi))
print("Mean Cost NL:",np.mean(cost_nl))
print("Mean Cost Greedy:",np.mean(cost_greedy))

print("Median Time FI:",np.median(time_fi))
print("Median Time NL:",np.median(time_nl))
print("Median Time Greedy:",np.median(time_greedy))


print("Median Memory FI:",np.median(mem_fi))
print("Median Memory NL:",np.median(mem_nl))
print("Median Memory Greedy:",np.median(mem_greedy))


print("Incorrect Selections FI:",incorr_fi)
print("Incorrect Selections NL:",incorr_nl)
print("Incorrect Selections Greedy:",incorr_greedy)


print("Cumu Cost FI:",np.sum(cost_fi))
print("Cumu Cost NL:",np.sum(cost_nl))
print("Cumu Cost Greedy:",np.sum(cost_greedy))


true_verifications = master_frame["Final Label"].iloc[verification_dataset.index]
print("FI: ", accuracy_score(true_verifications, greedy_fi_labels))
print("NL: ", accuracy_score(true_verifications, greedy_nl_labels_ver))
print("Greedy: ", accuracy_score(true_verifications, greedy_labels_ver))



import math

print("RMSE FI:",math.sqrt(met.mean_squared_error(vbs_cost, cost_fi)))
print("RMSE NL:",math.sqrt(met.mean_squared_error(vbs_cost, cost_nl)))
print("RMSE Greedy:",math.sqrt(met.mean_squared_error(vbs_cost,cost_greedy)))
print("RMSE SBS:",math.sqrt(met.mean_squared_error(vbs_cost,sbs_cost)))


(1-(np.median(cost_fi)/np.median(cost_nl) )  )*100
(1-(np.median(cost_greedy)/np.median(cost_nl) )  )*100

    
#######################################################################################
    


sbs_cost = []
vbs_cost = []

sbs_cumu_cost = []
vbs_cumu_cost = []

cover_cost = []
ucb_cost = []
e_greedy_cost = []
greedy_cost = []

cumu_cost_sbs = 0.0
cumu_cost_vbs = 0.0


ucb_curr_cost = 0.0
e_greedy_curr_cost = 0.0
greedy_curr_cost = 0.0

ucb_time = []
ucb_mem = []

e_greedy_time = []
e_greedy_mem = []

greedy_time = []
greedy_mem = []



ucb_incorrect_selections = np.zeros(shape=(len(gamma_arr),len(verification_indices)))
e_greedy_incorrect_selections = np.zeros(shape=(len(epsilon_arr),len(verification_indices)))
greedy_incorrect_selections = np.zeros(shape=(len(verification_indices)))


#for ucb

for i in range(0, len(gamma_arr)):
    ucb_labels_ver = ucb_data[str(gamma_arr[i])]["labels"][-len(verification_indices):]#[:len(online_indices)]#

    for j in range(0 , len(verification_indices)):
        idx = verification_indices[j]
        ucb_label = ucb_labels_ver[j]
        if ucb_label != "MCSP":
            ucb_label = ucb_label + " 0"
        if ucb_label not in master_frame["Pure Solution Algorithms"].iloc[idx]:
            ucb_incorrect_selections[i][j] = 1.0
    print("Incorrect selections for gamma ",gamma_arr[i],":",len(np.where(ucb_incorrect_selections[i]==1.0)[0]))
    
    

    
    
for i in range(0, len(epsilon_arr)):
    e_greedy_labels_ver = e_greedy_data[str(epsilon_arr[i])]["labels"][-len(verification_indices):]
    for j in range(0 , len(verification_indices)):
        idx = verification_indices[j]
        e_greedy_label = e_greedy_labels_ver[j]
        if e_greedy_label != "MCSP":
            e_greedy_label = e_greedy_label + " 0"
        if e_greedy_label not in master_frame["Pure Solution Algorithms"].iloc[idx]:
            e_greedy_incorrect_selections[i][j] = 1.0
    print("Incorrect selections for epsilon ",epsilon_arr[i],":",len(np.where(e_greedy_incorrect_selections[i]==1.0)[0]))

#greedy, epsilon 0.0
greedy_labels_ver = greedy_data["labels"][-len(verification_indices):]
for j in range(0, len(verification_indices)):
    idx = verification_indices[j]
    greedy_label = greedy_labels_ver[j]
    if greedy_label != "MCSP":
        greedy_label = greedy_label + " 0"
    if greedy_label not in master_frame["Pure Solution Algorithms"].iloc[idx]:
        greedy_incorrect_selections[j] = 1.0
print("Incorrect selections for greedy :",len(np.where(greedy_incorrect_selections==1.0)[0]))


vbs_time = []
vbs_mem = []

sbs_time = master_frame["MCSP Time"].iloc[total_indices]
sbs_mem = master_frame["MCSP Memory"].iloc[total_indices]

for i in range(0, len(total_indices)):
    idx = total_indices[i]
    curr_sbs_cost = getCost("MCSP", master_frame, idx)
    cumu_cost_sbs += curr_sbs_cost
    
    sbs_cumu_cost.append(cumu_cost_sbs)
    sbs_cost.append(curr_sbs_cost)
    
    curr_vbs_cost = getCost(master_frame["Final Label"].iloc[idx], master_frame, idx)
    cumu_cost_vbs += curr_vbs_cost
    vbs_cumu_cost.append(cumu_cost_vbs)
    vbs_cost.append(curr_vbs_cost)
    
    #cover_curr_cost = getCost(cover_labels[idx], master_frame, idx)
    ucb_label = ucb_data["0.05"]["labels"][i]
    e_greedy_label = e_greedy_data["0.3"]["labels"][i]
    greedy_label = greedy_data["labels"][i]#e_greedy_data["0.0"]["labels"][i]
    
    ucb_curr_cost = getCost(ucb_label, master_frame, idx)
    e_greedy_curr_cost = getCost(e_greedy_label, master_frame, idx)
    greedy_curr_cost = getCost(greedy_label, master_frame, idx)
    
    #cover_cost.append(cover_curr_cost)
    ucb_cost.append(ucb_curr_cost)
    e_greedy_cost.append(e_greedy_curr_cost)
    greedy_cost.append(greedy_curr_cost)
    
    
    if ucb_label != "MCSP":
        ucb_label = ucb_label + " 0"
        
    if e_greedy_label != "MCSP":
        e_greedy_label = e_greedy_label + " 0"
        
    if greedy_label != "MCSP":
        greedy_label = greedy_label + " 0"

     
    
    
    ucb_time.append(master_frame[ucb_label + " Time"].iloc[idx])
    ucb_mem.append(master_frame[ucb_label + " Memory"].iloc[idx])
    
    e_greedy_time.append(master_frame[e_greedy_label + " Time"].iloc[idx])
    e_greedy_mem.append(master_frame[e_greedy_label + " Memory"].iloc[idx])
    
    greedy_time.append(master_frame[greedy_label + " Time"].iloc[idx])
    greedy_mem.append(master_frame[greedy_label + " Memory"].iloc[idx])
    
    vbs_label = master_frame["Final Label"].iloc[idx]
    
    if vbs_label!="MCSP":
        vbs_label+=" 0"
    
    vbs_time.append(master_frame[vbs_label + " Time"].iloc[idx])
    vbs_mem.append(master_frame[vbs_label + " Memory"].iloc[idx])
    


############################################################################################################


ucb_cost_mod = {}
ucb_cost_mod["0.05"] = []

cumu_cost_mod = {}
cumu_cost_mod["0.3"] = []

cumulative_mod_costs_greedy = []

curr_ucb_0 = 0.0
curr_ucb_1 = 0.0

curr_e_greedy_0 = 0.0
curr_e_greedy_1 = 0.0

curr_greedy = 0.0

for i in range(0, len(online_indices)):
    idx = online_indices[i]
    
    ucb_label_1 = ucb_data["0.05"]["labels"][i]
    
    e_greedy_label_1 = e_greedy_data["0.3"]["labels"][i]
    
    
    greedy_label = greedy_data["labels"][i]
    

        
    curr_ucb_0  += getCost(ucb_label_0, master_frame, idx)
    curr_ucb_1  += getCost(ucb_label_1, master_frame, idx)
    
    curr_e_greedy_0  += getCost(e_greedy_label_0, master_frame, idx)
    curr_e_greedy_1  += getCost(e_greedy_label_1, master_frame, idx)

    curr_greedy += getCost(greedy_label, master_frame, idx)

    ucb_cost_mod["0.05"].append(curr_ucb_1)
    

    cumu_cost_mod["0.3"].append(curr_e_greedy_1)
    
    cumulative_mod_costs_greedy.append(curr_greedy)



sns.set_style("darkgrid")

plt.figure(figsize=(10,10))
plt.xticks(fontsize=25,rotation=-45)
plt.yticks(fontsize=25,rotation=45)

h = np.arange(1,len(cumulative_mod_costs_greedy)+1)

plt.plot(h, ucb_cost_mod[str(0.05)]/h, label="UCB, "+u'γ'+" : "+str(0.05),linewidth=3)

plt.plot(h, cumu_cost_mod[str(0.3)]/h, label="e-greedy, "+u'ε'+" : "+str(0.3),linewidth=3)

plt.plot(h, cumulative_mod_costs_greedy/h, label="greedy",linewidth=3)
plt.plot(h, sbs_cumu_cost[0:len(online_indices)]/h, label="sbs",linewidth=3)
plt.plot(h, vbs_cumu_cost[0:len(online_indices)]/h, label="vbs",linewidth=3)


plt.ylabel("Time + Memory", fontsize = 25)
plt.xlabel("Composition Tasks", fontsize = 25)
plt.legend(fontsize=25,  loc='lower right',ncol=1)#loc = 1,ncol=2)bbox_to_anchor=(1.05, 1),
plt.savefig("sbs_vbs_time_and_memory_bandits_cost_curve.pdf",bbox_inches="tight")



ind_cols = np.arange(0, len(h))
temp_dataset = pd.DataFrame(data=[ind_cols, ucb_cost_mod[str(0.05)]/h, cumu_cost_mod[str(0.3)]/h, cumulative_mod_costs_greedy/h, sbs_cumu_cost[0:len(online_indices)]/h, vbs_cumu_cost[0:len(online_indices)]/h])
temp_dataset = temp_dataset.T
temp_dataset.columns = ["ind","ucb","e_greedy","greedy","sbs","vbs"]

temp_dataset.to_csv("sbs_vbs_time_memory_cost.txt",sep=" ",index=False, header=False)
##############################################################################################################

ucb_time = np.asarray(ucb_time[-len(verification_indices):])
ucb_mem = np.asarray(ucb_mem[-len(verification_indices):])
e_greedy_time = np.asarray(e_greedy_time[-len(verification_indices):])
e_greedy_mem = np.asarray(e_greedy_mem[-len(verification_indices):])
greedy_time = np.asarray(greedy_time[-len(verification_indices):])
greedy_mem = np.asarray(greedy_mem[-len(verification_indices):])

sbs_time_ver = np.asarray(sbs_time[-len(verification_indices):])
sbs_mem_ver = np.asarray(sbs_mem[-len(verification_indices):])

vbs_time_ver = np.asarray(vbs_time[-len(verification_indices):])
vbs_mem_ver =  np.asarray(vbs_mem[-len(verification_indices):])

print("UCB Time Mean:", np.mean(ucb_time))
print("UCB Memory Mean:", np.mean(ucb_mem))

print("E-greedy Time Mean:", np.mean(e_greedy_time))
print("E-greedy Memory Mean:", np.mean(e_greedy_mem))

print("Greedy Time Mean:", np.mean(greedy_time))
print("Greedy Memory Mean:", np.mean(greedy_mem))

print("SBS Time Mean:", np.mean(sbs_time_ver))
print("SBS Memory Mean:", np.mean(sbs_mem_ver))

print("VBS Time Mean:", np.mean(vbs_time_ver))
print("VBS Memory Mean:", np.mean(vbs_mem_ver))


print("UCB Time Median:", np.median(ucb_time))
print("UCB Memory Median:", np.median(ucb_mem))

print("E-greedy Time Median:", np.median(e_greedy_time))
print("E-greedy Memory Median:", np.median(e_greedy_mem))

print("Greedy Time Median:", np.median(greedy_time))
print("Greedy Memory Median:", np.median(greedy_mem))

print("SBS Time Median:", np.median(sbs_time_ver))
print("SBS Memory Median:", np.median(sbs_mem_ver))

print("VBS Time Median:", np.median(vbs_time_ver))
print("VBS Memory Median:", np.median(vbs_mem_ver))




