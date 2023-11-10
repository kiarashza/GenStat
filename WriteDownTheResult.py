
# this python code is wite to select and rwite the result for the best configuration
import csv
import math
def my_round(x):
    x = math.floor(x * 100.0) / 100.0
    return x

def file(path):
    with open(path, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        rows = list(csvreader)
    return rows, rows[0]

def write(rows,path):
    file = open(path, 'w+', newline ='')

# writing the data into the file
    with file:
        write = csv.writer(file)
        write.writerows(rows)


def find_row_with_largest_index_value_in_csv(Config_name, csv, Index):
    # initialize variables to store the maximum index value and corresponding row
    max_index_value = 2
    max_index_row = None

    # open the CSV file and read all rows into a list

    just_to_check = []
    # iterate over all rows
    for row in csv:
        # check if the substring S is present in the first column of the current row
        if Config_name in row[0]:
            just_to_check.append(row)
            # if yes, check if the index I of the current row is greater than the current maximum index value
            if row[1]!="" and row[1]!="Error" and  float(row[Index]) < max_index_value:
                # if yes, update the maximum index value and corresponding row
                max_index_value = float(row[Index])
                max_index_row = row

    # return the row with the largest index value
    return max_index_row


def reOrder(selected_list,writing_order, precision_index, concat_wit=" ± "):
    reordered_list = []

    for row in selected_list[1:]:
        for indx in precision_index:
                row[indx] = float(row[indx])*100
        for i in range(len(row[1:-1])):
                row[i+1] = my_round(float(row[i+1]))
        print()
    for row in selected_list:
        new_row = [row[0]]
        for inex in  writing_order:
            if len(inex)==2:
                new_row.append(str(row[inex[0]])+concat_wit+str(row[inex[1]]))
            else: new_row.append(row[inex[0]])
        reordered_list.append(new_row)
    return reordered_list

# fileName = "EvalutionMetricsAndTables/LDVAE_reach3_RandGNNwithSubgraphStructuralProperties.csv"
# # configs = ["riangulargrid___whole_data_Norm1d_lr","lobster___whole_data_Norm1d_lr","/grid___whole_data_Norm1d_lr_", "ogbgmolbbbp___whole_data_Norm1d_lr","DD___whole_data_Norm1d_lr","MDBBINARY___whole_data_Norm1d_lr_","MUTAG___whole_data_Norm1d_lr_","PTC___whole_data_Norm1d_lr_"]
# configs = ["riangulargrid___reach3_whole_data_Norm1d_lr","lobster___reach3_whole_data_Norm1d_lr","/grid___reach3_whole_data_Norm1d_lr_", "ogbgmolbbbp___reach3_whole_data_Norm1d_lr","DD___reach3_whole_data_Norm1d_lr","MDBBINARY___reach3_whole_data_Norm1d_lr_","MUTAG___reach3_whole_data_Norm1d_lr_","PTC___reach3_whole_data_Norm1d_lr_"]
# file_nem = "LDVAE_reach3"
# writing_order = [[3,4],[1,2], [5,6], [7,8]]
# precision_inexes = [1,2,5,6,7,8]
# Index = 3 # the index to select by
#--------------------------------------------
# # # bigg and SS metric
# fileName = "/local-scratch/kiarash/AAAI/Arxived_BaseLines/bigg_result/data/RandomGNN_Structural_Feature_attri_orbit_too_report_of_dir_withPrecision.csv"
# configs = ["DD","MDBBINARY_lattice_graph","ogbg-molbbbp", "obster_Kernel","/tri","UTAG_lattice_graph","TC_lattice_graph","grid-BFS"]
# file_nem = "BiGG"
# writing_order = [[3,4],[1,2], [5,6], [7,8]]
# precision_inexes = [1,2,5,6,7,8]
# Index = 3


# # Random GNN
# fileName = "/localhome/kzahirni/Desktop/SelfSupervised_Evaluation/bigg/self_supervised_GNN_report_for_dir.csv"
# configs = ["DD","MDBBINARY_lattice_graph","ogbg-molbbbp", "obster_Kernel","/tri","UTAG_lattice_graph","TC_lattice_graph","grid-BFS"]
# file_nem = "BiGG"
# writing_order = [[4,3],[2,1], [8,7], [6,5]]
# precision_inexes = [1,2,5,6,7,8]
# Index = 5
#--------------------------------------------
## GRAPHRNN-MLP
# # Random GNN
# fileName = "/local-scratch/kiarash/AAAI/Arxived_BaseLines/Graphrnn/RandomGNN_Structural_Feature_attri_orbit_too_report_of_dir_withPrecision.csv"
# configs = ["DD-MLP","IMDb-MLP","OGB-RNN-MLP", "lobsterMLP","tri-MLP","MUTAG-MLP","PTC-MLP","GRIDRNN-MLP"]
# file_nem = "MLP"
# writing_order = [[3,4],[1,2], [5,6], [7,8]]
# precision_inexes = [1,2,5,6,7,8]
# Index = 3
# ## GRAPHRNN-RNN
# # # Random GNN
# fileName = "/local-scratch/kiarash/AAAI/Arxived_BaseLines/Graphrnn/RandomGNN_Structural_Feature_attri_orbit_too_report_of_dir_withPrecision.csv"
# configs = ["DD-RNN","IMdB-RNN","OGB-RNN-RNN", "lobsterRNN","tri-RNN","MuTAG_RNN","PTC-RNN","GRIDRNN-rnn"]
# file_nem = "RNN"
# writing_order = [[3,4],[1,2], [5,6], [7,8]]
# precision_inexes = [1,2,5,6,7,8]
# Index = 3
#--------------------------------------------
# # GraphVAE-MM
#
# fileName = "/localhome/kzahirni/Desktop/SelfSupervised_Evaluation/GhraphVAE-MM/self_supervised_GNN_report_for_dir.csv"
# configs = ["triangular_grid_graphGe","_lobster_graphGe","l_FC_grid_graph", "gbg-molbbbp_graphGe","_FC_DD_graphGeneratio","C_IMDBBINARY_gra","l_FC_PTC_graph","FC_MUTAG_grap"]
# file_nem = "GraphVAE-MM"
# writing_order = [[4,3],[2,1], [8,7], [6,5]]
# precision_inexes = [1,2,5,6,7,8]
# Index = 5

# # GraphVAE-MM
# # RAndom GNN
# fileName = "/local-scratch/kiarash/AAAI/Arxived_BaseLines/GraphVAE-MM/_Structural_Feature_attri_orbit_too_report_of_dir_withPrecision.csv"
# configs = ["triangular_grid_graphGe","_lobster_graphGe","l_FC_grid_graph", "gbg-molbbbp_graphGe","_FC_DD_graphGeneratio","C_IMDBBINARY_gra","l_FC_PTC_graph","FC_MUTAG_grap"]
# file_nem = "GraphVAE-MM"
# writing_order = [[3,4],[1,2], [5,6], [7,8]]
# precision_inexes = [1,2,5,6,7,8]
# Index = 3
#------------------------------------------------


# # Random GNN
# fileName = "/local-scratch/kiarash/AAAI/Arxived_BaseLines/GGRAN/RandomGNN_Structural_Feature_attri_orbit_too_report_of_dir_withPrecision.csv"
# configs = ["li_triangular_grid","li_lobster_","li_grid_2", "lli_ogbg-molbbbp_2","ulli_DD_2022-M","IMDBBINARY_lattice","GRAN_PTC","GRAN_MuTAG"]
# file_nem = "GRAN"
# writing_order = [[3,4],[1,2], [5,6], [7,8]]
# precision_inexes = [1,2,5,6,7,8]
# Index = 3
#-----------------------------------------------
# # LDVAE self supervised
# fileName = "BaseLineResult/SelfSupervised_Evaluation/ours/self_supervised_GNN_report_for_dir_LDVAE.csv"
# configs = ["riangulargrid___whole_data_Norm1d_lr","lobster___whole_data_Norm1d_lr","/grid___whole_data_Norm1d_lr_","DD___whole_data_Norm1d_lr","MDBBINARY___whole_data_Norm1d_lr_","MUTAG___whole_data_Norm1d_lr_","PTC___whole_data_Norm1d_lr_"]
# # configs = ["riangulargrid___reach3_whole_data_Norm1d_lr","lobster___reach3_whole_data_Norm1d_lr","/grid___reach3_whole_data_Norm1d_lr_", "ogbgmolbbbp___reach3_whole_data_Norm1d_lr","DD___reach3_whole_data_Norm1d_lr","MDBBINARY___reach3_whole_data_Norm1d_lr_","MUTAG___reach3_whole_data_Norm1d_lr_","PTC___reach3_whole_data_Norm1d_lr_"]
# file_nem = "reach2"
# Index = 4 # the index to select by
# writing_order = [[4,3],[2,1], [8,7], [6,5]]
# precision_inexes = [1,2,5,6,7,8]
# #-----------------------------------------------
# fileName = "BaseLineResult/SelfSupervised_Evaluation/ours/self_supervised_GNN_report_for_dir_LDVAE.csv"
# # configs = ["riangulargrid___whole_data_Norm1d_lr","lobster___whole_data_Norm1d_lr","/grid___whole_data_Norm1d_lr_","DD___whole_data_Norm1d_lr","MDBBINARY___whole_data_Norm1d_lr_","MUTAG___whole_data_Norm1d_lr_","PTC___whole_data_Norm1d_lr_"]
# configs = ["riangulargrid___reach3_whole_data_Norm1d_lr","lobster___reach3_whole_data_Norm1d_lr","/grid___reach3_whole_data_Norm1d_lr_" ,"DD___reach3_whole_data_Norm1d_lr","MDBBINARY___reach3_whole_data_Norm1d_lr_","MUTAG___reach3_whole_data_Norm1d_lr_","PTC___reach3_whole_data_Norm1d_lr_"]
# file_nem = "reach3"
# Index = 4 # the index to select by
# writing_order = [[4,3],[2,1], [8,7], [6,5]]
# precision_inexes = [1,2,5,6,7,8]
# #--------------------------------------------------------------------
# Statisticsa Based Baselines
# fileName = r"EvalutionMetricsAndTables/Statistics-BasedGGMs_RandGNNwithSubgraphStructuralProperties.csv"
# configs = ["triangular_grid_BTER","lobster_BTER",".1/grid_BTER", "ogbg-molbbbp_BTER","DD_BTER","/IMDBBINARY_BTER","PTC_BTER","MUTAG_BTER"]
# file_nem = "BTER_"
# writing_order = [[3,4],[1,2], [5,6], [7,8]]
# precision_inexes = [1,2,5,6,7,8]
# Index = 3
#--------------------------------------------------------------------
# # Statisticsa Based Baselines
# fileName = r"EvalutionMetricsAndTables/Statistics-BasedGGMs_RandGNNwithSubgraphStructuralProperties.csv"
# configs = ["triangular_grid_ChungLu","lobster_ChungLu",".1/grid_ChungL", "ogbg-molbbbp_ChungLu","DD_ChungLu","/IMDBBINARY_ChungLu","PTC_ChungLu","MUTAG_ChungLu"]
# file_nem = "ChungL"
# writing_order = [[3,4],[1,2], [5,6], [7,8]]
# precision_inexes = [1,2,5,6,7,8]
# Index = 3
#--------------------------------------------------------------------
# fileName = r"EvalutionMetricsAndTables/Statistics-BasedGGMs_RandGNNwithSubgraphStructuralProperties.csv"
# configs = ["triangular_grid_ErdosRenyi","lobster_ErdosRenyi",".1/grid_ErdosRenyi", "ogbg-molbbbp_ErdosRenyi","DD_ErdosRenyi","/IMDBBINARY_ErdosRenyi","PTC_ErdosRenyi","MUTAG_ErdosRenyi"]
# file_nem = "_ErdosRenyi"
# writing_order = [[3,4],[1,2], [5,6], [7,8]]
# precision_inexes = [1,2,5,6,7,8]
# Index = 3

#--------------------------------------------------------------------
# fileName = r"EvalutionMetricsAndTables/Statistics-BasedGGMs_RandGNNwithSubgraphStructuralProperties.csv"
# configs = ["triangular_grid_SBM","lobster_SBM",".1/grid_SBM", "ogbg-molbbbp_SBM","DD_SBM","/IMDBBINARY_SBM","PTC_SBM","MUTAG_SBM"]
# file_nem = "_SBM"
# writing_order = [[3,4],[1,2], [5,6], [7,8]]
# precision_inexes = [1,2,5,6,7,8]
# Index = 3
#

#--------------------------------------------------
## Ideal SCORE/ Random GNN
fileName = r"/local-scratch/kiarash/AAAI/Arxived_BaseLines/RandomGNN_WithStructuralPorperites_Ideal_report_of_dir_withPrecision.csv"
configs = ["data/tri","lobster_Kernel","grid-BFS", "ogbg-molbbbp","/DD","/IMDBBINARY_lattice_graph","PTC_lattice_graph","MUTAG_lattice_graph"]
file_nem = "_SBM"
writing_order = [[3,4],[1,2], [5,6], [7,8]]
precision_inexes = [1,2,5,6,7,8]
Index = 3
#--------------------------------------------------
## SelfSupervised
# Bigg
fileName = r"/local-scratch/kiarash/AAAI/Arxived_BaseLines/SelfSupervisedResult/BiGG_self_supervised_GNN_WithStructural_properties_report_for_dir.csv"
configs = ["data/tri","lobster_Kernel","grid-BFS", "ogbg-molbbbp","/DD","/IMDBBINARY_lattice_graph","PTC_lattice_graph","MUTAG_lattice_graph"]
file_nem = "BiGG"
writing_order = [[4,3],[2,1], [6,5], [8,7]]
precision_inexes = [1,2,5,6,7,8]


# GraphVAE-MM
fileName = r"/local-scratch/kiarash/AAAI/Arxived_BaseLines/SelfSupervisedResult/GraphVAE-MM_self_supervised_GNN_WithStructural_properties_report_for_dir.csv"
configs = ["triangular_grid_graphGe","_lobster_graphGe","l_FC_grid_graph", "gbg-molbbbp_graphGe","_FC_DD_graphGeneratio","C_IMDBBINARY_gra","l_FC_PTC_graph","FC_MUTAG_grap"]
file_nem = "GraphVAE-MM"
writing_order = [[4,3],[2,1], [6,5], [8,7]]
precision_inexes = [1,2,5,6,7,8]


# GRAN
fileName = r"/local-scratch/kiarash/AAAI/Arxived_BaseLines/SelfSupervisedResult/GRAN_self_supervised_GNN_WithStructural_properties_report_for_dir.csv"
configs = ["li_triangular_grid","li_lobster_","li_grid_2", "lli_ogbg-molbbbp_2","ulli_DD_2022-M","GRAN_PTC","GRAN_MuTAG"]
file_nem = "GRAN"
writing_order = [[4,3],[2,1], [6,5], [8,7]]
precision_inexes = [1,2,5,6,7,8]

# Self Supervised for StatBased Models

# #--------------------------------------------------------------------
# Statisticsa Based Baselines
fileName = r"/local-scratch/kiarash/AAAI/Arxived_BaseLines/SelfSupervisedEval/StatiticBasedModels_self_supervised_GNN_WithStructural_properties_report_for_dir.csv"
configs = ["triangular_grid_BTER","lobster_BTER",".1/grid_BTER", "ogbg-molbbbp_BTER","DD_BTER","/IMDBBINARY_BTER","PTC_BTER","MUTAG_BTER"]
file_nem = "BTER_"
writing_order = [[4,3],[2,1], [6,5], [8,7]]
precision_inexes = [1,2,5,6,7,8]
Index = 3
#--------------------------------------------------------------------
# Statisticsa Based Baselines
fileName = r"/local-scratch/kiarash/AAAI/Arxived_BaseLines/SelfSupervisedEval/StatiticBasedModels_self_supervised_GNN_WithStructural_properties_report_for_dir.csv"
configs = ["triangular_grid_ChungLu","lobster_ChungLu",".1/grid_ChungL", "ogbg-molbbbp_ChungLu","DD_ChungLu","/IMDBBINARY_ChungLu","PTC_ChungLu","MUTAG_ChungLu"]
file_nem = "ChungL"
writing_order = [[4,3],[2,1], [6,5], [8,7]]
precision_inexes = [1,2,5,6,7,8]
Index = 3
#--------------------------------------------------------------------
fileName = r"/local-scratch/kiarash/AAAI/Arxived_BaseLines/SelfSupervisedEval/StatiticBasedModels_self_supervised_GNN_WithStructural_properties_report_for_dir.csv"
configs = ["triangular_grid_ErdosRenyi","lobster_ErdosRenyi",".1/grid_ErdosRenyi", "ogbg-molbbbp_ErdosRenyi","DD_ErdosRenyi","/IMDBBINARY_ErdosRenyi","PTC_ErdosRenyi","MUTAG_ErdosRenyi"]
file_nem = "_ErdosRenyi"
writing_order = [[4,3],[2,1], [6,5], [8,7]]
precision_inexes = [1,2,5,6,7,8]
Index = 3

#--------------------------------------------------------------------
fileName = r"/local-scratch/kiarash/AAAI/Arxived_BaseLines/SelfSupervisedEval/StatiticBasedModels_self_supervised_GNN_WithStructural_properties_report_for_dir.csv"
configs = ["triangular_grid_SBM","lobster_SBM",".1/grid_SBM", "ogbg-molbbbp_SBM","DD_SBM","/IMDBBINARY_SBM","PTC_SBM","MUTAG_SBM"]
file_nem = "_SBM"
writing_order = [[4,3],[2,1], [6,5], [8,7]]
precision_inexes = [1,2,5,6,7,8]
Index = 3
#
#--------------------------------------------------------------------
#--------------------------------------------------------------------
#--------------------------------------------------------------------
# Ideal Score
fileName = r"/local-scratch/kiarash/AAAI/Arxived_BaseLines/SelfSupervisedEval/_idealScore.csv"
configs = ["/tri","obster_Kernel","grid-BFS", "ogbg-molbbbp","IMDBBINARY","PTC","MUTAG","DD"]
file_nem = "Ideal"
writing_order = [[4,3],[2,1], [6,5], [8,7]]
precision_inexes = [1,2,5,6,7,8]
Index = 3

#--------------------------------------------------------------------
# Bigg Self supervised
fileName = r"/local-scratch/kiarash/AAAI/Arxived_BaseLines/SelfSupervisedEval/Bigg_self_supervised_GNN_WithStructural_properties_report_for_dir.csv"
configs = ["DD","MDBBINARY_lattice_graph","ogbg-molbbbp", "obster_Kernel","/tri","UTAG_lattice_graph","TC_lattice_graph","grid-BFS"]
file_nem = "Ideal"
writing_order = [[4,3],[2,1], [6,5], [8,7]]
precision_inexes = [1,2,5,6,7,8]
Index = 3

# # Gran
# /localhome/kzahirni/Desktop/SelfSupervised_Evaluation/GRAN
fileName = r"/local-scratch/kiarash/AAAI/Arxived_BaseLines/SelfSupervisedEval/GRAN_self_supervised_GNN_WithStructural_properties_report_for_dir.csv"
configs = ["li_triangular_grid","li_lobster_","li_grid_2", "lli_ogbg-molbbbp_2","ulli_DD_2022-M","GRAN_PTC","GRAN_MuTAG"]
file_nem = "GRAN"
writing_order = [[4,3],[2,1], [8,7], [6,5]]
precision_inexes = [1,2,5,6,7,8]
# Index = 5
## GRAPHRNN-MLP
# # Random GNN
fileName = r"/local-scratch/kiarash/AAAI/Arxived_BaseLines/SelfSupervisedEval/GRaphRNN_self_supervised_GNN_WithStructural_properties_report_for_dir.csv"
configs = ["DD-MLP","IMDb-MLP","OGB-RNN-MLP", "lobsterMLP","tri-MLP","MUTAG-MLP","PTC-MLP","GRIDRNNMLP"]
file_nem = "MLP"
writing_order = [[4,3],[2,1], [8,7], [6,5]]
precision_inexes = [1,2,5,6,7,8]
Index = 3
# ## GRAPHRNN-RNN
# # # Random GNN
# fileName = r"/local-scratch/kiarash/AAAI/Arxived_BaseLines/SelfSupervisedEval/GRaphRNN_self_supervised_GNN_WithStructural_properties_report_for_dir.csv"
# configs = ["DD-RNN","IMdB-RNN","OGB-RNN-RNN", "lobsterRNN","tri-RNN","MuTAG_RNN","PTC-RNN","GRIDRNN-rnn"]
# file_nem = "RNN"
# writing_order = [[4,3],[2,1], [8,7], [6,5]]
# precision_inexes = [1,2,5,6,7,8]
# Index = 3


# LDVAE
fileName = "/local-scratch/kiarash/AAAI/Arxived_BaseLines/TheStructuralProperties/2Reach_self_supervised_GNN_WithStructural_properties_report_for_dir.csv"
# configs = ["riangulargrid___whole_data_Norm1d_lr","lobster___whole_data_Norm1d_lr","/grid___whole_data_Norm1d_lr_", "ogbgmolbbbp___whole_data_Norm1d_lr","DD___whole_data_Norm1d_lr","MDBBINARY___whole_data_Norm1d_lr_","MUTAG___whole_data_Norm1d_lr_","PTC___whole_data_Norm1d_lr_"]
# configs = ["riangulargrid___reach3","lobster___reach3","/grid___reach3", "ogbgmolbbbp___reach3","DD___reach3","MDBBINARY___reach3","MUTAG___reach3","PTC___reach3"]
configs = ["riangulargrid___NewArchi_2Step_","lobster___NewArchi_2Step_","/grid___NewArchi_2Step_", "ogbgmolbbbp___NewArchi_2Step_","DD___NewArchi_2Step_","MDBBINARY___NewArchi_2Step_","MUTAG___NewArchi_2Step_","PTC___NewArchi_2Step_"]

# file_nem = "LDVAE_reach3"
file_nem = "LDVAE_reach2_f1pr"
writing_order = [[4,3],[2,1], [6,5], [8,7]]
precision_inexes = [1,2,5,6,7,8]
Index = 4 # the index to select by



fileName = fileName
csv_file,head = file(fileName)

result = [head]
for conf in configs:
    result.append(find_row_with_largest_index_value_in_csv(conf,csv_file,Index))


reordered_list = reOrder(result,writing_order,precision_inexes)

# Write on the Disck
fileName = fileName+"_"+file_nem # dir + file name
write(reordered_list,fileName+"_Print.csv")
result
