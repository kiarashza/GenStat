from scipy.stats import spearmanr
import numpy as np

def calculate_pearson_correlation(list1, list2):
    if len(list1) != len(list2):
        raise ValueError("Both lists must have the same length.")

    pearson_corr = np.corrcoef(list1, list2)[0, 1]
    return pearson_corr


def calculate_spearman_rank_correlation(list1, list2):
    if len(list1) != len(list2):
        raise ValueError("Both lists must have the same length.")

    spearman_corr, _ = spearmanr(list1, list2)
    return spearman_corr

def calculate_mse(list1, list2):
    if len(list1) != len(list2):
        raise ValueError("Both lists must have the same length.")

    squared_errors = [(x - y) ** 2 for x, y in zip(list1, list2)]
    mse = sum(squared_errors) / len(list1)
    return mse

import csv

def read_csv_file(file_path):
    data = []
    with open(file_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            data.append(row)
    return data

def toCSV(filename, list_):
    with open(filename, 'w') as f:
        # using csv.writer method from CSV package
        write = csv.writer(f)
        write.writerows(list_)
path = "C:/git/AAAI/Nips-2023/LastCandidates/"
report_file_path = path+'GNN_effectiveness_3GGM.csv'  # Replace with the actual file path


BenchMark = read_csv_file(report_file_path)

BenchMark = np.array(BenchMark)



#AUC
# Models_index= {"GenStat":23,"BTER":33,"BiGG":13}
# original_index = 3
# Metric = "Test_auc"
#Accuracy
Models_index= {"bigg":16,"GenStat":11}
original_index = 4 # the index where accuracy should be read
Metric = "Test_ACURACY"

report = [[""]]
for model, column in Models_index.items():
    report[-1].append(model)
    report[-1].append(model)
    report[-1].append(model)
report.append([""])

for model, column in Models_index.items():
        report[-1].append("MSE")
        report[-1].append("pearson")
        report[-1].append("spearman")

for dataset in np.unique(BenchMark[2:,0]):
    d_report = [dataset]
    datset_index = BenchMark[:,0]==dataset
    for model,column in Models_index.items():
        or_list = BenchMark[datset_index,original_index]
        dataset_list = BenchMark[datset_index,column]
        or_list = [float(n) for n in or_list]
        dataset_list = [float(n) for n in dataset_list]
        MSE = calculate_mse(or_list,dataset_list)
        pearson = calculate_pearson_correlation(or_list,dataset_list)
        spearman = calculate_spearman_rank_correlation(or_list,dataset_list)
        d_report.extend([round(x,4) for x in [MSE,pearson,spearman]])
    report.append(d_report)
toCSV(path+Metric+"og_coreelationOfGNNS_.csv",report)
