
# this python code is wite to select and rwite the result for the best configuration
import csv
import math
def my_round(x,digit=4):
    digit = 10**digit
    x = math.floor(x * digit) / digit
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




# LDVAE
# fileName = "GenStatReportingResult/Candidates.csv"
fileName = "C:/git/AAAI/Nips-2023/LastCandidates/ThesummeyReport.csv"

Index = 4 # the index to select by



fileName = fileName
csv_file,head = file(fileName)

result = [head]
for rows  in csv_file[1:]:
    new_row = rows[:2]
    new_row.append(str(100*my_round(float(rows[2])))+ "+"+str(100*my_round(float(rows[3]))))
    new_row.append(str(my_round(float(rows[4]),4)) + "+" + str(my_round(float(rows[5]),4)))
    new_row.append(rows[-1])
    result.append(new_row)

# Write on the Disck

write(result,fileName+"_CleanedForpaper.csv")
result
