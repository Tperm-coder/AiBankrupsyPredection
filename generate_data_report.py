file_path = "Bankruptcy.csv"

# first :  remove duplicate records

# second possiblities for cleaning :
# 1 : fill empty with mean , mode , median
# 2 : drop columns with x% of emptyness (play with x)
# 3 : linear interpolation
# 4 : fill with outliers values (habda)
# 5 : fill with linear regression average

# third : apply normalization box cox

# for each permutation apply the three classification techniques and maximize the accuracy

file = open(file_path,'r')
data = file.read().split('\n')

data.pop()
file.close()

empty_cells_count = 0
rows_with_empty_cell_count = 0
empty_rows_count = 0

row_set = set()
map_col_empty_cells_count = {}


map_col_mean = {}
map_col_mode = {}
map_col_median = {}

for i in range(65) :
    map_col_empty_cells_count[i] = 0
    map_col_mean[i] = 0
    map_col_mode[i] = 0
    map_col_median[i] = 0


for row in range(len(data)) :

    temp = data[row].replace(" ",'')
    if (temp == "") :
        empty_rows_count += 1

    row = data[row].split(',')
    row_set.add(','.join(row))

    contains_empty_cell = False
    for i in range(len(row)) :

        if (row[i] == "" or row[i] == '?') :
            contains_empty_cell = True
            map_col_empty_cells_count[int(i)] =  map_col_empty_cells_count[int(i)] + 1
            empty_cells_count += 1

    if (contains_empty_cell) :
        rows_with_empty_cell_count += 1;



duplicated_row = len(data) - len(row_set)
cells_count = len(data) * 65

report_string = ""

report_string += ("Number of rows is : " + str(len(data)) + '\n')
report_string += ("Number of columns is : 65") + '\n'
report_string += ("Number of cells is : " + str(cells_count)) + '\n\n'

report_string += ("Duplicated rows count is : " + str(duplicated_row)) + '\n'
report_string += ("Percentage of duplicated_rows  : " + str((duplicated_row/len(data))*100) + ' %\n\n')

report_string += ("Number of rows that have empty cells is  : " + str(rows_with_empty_cell_count)) + '\n'
report_string += ("Percentage of rows that have empty cells is : " + str((rows_with_empty_cell_count/len(data))*100) +  " %") +'\n\n'

report_string += ("Empty cell count is : " + str(empty_cells_count)) + '\n'
report_string += ("Percentage of empty cells is : " + str((empty_cells_count/cells_count)*100) + " %") + '\n\n'

report_string += ("Number of empty rows is  : " + str(empty_rows_count) + '\n\n')

for w in sorted(map_col_empty_cells_count, key=map_col_empty_cells_count.get, reverse=True):
    report_string += ("Col No. " + str(int(w) + 1) + " has " + str(map_col_empty_cells_count[int(w)]) + " empty cells") +'\n'

file = open("DatasetReport.txt",'w')
file.write(report_string)
file.close()

print("Done")
