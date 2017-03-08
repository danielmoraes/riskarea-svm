import sys, os, math
from util import *
# get random subsets lines to cross validation
def random_subsets_lines(lines, fold):
    subsets = []
    remaining = lines
    sub_size = math.floor(len(remaining)/fold)
    for i in range(fold-1):
        random.seed()
        sub_lines = random_lines_selection(remaining, int(sub_size))
        subsets.append(sub_lines)
        remaining = [line for line in remaining if line not in sub_lines]
    subsets.append(remaining)
    return subsets

# checking params
if len(sys.argv) <= 2:
    print('Usage: {0} dataset_file number_of_folds'.format(sys.argv[0]))
    raise SystemExit
else:
    assert os.path.exists(sys.argv[1]),'dataset file not found'
# setting initial configuration
ds_path = sys.argv[1]
folds = int(sys.argv[2])
ds_total = sum(1 for line in open(ds_path,'r'))
ds_lines = [i for i in range(ds_total)]
# selecting subsets lines
subset_lines = random_subsets_lines(ds_lines, folds)
subset_files = [[open(ds_path + '.fold' + str(i) + '.train', 'w'), open(ds_path + '.fold' + str(i) + '.test', 'w')] for i in range(folds)]
# writing subsets files
for i in range(folds):
    ds_file = open(ds_path, 'r')
    prev_selected_linenum = -1
    for j in xrange(len(subset_lines[i])):
        for cnt in xrange(subset_lines[i][j]-prev_selected_linenum-1):
            line = ds_file.readline()
            subset_files[i][0].write(line)
        subset_files[i][1].write(ds_file.readline())
        prev_selected_linenum = subset_lines[i][j]
    subset_files[i][1].close()

    for line in ds_file:
        subset_files[i][0].write(line)
    subset_files[i][0].close()
    ds_file.close()



