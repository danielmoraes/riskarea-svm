import sys, os, math
from util import *

# checking params
if len(sys.argv) <= 1:
    print('Usage: {0} dataset_file'.format(sys.argv[0]))
    raise SystemExit
else:
    assert os.path.exists(sys.argv[1]),'dataset file not found'

classes_idx = {'bedroom': 1,
               'CALsuburb': 2,
               'industrial': 3,
               'kitchen': 4,
               'livingroom': 5,
               'MITcoast': 6,
               'MITforest': 7,
               'MIThighway': 8,
               'MITinsidecity': 9,
               'MITmountain': 10,
               'MITopencountry': 11,
               'MITstreet': 12,
               'MITtallbuilding': 13,
               'PARoffice': 14,
               'store': 15}

ds_path = sys.argv[1]
f = open(ds_path + '.libsvm', 'w')
for line in open(ds_path, 'r'):
    line = line.split(' ')
    label = classes_idx[line[len(line)-1].replace('\n', '')]
    features = line[0:len(line)-1]
    features_map = dict(zip([i+1 for i in range(len(features))], [float(i) for i in features]))
    out_line = str(label) + ' ' + str(features_map).replace(',', '').replace('{', '').replace('}', '').replace(': ', ':') + '\n'
    f.write(out_line)
f.close()

