import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import re
import argparse
import sys
# Precision, recall, F1 score for each method
PCA = (0.98, 0.67, 0.79)
LSTM = (0.9526, 0.9903, 0.9711)

#fig, ax = plt.subplots()

#index = np.arange(3)
#bar_width = 0.2
#opacity = 0.4

#rects1 = ax.bar(index, PCA, bar_width, alpha=opacity, color='b', label='PCA')
#rects2 = ax.bar(index + bar_width, LSTM, bar_width, alpha=opacity, color='r', label='LSTM')

#ax.set_xlabel('Measure')
#ax.set_ylabel('Scores')
#ax.set_title('Scores by different models')
#ax.set_xticks(index + bar_width / 2)
#ax.set_xticklabels(('Precesion', 'Recall', 'F1-score'))
#ax.legend()
#plt.show()


parser = argparse.ArgumentParser()
parser.add_argument('-dataset', default='OpenStack', type=str)
parser.add_argument('-parameter', default='window', type=str)
try:    
    args = parser.parse_args()
    dataset_folder = args.dataset
    parameter = args.parameter
    parameters = [
            'window',
            'batch_size',
            'epoch',
            'layers',
            'hidden',
            'candidates'
    ]
    dataset_folders = [
            'OpenStack',
            'HDFS'
    ]
    if parameter not in parameters:
        raise ValueError("Inserted the wrong parameter.\nParameters that can be analyzed are: {} ".format(parameters))
    if dataset_folder not in dataset_folders:
        raise ValueError("Inserted the wrong dataset.\nDatasets that can be analyzed are: {} ".format(dataset_folders))
except ValueError as e:
    print(e)
    sys.exit(2)

metrics_path = 'metrics/'+dataset_folder+'/varying_'+parameter
#x = [8, 9, 10, 11]
#FP = [605, 588, 495, 860]
#FN = [465, 333, 108, 237]
#TP = [4123 - FN[i] for i in range(4)]
#P = [TP[i] / (TP[i] + FP[i]) for i in range(4)]
#R = [TP[i] / (TP[i] + FN[i]) for i in range(4)]
#F1 = [2 * P[i] * R[i] / (P[i] + R[i]) for i in range(4)]

def extract_parameter(filename,param):
    match = re.search(re.escape(param)+r'=(\d+)', filename)
    return int(match.group(1)) if match else 0


x =[]
FP =[]
FN =[]
TP =[]
P =[]
R =[]
F1 =[]
files = sorted([f for f in os.listdir(metrics_path) if f.endswith(".pt")], key=lambda f: extract_parameter(f,parameter))

for filename in files:
    print(filename)
    if filename.endswith(".pt"):
        filepath = os.path.join(metrics_path, filename)
        metrics = torch.load(filepath)
        
        # Extract metrics
        x.append(metrics.get(parameter,0.0))
        FP.append(metrics.get('FP', 0.0))
        FN.append(metrics.get('FN', 0.0))
        TP.append(metrics.get('TP', 0.0))
        F1.append(metrics.get('F1-score', 0.0)/100)
        P.append(metrics.get('Precision', 0.0)/100)
        R.append(metrics.get('Recall', 0.0)/100)
print(x)

l1 = plt.plot(x, P, ':rx')
l2 = plt.plot(x, R, ':b+')
l3 = plt.plot(x, F1, ':k^')
plt.xlabel(parameter)
plt.ylabel('scores')
plt.legend((l1[0], l2[0], l3[0]), ('Precision', 'Recall', 'F1-score'))
plt.xticks(x)
plt.ylim((0.0, 1))
plt.show()
