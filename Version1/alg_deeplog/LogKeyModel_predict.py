import sys
import torch
import torch.nn as nn
import time
import argparse
import os 
# Device configuration
device = torch.device("cpu")
print("device: {}".format(device))

def generate(name):
    ''' 
    reads data from file and forms a set (or list) of
    sessions 'OpenStack', where each session is a sequence 
    of data logs
    '''
    dataset = set()    # For these results (better than DeepLog paper ones)
    #dataset = []     # For DeepLog paper results

    with open( name, 'r') as f:
        for ln in f.readlines():
            ln = list(map(lambda n: n - 1, map(int, ln.strip().split())))
            ln = ln + [-1] * (window_size + 1 - len(ln))
            dataset.add(tuple(ln))
            #dataset.append(tuple(ln))
    print('Number of sessions({}): {}'.format(name, len(dataset)))
    return dataset

# Subclass of nn.Module in PyTorch
class Model(nn.Module):

    # Class initialization methood
    def __init__(self, input_size, hidden_size, num_layers, num_keys):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layer initialization
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # Linear layer initialization
        self.fc = nn.Linear(hidden_size, num_keys)

    # Forward operation method
    def forward(self, x):
        ''' 
        During the forward operation, the input sequence 
        passes through the LSTM layer and the final output of 
        the last time step is projected onto a smaller output 
        space using a linear layer.
        
        x: input sequence 

        '''
        # Initial hidden layers 
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        
        # Pass input sequence x through LSTM layer, obtaining output
        # output of interal layers is ignored
        out, _ = self.lstm(x, (h0, c0))
        
        #takes only last temporal step of output
        out = self.fc(out[:, -1, :])
        return out


if __name__ == '__main__':

    # Hyperparameters
    num_classes = 28
    input_size = 1 
    batch_size = 64

    # Dataset considered 
    #dataset_folder = 'OpenStack/'
    dataset_normal = 'test_normal'
    dataset_abnormal = 'test_abnormal'

    # Parser of arguments from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', default='OpenStack', type=str)
    parser.add_argument('-num_epochs', default=300, type=int)
    parser.add_argument('-num_layers', default=2, type=int)
    parser.add_argument('-hidden_size', default=64, type=int)
    parser.add_argument('-window_size', default=10, type=int)
    parser.add_argument('-num_candidates', default=9, type=int)
    parser.add_argument('-parameter', default='window', type=str)
    
    try:
        args = parser.parse_args()
        dataset_folder = args.dataset
        num_epochs = args.num_epochs
        num_layers = args.num_layers         # NN layers
        hidden_size = args.hidden_size       # Size of hidden layer
        window_size = args.window_size       # Window size for input data generation
        num_candidates = args.num_candidates # Number of candidates to consider for the evaluation
        parameter = args.parameter           # parameter to analyze

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
    
    # Path of the model already trained
    model_name = 'Adam_{}_batch_size={}_epoch={}_layers={}_hidden={}_window={}_parameter={}'.format(str(dataset_folder),batch_size, num_epochs,num_layers,hidden_size,window_size,str(parameter))
    dataset_folder = dataset_folder+'/'
    model_path = 'models_'+dataset_folder+model_name+'.pt'
    print(model_path)
    # Path for saving metrics 
    metrics_path = 'metrics/'+dataset_folder+'varying_'+parameter

    # Model definition
    model = Model(input_size, hidden_size, num_layers, num_classes).to(device)
    
    # Load model weights from trained model
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print('model_path: {}'.format(model_path))

    # Generation of normal & abnormal datasets
    test_normal_loader = generate('data_parsed/'+dataset_folder+dataset_normal)
    test_abnormal_loader = generate('data_parsed/'+dataset_folder+dataset_abnormal)
    

    # Initialization of True positives & False positives counters
    TP = 0  # Correct predictions of abnormal sequences
    FP = 0  # Wrong predictions on normal sequences as abnormal

    # Test the model
    start_time = time.time()

    # Evaluation of the model on test_normal_loader data
    with torch.no_grad(): # Operations should't track the gradient
        for line in test_normal_loader:

            # For each data sequence 
            for i in range(len(line) - window_size):

                # Sequence & corresponding real label extraction
                seq = line[i:i + window_size]
                label = line[i + window_size]

                # Conversion in PyTorch tensors
                seq = torch.tensor(seq, dtype=torch.float).view(-1, window_size, input_size).to(device)
                label = torch.tensor(label).view(-1).to(device)

                # Compute output of the model given the sequence
                # It is a matrix, where each row is a probability vector 
                # for each output class
                # Example [0.1, 0.4, 0.0001, 0.2, ..., 0.05]
                output = model(seq)

                # Sort class indices by probability values in crescent order
                # and select of the last 'num_candidates' indices
                # corresponding to the most probable classes
                predicted = torch.argsort(output, 1)[0][-num_candidates:]
                if label not in predicted:
                    # Increment error
                    FP += 1
                    break

    # Evaluation of the model on test_normal_loader data
    with torch.no_grad(): # Operations should't track the gradient
        for line in test_abnormal_loader:

            # For each data sequence 
            for i in range(len(line) - window_size):

                # Sequence & corresponding real label extraction
                seq = line[i:i + window_size]
                label = line[i + window_size]

                # Conversion in PyTorch tensors
                seq = torch.tensor(seq, dtype=torch.float).view(-1, window_size, input_size).to(device)
                label = torch.tensor(label).view(-1).to(device)
                
                # Compute output of the model given the sequence
                # It is a matrix, where each row is a probability vector 
                # for each output class
                # Example [0.1, 0.4, 0.0001, 0.2, ..., 0.05]
                output = model(seq)

                # Sort class indices by probability values in crescent order
                # and select of the last 'num_candidates' indices
                # corresponding to the most probable classes
                predicted = torch.argsort(output, 1)[0][-num_candidates:]
                if label not in predicted:
                    # Increment error
                    TP += 1
                    break

    elapsed_time = time.time() - start_time
    print('elapsed_time: {:.3f}s'.format(elapsed_time))

    # Compute false negatives, precision, recall and F1-measure
    FN = len(test_abnormal_loader) - TP # Wrong predictions of abnormal sequences as normal
    P = 100 * TP / (TP + FP)
    R = 100 * TP / (TP + FN)
    F1 = 2 * P * R / (P + R)
    
    metrics = {
        'FN': FN,
        'FP': FP,
        'TP': TP,
        'Precision': P,
        'Recall': R,
        'F1-score':F1,
        'window':window_size,
        'batch_size': batch_size,
        'epoch': num_epochs,
        'layers': num_layers,
        'hidden': hidden_size,
        'candidates': num_candidates,
        'parameter': parameter
    }
    print('true positive (TP): {}, false positive (FP): {}, false negative (FN): {}, Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%'.format(TP,FP, FN, P, R, F1))
    print('Finished Predicting')
    if not os.path.isdir(metrics_path):
        os.makedirs(metrics_path)
    torch.save(metrics, metrics_path + '/' + model_name + '.pt')
