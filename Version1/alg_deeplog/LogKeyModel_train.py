import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader
import argparse
import os

# Device configuration
if not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        print("MPS not available because the current PyTorch install was not "
              "built with MPS enabled.")
    else:
        print("MPS not available because the current MacOS version is not 12.3+ "
              "and/or you do not have an MPS-enabled device on this machine.")
    device = torch.device("cpu")
else:
    device = torch.device("mps")
print("device: {}".format(device))

# Loading Dataset
def generate(name):
    num_sessions = 0
    inputs = []
    outputs = []
    with open( name, 'r') as f:
        for line in f.readlines():
            num_sessions += 1
            line = tuple(map(lambda n: n - 1, map(int, line.strip().split())))
            for i in range(len(line) - window_size):
                inputs.append(line[i:i + window_size])
                outputs.append(line[i + window_size])
    print('Number of sessions({}): {}'.format(name, num_sessions))
    print('Number of seqs({}): {}'.format(name, len(inputs)))
    dataset = TensorDataset(torch.tensor(inputs, dtype=torch.float), torch.tensor(outputs))
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

    # Dataset considered 
    #dataset_folder = 'OpenStack'
    dataset_train = '/train'
    # Hyperparameters
    num_classes = 28    # model classes
    batch_size = 64   # batch size for training
    input_size = 1      # dimension of the input for each step
    
    
    # Parser of arguments from command line
    ''' 
    example: python3 LogKeyModel_train.py -num_layers 3 -hidden_size 128 -window_size 20
    default values (the best ones) are:
    num_layers = 2
    hidden_size = 64
    window_size = 10
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', default='OpenStack', type=str)
    parser.add_argument('-num_epochs', default=300, type=int)
    parser.add_argument('-num_layers', default=2, type=int)
    parser.add_argument('-hidden_size', default=64, type=int)
    parser.add_argument('-window_size', default=10, type=int)
    parser.add_argument('-parameter', default='window', type=str)

    try:
        args = parser.parse_args()
        dataset_folder = args.dataset
        num_epochs = args.num_epochs
        num_layers = args.num_layers         # NN layers
        hidden_size = args.hidden_size       # Size of hidden layer
        window_size = args.window_size       # Window size for input data generation
        parameter = args.parameter           # parameter to analyze
        model_dir = 'models_'+dataset_folder # path to the directory in which to save the trained model
        log_dir = 'logs_'+dataset_folder
        parameters = [
            'window',
            'batch_size',
            'epoch',
            'layers',
            'hidden',
            'candidates'
        ]
        if parameter not in parameters:
            raise ValueError("Inserted the wrong parameter.\nParameters that can be analyzed are: {} ".format(parameters))
    except ValueError as e:
        print(e)
        sys.exit(2)

    # log string, for the model's name
    log = 'Adam_{}_batch_size={}_epoch={}_layers={}_hidden={}_window={}_parameter={}'.format(str(dataset_folder),str(batch_size), str(num_epochs),str(num_layers),str(hidden_size),str(window_size),str(parameter))
    
    # Model definition
    model = Model(input_size, hidden_size, num_layers, num_classes).to(device)
    
    # Dataset generation
    seq_dataset = generate('data_parsed/'+dataset_folder+dataset_train)
    
    dataloader = DataLoader(seq_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    
    # TensorBoard object for metrics registration
    writer = SummaryWriter(log_dir=log_dir+'/'+log)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # Train the model
    start_time = time.time()
    total_step = len(dataloader)

    # Loop over the dataset 'num_epochs' times
    for epoch in range(num_epochs): 
        train_loss = 0
        for step, (seq, label) in enumerate(dataloader):
           
            # Forward pass
            seq = seq.clone().detach().view(-1, window_size, input_size).to(device)
            output = model(seq)
            loss = criterion(output, label.to(device))

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            writer.add_graph(model, seq)
        print('Epoch [{}/{}], train_loss: {:.4f}'.format(epoch + 1, num_epochs, train_loss / total_step))
        writer.add_scalar('train_loss', train_loss / total_step, epoch + 1)
    elapsed_time = time.time() - start_time
    print('elapsed_time: {:.3f}s'.format(elapsed_time))

    # Save model
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    torch.save(model.state_dict(), model_dir + '/' + log + '.pt')

    # Close TensorBoard
    writer.close()
    print('Finished Training')
