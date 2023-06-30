import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import tkinter as tk
from datetime import datetime
import os
import torchsummary
import matplotlib.offsetbox as offsetbox
import time
import signal
import sys
import matplotlib.pyplot as plt

#Forward recurrent neural network (one-step ahead prediction):
# Mapping V (past) and Q (past) to Q(next): { q_{t-k} ... q_t, V_{t+1-k} ... V_{t+1} } -> q_{t+1}
#NOTE - if running without GPU, need to remove LEMs, and remove .cuda() from the code

#LEMs from Prof Siddharth Mehta's group --------------------------------#
import math
import lem_cuda
class LEMFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, weights, weights_lin_z, bias,
                bias_lin_z, initial_y_state, initial_z_state, dt):
        all_y_states, all_z_states, all_X, all_X2, \
        all_multi_scales, all_lin_new_z_state = lem_cuda.forward(inputs, weights,
                                                                 weights_lin_z, bias, bias_lin_z,
                                                                 initial_y_state, initial_z_state, dt)
        ctx.save_for_backward(all_X, all_X2, all_multi_scales,
                              all_lin_new_z_state, weights, weights_lin_z, bias,
                              bias_lin_z, initial_y_state, initial_z_state, dt)
        return all_y_states, all_z_states

    @staticmethod
    def backward(ctx, grad_y_states, grad_z_states):
        outputs = lem_cuda.backward(grad_y_states.contiguous(), grad_z_states.contiguous(), *ctx.saved_tensors)
        d_inputs, d_weights, d_weights_lin_z, d_bias, d_bias_lin_z, d_initial_y_state, d_initial_z_state = outputs
        return None, d_weights, d_weights_lin_z, d_bias, d_bias_lin_z, d_initial_y_state, d_initial_z_state, None


class LEMcuda(torch.nn.Module):
    def __init__(self, ninp, nhid, dt):
        super(LEMcuda, self).__init__()
        self.ninp = ninp
        self.nhid = nhid
        self.weights = torch.nn.Parameter(torch.empty(3 * nhid, ninp + nhid))
        self.weights_lin_z = torch.nn.Parameter(torch.empty(nhid, ninp + nhid))
        self.bias = torch.nn.Parameter(torch.empty(3 * nhid))
        self.bias_lin_z = torch.nn.Parameter(torch.empty(nhid))
        self.dt = torch.tensor(dt).reshape(1, 1).cuda()

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.nhid)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, +stdv)

    def forward(self, input, states=None):
        if states is None:
            y = input.data.new(input.size(1), self.nhid).zero_()
            z = input.data.new(input.size(1), self.nhid).zero_()
            states = (y,z)
        return LEMFunction.apply(input.contiguous(), self.weights, 
                                 self.weights_lin_z, self.bias, 
                                 self.bias_lin_z, *states, self.dt)

#--------------------End of LEM code--------------------#

global model
global test_x
global test_y
global train_y
global loss
global tag
global writer
global test_input_voltage
global entire_displacement
global split_tracker_train
global split_tracker_test
global master_q

#Function to catch SIG INT midway through training
def interrupt_handler(sig, frame):
    global test_y
    prediction, final_test_loss = test_model(model, test_x, test_y, loss) #So that we don't end up calling it twice if we've been tracking test loss
    writer.add_scalars('Train vs Test Loss', tag_scalar_dict={'Test loss': final_test_loss}, global_step=int(tag.split("_")[0]))
    plot_results(prediction, train_y, test_y, tag)
    torch.save(model.state_dict(), './Models/' + tag + '/Model')
    sys.exit()

def load_arrays(V_file_name, q_file_name):
    V_original = np.loadtxt(V_file_name, dtype = np.float32)
    q_original = np.loadtxt(q_file_name, dtype = np.float32)
    assert(len(V_original) == len(q_original))
    return V_original, q_original

#Create look back windows for input to neural network. Split dataset randomly into train and test points based on specified ratio
def create_time_series(V_original, q_original, look_back, train_ratio):
    global split_tracker_test
    global split_tracker_train
    split_tracker_test = []
    split_tracker_train = []
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    for i in range(look_back, len(V_original)):
            if (np.random.uniform() < train_ratio):
                #Goes in train
                x_train.append([V_original[i-look_back+1:i+1], q_original[i-look_back:i]])
                y_train.append(q_original[i])
                split_tracker_train.append(i*1e-2)
            else:
                #Goes in test
                x_test.append([V_original[i-look_back+1:i+1], q_original[i-look_back:i]])
                y_test.append(q_original[i])
                split_tracker_test.append(i*1e-2)
    assert(len(x_train) == len(y_train))
    assert(len(x_test) == len(y_test))
    return np.transpose(np.array(x_train), (0, 2, 1)), np.expand_dims(np.array(y_train), 1), np.transpose(np.array(x_test), (0, 2, 1)), np.expand_dims(np.array(y_test), 1)

def plot_dataset(x, y, l1, l2):
    time = np.arange(len(x))
    fig, axs = plt.subplots(2)
    axs[0].plot(time, x, label=l1)
    axs[1].plot(time, y, label=l2)
    axs[0].legend()
    axs[1].legend()

def plot_train_test_split(split_tracker_train, split_tracker_test, train_y, test_y):
    start = time.time()
    fig, axs = plt.subplots(1)
    axs.plot(split_tracker_train, train_y, linewidth=0.1, color='r')
    axs.plot(split_tracker_test, test_y, linewidth=0.1, color='g')
    print(time.time() - start)
    plt.show()

class hasel_data_set(Dataset):
    def __init__(self, x, y):
        self.x = torch.from_numpy(x).cuda()
        self.y = torch.from_numpy(y).cuda()

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

def load_data(x, y, batch_size):
    data = hasel_data_set(x, y)
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    assert(len(data_loader.dataset.x) == len(data_loader.dataset.y))
    return data_loader

def add_net_visualization(writer, model, data):
    writer.add_graph(model, data)
    writer.flush()
    writer.close()

def create_log_file(q_reference_file):
    counter_file = open('./Models/counts', 'r+')
    count = str(int(counter_file.read()) + 1)
    tag = count + "_" + q_reference_file.split('.')[1][6:]
    counter_file.seek(0)
    counter_file.write(count)
    counter_file.close()
    os.makedirs('./Models/' + tag)
    return tag

#NOTE - this function will NOT work for LEMs, since they are custom
def log_model(tag, model, example_input, learning_rate, epochs, batch_size, look_back, weight_decay, momentum, train_ratio):
    model_summary = str(torchsummary.summary(model, example_input, verbose=0))
    with open('./Models/' + tag +'/Log_' + tag + '.txt', 'a') as convert_file:
        convert_file.write(
            f"Look back: {look_back}\nEpochs: {epochs}\nTrain ratio: {train_ratio}\nBatch size: {batch_size}\nLearning rate: {learning_rate}\nWeight Decay: {weight_decay}\nMomentum: {momentum}\n"
            + model_summary)
    # pass #Uncomment and comment all previous lines for LEM

def plot_results(prediction, train_y, test_y, tag):
    global split_tracker_train
    global split_tracker_test
    prediction = prediction.detach().cpu().numpy()
    assert(prediction.shape == test_y.shape)
    error = prediction - test_y
    NRMSE = np.sqrt(((error)**2).mean()) / test_y.mean()

    fig, axs = plt.subplots(1)

    #Train/test split
    axs.set_title('Dataset')
    total_time = np.arange(stop=len(master_V)*1e-2, step=1e-2)
    axs.plot(total_time, master_V, linewidth=0.05, color='grey')
    axs.plot(split_tracker_train, train_y, linestyle='None', marker='.', markersize=0.8, color='r', label='Training Data')
    axs.plot(split_tracker_test, test_y, linestyle='None', marker='.', markersize=0.8, color='g', label='Test Data')
    axs.set_xlabel('Time (s)')
    axs.set_ylabel('Position (normalized, mm)')
    axs.legend()

    fig2, axs2 = plt.subplots(1)

    #Prediction vs actual
    axs2.plot(total_time, master_V, linewidth=0.05, color='grey')
    axs2.plot(split_tracker_test, test_y, marker='.', markersize=0.9, linestyle='None', label='Actual')
    axs2.plot(split_tracker_test, prediction, marker='.', markersize=0.9, linestyle='None', label='Model')
    at = offsetbox.AnchoredText(f"Fit: {(1-NRMSE)*100:.2f}%", loc='lower left', prop=dict(size=8),
    frameon=True, bbox_to_anchor=(0., 1.), bbox_transform=axs2.transAxes)
    axs2.add_artist(at)
    axs2.set_title('Test')
    axs2.set_xlabel('Time (s)')
    axs2.set_ylabel('Position (normalized, mm)')
    axs2.legend()

    fig3, axs3 = plt.subplots(1)

    #Prediction vs actual with lines
    axs3.plot(split_tracker_test, test_y, linewidth=0.2, label='Actual')
    axs3.plot(split_tracker_test, prediction, linewidth=0.2, label='Model')
    at = offsetbox.AnchoredText(f"Fit: {(1-NRMSE)*100:.2f}%", loc='lower left', prop=dict(size=8),
    frameon=True, bbox_to_anchor=(0., 1.), bbox_transform=axs3.transAxes)
    axs3.add_artist(at)
    axs3.set_title('Test')
    axs3.set_xlabel('Time (s)')
    axs3.set_ylabel('Position (normalized, mm)')
    axs3.legend()

    fig.tight_layout()
    fig2.tight_layout()
    fig3.tight_layout()
    # fig.canvas.manager.full_screen_toggle() #Full screen matplotlib
    # plt.savefig('./Models/' + tag + '/Results.png') # To save output of plot
    plt.show()

#Define custom neural modelwork
class hasel_nn(nn.Module):
    def __init__(self, look_back, hidden_size, layers):
        super().__init__()
        # self.LSTM = LEMcuda(2,hidden_size,1.) #Uncomment and comment next line for LEM
        self.LSTM = nn.LSTM(2, hidden_size, layers, batch_first=True, dropout=0)
        self.output_activation = nn.Sequential(
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        # x = torch.transpose(x, 0, 1) #Uncomment for LEM
        output, _ = self.LSTM(x)
        final = self.output_activation(output[:,-1,:])
        # final = self.output_activation(output[-1:,:,:].squeeze()) #Uncomment and comment previous line for LEM
        return final

def train_model(training_data, model, epochs, loss_fn, optimizer, logger, track_validation_loss, test_x, test_y):
    threshold_counter = 0
    log_iteration = 0
    prediction, test_loss = (0, 0)
    stop = False
    for i in range(epochs):
        if (not stop):
            for batch, (X, Y) in enumerate(training_data):
                model.train()
                output = model(X)
                assert(output.shape == Y.shape)
                loss = loss_fn(output, Y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                logger.add_scalar('Loss (Iteration)', loss.item(), log_iteration)
                #Early stopping threshold (hardcoded for now)
                if (loss.item() < 8e-6):
                    threshold_counter += 1
                    if threshold_counter > 10:
                        stop = True
                else:
                    threshold_counter = 0
                log_iteration += 1
                if (batch % 99 == 0):
                    print(f"Loss: {loss.item()}, {(batch+1)*len(X)}/{len(training_data.dataset)}")
                if(i == 0 and batch == 0):
                        if (track_validation_loss):
                            prediction, test_loss = test_model(model, test_x, test_y, loss_fn)
                            logger.add_scalars('Loss (Epoch)', tag_scalar_dict={'Train loss': loss.item(), 'Test loss': test_loss}, global_step=i)
                            print(f"Epoch {i}, train loss: {loss.item()}, test loss: {test_loss}")
                        else:
                            logger.add_scalar('Loss (Epoch)', loss.item(), i)
                            print(f"Epoch {i}, train loss: {loss.item()}")
            if (track_validation_loss):
                prediction, test_loss = test_model(model, test_x, test_y, loss_fn)
                logger.add_scalars('Loss (Epoch)', tag_scalar_dict={'Train loss': loss.item(), 'Test loss': test_loss}, global_step=i+1)
                print(f"Epoch {i+1}, train loss: {loss.item()}, test loss: {test_loss}")
            else:
                logger.add_scalar('Loss (Epoch)', loss.item(), i+1)
                print(f"Epoch {i+1}, train loss: {loss.item()}")
    return loss.item(), prediction, test_loss

#Test single-step ahead forecasting
def test_model(model, test_x, test_y, loss_fn):
    start = time.time()
    model.eval()
    prediction = model(torch.from_numpy(test_x).cuda())
    loss = loss_fn(prediction, torch.tensor(test_y).cuda())
    print(f"loss: {loss.item()}")
    nrmse = np.sqrt(loss.item()) / test_y.mean()
    print(f"Fit: {(1-nrmse)*100:.2f}%")
    print(f"TIME: {time.time() - start}")
    return prediction, loss.item()

#Accepts numpy arrays for train and test set
def run_train_test(model, loss, optimizer,
                   train_x, train_y, test_x, test_y, train_ratio,
                   look_back, learning_rate, epochs, batch_size, weight_decay, momentum,
                   compare, tolerance,
                   use_cached, model_name,
                   tag, track_validation_loss):
    global writer

    #Load in training data as dataloader
    training_data = load_data(train_x, train_y, batch_size)

    writer = SummaryWriter(log_dir= 'Runs/' + tag)

    #Add neural network visualization
    example_input = next(iter(training_data))[0]
    # add_net_visualization(writer, model, example_input)
    print(f"example input: {example_input.shape}")
    #Log the parameters used for this network
    log_model(tag, model, example_input, learning_rate, epochs, batch_size, look_back, weight_decay, momentum, train_ratio)

    final_train_loss = 0

    # Load model state from existing file
    if (use_cached):
        model.load_state_dict(torch.load(model_name, map_location=torch.device('cpu')))

    #Train the network
    else:
        final_train_loss, prediction, final_test_loss = train_model(training_data, model, epochs, loss, optimizer, writer, track_validation_loss, test_x, test_y)
        torch.save(model.state_dict(), './Models/' + tag + '/Model')

    #Test the network & plot the results
    if (compare):
        if (not track_validation_loss or use_cached): #Run the test, either if we are using a cached model, or if we hadn't been tracking test losses when running the training model
            prediction, final_test_loss = test_model(model, test_x, test_y, loss) #So that we don't end up calling it twice if we've been tracking test loss
        writer.add_scalars('Train vs Test Loss', tag_scalar_dict={'Train loss': final_train_loss, 'Test loss': final_test_loss}, global_step=int(tag.split("_")[0]))
        plot_results(prediction, train_y, test_y, tag)


if __name__ == '__main__':
    #Hyperparameters
    look_back = 5
    learning_rate = 1e-3
    epochs = 1000 #For downsampled
    train_ratio = 0.8
    batch_size = 3000 #1s of data per batch
    weight_decay = 0 #1e-5 #If you want L2 regularization
    momentum = 0.9 #For SGD with momentum
    hidden_size = 32
    layers = 3

    #Load in dataset
    v_reference_file = './Data/Voltage.txt'
    q_reference_file = './Data/Position.txt'
    master_V, master_q = load_arrays(v_reference_file, q_reference_file)

    #Normalize dataset to 0, 1
    normalize = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)) #For -1 to 1, use: (2 * (x - x.min()) / (x.max() - x.min())) - 1
    master_V = normalize(master_V)
    master_q = normalize(master_q)

    #Divide into train and test set while creating time series
    start = time.time()
    train_x, train_y, test_x, test_y = create_time_series(master_V, master_q, look_back, train_ratio)
    # print(f"train_x: {train_x.shape}, train_y: {train_y.shape}")
    # print(f"test_x: {test_x.shape}, test_y: {test_y.shape}")
    print(f"time for creating time series: {time.time() - start}")
    # plot_train_test_split(split_tracker_train, split_tracker_test, train_y, test_y)

    #Initialize model, loss & optimizer
    model = hasel_nn(look_back, hidden_size, layers)
    model.to(torch.device("cuda:0"))
    loss = nn.MSELoss()
    # optimizer = torch.optim.SGD(params=model.parameters(), lr = learning_rate, weight_decay=weight_decay, momentum=momentum)
    optimizer = torch.optim.Adam(params=model.parameters(), lr = learning_rate, weight_decay=weight_decay)

    #Set flags
    compare = True #Flag to compare test result
    tolerance = 0.01
    use_cached = True #Flag to use a cached model or not
    model_name = 'Models/All_Data_LEM_32h/Model' #Model to load from cache
    track_validation = True

    #Create log directory
    tag = create_log_file(q_reference_file)

    signal.signal(signal.SIGINT, interrupt_handler)
    entire_displacement = master_q

    run_train_test(model, loss, optimizer,
                   train_x, train_y, test_x, test_y, train_ratio,
                   look_back, learning_rate, epochs, batch_size, weight_decay, momentum,
                   compare, tolerance,
                   use_cached, model_name,
                   tag, track_validation)