import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from datetime import datetime
import os
import torchsummary
import matplotlib.offsetbox as offsetbox
import time
import signal

import sys

#Forward recurrent neural network (one-step ahead AND multi-step ahead prediction):
# Mapping V (past) and Q (past) to Q(next): { q_{t-k} ... q_t, V_{t+1-k} ... V_{t+1} } -> q_{t+1}
#NOTE: if not using GPU, disable .cuda() everywhere in code

global model
global test_v
global test_q
global test_y
global loss
global tag
global writer
global test_input_voltage
global entire_displacement
global test_q_feedback

#Function to catch SIG INT midway through training
def interrupt_handler(sig, frame):
    global test_y
    prediction, test_y, final_test_loss, prediction_feedback, final_test_loss_feedback = test_model(model, test_v, test_q, test_y, loss, test_q_feedback) #So that we don't end up calling it twice if we've been tracking test loss
    writer.add_scalars('Train vs Test Loss', tag_scalar_dict={'Test loss': final_test_loss, 'Test loss feedback': final_test_loss_feedback}, global_step=int(tag.split("_")[0]))
    plot_results(prediction, prediction_feedback, test_y, tolerance, test_input_voltage, entire_displacement, train_ratio)
    torch.save(model.state_dict(), './Models/' + tag + '/Model')
    sys.exit()

def load_arrays(V_file_name, q_file_name):
    V_original = np.loadtxt(V_file_name)
    q_original = np.loadtxt(q_file_name)
    assert(len(V_original) == len(q_original))
    return V_original, q_original

#Create look back windows for input to neural network. Split dataset randomly into train and test points based on specified ratio
def create_time_series(V_original, q_original, look_back):
    x = []
    y = []
    for i in range(len(V_original)):
        if (i >= look_back):
            x.append([V_original[i-look_back+1:i+1], q_original[i-look_back:i]])
            y.append(q_original[i])
    assert(len(x) == len(y))
    return np.transpose(np.array(x), (0, 2, 1)), np.expand_dims(np.array(y), 1)

#Create look back time series for test set
def create_test_time_series(V_original, q_original, look_back):
    v = []
    q = []
    y = []
    q_feedback = []
    for i in range(len(V_original)):
        if (i >= look_back):
            v.append(V_original[i-look_back+1:i+1])
            q.append(q_original[i-look_back:i])
            if (i == look_back):
                q_feedback.append(q_original[i-look_back:i])
            y.append(q_original[i])
    assert(len(v) == len(y))
    return np.expand_dims(np.array(v), 1), np.expand_dims(np.array(q), 1), np.expand_dims(np.array(y), 1), np.array(q_feedback)

def split_dataset(master_x, master_y, train_ratio):
    assert(len(master_x) == len(master_y))
    train_x = master_x[:round(train_ratio*len(master_x))]
    train_y = master_y[:round(train_ratio*len(master_y))]
    test_x = master_x[round(train_ratio*len(master_x)):]
    test_y = master_y[round(train_ratio*len(master_y)):]
    return train_x, train_y, test_x, test_y

#Tmporary custom split for chirp signal 87.5s-134s train for now
def split_dataset_temp(master_x, master_y, train_ratio):
    assert(len(master_x) == len(master_y))
    train_x = master_x[np.r_[0:8750,13400:27150]]
    train_y = master_y[np.r_[0:8750,13400:27150]]
    test_x = master_x[8750:13400]
    test_y = master_y[8750:13400]
    return train_x, train_y, test_x, test_y

def plot_dataset(x, y, l1, l2):
    time = np.arange(len(x))
    fig, axs = plt.subplots(2)
    axs[0].plot(time, x, label=l1, linewidth=0.1, marker='.', markersize=0.35, linestyle='None')
    axs[1].plot(time, y, label=l2, linewidth=0.1, marker='.', markersize=0.35, linestyle='None')
    axs[0].legend()
    axs[0].legend()
    axs[1].legend()

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
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=True) #Test shuffle here
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
    print(f"TAG: {tag}")
    counter_file.seek(0)
    counter_file.write(count)
    counter_file.close()
    os.makedirs('./Models/' + tag)
    return tag

def log_model(tag, model, example_input, learning_rate, epochs, batch_size, look_back, weight_decay, momentum, train_ratio):
    model_summary = str(torchsummary.summary(model, example_input, verbose=0))
    with open('./Models/' + tag +'/Log_' + tag + '.txt', 'a') as convert_file:
        convert_file.write(
            f"Look back: {look_back}\nEpochs: {epochs}\nTrain ratio: {train_ratio}\nBatch size: {batch_size}\nLearning rate: {learning_rate}\nWeight Decay: {weight_decay}\nMomentum: {momentum}\n"
            + model_summary)

def plot_results(prediction, prediction_feedback, test_y, tolerance, test_input_voltage, entire_displacement, train_ratio):
    assert(prediction.shape == test_y.shape and prediction_feedback.shape == test_y.shape)
    # tol = tolerance * prediction
    error = prediction - test_y
    error_feedback = prediction_feedback - test_y
    NRMSE = np.sqrt(((error)**2).mean()) / test_y.mean()
    NRMSE_feedback = np.sqrt(((error_feedback)**2).mean()) / test_y.mean()

    fig, axs = plt.subplots(4)

    #Train/test split
    total_input_time = len(entire_displacement)*1e-4
    total_time = np.linspace(0, total_input_time, len(entire_displacement)) #Hardcoded 1e-4 as sample time for now
    axs[0].set_title('Original Data')
    axs[0].plot(total_time, entire_displacement, label='q')
    train_boundary = train_ratio * total_input_time
    axs[0].fill_between([0, train_boundary], 0, 1, alpha=0.2, label='Train Set')
    axs[0].fill_between([train_boundary, len(entire_displacement)*1e-4], 0, 1, alpha=0.2, label='Test Set')
    axs[0].legend()

    #Prediction vs actual
    x = np.linspace(0, len(test_y)*1e-4, len(test_y)) #Hardcoded 1e-4 as sample time for now
    axs[1].plot(x, prediction, label='Predicted Displacement')
    axs[1].plot(x, test_y, label='Actual Displacement')
    at = offsetbox.AnchoredText(f"Fit: {(1-NRMSE)*100:.2f}%", loc='lower left', prop=dict(size=8), 
    frameon=True, bbox_to_anchor=(0., 1.), bbox_transform=axs[1].transAxes) 
    axs[1].add_artist(at)
    axs[1].set_title('Test run')
    axs[1].legend()

    #Prediction feedback vs actual
    axs[2].plot(x, prediction_feedback, label='Predicted Displacement (feedback)')
    axs[2].plot(x, test_y, label='Actual Displacement')
    at = offsetbox.AnchoredText(f"Fit: {(1-NRMSE_feedback)*100:.2f}%", loc='lower left', prop=dict(size=8), 
    frameon=True, bbox_to_anchor=(0., 1.), bbox_transform=axs[2].transAxes) 
    axs[2].add_artist(at)
    axs[2].set_title('Test run (feedback)')
    axs[2].legend()

    #Input voltage
    x_for_voltage = np.linspace(0, len(test_input_voltage)*1e-4, len(test_input_voltage)) #Hardcoded 1e-4 for now
    axs[3].plot(x_for_voltage, test_input_voltage, label='Voltage')
    axs[3].legend()
    fig.tight_layout()
    fig.canvas.manager.full_screen_toggle()
    plt.show()

#Define custom neural modelwork
class hasel_nn(nn.Module):
    def __init__(self, look_back, hidden_size, layers):
        super().__init__()
        self.LSTM = nn.LSTM(2, hidden_size, layers, dtype=float, batch_first=True)
        self.output_activation = nn.Sequential(
            nn.Linear(hidden_size, 1, dtype=float),
        )

    def forward(self, x):
        output, _ = self.LSTM(x)
        final = self.output_activation(output[:,-1,:])
        return final

def train_model(training_data, model, epochs, loss_fn, optimizer, logger, track_validation_loss, test_v, test_q, test_y, test_q_feedback):
    model.train()
    threshold_counter = 0
    log_iteration = 0
    prediction, test_loss, prediction_feedback, test_loss_feedback = (0, 0, 0, 0)
    stop = False
    for i in range(epochs):
        if (not stop):
            for batch, (X, Y) in enumerate(training_data):
                output = model(X)
                assert(output.shape == Y.shape)
                loss = loss_fn(output, Y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                logger.add_scalar('Loss (Iteration)', loss.item(), log_iteration)
                #Early stopping threshold (hardcoded for now)
                if (loss.item() < 4e-5):
                    threshold_counter += 1
                    if threshold_counter > 5:
                        stop = True
                else:
                    threshold_counter = 0
                log_iteration += 1
                if (batch % 99 == 0):
                    print(f"Loss: {loss.item()}, {(batch+1)*len(X)}/{len(training_data.dataset)}")
                if(i == 0 and batch == 0):
                        if (track_validation_loss):
                            prediction, test_y, test_loss, prediction_feedback, test_loss_feedback = test_model(model, test_v, test_q, test_y, loss_fn, test_q_feedback)
                            logger.add_scalars('Loss (Epoch)', tag_scalar_dict={'Train loss': loss.item(), 'Test loss': test_loss, 'Test loss feedback': test_loss_feedback}, global_step=i)
                            print(f"Epoch {i}, train loss: {loss.item()}, test loss: {test_loss}, feedback test loss: {test_loss_feedback}")
                        else:
                            logger.add_scalar('Loss (Epoch)', loss.item(), i)
                            print(f"Epoch {i}, train loss: {loss.item()}")
            if (track_validation_loss):
                prediction, test_y, test_loss, prediction_feedback, test_loss_feedback = test_model(model, test_v, test_q, test_y, loss_fn, test_q_feedback)
                logger.add_scalars('Loss (Epoch)', tag_scalar_dict={'Train loss': loss.item(), 'Test loss': test_loss, 'Test loss feedback': test_loss_feedback}, global_step=i+1)
                print(f"Epoch {i+1}, train loss: {loss.item()}, test loss: {test_loss}, feedback test loss: {test_loss_feedback}")
            else:
                logger.add_scalar('Loss (Epoch)', loss.item(), i+1)
                print(f"Epoch {i+1}, train loss: {loss.item()}")
    return loss.item(), prediction, test_y, test_loss, prediction_feedback, test_loss_feedback

#Test model for both single-step and multi-step ahead forecasting
def test_model(model, test_v, test_q, test_y, loss_fn, test_q_feedback):
    start = time.time()
    model.eval()
    y_hat = 0
    y_hat_feedback = 0
    for i, v in enumerate(test_v):
        input_data = np.transpose(np.stack((v, test_q[i]), 1), (0, 2, 1))
        input_data_feedback = np.transpose(np.stack((v, test_q_feedback), 1), (0, 2, 1))
        prediction = np.array([[model(torch.from_numpy(input_data).cuda()).item()]])
        prediction_feedback = np.array([[model(torch.from_numpy(input_data_feedback).cuda()).item()]])
        if (i == 0):
            y_hat = np.copy(prediction)
            y_hat_feedback = np.copy(prediction_feedback)
        else:
            y_hat = np.append(y_hat, prediction, 0)
            y_hat_feedback = np.append(y_hat_feedback, prediction_feedback, 0)
        test_q_feedback = np.append(test_q_feedback[:,1:], prediction_feedback, 1)

    y_hat_tensor = torch.tensor(y_hat).cuda()
    y_hat_feedback_tensor = torch.tensor(y_hat_feedback).cuda()
    test_y_tensor = torch.tensor(test_y).cuda()

    loss = loss_fn(y_hat_tensor, test_y_tensor)
    loss_feedback = loss_fn(y_hat_feedback_tensor, test_y_tensor)
    print(f"loss: {loss.item()}")
    print(f"loss_feedback: {loss_feedback.item()}")
    nrmse = np.sqrt(loss.item()) / test_y.mean()
    nrmse_feedback = np.sqrt(loss_feedback.item()) / test_y.mean()
    print(f"Fit: {(1-nrmse)*100:.2f}%")
    print(f"Fit feedback: {(1-nrmse_feedback)*100:.2f}%")
    print(f"TIME: {time.time() - start}")
    return y_hat, test_y, loss.item(), y_hat_feedback, loss_feedback.item()

#Accepts numpy arrays for train and test set
def run_train_test(model, loss, optimizer,
                   train_x, train_y, test_v, test_q, test_y, test_q_feedback, train_ratio, 
                   look_back, learning_rate, epochs, batch_size, weight_decay, momentum,
                   test_input_voltage, entire_displacement, compare, tolerance, 
                   use_cached, model_name, 
                   tag, track_validation_loss):
    global writer
    
    #Load in training data as dataloader
    training_data = load_data(train_x, train_y, batch_size)

    writer = SummaryWriter(log_dir= 'Runs/' + tag)

    #Add neural network visualization
    example_input = next(iter(training_data))[0]
    add_net_visualization(writer, model, example_input)
    print(f"example input: {example_input.shape}")
    #Log the parameters used for this network
    log_model(tag, model, example_input, learning_rate, epochs, batch_size, look_back, weight_decay, momentum, train_ratio)

    final_train_loss = 0
    
    # Load model state from existing file
    if (use_cached):
        model.load_state_dict(torch.load(model_name, map_location=torch.device('cpu')))
        
    #Train the network
    else:
        final_train_loss, prediction, test_y, final_test_loss, prediction_feedback, final_test_loss_feedback = train_model(training_data, model, epochs, loss, optimizer, writer, track_validation_loss, test_v, test_q, test_y, test_q_feedback)
        torch.save(model.state_dict(), './Models/' + tag + '/Model_' + tag)

    #Test the network on the training set itself & plot the results
    if (compare):
        if (not track_validation_loss or use_cached): #Run the test, either if we are using a cached model, or if we hadn't been tracking test losses when running the training model
            prediction, test_y, final_test_loss, prediction_feedback, final_test_loss_feedback = test_model(model, test_v, test_q, test_y, loss, test_q_feedback) #So that we don't end up calling it twice if we've been tracking test loss
        writer.add_scalars('Train vs Test Loss', tag_scalar_dict={'Train loss': final_train_loss, 'Test loss': final_test_loss, 'Test loss feedback': final_test_loss_feedback}, global_step=int(tag.split("_")[0]))
        plot_results(prediction, prediction_feedback, test_y, tolerance, test_input_voltage, entire_displacement, train_ratio)

        
if __name__ == '__main__':
    #Hyperparameters
    look_back = 800
    learning_rate = 0.1
    epochs = 320 #For downsampled
    train_ratio = 0.9
    batch_size = 40000 #1s of data per batch
    weight_decay = 0 #1e-5 #If you want L2 regularization
    momentum = 0.9 #For SGD with momentum
    hidden_size = 16 #Size of RNN layer
    layers = 1 #Number of RNN layers

    #Load in dataset
    v_reference_file = './Data/Sine_Voltage_100_DS.txt'
    q_reference_file = './Data/Sine_100_DS.txt'
    master_V, master_q = load_arrays(v_reference_file, q_reference_file) 
    # plot_dataset(master_V, master_q, 'Voltage', 'Displacement')

    #Normalize dataset to 0, 1
    normalize = lambda x: (x - x.min()) / (x.max() - x.min()) #For -1 to 1, use: (2 * (x - x.min()) / (x.max() - x.min())) - 1
    master_V = normalize(master_V)
    master_q = normalize(master_q)
    # plot_dataset(master_V, master_q, "Master V", "Master q")

    #Divide into train and test set
    train_V, train_q, test_V, test_q_orig = split_dataset(master_V, master_q, train_ratio) #For all experiments except chirp
    # train_V, train_q, test_V, test_q_orig = split_dataset_temp(master_V, master_q, train_ratio) #For chirp experiment
    print(f"{train_V.shape}, {train_q.shape}, {test_V.shape}, {test_q_orig.shape}")
    del master_V
    plot_dataset(train_V, train_q, "Train V", "Train q")
    plot_dataset(test_V, test_q_orig, "Test V", "Test q")
    plt.show()

    #Get time series for train and test set
    train_x, train_y = create_time_series(train_V, train_q, look_back)
    test_v, test_q, test_y, test_q_feedback = create_test_time_series(test_V, test_q_orig, look_back)
    print(f"train_x: {train_x.shape}, train_y: {train_y.shape}")
    print(f"test_v: {test_v.shape}, test_q: {test_q.shape}, test_y: {test_y.shape}")

    #Initialize model, loss & optimizer
    model = hasel_nn(look_back, hidden_size, layers)
    model.to(torch.device("cuda:0"))
    loss = nn.MSELoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr = learning_rate, weight_decay=weight_decay, momentum=momentum)
    # optimizer = torch.optim.Adam(params=model.parameters(), lr = learning_rate, weight_decay=weight_decay)
    
    #Set flags
    compare = True #Flag to compare test result
    tolerance = 0.01
    use_cached = True #Flag to use a cached model or not
    model_name = 'Models/BEST_MULTI_RNN_SINE_100_DS/Model (1)' #Model to load from cache
    track_validation = True

    #Create log directory
    tag = create_log_file(q_reference_file)

    signal.signal(signal.SIGINT, interrupt_handler)
    test_input_voltage = test_V
    entire_displacement = master_q

    run_train_test(model, loss, optimizer,
                   train_x, train_y, test_v, test_q, test_y, test_q_feedback, train_ratio,
                   look_back, learning_rate, epochs, batch_size, weight_decay, momentum,
                   test_V, master_q, compare, tolerance, 
                   use_cached, model_name,
                   tag, track_validation)