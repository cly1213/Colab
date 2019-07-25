import glob
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf

from model import RNN

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

TrainingSet_ratio = 0.8
Data_racial = 0.995
Visualize = True #True or False
window_size = 256
batch_size = 128

def read_dataset():
    files = os.path.join('dataset','Learning_set-Bearing2_1-acc.csv')
    total_sequence = []
    file_name = os.path.basename(files)
    print('Now processing ', file_name, '...')
    
    #write your code here
    df = pd.read_csv(files, index_col=False)
    
    if Visualize:
        visualization(df['x'], file_name)
    
    df = df.drop(['datetime'], 1)[int(len(df)*Data_racial):]

    for time_step in range(0, len(df) - window_size, 1):
        sequence_data = df[time_step:time_step+window_size].values
        total_sequence.append(sequence_data)

    print(np.array(total_sequence).shape)
    return np.array(total_sequence)

def visualization(data, file_name):
    #write your code here
    plt.title('Vibration of ' + file_name)
    plt.xlabel('Time')
    plt.ylabel('Accelerate')
    plt.plot(data)
    plt.show()
    #print()

def model_testing(sess, test_data):
    loss_sum = []
    #write your code here
    print('Testing Loss: ', np.average(loss_sum))

if __name__ == '__main__':
    
    all_data = read_dataset()
    
    #write your code here
    training_data, testing_data = np.split(all_data, [int(TrainingSet_ratio * len(all_data))])
    
    model = RNN(batch_size, sequence_len= 255, num_dim = 2)
    tf.set_random_seed(99)
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    for batch_idx in range(len(all_data) // batch_size):
        batch_input = training_data[batch_idx:batch_idx+batch_size]
        _, loss, pred = sess.run([model.train_network, model.cost, model.predictions], feed_dict={model.inputs: batch_input[:,:-1], model.labels: batch_input[:,-1], model.is_training: True,model.lr: 0.001})
        
        if batch_idx % 11 ==0:
            print('Iteration', batch_idx, 'Traing_Loss:', loss)
