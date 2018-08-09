'''
created : abib Duut
Date :  Mi 8. Aug 13:00:46 CEST 2018

'''

print('loading modules...')
"Modules imports"
import mxnet as mx
import numpy as np
import time
print('succesfully loaded modules!')


print('busy reading hyperparameters...')
'Hyperparameters'
maxsentence_len = 32
stride = 3
sentences = []
next_characters = []
epochs = 20
batch_size = 30
print('successfully read hyperparameters!')


print('busy loading data and preprocessing corpus...')
"reading data and preprocessing"
path= '/home/dlrig/Downloads/deep_nlp-master/tseliot.txt'
tsl_text =open(path).read().lower()


for i in range(0, len(tsl_text)- maxsentence_len, stride) :
    sentences.append(tsl_text[i: i + maxsentence_len])
    next_characters.append(tsl_text[i+maxsentence_len])
print("Nos. of Sequences :", len(sentences))
print('successfully loaded and preprocessed data!')


print('building and initializing model...')
'Building model'
lstm_model = mx.gluon.rnn.SequentialRNNCell()
with lstm_model.name_scope():
    lstm_model.add(mx.gluon.rnn.LSTMCell(30, 10))
    lstm_model.add(mx.gluon.rnn.LSTMCell(20))
    lstm_model.add(mx.gluon.nn.Dense(5, flatten=False))
states = lstm_model
lstm_model.initialize()

lstm_model_trainer = mx.gluon.Trainer(lstm_model.collect_params(),'rmsprop', {'learning_rate':args_lr, 'momentum': 0, 'wd':0})
print('succesfully built model!')



