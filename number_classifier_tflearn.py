#!/usr/bin/env PYTHONIOENCODING="utf-8" python
import tflearn
import pyaudio
import speech_data
import numpy
import os

# Simple spoken digit recognition demo, with 98% accuracy in under a minute

# Training Step: 544  | total loss: 0.15866
# | Adam | epoch: 034 | loss: 0.15866 - acc: 0.9818 -- iter: 0000/1000

batch=speech_data.wave_batch_generator(10000,target=speech_data.Target.digits)
X,Y=next(batch)
Y = [numpy.hstack([y, numpy.array([0, 0, 0, 0, 0, 0])]) for y in Y]
# Y = map(lambda a: , Y)
print (type(Y))
# print (np.hstack([Y[0], np.array([0, 0, 0, 0, 0, 0])]))
number_classes=16 # Digits

def save_layer_parameters(model, folder):
    if not os.path.isdir(folder):
        os.makedirs(folder)
    
    print ("Saving layer parameters to folder: " + folder)

    weight_f1_var = tflearn.variables.get_layer_variables_by_name('f1')
    weight_f1 = model.get_weights(weight_f1_var[0])
    # print (weight_f1)
    print (weight_f1.shape)

    with open(folder + '/weight_f1.npy', 'wb') as f:
        numpy.save(f, weight_f1)

    with model.session.as_default():
        bias_f1 = tflearn.variables.get_value(weight_f1_var[1])
        # print (bias_f1)
        print (bias_f1.shape)

    with open(folder + '/bias_f1.npy', 'wb') as f:
        numpy.save(f, bias_f1)

    weight_f2_var = tflearn.variables.get_layer_variables_by_name('f2')
    weight_f2 = model.get_weights(weight_f2_var[0])
    # print (weight_f2)
    print (weight_f2.shape)

    with open(folder + '/weight_f2.npy', 'wb') as f:
        numpy.save(f, weight_f2)

    with model.session.as_default():
        bias_f2 = tflearn.variables.get_value(weight_f2_var[1])
        # print (bias_f2)
        print (bias_f2.shape)

    with open(folder + '/bias_f2.npy', 'wb') as f:
        numpy.save(f, bias_f2)

# Classification
tflearn.init_graph(num_cores=8, gpu_memory_fraction=0.5)

net = tflearn.input_data(shape=[None, 8192])
net = tflearn.fully_connected(net, 64, name='f1')
net = tflearn.dropout(net, 0.5, name='dp')
net = tflearn.fully_connected(net, number_classes, activation='softmax', name='f2')
net = tflearn.regression(net, optimizer='sgd', loss='categorical_crossentropy')

model = tflearn.DNN(net)

save_layer_parameters(model, 'sgd_init')

model.fit(X, Y,n_epoch=175,show_metric=True,snapshot_step=100)
# Overfitting okay for now

demo_file = "5_Vicki_260.wav"
demo=speech_data.load_wav_file(speech_data.path + demo_file)
with open('5_Vicki_260.npy', 'wb') as f:
    numpy.save(f, numpy.expand_dims(numpy.array(demo), axis=0))
print ("DEMO")
# print (demo)
print (len(demo))
result=model.predict([demo])
print ("-------------")
# print (result)
print (len(result))
result=numpy.argmax(result)
print("predicted digit for %s : result = %d "%(demo_file,result))

save_layer_parameters(model, 'sgd_trained')
model.save('model.tflearn.sgd_trained')
