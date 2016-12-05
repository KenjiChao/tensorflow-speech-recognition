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

# Classification
tflearn.init_graph(num_cores=8, gpu_memory_fraction=0.5)

net = tflearn.input_data(shape=[None, 8192])
net = tflearn.fully_connected(net, 64, name='f1')
net = tflearn.dropout(net, 0.5, name='dp')
net = tflearn.fully_connected(net, number_classes, activation='softmax', name='f2')
net = tflearn.regression(net, optimizer='sgd', loss='categorical_crossentropy')

model = tflearn.DNN(net)
model.load('pre-trained/model.tflearn.sgd_trained')

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
