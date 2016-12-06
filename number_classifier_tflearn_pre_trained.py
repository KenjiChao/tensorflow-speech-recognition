#!/usr/bin/env PYTHONIOENCODING="utf-8" python
import tflearn
import pyaudio
import speech_data
import numpy
import os
import time
from memory_profiler import profile

# Simple spoken digit recognition demo, with 98% accuracy in under a minute

# Training Step: 544  | total loss: 0.15866
# | Adam | epoch: 034 | loss: 0.15866 - acc: 0.9818 -- iter: 0000/1000

def main():
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
    totalTime = 0
    totalAcc = 0
    numTimes = 100
    for i in range(numTimes):
        t = time.time()
        result = model.predict(X)
        print ("-------------")

        result = numpy.array([numpy.argmax(r) for r in result])
        answers = numpy.array([numpy.argmax(answer) for answer in Y])

        print (i, ">>>", (result == answers).sum() / float(len(answers)), "time: ", time.time() - t)  
        totalAcc = totalAcc + (result == answers).sum() / float(len(answers))
        totalTime = totalTime + time.time() - t

    print("Avg. Acc. = ", totalAcc / numTimes)
    print("Avg. time = ", totalTime / numTimes)

if __name__ == '__main__':
    main()