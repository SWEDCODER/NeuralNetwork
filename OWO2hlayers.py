import numpy as np
import sys
from math import *
from PIL import Image
'''
np.set_printoptions(threshold=sys.maxsize)
'''
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    return x * (1 - x)

'''
training_inputss = []
for i in range(1,3):
    file = open("cropped/output"+str(i)+".txt", "r")
    training_inputss.append(eval(file.read()))

    training_inputs = np.asarray(training_inputss)
    print(training_inputs)
    print("\n")
'''
'''
for i,l in enumerate(list(str(file.read()).replace(",", ""))):
    ooga = int((list(str(file.read()).replace(",", "")))[i])
print(ooga)
'''






training_inputs = []
test_data = []
for i in range(1,301):
    img = Image.open('cropped/3.'+str(i)+'.png').convert('L')

    np_img=np.array(img)
    np_img = ~np_img
    np_img[np_img > 0] = 1
    np_img = np_img.flatten(order='C')
    #TODO: TEST DATA
    if i != 301:
        training_inputs.append(np_img)
    else:
        test_data.append(np_img)


training_inputs = np.asarray(training_inputs)


print(training_inputs)










training_outputs = np.array([[0,
0.3,
0.3,
0.3,
0.3,
0.3,
0.3,
0.3,
0.3,
0.3,
0.3,
0.3,
0.3,
0.3,
0.3,
0.3,
0.3,
0.3,
0.3,
0.3,
0.3,
0.3,
0.3,
0.3,
0.3,
0.3,
0.3,
0.3,
0.3,
0.3,
0.3,
0.3,
0.3,
0.3,
0.3,
0.3,
0.3,
0.3,
0.3,
0.3,
0.3,
0.3,
0.3,
0.3,
0.3,
0.3,
0.3,
0.3,
0.3,
0.3,
0.3,
0.3,
0.3,
0.3,
0.3,
0.3,
0.3,
0.3,
0.3,
0.3,
0.3,
0.3,
0.3,
0.3,
0.3,
0.3,
0.3,
0.3,
0.3,
0.3,
0.3,
0.3,
0.3,
0.3,
0.3,
0.3,
0.3,
0.3,
0.3,
0.3,
0.3,
0.3,
0.3,
0.3,
0.3,
0.3,
0.3,
0.3,
0.3,
0.3,
0.3,
0.3,
0.3,
0.3,
0.3,
0.3,
0.3,
0.3,
0.3,
0.3,
0.4,
0.4,
0.4,
0.4,
0.4,
0.4,
0.4,
0.4,
0.4,
0.4,
0.4,
0.4,
0.4,
0.4,
0.4,
0.4,
0.4,
0.4,
0.4,
0.4,
0.4,
0.4,
0.4,
0.4,
0.4,
0.4,
0.4,
0.4,
0.4,
0.4,
0.4,
0.4,
0.4,
0.4,
0.4,
0.4,
0.4,
0.4,
0.4,
0.4,
0.4,
0.4,
0.4,
0.4,
0.4,
0.4,
0.4,
0.4,
0.4,
0.4,
0.4,
0.4,
0.4,
0.4,
0.4,
0.4,
0.4,
0.4,
0.4,
0.4,
0.4,
0.4,
0.4,
0.4,
0.4,
0.4,
0.4,
0.4,
0.4,
0.4,
0.4,
0.4,
0.4,
0.4,
0.4,
0.4,
0.4,
0.4,
0.4,
0.4,
0.4,
0.4,
0.4,
0.4,
0.4,
0.4,
0.4,
0.4,
0.4,
0.4,
0.4,
0.4,
0.4,
0.4,
0.4,
0.4,
0.4,
0.4,
0.4,
0.4,
0.5,
0.5,
0.5,
0.5,
0.5,
0.5,
0.5,
0.5,
0.5,
0.5,
0.5,
0.5,
0.5,
0.5,
0.5,
0.5,
0.5,
0.5,
0.5,
0.5,
0.5,
0.5,
0.5,
0.5,
0.5,
0.5,
0.5,
0.5,
0.5,
0.5,
0.5,
0.5,
0.5,
0.5,
0.5,
0.5,
0.5,
0.5,
0.5,
0.5,
0.5,
0.5,
0.5,
0.5,
0.5,
0.5,
0.5,
0.5,
0.5,
0.5,
0.5,
0.5,
0.5,
0.5,
0.5,
0.5,
0.5,
0.5,
0.5,
0.5,
0.5,
0.5,
0.5,
0.5,
0.5,
0.5,
0.5,
0.5,
0.5,
0.5,
0.5,
0.5,
0.5,
0.5,
0.5,
0.5,
0.5,
0.5,
0.5,
0.5,
0.5,
0.5,
0.5,
0.5,
0.5,
0.5,
0.5,
0.5,
0.5,
0.5,
0.5,
0.5,
0.5,
0.5,
0.5,
0.5,
0.5,
0.5,
0.5,
0.5]]).T

np.random.seed(1)

def makeWeights(brain):
    weights=[]
    for p,l in enumerate(brain[1:]):
        weight=2*np.random.random((brain[p],l)) - 1
        weights.append(weight)
    return weights
weights = makeWeights([1024,4,4,4,1])
'''
weights = []
weights.append(2*np.random.random((3,4)) - 1)
weights.append(2*np.random.random((4,1)) - 1)
'''
print('Random starting weights: ')

print(weights)

for iteration in range(5000):
    layer0 = training_inputs
    layer1 = sigmoid(np.dot(layer0, weights[0]))
    layer2 = sigmoid(np.dot(layer1, weights[1]))
    layer3 = sigmoid(np.dot(layer2, weights[2]))
    layer4 = sigmoid(np.dot(layer3, weights[3]))

    layer4_error = training_outputs - layer4
    layer4_adj = layer4_error*sigmoid_deriv(layer4)
    layer3_error = layer4_adj.dot(weights[3].T)
    if(iteration % 1) == 0:
        print("Error: " + str(np.mean(np.abs(layer3_error))) + " Iteration: " + str(iteration))
    layer3_adj = layer3_error*sigmoid_deriv(layer3)
    layer2_error = layer3_adj.dot(weights[2].T)
    layer2_adj = layer2_error*sigmoid_deriv(layer2)
    layer1_error = layer2_adj.dot(weights[1].T)
    layer1_adj = layer1_error*sigmoid_deriv(layer1)

    weights[3] += layer3.T.dot(layer4_adj)
    weights[2] += layer2.T.dot(layer3_adj)
    weights[1] += layer1.T.dot(layer2_adj)
    weights[0] += layer0.T.dot(layer1_adj)

np.asarray(weights)
weightsfile0 = open("weights0.txt", "w+")
np.savetxt(weightsfile0, weights[0] , fmt="%i",delimiter=',', newline=',')
weightsfile0.close()
weightsfile1 = open("weights1.txt", "w+")
np.savetxt(weightsfile1, weights[1] , fmt="%i",delimiter=',', newline=',')
weightsfile1.close()
weightsfile2 = open("weights2.txt", "w+")
np.savetxt(weightsfile2, weights[2] , fmt="%i",delimiter=',', newline=',')
weightsfile2.close()
weightsfile3 = open("weights3.txt", "w+")
np.savetxt(weightsfile3, weights[3] , fmt="%i",delimiter=',', newline=',')
weightsfile3.close()
print("Output After Training : ")
print(layer4)
def die(layer4):
    for values in layer4:
        for val in values:
            rooo = round(val)
            np.asarray(rooo)
            return rooo
print(die(layer4))





print("Test data results : ")
#TODO: TEST DATA
'''
layer0=np.array([test_data])
layer1 = sigmoid(np.dot(layer0, weights[0]))
layer2 = sigmoid(np.dot(layer1, weights[1]))
layer3 = sigmoid(np.dot(layer2, weights[2]))
layer4 = sigmoid(np.dot(layer3, weights[3]))
print(layer4)
'''
