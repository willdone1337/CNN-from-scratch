import numpy as np
import matplotlib.pyplot as plt
import sys



np.random.seed(1)


from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

images, labels = (x_train[0:1000].reshape(1000,28*28) / 255, #transform train images to matrix 28*28 and devide to number of color in palitra
                  y_train[0:1000]) 


one_hot_labels = np.zeros((len(labels),10))



for i,l in enumerate(labels): 
    one_hot_labels[i][l] = 1
labels = one_hot_labels #for each labels create vector with index

test_images = x_test.reshape(len(x_test),28*28) / 255 # same transfrom with  train images
test_labels = np.zeros((len(y_test),10))
for i,l in enumerate(y_test):
    test_labels[i][l] = 1 

def tanh(x): #activation function
    return np.tanh(x) #  (2/(1+e**(-2x)))-1

def tanh2deriv(output): #derivative of tanh for defining change of weights values 
    return 1 - (output ** 2)

def softmax(x): #activation function where sum of all neurals ==1
    temp = np.exp(x)
    return temp / np.sum(temp, axis=1, keepdims=True)

alpha, iterations = (2, 300)
pixels_per_image, num_labels = (784, 10)
batch_size = 128

input_rows = 28
input_cols = 28

kernel_rows = 3
kernel_cols = 3
num_kernels = 16

hidden_size = ((input_rows - kernel_rows) *
               (input_cols - kernel_cols)) * num_kernels #quanity of layer between input and layer_2(result)

# weights_0_1 = 0.02*np.random.random((pixels_per_image,hidden_size))-0.01
kernels = 0.02*np.random.random((kernel_rows*kernel_cols,
                                 num_kernels))-0.01

weight_1_2 = 0.2*np.random.random((hidden_size,
                                    num_labels)) - 0.1


def get_image_section(layer,row_from, row_to, col_from, col_to): #matrix 3x3
    section = layer[:,row_from:row_to,col_from:col_to]
    return section.reshape(-1,1,row_to-row_from, col_to-col_from)

for j in range(iterations):
    correct_cnt = 0
    for i in range(int(len(images) / batch_size)):
        batch_start, batch_end=((i * batch_size),((i+1)*batch_size))
        layer_0 = images[batch_start:batch_end]
        layer_0 = layer_0.reshape(layer_0.shape[0],28,28)
        layer_0.shape

        sects=list()

        for row_start in range(layer_0.shape[1] - kernel_rows):
            for col_start in range(layer_0.shape[2] - kernel_cols):
                sect = get_image_section(layer_0,
                                         row_start,
                                         row_start + kernel_rows,
                                         col_start,
                                         col_start + kernel_cols)



                sects.append(sect)

        expanded_input=np.concatenate(sects,axis=1) #transform image sections to matrix(128,625,3,3)
        es=expanded_input.shape
        flattened_input=expanded_input.reshape(es[0]*es[1],-1) #aka conventional input



        kernel_output=flattened_input.dot(kernels)#conventional layer
        layer_1=tanh(kernel_output.reshape(es[0],-1))
        
        #drop out is metod for drop noises randomly divide layer_1 values and then for better correlation we must divide to part of off values, in our code it is 2 
        drop_out_mask=np.random.randint(2,size=layer_1.shape)
        layer_1*=drop_out_mask*2
        layer_2=softmax(np.dot(layer_1,weight_1_2))


        for k in range(batch_size):#compare label vecotr with our prediction verctor
            labelset = labels[batch_start + k:batch_start + k + 1]
            _inc = int(np.argmax(layer_2[k:k + 1]) ==
                       np.argmax(labelset))
            correct_cnt += _inc

        layer_2_delta = (labels[batch_start:batch_end] - layer_2) \ #difference between our predict and label
                        / (batch_size * layer_2.shape[0]) 
        
        #backpropogation
        #!-st stage is :multiply to error signals from layer_2_delta(errors) to weights where come from(weights_0_1) 
        #2-nd stage is :multiply layer_1_delta to input(flattened layer)
        layer_1_delta = layer_2_delta.dot(weight_1_2.T) * \ 
                        tanh2deriv(layer_1) 
        layer_1_delta *= drop_out_mask
        layer_1_delta_reshape = layer_1_delta.reshape(kernel_output.shape)
        k_update = flattened_input.T.dot(layer_1_delta_reshape)
        kernels -= alpha * k_update#gradient descent 
        
        weight_1_2 += alpha * layer_1.T.dot(layer_2_delta) #gradient descent(multiply delta of layer to input and alpha for better angle) 

    test_correct_count=0

    for i in range(len(test_images)):

        layer_0 = test_images[i:i + 1]
        layer_0 = layer_0.reshape(layer_0.shape[0], 28, 28)
        layer_0.shape

        sects = list()

        for row_start in range(layer_0.shape[1] - kernel_rows):
            for col_start in range(layer_0.shape[2] - kernel_cols):
                sect = get_image_section(layer_0,
                                         row_start,
                                         row_start + kernel_rows,
                                         col_start,
                                         col_start + kernel_cols)
                sects.append(sect)

        expanded_input = np.concatenate(sects, axis=1)
        es = expanded_input.shape
        flattened_input = expanded_input.reshape(es[0] * es[1], -1)

        kernel_output = flattened_input.dot(kernels)
        layer_1 = tanh(kernel_output.reshape(es[0], -1))
        layer_2 = np.dot(layer_1, weight_1_2)

        test_correct_count += int(np.argmax(layer_2) ==

                                  np.argmax(test_labels[i:i + 1]))

        # First 5 images
        if j%30==0 and i<5:
                fig, ax = plt.subplots()
                im=ax.imshow(layer_2,cmap="binary")
                ax.set_title(f"Post training accuracy the {i+1}/5 of MNIST Test images in {j} iteration")
                plt.show()
                fig, ax = plt.subplots()
                im = ax.imshow(test_labels[i:i+1],cmap="binary")
                ax.set_title(f"Label the {i+1}/5 of MNIST Test images in {j} iteration")
                plt.show()

    if (j % 1 == 0):
        sys.stdout.write("\n" + \
                         "Iteration:" + str(j) + \
                         " Test-Acc:" + str(test_correct_count/ float(len(test_images))) + \
                         " Train-Acc:" + str(correct_cnt / float(len(images))))

