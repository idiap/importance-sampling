# Examples

In the [repository][github_is], we have included examples that use importance
sampling to train neural networks. They are all python scripts that can be run
without arguments but also accept command line arguments which can be seen by
passing `-h` or `--help`.

## MNIST MLP

In this example we train a fully connected network to classify MNIST digits on
the CPU.  The example uses the `ImportanceTraining` class with a presampling
factor of 5. If the example is run with the `--uniform` argument it uses plain
Keras to train.

Although we chose to show the difference in terms of epochs for this example we
can see that importance sampling achieves better training loss and
classification accuracy even with respect to wall clock time.

<div class="fig col-2">
<img src="../img/mnist_mlp_train_loss.png" alt="Training Loss">
<img src="../img/mnist_mlp_test_acc.png" alt="Test accuracy">
<span>Results of training a fully connected neural network on MNIST with and
without importance sampling (orange and blue respectively).</span>
</div>

Here follows the terminal output

<pre style="height: 300px; overflow-y: scroll;"><code class="bash">$ python examples/mnist_mlp.py --uniform
Using TensorFlow backend.
60000 train samples
10000 test samples
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_1 (Dense)              (None, 512)               401920    
_________________________________________________________________
dense_2 (Dense)              (None, 512)               262656    
_________________________________________________________________
dense_3 (Dense)              (None, 10)                5130      
_________________________________________________________________
activation_1 (Activation)    (None, 10)                0         
=================================================================
Total params: 669,706
Trainable params: 669,706
Non-trainable params: 0
_________________________________________________________________
Train on 60000 samples, validate on 10000 samples
Epoch 1/10
60000/60000 [==============================] - 4s 61us/step - loss: 0.2284 - acc: 0.9326 - val_loss: 0.1255 - val_acc: 0.9643
Epoch 2/10
60000/60000 [==============================] - 3s 56us/step - loss: 0.0964 - acc: 0.9734 - val_loss: 0.1005 - val_acc: 0.9725
Epoch 3/10
60000/60000 [==============================] - 3s 56us/step - loss: 0.0692 - acc: 0.9826 - val_loss: 0.0862 - val_acc: 0.9785
Epoch 4/10
60000/60000 [==============================] - 3s 56us/step - loss: 0.0563 - acc: 0.9867 - val_loss: 0.0990 - val_acc: 0.9780
Epoch 5/10
60000/60000 [==============================] - 3s 56us/step - loss: 0.0473 - acc: 0.9894 - val_loss: 0.1028 - val_acc: 0.9770
Epoch 6/10
60000/60000 [==============================] - 3s 56us/step - loss: 0.0422 - acc: 0.9909 - val_loss: 0.0868 - val_acc: 0.9807
Epoch 7/10
60000/60000 [==============================] - 3s 56us/step - loss: 0.0362 - acc: 0.9928 - val_loss: 0.0976 - val_acc: 0.9793
Epoch 8/10
60000/60000 [==============================] - 3s 56us/step - loss: 0.0347 - acc: 0.9929 - val_loss: 0.0866 - val_acc: 0.9813
Epoch 9/10
60000/60000 [==============================] - 3s 56us/step - loss: 0.0320 - acc: 0.9938 - val_loss: 0.0975 - val_acc: 0.9798
Epoch 10/10
60000/60000 [==============================] - 3s 57us/step - loss: 0.0305 - acc: 0.9943 - val_loss: 0.0999 - val_acc: 0.9799
Test loss: 0.0999036713034
Test accuracy: 0.9799
$ python examples/mnist_mlp.py
Using TensorFlow backend.
60000 train samples
10000 test samples
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_1 (Dense)              (None, 512)               401920    
_________________________________________________________________
dense_2 (Dense)              (None, 512)               262656    
_________________________________________________________________
dense_3 (Dense)              (None, 10)                5130      
_________________________________________________________________
activation_1 (Activation)    (None, 10)                0         
=================================================================
Total params: 669,706
Trainable params: 669,706
Non-trainable params: 0
_________________________________________________________________
Epoch 1/10
468/468 [==============================] - 8s 18ms/step - loss: 0.2031 - accuracy: 0.6996 - val_loss: 0.0971 - val_accuracy: 0.9699
Epoch 2/10
468/468 [==============================] - 11s 23ms/step - loss: 0.0440 - accuracy: 0.5795 - val_loss: 0.0694 - val_accuracy: 0.9800
Epoch 3/10
468/468 [==============================] - 11s 23ms/step - loss: 0.0258 - accuracy: 0.6500 - val_loss: 0.0656 - val_accuracy: 0.9824
Epoch 4/10
468/468 [==============================] - 11s 23ms/step - loss: 0.0202 - accuracy: 0.6919 - val_loss: 0.0754 - val_accuracy: 0.9823
Epoch 5/10
468/468 [==============================] - 11s 23ms/step - loss: 0.0178 - accuracy: 0.7570 - val_loss: 0.0614 - val_accuracy: 0.9842
Epoch 6/10
468/468 [==============================] - 11s 23ms/step - loss: 0.0162 - accuracy: 0.7535 - val_loss: 0.0812 - val_accuracy: 0.9817
Epoch 7/10
468/468 [==============================] - 11s 23ms/step - loss: 0.0153 - accuracy: 0.7738 - val_loss: 0.0751 - val_accuracy: 0.9815
Epoch 8/10
468/468 [==============================] - 11s 23ms/step - loss: 0.0135 - accuracy: 0.7552 - val_loss: 0.0803 - val_accuracy: 0.9804
Epoch 9/10
468/468 [==============================] - 11s 23ms/step - loss: 0.0137 - accuracy: 0.7654 - val_loss: 0.0704 - val_accuracy: 0.9840
Epoch 10/10
468/468 [==============================] - 11s 23ms/step - loss: 0.0130 - accuracy: 0.8060 - val_loss: 0.0632 - val_accuracy: 0.9858
Test loss: 0.0711704681695
Test accuracy: 0.9858
</code></pre>

From the above output we can see the following important differences:

1. The training accuracy when using importance sampling is not representative
   of the actual training accuracy since we sample hard examples to train on
2. The name of the *accuracy* metric is different due to special handling of
   this metric by Keras
3. The first epoch takes a bit less with importance sampling because our
   algorithm automatically waits untill importance sampling can actually
   benefit the optimization

## MNIST CNN

In this example, we showcase the use of the `ConstantTimeImportanceTraining`
class. This trainer aims at keeping the per parameter update time constant
while improving the variance of the gradients. Once again, passing the command
line argument `--uniform` trains the network with plain Keras.

The following results compare the training loss and test error with respect to
the elapsed seconds. From the terminal output we observe that indeed the per
iteration time remains constant in contrast to the MLP example where the time
per epoch increases in comparison to uniform sampling.

<div class="fig col-2">
<img src="../img/mnist_cnn_train_loss.png" alt="Training Loss">
<img src="../img/mnist_cnn_test_acc.png" alt="Test accuracy">
<span>Results of training a small CNN on MNIST with and without
importance sampling (orange and blue respectively).</span>
</div>

<pre style="height:300px; overflow-y: scroll;"><code class="bash">$ python examples/mnist_cnn.py --uniform
Using TensorFlow backend.
x_train shape: (60000, 28, 28, 1)
60000 train samples
10000 test samples
Train on 60000 samples, validate on 10000 samples
Epoch 1/10
60000/60000 [==============================] - 57s 945us/step - loss: 0.2057 - acc: 0.9373 - val_loss: 0.0545 - val_acc: 0.9834
Epoch 2/10
60000/60000 [==============================] - 56s 937us/step - loss: 0.0512 - acc: 0.9857 - val_loss: 0.0421 - val_acc: 0.9877
Epoch 3/10
60000/60000 [==============================] - 56s 939us/step - loss: 0.0344 - acc: 0.9909 - val_loss: 0.0362 - val_acc: 0.9893
Epoch 4/10
60000/60000 [==============================] - 56s 939us/step - loss: 0.0260 - acc: 0.9934 - val_loss: 0.0321 - val_acc: 0.9911
Epoch 5/10
60000/60000 [==============================] - 56s 938us/step - loss: 0.0193 - acc: 0.9957 - val_loss: 0.0408 - val_acc: 0.9898
Epoch 6/10
60000/60000 [==============================] - 56s 938us/step - loss: 0.0151 - acc: 0.9969 - val_loss: 0.0405 - val_acc: 0.9892
Epoch 7/10
60000/60000 [==============================] - 56s 938us/step - loss: 0.0121 - acc: 0.9980 - val_loss: 0.0405 - val_acc: 0.9910
Epoch 8/10
60000/60000 [==============================] - 56s 938us/step - loss: 0.0095 - acc: 0.9990 - val_loss: 0.0373 - val_acc: 0.9913
Epoch 9/10
60000/60000 [==============================] - 56s 938us/step - loss: 0.0090 - acc: 0.9990 - val_loss: 0.0412 - val_acc: 0.9909
Epoch 10/10
60000/60000 [==============================] - 56s 940us/step - loss: 0.0079 - acc: 0.9992 - val_loss: 0.0450 - val_acc: 0.9904
Test loss: 0.0450472147308
Test accuracy: 0.9904
$ python examples/mnist_cnn.py
Using TensorFlow backend.
x_train shape: (60000, 28, 28, 1)
60000 train samples
10000 test samples
Epoch 1/10
468/468 [==============================] - 55s 118ms/step - loss: 0.1909 - accuracy: 0.5881 - val_loss: 0.0525 - val_accuracy: 0.9816
Epoch 2/10
468/468 [==============================] - 54s 116ms/step - loss: 0.0414 - accuracy: 0.5832 - val_loss: 0.0373 - val_accuracy: 0.9879
Epoch 3/10
468/468 [==============================] - 54s 116ms/step - loss: 0.0259 - accuracy: 0.6048 - val_loss: 0.0328 - val_accuracy: 0.9897
Epoch 4/10
468/468 [==============================] - 54s 116ms/step - loss: 0.0168 - accuracy: 0.7021 - val_loss: 0.0346 - val_accuracy: 0.9893
Epoch 5/10
468/468 [==============================] - 54s 116ms/step - loss: 0.0129 - accuracy: 0.7465 - val_loss: 0.0324 - val_accuracy: 0.9910
Epoch 6/10
468/468 [==============================] - 54s 116ms/step - loss: 0.0096 - accuracy: 0.8453 - val_loss: 0.0322 - val_accuracy: 0.9910
Epoch 7/10
468/468 [==============================] - 54s 116ms/step - loss: 0.0082 - accuracy: 0.8558 - val_loss: 0.0347 - val_accuracy: 0.9913
Epoch 8/10
468/468 [==============================] - 54s 116ms/step - loss: 0.0070 - accuracy: 0.9078 - val_loss: 0.0404 - val_accuracy: 0.9899
Epoch 9/10
468/468 [==============================] - 54s 116ms/step - loss: 0.0057 - accuracy: 0.9607 - val_loss: 0.0373 - val_accuracy: 0.9917
Epoch 10/10
468/468 [==============================] - 54s 116ms/step - loss: 0.0056 - accuracy: 0.9610 - val_loss: 0.0377 - val_accuracy: 0.9911
Test loss: 0.0424824692458
Test accuracy: 0.9911
</code></pre>

## CIFAR10 ResNet

In this final example, we showcase a more realistic scenario. We train a Wide
Residual Network to classify the CIFAR-10 images. The specific implementation
of WRN-28-2 can be found in [importance_sampling/models.py][models.py]. The
example accepts several parameters but we use the default values and
`--uniform` to train with plain Keras.

In order for the comparison between uniform and importance sampling to be fair,
we introduce a time budget (the training stops after a given duration). The
learning rate schedule is tied with that time budget and learning rate is
reduced at 50% and at 80% of the available time.

The training is done with an Nvidia GTX 1080 Ti and the following plots show
the progress of the training loss and the test error with respect to the
elapsed seconds. We observe that importance sampling automatically starts after
the first learning rate reduction and improves significantly both in terms of
training loss and test error.

<div class="fig col-2">
<img src="../img/cifar_cnn_train_loss.png" alt="Training Loss">
<img src="../img/cifar_cnn_test_acc.png" alt="Test Accuracy">
<span>Results of training a WRN-28-2 on CIFAR10 with and without
importance sampling (orange and blue respectively).</span>
</div>

<pre style="height:300px; overflow-y: scroll;"><code class="bash">$ python examples/cifar10_resnet.py --uniform
Using TensorFlow backend
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_1 (InputLayer)            (None, 32, 32, 3)    0                                            
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 32, 32, 16)   432         input_1[0][0]                    
__________________________________________________________________________________________________
layer_normalization_1 (LayerNor (None, 32, 32, 16)   17          conv2d_1[0][0]                   
__________________________________________________________________________________________________
activation_1 (Activation)       (None, 32, 32, 16)   0           layer_normalization_1[0][0]      
__________________________________________________________________________________________________
conv2d_2 (Conv2D)               (None, 32, 32, 32)   4608        activation_1[0][0]               
__________________________________________________________________________________________________
layer_normalization_2 (LayerNor (None, 32, 32, 32)   33          conv2d_2[0][0]                   
__________________________________________________________________________________________________
activation_2 (Activation)       (None, 32, 32, 32)   0           layer_normalization_2[0][0]      
__________________________________________________________________________________________________
conv2d_4 (Conv2D)               (None, 32, 32, 32)   512         conv2d_1[0][0]                   
__________________________________________________________________________________________________
conv2d_3 (Conv2D)               (None, 32, 32, 32)   9216        activation_2[0][0]               
__________________________________________________________________________________________________
add_1 (Add)                     (None, 32, 32, 32)   0           conv2d_4[0][0]                   
                                                                 conv2d_3[0][0]                   
__________________________________________________________________________________________________
layer_normalization_3 (LayerNor (None, 32, 32, 32)   33          add_1[0][0]                      
__________________________________________________________________________________________________
activation_3 (Activation)       (None, 32, 32, 32)   0           layer_normalization_3[0][0]      
__________________________________________________________________________________________________
conv2d_5 (Conv2D)               (None, 32, 32, 32)   9216        activation_3[0][0]               
__________________________________________________________________________________________________
layer_normalization_4 (LayerNor (None, 32, 32, 32)   33          conv2d_5[0][0]                   
__________________________________________________________________________________________________
activation_4 (Activation)       (None, 32, 32, 32)   0           layer_normalization_4[0][0]      
__________________________________________________________________________________________________
conv2d_6 (Conv2D)               (None, 32, 32, 32)   9216        activation_4[0][0]               
__________________________________________________________________________________________________
add_2 (Add)                     (None, 32, 32, 32)   0           add_1[0][0]                      
                                                                 conv2d_6[0][0]                   
__________________________________________________________________________________________________
layer_normalization_5 (LayerNor (None, 32, 32, 32)   33          add_2[0][0]                      
__________________________________________________________________________________________________
activation_5 (Activation)       (None, 32, 32, 32)   0           layer_normalization_5[0][0]      
__________________________________________________________________________________________________
conv2d_7 (Conv2D)               (None, 32, 32, 32)   9216        activation_5[0][0]               
__________________________________________________________________________________________________
layer_normalization_6 (LayerNor (None, 32, 32, 32)   33          conv2d_7[0][0]                   
__________________________________________________________________________________________________
activation_6 (Activation)       (None, 32, 32, 32)   0           layer_normalization_6[0][0]      
__________________________________________________________________________________________________
conv2d_8 (Conv2D)               (None, 32, 32, 32)   9216        activation_6[0][0]               
__________________________________________________________________________________________________
add_3 (Add)                     (None, 32, 32, 32)   0           add_2[0][0]                      
                                                                 conv2d_8[0][0]                   
__________________________________________________________________________________________________
layer_normalization_7 (LayerNor (None, 32, 32, 32)   33          add_3[0][0]                      
__________________________________________________________________________________________________
activation_7 (Activation)       (None, 32, 32, 32)   0           layer_normalization_7[0][0]      
__________________________________________________________________________________________________
conv2d_9 (Conv2D)               (None, 32, 32, 32)   9216        activation_7[0][0]               
__________________________________________________________________________________________________
layer_normalization_8 (LayerNor (None, 32, 32, 32)   33          conv2d_9[0][0]                   
__________________________________________________________________________________________________
activation_8 (Activation)       (None, 32, 32, 32)   0           layer_normalization_8[0][0]      
__________________________________________________________________________________________________
conv2d_10 (Conv2D)              (None, 32, 32, 32)   9216        activation_8[0][0]               
__________________________________________________________________________________________________
add_4 (Add)                     (None, 32, 32, 32)   0           add_3[0][0]                      
                                                                 conv2d_10[0][0]                  
__________________________________________________________________________________________________
layer_normalization_9 (LayerNor (None, 32, 32, 32)   33          add_4[0][0]                      
__________________________________________________________________________________________________
activation_9 (Activation)       (None, 32, 32, 32)   0           layer_normalization_9[0][0]      
__________________________________________________________________________________________________
conv2d_11 (Conv2D)              (None, 16, 16, 64)   18432       activation_9[0][0]               
__________________________________________________________________________________________________
layer_normalization_10 (LayerNo (None, 16, 16, 64)   65          conv2d_11[0][0]                  
__________________________________________________________________________________________________
activation_10 (Activation)      (None, 16, 16, 64)   0           layer_normalization_10[0][0]     
__________________________________________________________________________________________________
conv2d_13 (Conv2D)              (None, 16, 16, 64)   2048        add_4[0][0]                      
__________________________________________________________________________________________________
conv2d_12 (Conv2D)              (None, 16, 16, 64)   36864       activation_10[0][0]              
__________________________________________________________________________________________________
add_5 (Add)                     (None, 16, 16, 64)   0           conv2d_13[0][0]                  
                                                                 conv2d_12[0][0]                  
__________________________________________________________________________________________________
layer_normalization_11 (LayerNo (None, 16, 16, 64)   65          add_5[0][0]                      
__________________________________________________________________________________________________
activation_11 (Activation)      (None, 16, 16, 64)   0           layer_normalization_11[0][0]     
__________________________________________________________________________________________________
conv2d_14 (Conv2D)              (None, 16, 16, 64)   36864       activation_11[0][0]              
__________________________________________________________________________________________________
layer_normalization_12 (LayerNo (None, 16, 16, 64)   65          conv2d_14[0][0]                  
__________________________________________________________________________________________________
activation_12 (Activation)      (None, 16, 16, 64)   0           layer_normalization_12[0][0]     
__________________________________________________________________________________________________
conv2d_15 (Conv2D)              (None, 16, 16, 64)   36864       activation_12[0][0]              
__________________________________________________________________________________________________
add_6 (Add)                     (None, 16, 16, 64)   0           add_5[0][0]                      
                                                                 conv2d_15[0][0]                  
__________________________________________________________________________________________________
layer_normalization_13 (LayerNo (None, 16, 16, 64)   65          add_6[0][0]                      
__________________________________________________________________________________________________
activation_13 (Activation)      (None, 16, 16, 64)   0           layer_normalization_13[0][0]     
__________________________________________________________________________________________________
conv2d_16 (Conv2D)              (None, 16, 16, 64)   36864       activation_13[0][0]              
__________________________________________________________________________________________________
layer_normalization_14 (LayerNo (None, 16, 16, 64)   65          conv2d_16[0][0]                  
__________________________________________________________________________________________________
activation_14 (Activation)      (None, 16, 16, 64)   0           layer_normalization_14[0][0]     
__________________________________________________________________________________________________
conv2d_17 (Conv2D)              (None, 16, 16, 64)   36864       activation_14[0][0]              
__________________________________________________________________________________________________
add_7 (Add)                     (None, 16, 16, 64)   0           add_6[0][0]                      
                                                                 conv2d_17[0][0]                  
__________________________________________________________________________________________________
layer_normalization_15 (LayerNo (None, 16, 16, 64)   65          add_7[0][0]                      
__________________________________________________________________________________________________
activation_15 (Activation)      (None, 16, 16, 64)   0           layer_normalization_15[0][0]     
__________________________________________________________________________________________________
conv2d_18 (Conv2D)              (None, 16, 16, 64)   36864       activation_15[0][0]              
__________________________________________________________________________________________________
layer_normalization_16 (LayerNo (None, 16, 16, 64)   65          conv2d_18[0][0]                  
__________________________________________________________________________________________________
activation_16 (Activation)      (None, 16, 16, 64)   0           layer_normalization_16[0][0]     
__________________________________________________________________________________________________
conv2d_19 (Conv2D)              (None, 16, 16, 64)   36864       activation_16[0][0]              
__________________________________________________________________________________________________
add_8 (Add)                     (None, 16, 16, 64)   0           add_7[0][0]                      
                                                                 conv2d_19[0][0]                  
__________________________________________________________________________________________________
layer_normalization_17 (LayerNo (None, 16, 16, 64)   65          add_8[0][0]                      
__________________________________________________________________________________________________
activation_17 (Activation)      (None, 16, 16, 64)   0           layer_normalization_17[0][0]     
__________________________________________________________________________________________________
conv2d_20 (Conv2D)              (None, 8, 8, 128)    73728       activation_17[0][0]              
__________________________________________________________________________________________________
layer_normalization_18 (LayerNo (None, 8, 8, 128)    129         conv2d_20[0][0]                  
__________________________________________________________________________________________________
activation_18 (Activation)      (None, 8, 8, 128)    0           layer_normalization_18[0][0]     
__________________________________________________________________________________________________
conv2d_22 (Conv2D)              (None, 8, 8, 128)    8192        add_8[0][0]                      
__________________________________________________________________________________________________
conv2d_21 (Conv2D)              (None, 8, 8, 128)    147456      activation_18[0][0]              
__________________________________________________________________________________________________
add_9 (Add)                     (None, 8, 8, 128)    0           conv2d_22[0][0]                  
                                                                 conv2d_21[0][0]                  
__________________________________________________________________________________________________
layer_normalization_19 (LayerNo (None, 8, 8, 128)    129         add_9[0][0]                      
__________________________________________________________________________________________________
activation_19 (Activation)      (None, 8, 8, 128)    0           layer_normalization_19[0][0]     
__________________________________________________________________________________________________
conv2d_23 (Conv2D)              (None, 8, 8, 128)    147456      activation_19[0][0]              
__________________________________________________________________________________________________
layer_normalization_20 (LayerNo (None, 8, 8, 128)    129         conv2d_23[0][0]                  
__________________________________________________________________________________________________
activation_20 (Activation)      (None, 8, 8, 128)    0           layer_normalization_20[0][0]     
__________________________________________________________________________________________________
conv2d_24 (Conv2D)              (None, 8, 8, 128)    147456      activation_20[0][0]              
__________________________________________________________________________________________________
add_10 (Add)                    (None, 8, 8, 128)    0           add_9[0][0]                      
                                                                 conv2d_24[0][0]                  
__________________________________________________________________________________________________
layer_normalization_21 (LayerNo (None, 8, 8, 128)    129         add_10[0][0]                     
__________________________________________________________________________________________________
activation_21 (Activation)      (None, 8, 8, 128)    0           layer_normalization_21[0][0]     
__________________________________________________________________________________________________
conv2d_25 (Conv2D)              (None, 8, 8, 128)    147456      activation_21[0][0]              
__________________________________________________________________________________________________
layer_normalization_22 (LayerNo (None, 8, 8, 128)    129         conv2d_25[0][0]                  
__________________________________________________________________________________________________
activation_22 (Activation)      (None, 8, 8, 128)    0           layer_normalization_22[0][0]     
__________________________________________________________________________________________________
conv2d_26 (Conv2D)              (None, 8, 8, 128)    147456      activation_22[0][0]              
__________________________________________________________________________________________________
add_11 (Add)                    (None, 8, 8, 128)    0           add_10[0][0]                     
                                                                 conv2d_26[0][0]                  
__________________________________________________________________________________________________
layer_normalization_23 (LayerNo (None, 8, 8, 128)    129         add_11[0][0]                     
__________________________________________________________________________________________________
activation_23 (Activation)      (None, 8, 8, 128)    0           layer_normalization_23[0][0]     
__________________________________________________________________________________________________
conv2d_27 (Conv2D)              (None, 8, 8, 128)    147456      activation_23[0][0]              
__________________________________________________________________________________________________
layer_normalization_24 (LayerNo (None, 8, 8, 128)    129         conv2d_27[0][0]                  
__________________________________________________________________________________________________
activation_24 (Activation)      (None, 8, 8, 128)    0           layer_normalization_24[0][0]     
__________________________________________________________________________________________________
conv2d_28 (Conv2D)              (None, 8, 8, 128)    147456      activation_24[0][0]              
__________________________________________________________________________________________________
add_12 (Add)                    (None, 8, 8, 128)    0           add_11[0][0]                     
                                                                 conv2d_28[0][0]                  
__________________________________________________________________________________________________
layer_normalization_25 (LayerNo (None, 8, 8, 128)    129         add_12[0][0]                     
__________________________________________________________________________________________________
activation_25 (Activation)      (None, 8, 8, 128)    0           layer_normalization_25[0][0]     
__________________________________________________________________________________________________
global_average_pooling2d_1 (Glo (None, 128)          0           activation_25[0][0]              
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 10)           1290        global_average_pooling2d_1[0][0] 
__________________________________________________________________________________________________
activation_26 (Activation)      (None, 10)           0           dense_1[0][0]                    
==================================================================================================
Total params: 1,465,827
Trainable params: 1,465,827
Non-trainable params: 0
__________________________________________________________________________________________________
Epoch 1/1000000
391/391 [==============================] - 44s 114ms/step - loss: 2.9380 - acc: 0.1064 - val_loss: 2.6958 - val_acc: 0.1367
Epoch 2/1000000
391/391 [==============================] - 42s 108ms/step - loss: 2.3984 - acc: 0.1845 - val_loss: 2.1257 - val_acc: 0.2150
Epoch 3/1000000
391/391 [==============================] - 42s 109ms/step - loss: 2.0156 - acc: 0.2763 - val_loss: 1.8615 - val_acc: 0.3400
Epoch 4/1000000
391/391 [==============================] - 42s 107ms/step - loss: 1.7403 - acc: 0.3969 - val_loss: 2.0339 - val_acc: 0.3302
Epoch 5/1000000
391/391 [==============================] - 41s 106ms/step - loss: 1.5567 - acc: 0.4813 - val_loss: 1.5330 - val_acc: 0.4951
Epoch 6/1000000
391/391 [==============================] - 41s 105ms/step - loss: 1.4558 - acc: 0.5361 - val_loss: 1.3869 - val_acc: 0.5701
Epoch 7/1000000
391/391 [==============================] - 41s 105ms/step - loss: 1.3699 - acc: 0.5718 - val_loss: 1.3625 - val_acc: 0.5791
Epoch 8/1000000
391/391 [==============================] - 41s 106ms/step - loss: 1.3062 - acc: 0.6069 - val_loss: 1.5217 - val_acc: 0.5251
Epoch 9/1000000
391/391 [==============================] - 41s 104ms/step - loss: 1.2734 - acc: 0.6252 - val_loss: 1.2764 - val_acc: 0.6289
Epoch 10/1000000
391/391 [==============================] - 41s 106ms/step - loss: 1.2181 - acc: 0.6469 - val_loss: 1.1489 - val_acc: 0.6629
Epoch 11/1000000
391/391 [==============================] - 41s 104ms/step - loss: 1.1800 - acc: 0.6675 - val_loss: 1.0742 - val_acc: 0.7092
Epoch 12/1000000
391/391 [==============================] - 40s 104ms/step - loss: 1.1700 - acc: 0.6760 - val_loss: 1.1361 - val_acc: 0.6849
Epoch 13/1000000
391/391 [==============================] - 40s 103ms/step - loss: 1.1497 - acc: 0.6897 - val_loss: 1.2368 - val_acc: 0.6648
Epoch 14/1000000
391/391 [==============================] - 41s 105ms/step - loss: 1.1111 - acc: 0.7097 - val_loss: 1.1373 - val_acc: 0.7093
Epoch 15/1000000
391/391 [==============================] - 41s 105ms/step - loss: 1.1046 - acc: 0.7144 - val_loss: 1.1954 - val_acc: 0.6895
Epoch 16/1000000
391/391 [==============================] - 41s 105ms/step - loss: 1.0941 - acc: 0.7293 - val_loss: 1.0751 - val_acc: 0.7457
Epoch 17/1000000
391/391 [==============================] - 41s 105ms/step - loss: 1.0836 - acc: 0.7367 - val_loss: 1.1855 - val_acc: 0.7037
Epoch 18/1000000
391/391 [==============================] - 41s 104ms/step - loss: 1.0486 - acc: 0.7507 - val_loss: 1.0857 - val_acc: 0.7395
Epoch 19/1000000
391/391 [==============================] - 41s 105ms/step - loss: 1.0375 - acc: 0.7586 - val_loss: 1.1172 - val_acc: 0.7203
Epoch 20/1000000
391/391 [==============================] - 41s 105ms/step - loss: 1.0388 - acc: 0.7606 - val_loss: 1.0932 - val_acc: 0.7360
Epoch 21/1000000
391/391 [==============================] - 40s 103ms/step - loss: 1.0065 - acc: 0.7741 - val_loss: 0.9804 - val_acc: 0.7783
Epoch 22/1000000
391/391 [==============================] - 40s 103ms/step - loss: 1.0070 - acc: 0.7739 - val_loss: 1.0556 - val_acc: 0.7664
Epoch 23/1000000
391/391 [==============================] - 40s 104ms/step - loss: 1.0137 - acc: 0.7739 - val_loss: 0.9827 - val_acc: 0.7815
Epoch 24/1000000
391/391 [==============================] - 41s 104ms/step - loss: 1.0034 - acc: 0.7788 - val_loss: 1.0727 - val_acc: 0.7574
Epoch 25/1000000
391/391 [==============================] - 41s 105ms/step - loss: 0.9935 - acc: 0.7832 - val_loss: 0.9699 - val_acc: 0.7963
Epoch 26/1000000
391/391 [==============================] - 41s 105ms/step - loss: 1.0019 - acc: 0.7833 - val_loss: 1.1758 - val_acc: 0.7334
Epoch 27/1000000
391/391 [==============================] - 41s 104ms/step - loss: 1.0007 - acc: 0.7848 - val_loss: 1.1349 - val_acc: 0.7539
Epoch 28/1000000
391/391 [==============================] - 41s 106ms/step - loss: 0.9837 - acc: 0.7924 - val_loss: 0.9830 - val_acc: 0.7855
Epoch 29/1000000
391/391 [==============================] - 40s 103ms/step - loss: 0.9807 - acc: 0.7918 - val_loss: 0.9778 - val_acc: 0.7954
Epoch 30/1000000
391/391 [==============================] - 41s 104ms/step - loss: 0.9861 - acc: 0.7917 - val_loss: 0.9318 - val_acc: 0.8070
Epoch 31/1000000
391/391 [==============================] - 41s 106ms/step - loss: 0.9923 - acc: 0.7920 - val_loss: 1.0464 - val_acc: 0.7705
Epoch 32/1000000
391/391 [==============================] - 41s 104ms/step - loss: 0.9730 - acc: 0.7985 - val_loss: 1.0076 - val_acc: 0.7885
Epoch 33/1000000
391/391 [==============================] - 41s 105ms/step - loss: 0.9902 - acc: 0.7917 - val_loss: 0.9868 - val_acc: 0.7955
Epoch 34/1000000
391/391 [==============================] - 41s 105ms/step - loss: 0.9776 - acc: 0.7978 - val_loss: 1.0281 - val_acc: 0.7814
Epoch 35/1000000
391/391 [=============================] - 41s 105ms/step - loss: 0.9798 - acc: 0.7973 - val_loss: 0.9608 - val_acc: 0.8084
Epoch 36/1000000
391/391 [==============================] - 42s 107ms/step - loss: 0.9745 - acc: 0.7980 - val_loss: 0.9567 - val_acc: 0.8058
Epoch 37/1000000
391/391 [==============================] - 41s 106ms/step - loss: 0.9829 - acc: 0.7994 - val_loss: 1.0102 - val_acc: 0.7912
Epoch 38/1000000
391/391 [==============================] - 41s 106ms/step - loss: 0.9880 - acc: 0.8000 - val_loss: 0.9299 - val_acc: 0.8144
Epoch 39/1000000
391/391 [==============================] - 41s 104ms/step - loss: 0.9654 - acc: 0.8025 - val_loss: 0.9989 - val_acc: 0.7885
Epoch 40/1000000
391/391 [==============================] - 40s 103ms/step - loss: 0.9884 - acc: 0.8006 - val_loss: 1.0000 - val_acc: 0.7906
Epoch 41/1000000
391/391 [==============================] - 41s 104ms/step - loss: 0.9809 - acc: 0.8017 - val_loss: 1.0070 - val_acc: 0.8008
Epoch 42/1000000
391/391 [==============================] - 40s 103ms/step - loss: 0.9711 - acc: 0.8069 - val_loss: 1.0159 - val_acc: 0.8021
Epoch 43/1000000
391/391 [==============================] - 40s 103ms/step - loss: 0.9709 - acc: 0.8058 - val_loss: 1.0219 - val_acc: 0.7948
Epoch 44/1000000
391/391 [==============================] - 41s 105ms/step - loss: 0.9725 - acc: 0.8049 - val_loss: 0.9961 - val_acc: 0.7966
Epoch 45/1000000
391/391 [==============================] - 41s 106ms/step - loss: 0.9711 - acc: 0.8052 - val_loss: 1.1400 - val_acc: 0.7607
Epoch 46/1000000
391/391 [==============================] - 40s 103ms/step - loss: 0.9695 - acc: 0.8076 - val_loss: 0.9000 - val_acc: 0.8332
Epoch 47/1000000
391/391 [==============================] - 40s 103ms/step - loss: 0.9710 - acc: 0.8072 - val_loss: 0.9580 - val_acc: 0.8131
Epoch 48/1000000
391/391 [==============================] - 41s 104ms/step - loss: 0.9753 - acc: 0.8055 - val_loss: 0.9696 - val_acc: 0.8075
Epoch 49/1000000
391/391 [==============================] - 40s 103ms/step - loss: 0.9700 - acc: 0.8102 - val_loss: 1.1380 - val_acc: 0.7615
Epoch 50/1000000
391/391 [==============================] - 42s 108ms/step - loss: 0.9607 - acc: 0.8106 - val_loss: 0.9477 - val_acc: 0.8141
Epoch 51/1000000
391/391 [==============================] - 41s 106ms/step - loss: 0.9734 - acc: 0.8073 - val_loss: 0.9734 - val_acc: 0.8028
Epoch 52/1000000
391/391 [==============================] - 41s 105ms/step - loss: 0.9618 - acc: 0.8092 - val_loss: 0.9639 - val_acc: 0.8092
Epoch 53/1000000
391/391 [==============================] - 42s 108ms/step - loss: 0.9774 - acc: 0.8069 - val_loss: 1.0551 - val_acc: 0.7857
Epoch 54/1000000
391/391 [==============================] - 41s 106ms/step - loss: 0.9551 - acc: 0.8141 - val_loss: 1.1194 - val_acc: 0.7682
Epoch 55/1000000
391/391 [==============================] - 41s 104ms/step - loss: 0.9554 - acc: 0.8128 - val_loss: 1.0579 - val_acc: 0.7804
Epoch 56/1000000
391/391 [==============================] - 41s 106ms/step - loss: 0.9670 - acc: 0.8104 - val_loss: 0.9567 - val_acc: 0.8106
Epoch 57/1000000
391/391 [==============================] - 41s 106ms/step - loss: 0.9510 - acc: 0.8158 - val_loss: 1.0041 - val_acc: 0.7979
Epoch 58/1000000
391/391 [==============================] - 40s 103ms/step - loss: 0.9653 - acc: 0.8130 - val_loss: 1.0690 - val_acc: 0.7834
Epoch 59/1000000
391/391 [==============================] - 41s 105ms/step - loss: 0.9612 - acc: 0.8121 - val_loss: 0.9869 - val_acc: 0.8043
Epoch 60/1000000
391/391 [==============================] - 41s 104ms/step - loss: 0.9544 - acc: 0.8122 - val_loss: 0.9612 - val_acc: 0.8158
Epoch 61/1000000
391/391 [==============================] - 41s 104ms/step - loss: 0.9578 - acc: 0.8129 - val_loss: 1.0031 - val_acc: 0.8047
Epoch 62/1000000
391/391 [==============================] - 42s 107ms/step - loss: 0.9573 - acc: 0.8146 - val_loss: 0.9617 - val_acc: 0.8144
Epoch 63/1000000
391/391 [==============================] - 40s 104ms/step - loss: 0.9443 - acc: 0.8178 - val_loss: 1.0440 - val_acc: 0.7835
Epoch 64/1000000
391/391 [==============================] - 41s 105ms/step - loss: 0.9583 - acc: 0.8138 - val_loss: 0.9888 - val_acc: 0.7999
Epoch 65/1000000
391/391 [==============================] - 41s 105ms/step - loss: 0.9581 - acc: 0.8137 - val_loss: 1.0410 - val_acc: 0.7815
Epoch 66/1000000
391/391 [==============================] - 41s 105ms/step - loss: 0.9523 - acc: 0.8155 - val_loss: 0.8907 - val_acc: 0.8400
Epoch 67/1000000
391/391 [==============================] - 41s 104ms/step - loss: 0.9576 - acc: 0.8151 - val_loss: 1.1378 - val_acc: 0.7609
Epoch 68/1000000
391/391 [==============================] - 41s 105ms/step - loss: 0.9410 - acc: 0.8180 - val_loss: 1.0349 - val_acc: 0.7855
Epoch 69/1000000
391/391 [==============================] - 41s 104ms/step - loss: 0.9750 - acc: 0.8085 - val_loss: 0.9649 - val_acc: 0.8166
Epoch 70/1000000
391/391 [==============================] - 41s 104ms/step - loss: 0.9543 - acc: 0.8182 - val_loss: 0.9804 - val_acc: 0.8051
Epoch 71/1000000
391/391 [==============================] - 40s 103ms/step - loss: 0.9598 - acc: 0.8134 - val_loss: 1.0169 - val_acc: 0.7955
Epoch 72/1000000
391/391 [==============================] - 40s 103ms/step - loss: 0.9420 - acc: 0.8207 - val_loss: 1.0281 - val_acc: 0.7902
Epoch 73/1000000
391/391 [==============================] - 41s 104ms/step - loss: 0.9596 - acc: 0.8158 - val_loss: 1.0356 - val_acc: 0.8020
Epoch 74/1000000
391/391 [==============================] - 41s 105ms/step - loss: 0.9601 - acc: 0.8158 - val_loss: 0.9216 - val_acc: 0.8272
Epoch 75/1000000
391/391 [==============================] - 41s 104ms/step - loss: 0.9407 - acc: 0.8195 - val_loss: 0.9679 - val_acc: 0.8105
Epoch 76/1000000
391/391 [==============================] - 41s 105ms/step - loss: 0.9475 - acc: 0.8182 - val_loss: 0.9469 - val_acc: 0.8178
Epoch 77/1000000
391/391 [==============================] - 41s 104ms/step - loss: 0.9563 - acc: 0.8151 - val_loss: 0.9388 - val_acc: 0.8211
Epoch 78/1000000
391/391 [==============================] - 41s 104ms/step - loss: 0.9389 - acc: 0.8232 - val_loss: 0.9590 - val_acc: 0.8148
Epoch 79/1000000
391/391 [==============================] - 42s 107ms/step - loss: 0.9697 - acc: 0.8120 - val_loss: 0.9821 - val_acc: 0.8066
Epoch 80/1000000
391/391 [==============================] - 41s 105ms/step - loss: 0.9343 - acc: 0.8235 - val_loss: 1.0246 - val_acc: 0.7930
Epoch 81/1000000
391/391 [==============================] - 41s 106ms/step - loss: 0.9414 - acc: 0.8206 - val_loss: 0.9869 - val_acc: 0.8094
Epoch 82/1000000
391/391 [==============================] - 42s 107ms/step - loss: 0.9490 - acc: 0.8166 - val_loss: 0.9289 - val_acc: 0.8214
Epoch 83/1000000
391/391 [==============================] - 42s 108ms/step - loss: 0.9503 - acc: 0.8190 - val_loss: 0.9542 - val_acc: 0.8162
Epoch 84/1000000
391/391 [==============================] - 41s 105ms/step - loss: 0.9537 - acc: 0.8185 - val_loss: 0.9254 - val_acc: 0.8321
Epoch 85/1000000
391/391 [==============================] - 41s 105ms/step - loss: 0.9394 - acc: 0.8224 - val_loss: 0.9417 - val_acc: 0.8212
Epoch 86/1000000
391/391 [==============================] - 41s 104ms/step - loss: 0.9425 - acc: 0.8217 - val_loss: 1.0966 - val_acc: 0.7775
Epoch 87/1000000
391/391 [==============================] - 41s 105ms/step - loss: 0.9503 - acc: 0.8196 - val_loss: 0.9664 - val_acc: 0.8104
Epoch 88/1000000
391/391 [==============================] - 42s 107ms/step - loss: 0.9454 - acc: 0.8206 - val_loss: 0.9610 - val_acc: 0.8154
Epoch 89/1000000
391/391 [==============================] - 41s 105ms/step - loss: 0.9485 - acc: 0.8199 - val_loss: 1.0032 - val_acc: 0.8002
Epoch 90/1000000
391/391 [==============================] - 41s 105ms/step - loss: 0.9413 - acc: 0.8212 - val_loss: 1.0001 - val_acc: 0.8033
Epoch 91/1000000
391/391 [==============================] - 41s 106ms/step - loss: 0.9402 - acc: 0.8220 - val_loss: 1.0071 - val_acc: 0.8041
Epoch 92/1000000
391/391 [==============================] - 42s 107ms/step - loss: 0.9502 - acc: 0.8180 - val_loss: 0.9877 - val_acc: 0.8050
Epoch 93/1000000
391/391 [=============================] - 42s 106ms/step - loss: 0.9483 - acc: 0.8243 - val_loss: 0.8878 - val_acc: 0.8395
Epoch 94/1000000
391/391 [==============================] - 42s 107ms/step - loss: 0.9444 - acc: 0.8209 - val_loss: 0.9194 - val_acc: 0.8277
Epoch 95/1000000
391/391 [==============================] - 41s 104ms/step - loss: 0.9359 - acc: 0.8226 - val_loss: 0.9491 - val_acc: 0.8203
Epoch 96/1000000
391/391 [==============================] - 42s 107ms/step - loss: 0.9523 - acc: 0.8191 - val_loss: 0.9310 - val_acc: 0.8258
Epoch 97/1000000
391/391 [==============================] - 42s 107ms/step - loss: 0.9495 - acc: 0.8197 - val_loss: 0.9578 - val_acc: 0.8161
Epoch 98/1000000
391/391 [==============================] - 41s 104ms/step - loss: 0.9461 - acc: 0.8194 - val_loss: 1.0503 - val_acc: 0.7904
Epoch 99/1000000
391/391 [==============================] - 41s 106ms/step - loss: 0.9501 - acc: 0.8193 - val_loss: 0.9467 - val_acc: 0.8218
Epoch 100/1000000
391/391 [==============================] - 41s 104ms/step - loss: 0.9296 - acc: 0.8270 - val_loss: 1.0412 - val_acc: 0.7892
Epoch 101/1000000
391/391 [==============================] - 40s 103ms/step - loss: 0.9260 - acc: 0.8252 - val_loss: 0.9640 - val_acc: 0.8109
Epoch 102/1000000
391/391 [==============================] - 41s 105ms/step - loss: 0.9238 - acc: 0.8245 - val_loss: 0.9212 - val_acc: 0.8256
Epoch 103/1000000
391/391 [==============================] - 41s 104ms/step - loss: 0.9443 - acc: 0.8210 - val_loss: 0.9674 - val_acc: 0.8192
Epoch 104/1000000
391/391 [==============================] - 41s 104ms/step - loss: 0.9204 - acc: 0.8288 - val_loss: 0.9837 - val_acc: 0.8040
Epoch 105/1000000
391/391 [==============================] - 41s 104ms/step - loss: 0.9258 - acc: 0.8263 - val_loss: 1.0192 - val_acc: 0.8009
Epoch 106/1000000
391/391 [==============================] - 41s 106ms/step - loss: 0.9374 - acc: 0.8226 - val_loss: 1.0195 - val_acc: 0.7952
Epoch 107/1000000
391/391 [==============================] - 41s 105ms/step - loss: 0.9357 - acc: 0.8243 - val_loss: 0.9804 - val_acc: 0.8079
Epoch 108/1000000
391/391 [==============================] - 40s 104ms/step - loss: 0.9336 - acc: 0.8227 - val_loss: 0.9091 - val_acc: 0.8347
Epoch 109/1000000
391/391 [==============================] - 42s 107ms/step - loss: 0.9401 - acc: 0.8240 - val_loss: 0.9678 - val_acc: 0.8166
Epoch 110/1000000
391/391 [==============================] - 42s 106ms/step - loss: 0.9233 - acc: 0.8281 - val_loss: 0.9647 - val_acc: 0.8094
Epoch 111/1000000
391/391 [==============================] - 41s 105ms/step - loss: 0.9386 - acc: 0.8217 - val_loss: 0.9642 - val_acc: 0.8143
Epoch 112/1000000
391/391 [==============================] - 42s 106ms/step - loss: 0.9241 - acc: 0.8277 - val_loss: 0.8984 - val_acc: 0.8307
Epoch 113/1000000
391/391 [==============================] - 41s 105ms/step - loss: 0.9342 - acc: 0.8221 - val_loss: 0.9422 - val_acc: 0.8213
Epoch 114/1000000
391/391 [==============================] - 42s 108ms/step - loss: 0.9282 - acc: 0.8278 - val_loss: 1.0228 - val_acc: 0.8059
Epoch 115/1000000
391/391 [==============================] - 42s 107ms/step - loss: 0.9336 - acc: 0.8232 - val_loss: 1.0238 - val_acc: 0.8116
Epoch 116/1000000
391/391 [==============================] - 41s 105ms/step - loss: 0.9350 - acc: 0.8254 - val_loss: 0.9967 - val_acc: 0.8084
Epoch 117/1000000
391/391 [==============================] - 41s 106ms/step - loss: 0.9318 - acc: 0.8236 - val_loss: 0.8943 - val_acc: 0.8409
Epoch 118/1000000
391/391 [==============================] - 42s 106ms/step - loss: 0.9224 - acc: 0.8283 - val_loss: 0.9498 - val_acc: 0.8228
Epoch 119/1000000
391/391 [==============================] - 41s 105ms/step - loss: 0.9276 - acc: 0.8262 - val_loss: 0.9516 - val_acc: 0.8199
Epoch 120/1000000
391/391 [==============================] - 41s 105ms/step - loss: 0.9365 - acc: 0.8246 - val_loss: 1.1437 - val_acc: 0.7758
Epoch 121/1000000
391/391 [==============================] - 42s 106ms/step - loss: 0.9450 - acc: 0.8208 - val_loss: 0.9339 - val_acc: 0.8288
Epoch 122/1000000
391/391 [==============================] - 41s 106ms/step - loss: 0.9346 - acc: 0.8237 - val_loss: 1.0032 - val_acc: 0.8031
Epoch 123/1000000
391/391 [==============================] - 41s 106ms/step - loss: 0.9412 - acc: 0.8233 - val_loss: 0.9567 - val_acc: 0.8202
Epoch 124/1000000
391/391 [==============================] - 41s 105ms/step - loss: 0.9305 - acc: 0.8254 - val_loss: 0.9239 - val_acc: 0.8324
Epoch 125/1000000
391/391 [==============================] - 41s 105ms/step - loss: 0.9415 - acc: 0.8245 - val_loss: 0.9176 - val_acc: 0.8304
Epoch 126/1000000
391/391 [==============================] - 41s 105ms/step - loss: 0.9305 - acc: 0.8285 - val_loss: 0.9336 - val_acc: 0.8255
Epoch 127/1000000
391/391 [==============================] - 41s 106ms/step - loss: 0.9160 - acc: 0.8302 - val_loss: 1.0307 - val_acc: 0.7973
Epoch 128/1000000
391/391 [==============================] - 41s 104ms/step - loss: 0.9249 - acc: 0.8273 - val_loss: 0.8957 - val_acc: 0.8338
Epoch 129/1000000
391/391 [==============================] - 41s 106ms/step - loss: 0.9333 - acc: 0.8252 - val_loss: 0.9775 - val_acc: 0.8115
Epoch 130/1000000
391/391 [==============================] - 42s 107ms/step - loss: 0.9295 - acc: 0.8278 - val_loss: 0.9515 - val_acc: 0.8223
Epoch 131/1000000
391/391 [==============================] - 41s 104ms/step - loss: 0.9182 - acc: 0.8293 - val_loss: 0.9819 - val_acc: 0.8052
Epoch 132/1000000
391/391 [==============================] - 42s 107ms/step - loss: 0.8053 - acc: 0.8626 - val_loss: 0.7610 - val_acc: 0.8733
Epoch 133/1000000
391/391 [==============================] - 41s 104ms/step - loss: 0.6381 - acc: 0.9046 - val_loss: 0.6668 - val_acc: 0.8898
Epoch 134/1000000
391/391 [==============================] - 41s 106ms/step - loss: 0.5810 - acc: 0.9104 - val_loss: 0.6276 - val_acc: 0.8917
Epoch 135/1000000
391/391 [==============================] - 41s 105ms/step - loss: 0.5396 - acc: 0.9146 - val_loss: 0.6133 - val_acc: 0.8891
Epoch 136/1000000
391/391 [==============================] - 41s 104ms/step - loss: 0.5158 - acc: 0.9152 - val_loss: 0.5952 - val_acc: 0.8862
Epoch 137/1000000
391/391 [==============================] - 41s 104ms/step - loss: 0.5056 - acc: 0.9138 - val_loss: 0.5851 - val_acc: 0.8857
Epoch 138/1000000
391/391 [==============================] - 41s 105ms/step - loss: 0.4968 - acc: 0.9121 - val_loss: 0.5666 - val_acc: 0.8907
Epoch 139/1000000
391/391 [==============================] - 40s 103ms/step - loss: 0.4854 - acc: 0.9133 - val_loss: 0.5753 - val_acc: 0.8822
Epoch 140/1000000
391/391 [==============================] - 40s 103ms/step - loss: 0.4863 - acc: 0.9121 - val_loss: 0.6243 - val_acc: 0.8722
Epoch 141/1000000
391/391 [==============================] - 41s 104ms/step - loss: 0.4787 - acc: 0.9136 - val_loss: 0.5991 - val_acc: 0.8795
Epoch 142/1000000
391/391 [==============================] - 41s 104ms/step - loss: 0.4832 - acc: 0.9114 - val_loss: 0.6018 - val_acc: 0.8758
Epoch 143/1000000
391/391 [==============================] - 41s 104ms/step - loss: 0.4811 - acc: 0.9122 - val_loss: 0.6176 - val_acc: 0.8729
Epoch 144/1000000
391/391 [==============================] - 41s 106ms/step - loss: 0.4769 - acc: 0.9140 - val_loss: 0.6176 - val_acc: 0.8731
Epoch 145/1000000
391/391 [==============================] - 41s 106ms/step - loss: 0.4785 - acc: 0.9119 - val_loss: 0.6081 - val_acc: 0.8762
Epoch 146/1000000
391/391 [==============================] - 41s 105ms/step - loss: 0.4827 - acc: 0.9120 - val_loss: 0.6146 - val_acc: 0.8758
Epoch 147/1000000
391/391 [==============================] - 42s 107ms/step - loss: 0.4823 - acc: 0.9135 - val_loss: 0.5786 - val_acc: 0.8849
Epoch 148/1000000
391/391 [==============================] - 41s 104ms/step - loss: 0.4761 - acc: 0.9170 - val_loss: 0.5880 - val_acc: 0.8837
Epoch 149/1000000
391/391 [==============================] - 41s 104ms/step - loss: 0.4852 - acc: 0.9134 - val_loss: 0.5907 - val_acc: 0.8861
Epoch 150/1000000
391/391 [==============================] - 41s 104ms/step - loss: 0.4736 - acc: 0.9183 - val_loss: 0.5912 - val_acc: 0.8865
Epoch 151/1000000
391/391 [==============================] - 41s 104ms/step - loss: 0.4763 - acc: 0.9169 - val_loss: 0.5885 - val_acc: 0.8868
Epoch 152/1000000
391/391 [==============================] - 41s 105ms/step - loss: 0.4896 - acc: 0.9127 - val_loss: 0.6091 - val_acc: 0.8752
Epoch 153/1000000
391/391 [==============================] - 41s 104ms/step - loss: 0.4845 - acc: 0.9171 - val_loss: 0.6096 - val_acc: 0.8822
Epoch 154/1000000
391/391 [==============================] - 41s 105ms/step - loss: 0.4741 - acc: 0.9200 - val_loss: 0.6090 - val_acc: 0.8823
Epoch 155/1000000
391/391 [==============================] - 41s 104ms/step - loss: 0.4798 - acc: 0.9182 - val_loss: 0.6183 - val_acc: 0.8787
Epoch 156/1000000
391/391 [==============================] - 41s 105ms/step - loss: 0.4760 - acc: 0.9200 - val_loss: 0.5938 - val_acc: 0.8819
Epoch 157/1000000
391/391 [==============================] - 41s 104ms/step - loss: 0.4804 - acc: 0.9185 - val_loss: 0.5861 - val_acc: 0.8869
Epoch 158/1000000
391/391 [==============================] - 41s 105ms/step - loss: 0.4769 - acc: 0.9200 - val_loss: 0.5919 - val_acc: 0.8884
Epoch 159/1000000
391/391 [==============================] - 41s 105ms/step - loss: 0.4847 - acc: 0.9175 - val_loss: 0.5802 - val_acc: 0.8868
Epoch 160/1000000
391/391 [==============================] - 41s 104ms/step - loss: 0.4822 - acc: 0.9190 - val_loss: 0.5816 - val_acc: 0.8907
Epoch 161/1000000
391/391 [==============================] - 41s 106ms/step - loss: 0.4894 - acc: 0.9171 - val_loss: 0.6214 - val_acc: 0.8747
Epoch 162/1000000
391/391 [==============================] - 42s 107ms/step - loss: 0.4818 - acc: 0.9209 - val_loss: 0.6153 - val_acc: 0.8827
Epoch 163/1000000
391/391 [==============================] - 41s 106ms/step - loss: 0.4872 - acc: 0.9186 - val_loss: 0.6240 - val_acc: 0.8785
Epoch 164/1000000
391/391 [==============================] - 41s 106ms/step - loss: 0.4741 - acc: 0.9234 - val_loss: 0.5811 - val_acc: 0.8924
Epoch 165/1000000
391/391 [==============================] - 41s 104ms/step - loss: 0.4831 - acc: 0.9200 - val_loss: 0.6372 - val_acc: 0.8720
Epoch 166/1000000
391/391 [==============================] - 41s 106ms/step - loss: 0.4864 - acc: 0.9195 - val_loss: 0.6043 - val_acc: 0.8879
Epoch 167/1000000
391/391 [==============================] - 42s 107ms/step - loss: 0.4815 - acc: 0.9224 - val_loss: 0.5960 - val_acc: 0.8916
Epoch 168/1000000
391/391 [==============================] - 42s 108ms/step - loss: 0.4792 - acc: 0.9231 - val_loss: 0.6158 - val_acc: 0.8777
Epoch 169/1000000
391/391 [==============================] - 42s 108ms/step - loss: 0.4790 - acc: 0.9241 - val_loss: 0.5996 - val_acc: 0.8879
Epoch 170/1000000
391/391 [==============================] - 42s 107ms/step - loss: 0.4831 - acc: 0.9208 - val_loss: 0.6415 - val_acc: 0.8785
Epoch 171/1000000
391/391 [==============================] - 41s 106ms/step - loss: 0.4823 - acc: 0.9230 - val_loss: 0.5921 - val_acc: 0.8909
Epoch 172/1000000
391/391 [==============================] - 42s 107ms/step - loss: 0.4849 - acc: 0.9236 - val_loss: 0.5912 - val_acc: 0.8921
Epoch 173/1000000
391/391 [==============================] - 42s 107ms/step - loss: 0.4798 - acc: 0.9242 - val_loss: 0.6081 - val_acc: 0.8827
Epoch 174/1000000
391/391 [==============================] - 41s 104ms/step - loss: 0.4835 - acc: 0.9228 - val_loss: 0.6486 - val_acc: 0.8708
Epoch 175/1000000
391/391 [==============================] - 42s 108ms/step - loss: 0.4822 - acc: 0.9230 - val_loss: 0.6229 - val_acc: 0.8809
Epoch 176/1000000
391/391 [=============================] - 42s 106ms/step - loss: 0.4821 - acc: 0.9232 - val_loss: 0.6232 - val_acc: 0.8805
Epoch 177/1000000
391/391 [==============================] - 43s 109ms/step - loss: 0.4777 - acc: 0.9239 - val_loss: 0.6079 - val_acc: 0.8868
Epoch 178/1000000
391/391 [==============================] - 42s 107ms/step - loss: 0.4755 - acc: 0.9256 - val_loss: 0.6163 - val_acc: 0.8828
Epoch 179/1000000
391/391 [==============================] - 42s 106ms/step - loss: 0.4895 - acc: 0.9208 - val_loss: 0.6248 - val_acc: 0.8838
Epoch 180/1000000
391/391 [==============================] - 42s 106ms/step - loss: 0.4833 - acc: 0.9239 - val_loss: 0.6180 - val_acc: 0.8846
Epoch 181/1000000
391/391 [==============================] - 41s 104ms/step - loss: 0.4816 - acc: 0.9241 - val_loss: 0.6274 - val_acc: 0.8811
Epoch 182/1000000
391/391 [==============================] - 41s 105ms/step - loss: 0.4767 - acc: 0.9269 - val_loss: 0.5805 - val_acc: 0.8995
Epoch 183/1000000
391/391 [==============================] - 41s 105ms/step - loss: 0.4819 - acc: 0.9235 - val_loss: 0.6092 - val_acc: 0.8940
Epoch 184/1000000
391/391 [==============================] - 41s 104ms/step - loss: 0.4846 - acc: 0.9229 - val_loss: 0.6278 - val_acc: 0.8841
Epoch 185/1000000
391/391 [==============================] - 41s 105ms/step - loss: 0.4856 - acc: 0.9247 - val_loss: 0.5998 - val_acc: 0.8886
Epoch 186/1000000
391/391 [==============================] - 41s 106ms/step - loss: 0.4815 - acc: 0.9253 - val_loss: 0.6054 - val_acc: 0.8917
Epoch 187/1000000
391/391 [==============================] - 41s 106ms/step - loss: 0.4914 - acc: 0.9226 - val_loss: 0.6118 - val_acc: 0.8901
Epoch 188/1000000
391/391 [==============================] - 41s 104ms/step - loss: 0.4849 - acc: 0.9265 - val_loss: 0.6184 - val_acc: 0.8874
Epoch 189/1000000
391/391 [==============================] - 41s 105ms/step - loss: 0.4896 - acc: 0.9208 - val_loss: 0.6367 - val_acc: 0.8776
Epoch 190/1000000
391/391 [==============================] - 41s 105ms/step - loss: 0.4875 - acc: 0.9250 - val_loss: 0.6569 - val_acc: 0.8766
Epoch 191/1000000
391/391 [==============================] - 41s 105ms/step - loss: 0.4814 - acc: 0.9266 - val_loss: 0.6414 - val_acc: 0.8832
Epoch 192/1000000
391/391 [==============================] - 41s 105ms/step - loss: 0.4907 - acc: 0.9238 - val_loss: 0.6470 - val_acc: 0.8781
Epoch 193/1000000
391/391 [==============================] - 42s 107ms/step - loss: 0.4819 - acc: 0.9269 - val_loss: 0.6373 - val_acc: 0.8849
Epoch 194/1000000
391/391 [==============================] - 42s 109ms/step - loss: 0.4823 - acc: 0.9249 - val_loss: 0.6135 - val_acc: 0.8878
Epoch 195/1000000
391/391 [==============================] - 42s 107ms/step - loss: 0.4848 - acc: 0.9255 - val_loss: 0.7121 - val_acc: 0.8607
Epoch 196/1000000
391/391 [==============================] - 41s 105ms/step - loss: 0.4864 - acc: 0.9253 - val_loss: 0.6125 - val_acc: 0.8885
Epoch 197/1000000
391/391 [==============================] - 41s 105ms/step - loss: 0.4811 - acc: 0.9258 - val_loss: 0.6961 - val_acc: 0.8685
Epoch 198/1000000
391/391 [==============================] - 41s 106ms/step - loss: 0.4916 - acc: 0.9244 - val_loss: 0.6557 - val_acc: 0.8798
Epoch 199/1000000
391/391 [==============================] - 41s 105ms/step - loss: 0.4871 - acc: 0.9259 - val_loss: 0.6187 - val_acc: 0.8901
Epoch 200/1000000
391/391 [==============================] - 41s 105ms/step - loss: 0.4878 - acc: 0.9249 - val_loss: 0.5913 - val_acc: 0.8926
Epoch 201/1000000
391/391 [==============================] - 41s 105ms/step - loss: 0.4921 - acc: 0.9242 - val_loss: 0.6210 - val_acc: 0.8852
Epoch 202/1000000
391/391 [==============================] - 41s 104ms/step - loss: 0.4851 - acc: 0.9269 - val_loss: 0.6329 - val_acc: 0.8811
Epoch 203/1000000
391/391 [==============================] - 41s 105ms/step - loss: 0.4857 - acc: 0.9277 - val_loss: 0.6520 - val_acc: 0.8754
Epoch 204/1000000
391/391 [==============================] - 41s 104ms/step - loss: 0.4757 - acc: 0.9299 - val_loss: 0.6694 - val_acc: 0.8722
Epoch 205/1000000
391/391 [==============================] - 41s 105ms/step - loss: 0.4847 - acc: 0.9262 - val_loss: 0.6381 - val_acc: 0.8822
Epoch 206/1000000
391/391 [==============================] - 42s 108ms/step - loss: 0.4854 - acc: 0.9270 - val_loss: 0.6727 - val_acc: 0.8800
Epoch 207/1000000
391/391 [==============================] - 41s 104ms/step - loss: 0.4835 - acc: 0.9276 - val_loss: 0.6127 - val_acc: 0.8908
Epoch 208/1000000
391/391 [==============================] - 42s 108ms/step - loss: 0.4835 - acc: 0.9277 - val_loss: 0.6093 - val_acc: 0.8912
Epoch 209/1000000
391/391 [==============================] - 42s 107ms/step - loss: 0.4796 - acc: 0.9280 - val_loss: 0.6262 - val_acc: 0.8901
Epoch 210/1000000
391/391 [==============================] - 43s 109ms/step - loss: 0.4818 - acc: 0.9278 - val_loss: 0.6254 - val_acc: 0.8866
Epoch 211/1000000
391/391 [==============================] - 43s 109ms/step - loss: 0.3739 - acc: 0.9645 - val_loss: 0.5482 - val_acc: 0.9145
Epoch 212/1000000
391/391 [==============================] - 42s 108ms/step - loss: 0.3431 - acc: 0.9729 - val_loss: 0.5312 - val_acc: 0.9206
Epoch 213/1000000
391/391 [==============================] - 41s 105ms/step - loss: 0.3265 - acc: 0.9760 - val_loss: 0.5439 - val_acc: 0.9177
Epoch 214/1000000
391/391 [==============================] - 41s 105ms/step - loss: 0.3150 - acc: 0.9778 - val_loss: 0.5375 - val_acc: 0.9176
Epoch 215/1000000
391/391 [==============================] - 41s 104ms/step - loss: 0.3024 - acc: 0.9802 - val_loss: 0.5433 - val_acc: 0.9158
Epoch 216/1000000
391/391 [==============================] - 41s 105ms/step - loss: 0.2912 - acc: 0.9819 - val_loss: 0.5630 - val_acc: 0.9133
Epoch 217/1000000
391/391 [==============================] - 41s 105ms/step - loss: 0.2822 - acc: 0.9827 - val_loss: 0.5210 - val_acc: 0.9197
Epoch 218/1000000
391/391 [==============================] - 42s 107ms/step - loss: 0.2724 - acc: 0.9844 - val_loss: 0.5364 - val_acc: 0.9168
Epoch 219/1000000
391/391 [==============================] - 42s 107ms/step - loss: 0.2678 - acc: 0.9842 - val_loss: 0.5404 - val_acc: 0.9131
Epoch 220/1000000
391/391 [==============================] - 42s 108ms/step - loss: 0.2611 - acc: 0.9844 - val_loss: 0.5202 - val_acc: 0.9210
Epoch 221/1000000
391/391 [==============================] - 41s 106ms/step - loss: 0.2543 - acc: 0.9853 - val_loss: 0.5172 - val_acc: 0.9204
Epoch 222/1000000
391/391 [==============================] - 41s 105ms/step - loss: 0.2484 - acc: 0.9859 - val_loss: 0.5221 - val_acc: 0.9184
Epoch 223/1000000
391/391 [==============================] - 42s 106ms/step - loss: 0.2452 - acc: 0.9853 - val_loss: 0.5150 - val_acc: 0.9169
Epoch 224/1000000
391/391 [==============================] - 42s 107ms/step - loss: 0.2398 - acc: 0.9853 - val_loss: 0.5098 - val_acc: 0.9194
Epoch 225/1000000
391/391 [==============================] - 41s 104ms/step - loss: 0.2370 - acc: 0.9844 - val_loss: 0.5146 - val_acc: 0.9167
Epoch 226/1000000
391/391 [==============================] - 41s 105ms/step - loss: 0.2338 - acc: 0.9857 - val_loss: 0.5084 - val_acc: 0.9192
Epoch 227/1000000
391/391 [==============================] - 42s 107ms/step - loss: 0.2310 - acc: 0.9846 - val_loss: 0.5260 - val_acc: 0.9119
Epoch 228/1000000
391/391 [==============================] - 42s 108ms/step - loss: 0.2255 - acc: 0.9853 - val_loss: 0.5054 - val_acc: 0.9162
Epoch 229/1000000
391/391 [==============================] - 41s 106ms/step - loss: 0.2238 - acc: 0.9848 - val_loss: 0.5052 - val_acc: 0.9166
Epoch 230/1000000
391/391 [==============================] - 41s 106ms/step - loss: 0.2161 - acc: 0.9866 - val_loss: 0.4971 - val_acc: 0.9180
Epoch 231/1000000
391/391 [==============================] - 40s 103ms/step - loss: 0.2147 - acc: 0.9856 - val_loss: 0.5069 - val_acc: 0.9144
Epoch 232/1000000
391/391 [==============================] - 41s 105ms/step - loss: 0.2173 - acc: 0.9838 - val_loss: 0.4938 - val_acc: 0.9162
Epoch 233/1000000
391/391 [==============================] - 41s 104ms/step - loss: 0.2101 - acc: 0.9853 - val_loss: 0.5050 - val_acc: 0.917
Epoch 234/1000000
391/391 [==============================] - 41s 104ms/step - loss: 0.2129 - acc: 0.9839 - val_loss: 0.4824 - val_acc: 0.9205
Epoch 235/1000000
391/391 [==============================] - 41s 104ms/step - loss: 0.2076 - acc: 0.9850 - val_loss: 0.4821 - val_acc: 0.9170
Epoch 236/1000000
391/391 [==============================] - 42s 108ms/step - loss: 0.2060 - acc: 0.9849 - val_loss: 0.4885 - val_acc: 0.9174
Epoch 237/1000000
391/391 [==============================] - 42s 106ms/step - loss: 0.2048 - acc: 0.9839 - val_loss: 0.4821 - val_acc: 0.9152
Epoch 238/1000000
391/391 [==============================] - 41s 106ms/step - loss: 0.2008 - acc: 0.9855 - val_loss: 0.5182 - val_acc: 0.9124
Epoch 239/1000000
391/391 [==============================] - 41s 106ms/step - loss: 0.2001 - acc: 0.9846 - val_loss: 0.4884 - val_acc: 0.9159
Epoch 240/1000000
391/391 [==============================] - 42s 108ms/step - loss: 0.1989 - acc: 0.9843 - val_loss: 0.5157 - val_acc: 0.9078
Epoch 241/1000000
391/391 [==============================] - 41s 104ms/step - loss: 0.2031 - acc: 0.9819 - val_loss: 0.4668 - val_acc: 0.9158
Epoch 242/1000000
391/391 [==============================] - 41s 105ms/step - loss: 0.1977 - acc: 0.9833 - val_loss: 0.4559 - val_acc: 0.9168
Epoch 243/1000000
391/391 [==============================] - 41s 104ms/step - loss: 0.1991 - acc: 0.9833 - val_loss: 0.4600 - val_acc: 0.9166
Epoch 244/1000000
391/391 [==============================] - 41s 104ms/step - loss: 0.1952 - acc: 0.9837 - val_loss: 0.4940 - val_acc: 0.9085
Epoch 245/1000000
391/391 [==============================] - 41s 104ms/step - loss: 0.1981 - acc: 0.9825 - val_loss: 0.4742 - val_acc: 0.9170
Epoch 246/1000000
391/391 [==============================] - 41s 104ms/step - loss: 0.1946 - acc: 0.9827 - val_loss: 0.4640 - val_acc: 0.9150
Epoch 247/1000000
391/391 [==============================] - 41s 104ms/step - loss: 0.1912 - acc: 0.9837 - val_loss: 0.4948 - val_acc: 0.9097
Epoch 248/1000000
391/391 [==============================] - 41s 104ms/step - loss: 0.1941 - acc: 0.9819 - val_loss: 0.5654 - val_acc: 0.8960
Epoch 249/1000000
391/391 [==============================] - 42s 107ms/step - loss: 0.1941 - acc: 0.9820 - val_loss: 0.4895 - val_acc: 0.9091
Epoch 250/1000000
391/391 [==============================] - 42s 107ms/step - loss: 0.1958 - acc: 0.9812 - val_loss: 0.4984 - val_acc: 0.9074
Epoch 251/1000000
391/391 [==============================] - 41s 106ms/step - loss: 0.1987 - acc: 0.9801 - val_loss: 0.4695 - val_acc: 0.9138
Epoch 252/1000000
391/391 [==============================] - 41s 105ms/step - loss: 0.1914 - acc: 0.9823 - val_loss: 0.4858 - val_acc: 0.9120
Epoch 253/1000000
391/391 [==============================] - 41s 106ms/step - loss: 0.1915 - acc: 0.9823 - val_loss: 0.4960 - val_acc: 0.9096
Epoch 254/1000000
391/391 [==============================] - 41s 105ms/step - loss: 0.1963 - acc: 0.9801 - val_loss: 0.4761 - val_acc: 0.9112
Epoch 255/1000000
391/391 [==============================] - 41s 105ms/step - loss: 0.1914 - acc: 0.9818 - val_loss: 0.4797 - val_acc: 0.9119
Epoch 256/1000000
391/391 [==============================] - 42s 108ms/step - loss: 0.1903 - acc: 0.9821 - val_loss: 0.4776 - val_acc: 0.9147
Epoch 257/1000000
391/391 [==============================] - 42s 107ms/step - loss: 0.1963 - acc: 0.9796 - val_loss: 0.4732 - val_acc: 0.9102
Epoch 258/1000000
391/391 [==============================] - 41s 106ms/step - loss: 0.1894 - acc: 0.9821 - val_loss: 0.4932 - val_acc: 0.9074
Epoch 259/1000000
391/391 [==============================] - 41s 106ms/step - loss: 0.1901 - acc: 0.9815 - val_loss: 0.4726 - val_acc: 0.9164
Epoch 260/1000000
391/391 [==============================] - 42s 106ms/step - loss: 0.1913 - acc: 0.9812 - val_loss: 0.4824 - val_acc: 0.9078
Epoch 261/1000000
391/391 [==============================] - 41s 106ms/step - loss: 0.1916 - acc: 0.9804 - val_loss: 0.4690 - val_acc: 0.9149
Epoch 262/1000000
391/391 [==============================] - 41s 104ms/step - loss: 0.1903 - acc: 0.9814 - val_loss: 0.4855 - val_acc: 0.9115
Epoch 263/1000000
10000/10000 [==============================] - 3s 336us/step
Test loss: 0.482680909777
Test accuracy: 0.9134
$ python examples/cifar10_resnet.py
Using TensorFlow backend
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_1 (InputLayer)            (None, 32, 32, 3)    0                                            
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 32, 32, 16)   432         input_1[0][0]                    
__________________________________________________________________________________________________
layer_normalization_1 (LayerNor (None, 32, 32, 16)   17          conv2d_1[0][0]                   
__________________________________________________________________________________________________
activation_1 (Activation)       (None, 32, 32, 16)   0           layer_normalization_1[0][0]      
__________________________________________________________________________________________________
conv2d_2 (Conv2D)               (None, 32, 32, 32)   4608        activation_1[0][0]               
__________________________________________________________________________________________________
layer_normalization_2 (LayerNor (None, 32, 32, 32)   33          conv2d_2[0][0]                   
__________________________________________________________________________________________________
activation_2 (Activation)       (None, 32, 32, 32)   0           layer_normalization_2[0][0]      
__________________________________________________________________________________________________
conv2d_4 (Conv2D)               (None, 32, 32, 32)   512         conv2d_1[0][0]                   
__________________________________________________________________________________________________
conv2d_3 (Conv2D)               (None, 32, 32, 32)   9216        activation_2[0][0]               
__________________________________________________________________________________________________
add_1 (Add)                     (None, 32, 32, 32)   0           conv2d_4[0][0]                   
                                                                 conv2d_3[0][0]                   
__________________________________________________________________________________________________
layer_normalization_3 (LayerNor (None, 32, 32, 32)   33          add_1[0][0]                      
__________________________________________________________________________________________________
activation_3 (Activation)       (None, 32, 32, 32)   0           layer_normalization_3[0][0]      
__________________________________________________________________________________________________
conv2d_5 (Conv2D)               (None, 32, 32, 32)   9216        activation_3[0][0]               
__________________________________________________________________________________________________
layer_normalization_4 (LayerNor (None, 32, 32, 32)   33          conv2d_5[0][0]                   
__________________________________________________________________________________________________
activation_4 (Activation)       (None, 32, 32, 32)   0           layer_normalization_4[0][0]      
__________________________________________________________________________________________________
conv2d_6 (Conv2D)               (None, 32, 32, 32)   9216        activation_4[0][0]               
__________________________________________________________________________________________________
add_2 (Add)                     (None, 32, 32, 32)   0           add_1[0][0]                      
                                                                 conv2d_6[0][0]                   
__________________________________________________________________________________________________
layer_normalization_5 (LayerNor (None, 32, 32, 32)   33          add_2[0][0]                      
__________________________________________________________________________________________________
activation_5 (Activation)       (None, 32, 32, 32)   0           layer_normalization_5[0][0]      
__________________________________________________________________________________________________
conv2d_7 (Conv2D)               (None, 32, 32, 32)   9216        activation_5[0][0]               
__________________________________________________________________________________________________
layer_normalization_6 (LayerNor (None, 32, 32, 32)   33          conv2d_7[0][0]                   
__________________________________________________________________________________________________
activation_6 (Activation)       (None, 32, 32, 32)   0           layer_normalization_6[0][0]      
__________________________________________________________________________________________________
conv2d_8 (Conv2D)               (None, 32, 32, 32)   9216        activation_6[0][0]               
__________________________________________________________________________________________________
add_3 (Add)                     (None, 32, 32, 32)   0           add_2[0][0]                      
                                                                 conv2d_8[0][0]                   
__________________________________________________________________________________________________
layer_normalization_7 (LayerNor (None, 32, 32, 32)   33          add_3[0][0]                      
__________________________________________________________________________________________________
activation_7 (Activation)       (None, 32, 32, 32)   0           layer_normalization_7[0][0]      
__________________________________________________________________________________________________
conv2d_9 (Conv2D)               (None, 32, 32, 32)   9216        activation_7[0][0]               
__________________________________________________________________________________________________
layer_normalization_8 (LayerNor (None, 32, 32, 32)   33          conv2d_9[0][0]                   
__________________________________________________________________________________________________
activation_8 (Activation)       (None, 32, 32, 32)   0           layer_normalization_8[0][0]      
__________________________________________________________________________________________________
conv2d_10 (Conv2D)              (None, 32, 32, 32)   9216        activation_8[0][0]               
__________________________________________________________________________________________________
add_4 (Add)                     (None, 32, 32, 32)   0           add_3[0][0]                      
                                                                 conv2d_10[0][0]                  
__________________________________________________________________________________________________
layer_normalization_9 (LayerNor (None, 32, 32, 32)   33          add_4[0][0]                      
__________________________________________________________________________________________________
activation_9 (Activation)       (None, 32, 32, 32)   0           layer_normalization_9[0][0]      
__________________________________________________________________________________________________
conv2d_11 (Conv2D)              (None, 16, 16, 64)   18432       activation_9[0][0]               
__________________________________________________________________________________________________
layer_normalization_10 (LayerNo (None, 16, 16, 64)   65          conv2d_11[0][0]                  
__________________________________________________________________________________________________
activation_10 (Activation)      (None, 16, 16, 64)   0           layer_normalization_10[0][0]     
__________________________________________________________________________________________________
conv2d_13 (Conv2D)              (None, 16, 16, 64)   2048        add_4[0][0]                      
__________________________________________________________________________________________________
conv2d_12 (Conv2D)              (None, 16, 16, 64)   36864       activation_10[0][0]              
__________________________________________________________________________________________________
add_5 (Add)                     (None, 16, 16, 64)   0           conv2d_13[0][0]                  
                                                                 conv2d_12[0][0]                  
__________________________________________________________________________________________________
layer_normalization_11 (LayerNo (None, 16, 16, 64)   65          add_5[0][0]                      
__________________________________________________________________________________________________
activation_11 (Activation)      (None, 16, 16, 64)   0           layer_normalization_11[0][0]     
__________________________________________________________________________________________________
conv2d_14 (Conv2D)              (None, 16, 16, 64)   36864       activation_11[0][0]              
__________________________________________________________________________________________________
layer_normalization_12 (LayerNo (None, 16, 16, 64)   65          conv2d_14[0][0]                  
__________________________________________________________________________________________________
activation_12 (Activation)      (None, 16, 16, 64)   0           layer_normalization_12[0][0]     
__________________________________________________________________________________________________
conv2d_15 (Conv2D)              (None, 16, 16, 64)   36864       activation_12[0][0]              
__________________________________________________________________________________________________
add_6 (Add)                     (None, 16, 16, 64)   0           add_5[0][0]                      
                                                                 conv2d_15[0][0]                  
__________________________________________________________________________________________________
layer_normalization_13 (LayerNo (None, 16, 16, 64)   65          add_6[0][0]                      
__________________________________________________________________________________________________
activation_13 (Activation)      (None, 16, 16, 64)   0           layer_normalization_13[0][0]     
__________________________________________________________________________________________________
conv2d_16 (Conv2D)              (None, 16, 16, 64)   36864       activation_13[0][0]              
__________________________________________________________________________________________________
layer_normalization_14 (LayerNo (None, 16, 16, 64)   65          conv2d_16[0][0]                  
__________________________________________________________________________________________________
activation_14 (Activation)      (None, 16, 16, 64)   0           layer_normalization_14[0][0]     
__________________________________________________________________________________________________
conv2d_17 (Conv2D)              (None, 16, 16, 64)   36864       activation_14[0][0]              
__________________________________________________________________________________________________
add_7 (Add)                     (None, 16, 16, 64)   0           add_6[0][0]                      
                                                                 conv2d_17[0][0]                  
__________________________________________________________________________________________________
layer_normalization_15 (LayerNo (None, 16, 16, 64)   65          add_7[0][0]                      
__________________________________________________________________________________________________
activation_15 (Activation)      (None, 16, 16, 64)   0           layer_normalization_15[0][0]     
__________________________________________________________________________________________________
conv2d_18 (Conv2D)              (None, 16, 16, 64)   36864       activation_15[0][0]              
__________________________________________________________________________________________________
layer_normalization_16 (LayerNo (None, 16, 16, 64)   65          conv2d_18[0][0]                  
__________________________________________________________________________________________________
activation_16 (Activation)      (None, 16, 16, 64)   0           layer_normalization_16[0][0]     
__________________________________________________________________________________________________
conv2d_19 (Conv2D)              (None, 16, 16, 64)   36864       activation_16[0][0]              
__________________________________________________________________________________________________
add_8 (Add)                     (None, 16, 16, 64)   0           add_7[0][0]                      
                                                                 conv2d_19[0][0]                  
__________________________________________________________________________________________________
layer_normalization_17 (LayerNo (None, 16, 16, 64)   65          add_8[0][0]                      
__________________________________________________________________________________________________
activation_17 (Activation)      (None, 16, 16, 64)   0           layer_normalization_17[0][0]     
__________________________________________________________________________________________________
conv2d_20 (Conv2D)              (None, 8, 8, 128)    73728       activation_17[0][0]              
__________________________________________________________________________________________________
layer_normalization_18 (LayerNo (None, 8, 8, 128)    129         conv2d_20[0][0]                  
__________________________________________________________________________________________________
activation_18 (Activation)      (None, 8, 8, 128)    0           layer_normalization_18[0][0]     
__________________________________________________________________________________________________
conv2d_22 (Conv2D)              (None, 8, 8, 128)    8192        add_8[0][0]                      
__________________________________________________________________________________________________
conv2d_21 (Conv2D)              (None, 8, 8, 128)    147456      activation_18[0][0]              
__________________________________________________________________________________________________
add_9 (Add)                     (None, 8, 8, 128)    0           conv2d_22[0][0]                  
                                                                 conv2d_21[0][0]                  
__________________________________________________________________________________________________
layer_normalization_19 (LayerNo (None, 8, 8, 128)    129         add_9[0][0]                      
__________________________________________________________________________________________________
activation_19 (Activation)      (None, 8, 8, 128)    0           layer_normalization_19[0][0]     
__________________________________________________________________________________________________
conv2d_23 (Conv2D)              (None, 8, 8, 128)    147456      activation_19[0][0]              
__________________________________________________________________________________________________
layer_normalization_20 (LayerNo (None, 8, 8, 128)    129         conv2d_23[0][0]                  
__________________________________________________________________________________________________
activation_20 (Activation)      (None, 8, 8, 128)    0           layer_normalization_20[0][0]     
__________________________________________________________________________________________________
conv2d_24 (Conv2D)              (None, 8, 8, 128)    147456      activation_20[0][0]              
__________________________________________________________________________________________________
add_10 (Add)                    (None, 8, 8, 128)    0           add_9[0][0]                      
                                                                 conv2d_24[0][0]                  
__________________________________________________________________________________________________
layer_normalization_21 (LayerNo (None, 8, 8, 128)    129         add_10[0][0]                     
__________________________________________________________________________________________________
activation_21 (Activation)      (None, 8, 8, 128)    0           layer_normalization_21[0][0]     
__________________________________________________________________________________________________
conv2d_25 (Conv2D)              (None, 8, 8, 128)    147456      activation_21[0][0]              
__________________________________________________________________________________________________
layer_normalization_22 (LayerNo (None, 8, 8, 128)    129         conv2d_25[0][0]                  
__________________________________________________________________________________________________
activation_22 (Activation)      (None, 8, 8, 128)    0           layer_normalization_22[0][0]     
__________________________________________________________________________________________________
conv2d_26 (Conv2D)              (None, 8, 8, 128)    147456      activation_22[0][0]              
__________________________________________________________________________________________________
add_11 (Add)                    (None, 8, 8, 128)    0           add_10[0][0]                     
                                                                 conv2d_26[0][0]                  
__________________________________________________________________________________________________
layer_normalization_23 (LayerNo (None, 8, 8, 128)    129         add_11[0][0]                     
__________________________________________________________________________________________________
activation_23 (Activation)      (None, 8, 8, 128)    0           layer_normalization_23[0][0]     
__________________________________________________________________________________________________
conv2d_27 (Conv2D)              (None, 8, 8, 128)    147456      activation_23[0][0]              
__________________________________________________________________________________________________
layer_normalization_24 (LayerNo (None, 8, 8, 128)    129         conv2d_27[0][0]                  
__________________________________________________________________________________________________
activation_24 (Activation)      (None, 8, 8, 128)    0           layer_normalization_24[0][0]     
__________________________________________________________________________________________________
conv2d_28 (Conv2D)              (None, 8, 8, 128)    147456      activation_24[0][0]              
__________________________________________________________________________________________________
add_12 (Add)                    (None, 8, 8, 128)    0           add_11[0][0]                     
                                                                 conv2d_28[0][0]                  
__________________________________________________________________________________________________
layer_normalization_25 (LayerNo (None, 8, 8, 128)    129         add_12[0][0]                     
__________________________________________________________________________________________________
activation_25 (Activation)      (None, 8, 8, 128)    0           layer_normalization_25[0][0]     
__________________________________________________________________________________________________
global_average_pooling2d_1 (Glo (None, 128)          0           activation_25[0][0]              
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 10)           1290        global_average_pooling2d_1[0][0] 
__________________________________________________________________________________________________
activation_26 (Activation)      (None, 10)           0           dense_1[0][0]                    
==================================================================================================
Total params: 1,465,827
Trainable params: 1,465,827
Non-trainable params: 0
__________________________________________________________________________________________________
Epoch 1/1000000
391/391 [==============================] - 45s 114ms/step - loss: 2.8618 - accuracy: 0.1474 - val_loss: 2.0379 - val_accuracy: 0.2144
Epoch 2/1000000
391/391 [==============================] - 41s 106ms/step - loss: 2.2304 - accuracy: 0.2536 - val_loss: 1.7864 - val_accuracy: 0.3033
Epoch 3/1000000
391/391 [==============================] - 42s 107ms/step - loss: 1.8926 - accuracy: 0.3537 - val_loss: 1.7058 - val_accuracy: 0.3629
Epoch 4/1000000
391/391 [==============================] - 42s 107ms/step - loss: 1.7106 - accuracy: 0.4178 - val_loss: 1.4440 - val_accuracy: 0.4619
Epoch 5/1000000
391/391 [==============================] - 42s 108ms/step - loss: 1.5761 - accuracy: 0.4749 - val_loss: 1.3550 - val_accuracy: 0.5138
Epoch 6/1000000
391/391 [==============================] - 41s 106ms/step - loss: 1.4463 - accuracy: 0.5344 - val_loss: 1.1383 - val_accuracy: 0.5858
Epoch 7/1000000
391/391 [==============================] - 42s 107ms/step - loss: 1.3773 - accuracy: 0.5729 - val_loss: 1.1476 - val_accuracy: 0.5847
Epoch 8/1000000
391/391 [==============================] - 42s 107ms/step - loss: 1.3316 - accuracy: 0.5962 - val_loss: 1.1391 - val_accuracy: 0.6038
Epoch 9/1000000
391/391 [==============================] - 42s 107ms/step - loss: 1.2624 - accuracy: 0.6276 - val_loss: 0.9808 - val_accuracy: 0.6458
Epoch 10/1000000
391/391 [==============================] - 42s 107ms/step - loss: 1.2161 - accuracy: 0.6471 - val_loss: 0.9586 - val_accuracy: 0.6652
Epoch 11/1000000
391/391 [==============================] - 42s 107ms/step - loss: 1.1728 - accuracy: 0.6709 - val_loss: 0.9819 - val_accuracy: 0.6673
Epoch 12/1000000
391/391 [==============================] - 42s 107ms/step - loss: 1.1418 - accuracy: 0.6930 - val_loss: 1.0263 - val_accuracy: 0.6330
Epoch 13/1000000
391/391 [==============================] - 42s 108ms/step - loss: 1.1286 - accuracy: 0.7016 - val_loss: 0.8451 - val_accuracy: 0.7032
Epoch 14/1000000
391/391 [==============================] - 42s 107ms/step - loss: 1.0867 - accuracy: 0.7252 - val_loss: 0.6922 - val_accuracy: 0.7666
Epoch 15/1000000
391/391 [==============================] - 42s 107ms/step - loss: 1.0489 - accuracy: 0.7404 - val_loss: 0.6907 - val_accuracy: 0.7728
Epoch 16/1000000
391/391 [==============================] - 42s 107ms/step - loss: 1.0460 - accuracy: 0.7464 - val_loss: 0.8025 - val_accuracy: 0.7259
Epoch 17/1000000
391/391 [==============================] - 42s 106ms/step - loss: 1.0208 - accuracy: 0.7613 - val_loss: 0.6967 - val_accuracy: 0.7608
Epoch 18/1000000
391/391 [==============================] - 42s 106ms/step - loss: 1.0158 - accuracy: 0.7630 - val_loss: 0.7540 - val_accuracy: 0.7440
Epoch 19/1000000
391/391 [==============================] - 41s 106ms/step - loss: 1.0014 - accuracy: 0.7739 - val_loss: 0.7563 - val_accuracy: 0.7435
Epoch 20/1000000
391/391 [==============================] - 42s 107ms/step - loss: 0.9912 - accuracy: 0.7771 - val_loss: 0.6857 - val_accuracy: 0.7691
Epoch 21/1000000
391/391 [==============================] - 42s 107ms/step - loss: 0.9944 - accuracy: 0.7767 - val_loss: 0.6769 - val_accuracy: 0.7792
Epoch 22/1000000
391/391 [==============================] - 42s 107ms/step - loss: 0.9841 - accuracy: 0.7838 - val_loss: 0.7571 - val_accuracy: 0.7500
Epoch 23/1000000
391/391 [==============================] - 42s 107ms/step - loss: 0.9798 - accuracy: 0.7873 - val_loss: 0.6389 - val_accuracy: 0.7823
Epoch 24/1000000
391/391 [==============================] - 41s 106ms/step - loss: 0.9923 - accuracy: 0.7855 - val_loss: 0.5956 - val_accuracy: 0.7939
Epoch 25/1000000
391/391 [==============================] - 42s 107ms/step - loss: 0.9743 - accuracy: 0.7923 - val_loss: 0.6064 - val_accuracy: 0.7924
Epoch 26/1000000
391/391 [==============================] - 42s 106ms/step - loss: 0.9842 - accuracy: 0.7902 - val_loss: 0.5862 - val_accuracy: 0.7961
Epoch 27/1000000
391/391 [==============================] - 41s 106ms/step - loss: 0.9589 - accuracy: 0.7976 - val_loss: 0.6102 - val_accuracy: 0.7899
Epoch 28/1000000
391/391 [==============================] - 41s 105ms/step - loss: 0.9712 - accuracy: 0.7951 - val_loss: 0.5583 - val_accuracy: 0.8115
Epoch 29/1000000
391/391 [==============================] - 42s 108ms/step - loss: 0.9808 - accuracy: 0.7942 - val_loss: 0.6415 - val_accuracy: 0.7846
Epoch 30/1000000
391/391 [==============================] - 42s 107ms/step - loss: 0.9651 - accuracy: 0.7994 - val_loss: 0.6237 - val_accuracy: 0.7826
Epoch 31/1000000
391/391 [==============================] - 41s 106ms/step - loss: 0.9553 - accuracy: 0.8049 - val_loss: 0.6349 - val_accuracy: 0.7900
Epoch 32/1000000
391/391 [==============================] - 42s 107ms/step - loss: 0.9673 - accuracy: 0.8013 - val_loss: 0.5543 - val_accuracy: 0.8139
Epoch 33/1000000
391/391 [==============================] - 41s 106ms/step - loss: 0.9550 - accuracy: 0.8040 - val_loss: 0.5577 - val_accuracy: 0.8091
Epoch 34/1000000
391/391 [==============================] - 42s 107ms/step - loss: 0.9599 - accuracy: 0.8049 - val_loss: 0.6227 - val_accuracy: 0.7873
Epoch 35/1000000
391/391 [==============================] - 41s 105ms/step - loss: 0.9577 - accuracy: 0.8065 - val_loss: 0.6008 - val_accuracy: 0.7942
Epoch 36/1000000
391/391 [=============================] - 41s 106ms/step - loss: 0.9561 - accuracy: 0.8039 - val_loss: 0.6736 - val_accuracy: 0.7752
Epoch 37/1000000
391/391 [==============================] - 42s 107ms/step - loss: 0.9628 - accuracy: 0.8078 - val_loss: 0.5997 - val_accuracy: 0.7989
Epoch 38/1000000
391/391 [==============================] - 42s 107ms/step - loss: 0.9604 - accuracy: 0.8069 - val_loss: 0.5559 - val_accuracy: 0.8101
Epoch 39/1000000
391/391 [==============================] - 41s 106ms/step - loss: 0.9434 - accuracy: 0.8111 - val_loss: 0.5898 - val_accuracy: 0.8008
Epoch 40/1000000
391/391 [==============================] - 41s 105ms/step - loss: 0.9588 - accuracy: 0.8078 - val_loss: 0.5802 - val_accuracy: 0.8084
Epoch 41/1000000
391/391 [==============================] - 41s 105ms/step - loss: 0.9565 - accuracy: 0.8091 - val_loss: 0.6098 - val_accuracy: 0.7904
Epoch 42/1000000
391/391 [==============================] - 41s 104ms/step - loss: 0.9537 - accuracy: 0.8101 - val_loss: 0.6275 - val_accuracy: 0.7922
Epoch 43/1000000
391/391 [==============================] - 42s 107ms/step - loss: 0.9422 - accuracy: 0.8127 - val_loss: 0.5963 - val_accuracy: 0.8004
Epoch 44/1000000
391/391 [==============================] - 42s 107ms/step - loss: 0.9606 - accuracy: 0.8091 - val_loss: 0.5746 - val_accuracy: 0.8100
Epoch 45/1000000
391/391 [==============================] - 42s 108ms/step - loss: 0.9453 - accuracy: 0.8143 - val_loss: 0.5981 - val_accuracy: 0.8050
Epoch 46/1000000
391/391 [==============================] - 41s 106ms/step - loss: 0.9382 - accuracy: 0.8162 - val_loss: 0.5571 - val_accuracy: 0.8105
Epoch 47/1000000
391/391 [==============================] - 42s 107ms/step - loss: 0.9478 - accuracy: 0.8119 - val_loss: 0.6097 - val_accuracy: 0.7975
Epoch 48/1000000
391/391 [==============================] - 41s 106ms/step - loss: 0.9435 - accuracy: 0.8143 - val_loss: 0.5251 - val_accuracy: 0.8244
Epoch 49/1000000
391/391 [==============================] - 42s 107ms/step - loss: 0.9395 - accuracy: 0.8161 - val_loss: 0.6848 - val_accuracy: 0.7736
Epoch 50/1000000
391/391 [==============================] - 41s 106ms/step - loss: 0.9350 - accuracy: 0.8174 - val_loss: 0.5727 - val_accuracy: 0.8112
Epoch 51/1000000
391/391 [==============================] - 42s 107ms/step - loss: 0.9349 - accuracy: 0.8173 - val_loss: 0.5645 - val_accuracy: 0.8098
Epoch 52/1000000
391/391 [==============================] - 42s 107ms/step - loss: 0.9311 - accuracy: 0.8191 - val_loss: 0.7249 - val_accuracy: 0.7676
Epoch 53/1000000
391/391 [==============================] - 41s 106ms/step - loss: 0.9393 - accuracy: 0.8162 - val_loss: 0.5679 - val_accuracy: 0.8056
Epoch 54/1000000
391/391 [==============================] - 41s 106ms/step - loss: 0.9416 - accuracy: 0.8156 - val_loss: 0.5429 - val_accuracy: 0.8198
Epoch 55/1000000
391/391 [==============================] - 42s 107ms/step - loss: 0.9415 - accuracy: 0.8169 - val_loss: 0.5271 - val_accuracy: 0.8192
Epoch 56/1000000
391/391 [==============================] - 42s 106ms/step - loss: 0.9411 - accuracy: 0.8170 - val_loss: 0.5952 - val_accuracy: 0.8031
Epoch 57/1000000
391/391 [==============================] - 41s 105ms/step - loss: 0.9317 - accuracy: 0.8203 - val_loss: 0.5988 - val_accuracy: 0.7990
Epoch 58/1000000
391/391 [==============================] - 42s 107ms/step - loss: 0.9409 - accuracy: 0.8161 - val_loss: 0.4925 - val_accuracy: 0.8309
Epoch 59/1000000
391/391 [==============================] - 42s 107ms/step - loss: 0.9476 - accuracy: 0.8169 - val_loss: 0.5116 - val_accuracy: 0.8234
Epoch 60/1000000
391/391 [==============================] - 42s 107ms/step - loss: 0.9349 - accuracy: 0.8201 - val_loss: 0.5197 - val_accuracy: 0.8256
Epoch 61/1000000
391/391 [==============================] - 41s 106ms/step - loss: 0.9260 - accuracy: 0.8229 - val_loss: 0.6445 - val_accuracy: 0.7881
Epoch 62/1000000
391/391 [==============================] - 41s 106ms/step - loss: 0.9317 - accuracy: 0.8192 - val_loss: 0.4849 - val_accuracy: 0.8350
Epoch 63/1000000
391/391 [==============================] - 42s 107ms/step - loss: 0.9387 - accuracy: 0.8178 - val_loss: 0.6540 - val_accuracy: 0.7839
Epoch 64/1000000
391/391 [==============================] - 41s 106ms/step - loss: 0.9263 - accuracy: 0.8195 - val_loss: 0.5168 - val_accuracy: 0.8212
Epoch 65/1000000
391/391 [==============================] - 42s 107ms/step - loss: 0.9302 - accuracy: 0.8223 - val_loss: 0.5854 - val_accuracy: 0.8067
Epoch 66/1000000
391/391 [==============================] - 42s 106ms/step - loss: 0.9324 - accuracy: 0.8187 - val_loss: 0.5256 - val_accuracy: 0.8236
Epoch 67/1000000
391/391 [==============================] - 42s 106ms/step - loss: 0.9342 - accuracy: 0.8200 - val_loss: 0.5621 - val_accuracy: 0.8105
Epoch 68/1000000
391/391 [==============================] - 42s 107ms/step - loss: 0.9267 - accuracy: 0.8215 - val_loss: 0.5627 - val_accuracy: 0.8048
Epoch 69/1000000
391/391 [==============================] - 41s 105ms/step - loss: 0.9359 - accuracy: 0.8207 - val_loss: 0.5791 - val_accuracy: 0.8099
Epoch 70/1000000
391/391 [==============================] - 42s 106ms/step - loss: 0.9326 - accuracy: 0.8205 - val_loss: 0.5548 - val_accuracy: 0.8081
Epoch 71/1000000
391/391 [==============================] - 42s 107ms/step - loss: 0.9307 - accuracy: 0.8229 - val_loss: 0.5781 - val_accuracy: 0.8091
Epoch 72/1000000
391/391 [==============================] - 42s 108ms/step - loss: 0.9330 - accuracy: 0.8205 - val_loss: 0.5420 - val_accuracy: 0.8168
Epoch 73/1000000
391/391 [==============================] - 42s 106ms/step - loss: 0.9345 - accuracy: 0.8205 - val_loss: 0.4809 - val_accuracy: 0.8407
Epoch 74/1000000
391/391 [==============================] - 42s 107ms/step - loss: 0.9251 - accuracy: 0.8230 - val_loss: 0.5429 - val_accuracy: 0.8222
Epoch 75/1000000
391/391 [==============================] - 42s 107ms/step - loss: 0.9321 - accuracy: 0.8190 - val_loss: 0.5324 - val_accuracy: 0.8236
Epoch 76/1000000
391/391 [==============================] - 42s 106ms/step - loss: 0.9224 - accuracy: 0.8269 - val_loss: 0.4952 - val_accuracy: 0.8359
Epoch 77/1000000
391/391 [==============================] - 41s 106ms/step - loss: 0.9207 - accuracy: 0.8257 - val_loss: 0.5444 - val_accuracy: 0.8141
Epoch 78/1000000
391/391 [==============================] - 42s 107ms/step - loss: 0.9233 - accuracy: 0.8241 - val_loss: 0.5818 - val_accuracy: 0.8056
Epoch 79/1000000
391/391 [==============================] - 42s 107ms/step - loss: 0.9169 - accuracy: 0.8270 - val_loss: 0.5800 - val_accuracy: 0.7999
Epoch 80/1000000
391/391 [==============================] - 42s 106ms/step - loss: 0.9199 - accuracy: 0.8249 - val_loss: 0.4955 - val_accuracy: 0.8337
Epoch 81/1000000
391/391 [==============================] - 42s 106ms/step - loss: 0.9260 - accuracy: 0.8237 - val_loss: 0.4912 - val_accuracy: 0.8339
Epoch 82/1000000
391/391 [==============================] - 41s 106ms/step - loss: 0.9105 - accuracy: 0.8291 - val_loss: 0.5494 - val_accuracy: 0.8172
Epoch 83/1000000
391/391 [==============================] - 42s 106ms/step - loss: 0.9294 - accuracy: 0.8213 - val_loss: 0.5765 - val_accuracy: 0.8092
Epoch 84/1000000
391/391 [==============================] - 42s 107ms/step - loss: 0.9195 - accuracy: 0.8267 - val_loss: 0.6724 - val_accuracy: 0.7785
Epoch 85/1000000
391/391 [==============================] - 42s 107ms/step - loss: 0.9236 - accuracy: 0.8276 - val_loss: 0.5385 - val_accuracy: 0.8221
Epoch 86/1000000
391/391 [==============================] - 42s 108ms/step - loss: 0.9436 - accuracy: 0.8184 - val_loss: 0.5195 - val_accuracy: 0.8220
Epoch 87/1000000
391/391 [=============================] - 41s 105ms/step - loss: 0.9341 - accuracy: 0.8246 - val_loss: 0.6649 - val_accuracy: 0.7771
Epoch 88/1000000
391/391 [==============================] - 42s 107ms/step - loss: 0.9215 - accuracy: 0.8276 - val_loss: 0.5140 - val_accuracy: 0.8297
Epoch 89/1000000
391/391 [==============================] - 42s 107ms/step - loss: 0.9323 - accuracy: 0.8228 - val_loss: 0.4970 - val_accuracy: 0.8356
Epoch 90/1000000
391/391 [==============================] - 42s 108ms/step - loss: 0.9273 - accuracy: 0.8253 - val_loss: 0.5083 - val_accuracy: 0.8267
Epoch 91/1000000
391/391 [==============================] - 42s 107ms/step - loss: 0.9099 - accuracy: 0.8294 - val_loss: 0.5002 - val_accuracy: 0.8278
Epoch 92/1000000
391/391 [==============================] - 42s 107ms/step - loss: 0.9244 - accuracy: 0.8241 - val_loss: 0.5479 - val_accuracy: 0.8167
Epoch 93/1000000
391/391 [==============================] - 41s 106ms/step - loss: 0.9235 - accuracy: 0.8272 - val_loss: 0.5294 - val_accuracy: 0.8191
Epoch 94/1000000
391/391 [==============================] - 42s 107ms/step - loss: 0.9250 - accuracy: 0.8234 - val_loss: 0.5718 - val_accuracy: 0.8070
Epoch 95/1000000
391/391 [==============================] - 42s 107ms/step - loss: 0.9181 - accuracy: 0.8280 - val_loss: 0.5027 - val_accuracy: 0.8315
Epoch 96/1000000
391/391 [==============================] - 42s 106ms/step - loss: 0.9275 - accuracy: 0.8227 - val_loss: 0.4765 - val_accuracy: 0.8420
Epoch 97/1000000
391/391 [==============================] - 42s 106ms/step - loss: 0.9163 - accuracy: 0.8291 - val_loss: 0.5639 - val_accuracy: 0.8081
Epoch 98/1000000
391/391 [==============================] - 41s 105ms/step - loss: 0.9354 - accuracy: 0.8234 - val_loss: 0.4848 - val_accuracy: 0.8348
Epoch 99/1000000
391/391 [==============================] - 41s 106ms/step - loss: 0.9127 - accuracy: 0.8292 - val_loss: 0.5609 - val_accuracy: 0.8107
Epoch 100/1000000
391/391 [==============================] - 41s 105ms/step - loss: 0.9403 - accuracy: 0.8227 - val_loss: 0.5251 - val_accuracy: 0.8207
Epoch 101/1000000
391/391 [==============================] - 41s 106ms/step - loss: 0.9225 - accuracy: 0.8272 - val_loss: 0.4599 - val_accuracy: 0.8442
Epoch 102/1000000
391/391 [==============================] - 42s 107ms/step - loss: 0.9141 - accuracy: 0.8289 - val_loss: 0.6462 - val_accuracy: 0.7925
Epoch 103/1000000
391/391 [==============================] - 41s 105ms/step - loss: 0.9278 - accuracy: 0.8239 - val_loss: 0.5226 - val_accuracy: 0.8277
Epoch 104/1000000
391/391 [==============================] - 41s 105ms/step - loss: 0.9281 - accuracy: 0.8271 - val_loss: 0.5904 - val_accuracy: 0.8013
Epoch 105/1000000
391/391 [==============================] - 41s 105ms/step - loss: 0.9297 - accuracy: 0.8231 - val_loss: 0.5923 - val_accuracy: 0.8001
Epoch 106/1000000
391/391 [==============================] - 41s 105ms/step - loss: 0.9228 - accuracy: 0.8284 - val_loss: 0.5622 - val_accuracy: 0.8097
Epoch 107/1000000
391/391 [==============================] - 41s 105ms/step - loss: 0.9294 - accuracy: 0.8270 - val_loss: 0.5350 - val_accuracy: 0.8202
Epoch 108/1000000
391/391 [==============================] - 41s 106ms/step - loss: 0.9182 - accuracy: 0.8288 - val_loss: 0.5958 - val_accuracy: 0.8015
Epoch 109/1000000
391/391 [==============================] - 41s 105ms/step - loss: 0.9135 - accuracy: 0.8299 - val_loss: 0.5402 - val_accuracy: 0.8171
Epoch 110/1000000
391/391 [==============================] - 41s 105ms/step - loss: 0.9176 - accuracy: 0.8276 - val_loss: 0.5341 - val_accuracy: 0.8191
Epoch 111/1000000
391/391 [==============================] - 41s 106ms/step - loss: 0.9227 - accuracy: 0.8275 - val_loss: 0.4907 - val_accuracy: 0.8316
Epoch 112/1000000
391/391 [==============================] - 41s 106ms/step - loss: 0.9162 - accuracy: 0.8287 - val_loss: 0.4940 - val_accuracy: 0.8296
Epoch 113/1000000
391/391 [==============================] - 41s 106ms/step - loss: 0.9122 - accuracy: 0.8306 - val_loss: 0.6474 - val_accuracy: 0.7828
Epoch 114/1000000
391/391 [==============================] - 41s 105ms/step - loss: 0.9106 - accuracy: 0.8305 - val_loss: 0.6062 - val_accuracy: 0.7925
Epoch 115/1000000
391/391 [==============================] - 42s 107ms/step - loss: 0.9198 - accuracy: 0.8273 - val_loss: 0.5709 - val_accuracy: 0.8148
Epoch 116/1000000
391/391 [==============================] - 41s 106ms/step - loss: 0.9079 - accuracy: 0.8326 - val_loss: 0.5106 - val_accuracy: 0.8304
Epoch 117/1000000
391/391 [==============================] - 41s 105ms/step - loss: 0.9190 - accuracy: 0.8298 - val_loss: 0.5031 - val_accuracy: 0.8382
Epoch 118/1000000
391/391 [==============================] - 41s 105ms/step - loss: 0.9086 - accuracy: 0.8308 - val_loss: 0.5333 - val_accuracy: 0.8185
Epoch 119/1000000
391/391 [==============================] - 41s 106ms/step - loss: 0.9173 - accuracy: 0.8280 - val_loss: 0.5819 - val_accuracy: 0.7986
Epoch 120/1000000
391/391 [==============================] - 41s 105ms/step - loss: 0.9226 - accuracy: 0.8278 - val_loss: 0.5943 - val_accuracy: 0.7932
Epoch 121/1000000
391/391 [==============================] - 41s 105ms/step - loss: 0.9202 - accuracy: 0.8269 - val_loss: 0.5029 - val_accuracy: 0.8267
Epoch 122/1000000
391/391 [==============================] - 42s 106ms/step - loss: 0.9148 - accuracy: 0.8309 - val_loss: 0.5111 - val_accuracy: 0.8264
Epoch 123/1000000
391/391 [==============================] - 41s 106ms/step - loss: 0.9114 - accuracy: 0.8291 - val_loss: 0.5850 - val_accuracy: 0.8072
Epoch 124/1000000
391/391 [==============================] - 41s 105ms/step - loss: 0.9188 - accuracy: 0.8290 - val_loss: 0.5057 - val_accuracy: 0.8329
Epoch 125/1000000
391/391 [==============================] - 41s 105ms/step - loss: 0.9189 - accuracy: 0.8304 - val_loss: 0.5643 - val_accuracy: 0.8105
Epoch 126/1000000
391/391 [==============================] - 41s 105ms/step - loss: 0.9072 - accuracy: 0.8330 - val_loss: 0.5258 - val_accuracy: 0.8247
Epoch 127/1000000
391/391 [==============================] - 41s 106ms/step - loss: 0.9022 - accuracy: 0.8330 - val_loss: 0.5785 - val_accuracy: 0.8092
Epoch 128/1000000
391/391 [==============================] - 41s 106ms/step - loss: 0.9065 - accuracy: 0.8304 - val_loss: 0.5543 - val_accuracy: 0.8088
Epoch 129/1000000
391/391 [==============================] - 41s 106ms/step - loss: 0.9185 - accuracy: 0.8271 - val_loss: 0.5529 - val_accuracy: 0.8133
Epoch 130/1000000
391/391 [==============================] - 42s 106ms/step - loss: 0.9012 - accuracy: 0.8338 - val_loss: 0.4363 - val_accuracy: 0.8528
Epoch 131/1000000
391/391 [==============================] - 64s 164ms/step - loss: 0.6621 - accuracy: 0.5326 - val_loss: 0.3127 - val_accuracy: 0.8939
Epoch 132/1000000
391/391 [==============================] - 71s 180ms/step - loss: 0.5515 - accuracy: 0.4494 - val_loss: 0.3238 - val_accuracy: 0.8940
Epoch 133/1000000
391/391 [==============================] - 70s 180ms/step - loss: 0.4912 - accuracy: 0.4465 - val_loss: 0.3039 - val_accuracy: 0.8993
Epoch 134/1000000
391/391 [==============================] - 71s 181ms/step - loss: 0.4457 - accuracy: 0.4570 - val_loss: 0.3145 - val_accuracy: 0.8977
Epoch 135/1000000
391/391 [==============================] - 71s 180ms/step - loss: 0.4063 - accuracy: 0.4635 - val_loss: 0.3357 - val_accuracy: 0.8966
Epoch 136/1000000
391/391 [==============================] - 71s 182ms/step - loss: 0.3854 - accuracy: 0.4599 - val_loss: 0.3440 - val_accuracy: 0.8948
Epoch 137/1000000
391/391 [==============================] - 71s 181ms/step - loss: 0.3662 - accuracy: 0.4661 - val_loss: 0.3529 - val_accuracy: 0.8877
Epoch 138/1000000
391/391 [==============================] - 71s 181ms/step - loss: 0.3550 - accuracy: 0.4578 - val_loss: 0.3261 - val_accuracy: 0.8973
Epoch 139/1000000
391/391 [==============================] - 71s 181ms/step - loss: 0.3463 - accuracy: 0.4654 - val_loss: 0.3446 - val_accuracy: 0.8923
Epoch 140/1000000
391/391 [==============================] - 71s 180ms/step - loss: 0.3415 - accuracy: 0.4581 - val_loss: 0.3487 - val_accuracy: 0.8961
Epoch 141/1000000
391/391 [==============================] - 70s 180ms/step - loss: 0.3326 - accuracy: 0.4631 - val_loss: 0.3287 - val_accuracy: 0.8996
Epoch 142/1000000
391/391 [==============================] - 70s 179ms/step - loss: 0.3366 - accuracy: 0.4536 - val_loss: 0.3619 - val_accuracy: 0.8880
Epoch 143/1000000
391/391 [==============================] - 71s 181ms/step - loss: 0.3349 - accuracy: 0.4592 - val_loss: 0.3441 - val_accuracy: 0.8960
Epoch 144/1000000
391/391 [==============================] - 71s 181ms/step - loss: 0.3276 - accuracy: 0.4595 - val_loss: 0.3354 - val_accuracy: 0.8993
Epoch 145/1000000
391/391 [==============================] - 70s 180ms/step - loss: 0.3291 - accuracy: 0.4608 - val_loss: 0.3579 - val_accuracy: 0.8904
Epoch 146/1000000
391/391 [==============================] - 71s 180ms/step - loss: 0.3260 - accuracy: 0.4613 - val_loss: 0.3424 - val_accuracy: 0.8985
Epoch 147/1000000
391/391 [==============================] - 71s 180ms/step - loss: 0.3263 - accuracy: 0.4609 - val_loss: 0.3621 - val_accuracy: 0.8889
Epoch 148/1000000
391/391 [==============================] - 70s 179ms/step - loss: 0.3262 - accuracy: 0.4591 - val_loss: 0.3463 - val_accuracy: 0.8985
Epoch 149/1000000
391/391 [==============================] - 71s 181ms/step - loss: 0.3268 - accuracy: 0.4643 - val_loss: 0.3232 - val_accuracy: 0.9004
Epoch 150/1000000
391/391 [==============================] - 71s 180ms/step - loss: 0.3206 - accuracy: 0.4632 - val_loss: 0.3300 - val_accuracy: 0.9008
Epoch 151/1000000
391/391 [==============================] - 71s 181ms/step - loss: 0.3167 - accuracy: 0.4631 - val_loss: 0.3517 - val_accuracy: 0.8943
Epoch 152/1000000
391/391 [==============================] - 71s 181ms/step - loss: 0.3232 - accuracy: 0.4630 - val_loss: 0.3768 - val_accuracy: 0.8907
Epoch 153/1000000
391/391 [==============================] - 71s 182ms/step - loss: 0.3217 - accuracy: 0.4610 - val_loss: 0.3712 - val_accuracy: 0.8885
Epoch 154/1000000
391/391 [==============================] - 71s 181ms/step - loss: 0.3154 - accuracy: 0.4628 - val_loss: 0.3394 - val_accuracy: 0.9010
Epoch 155/1000000
391/391 [==============================] - 71s 180ms/step - loss: 0.3240 - accuracy: 0.4644 - val_loss: 0.3653 - val_accuracy: 0.8984
Epoch 156/1000000
391/391 [==============================] - 71s 181ms/step - loss: 0.3208 - accuracy: 0.4627 - val_loss: 0.3533 - val_accuracy: 0.9009
Epoch 157/1000000
391/391 [==============================] - 71s 181ms/step - loss: 0.3150 - accuracy: 0.4717 - val_loss: 0.3380 - val_accuracy: 0.9039
Epoch 158/1000000
391/391 [==============================] - 71s 181ms/step - loss: 0.3175 - accuracy: 0.4641 - val_loss: 0.3640 - val_accuracy: 0.8965
Epoch 159/1000000
391/391 [==============================] - 71s 181ms/step - loss: 0.3131 - accuracy: 0.4615 - val_loss: 0.3539 - val_accuracy: 0.9003
Epoch 160/1000000
391/391 [==============================] - 71s 182ms/step - loss: 0.3131 - accuracy: 0.4627 - val_loss: 0.3607 - val_accuracy: 0.8939
Epoch 161/1000000
391/391 [==============================] - 71s 181ms/step - loss: 0.3156 - accuracy: 0.4714 - val_loss: 0.3674 - val_accuracy: 0.8993
Epoch 162/1000000
391/391 [==============================] - 71s 182ms/step - loss: 0.3144 - accuracy: 0.4671 - val_loss: 0.3902 - val_accuracy: 0.8929
Epoch 163/1000000
391/391 [==============================] - 72s 183ms/step - loss: 0.3159 - accuracy: 0.4629 - val_loss: 0.3306 - val_accuracy: 0.9057
Epoch 164/1000000
391/391 [==============================] - 71s 183ms/step - loss: 0.3123 - accuracy: 0.4694 - val_oss: 0.3304 - val_accuracy: 0.9043
Epoch 165/1000000
391/391 [==============================] - 71s 182ms/step - loss: 0.3165 - accuracy: 0.4632 - val_loss: 0.3431 - val_accuracy: 0.9032
Epoch 166/1000000
391/391 [==============================] - 71s 180ms/step - loss: 0.3161 - accuracy: 0.4630 - val_loss: 0.3720 - val_accuracy: 0.8947
Epoch 167/1000000
391/391 [==============================] - 70s 179ms/step - loss: 0.3136 - accuracy: 0.4645 - val_loss: 0.3658 - val_accuracy: 0.8958
Epoch 168/1000000
391/391 [==============================] - 71s 183ms/step - loss: 0.3124 - accuracy: 0.4669 - val_loss: 0.3670 - val_accuracy: 0.8963
Epoch 169/1000000
391/391 [==============================] - 71s 182ms/step - loss: 0.3164 - accuracy: 0.4610 - val_loss: 0.3482 - val_accuracy: 0.9008
Epoch 170/1000000
391/391 [==============================] - 71s 181ms/step - loss: 0.3086 - accuracy: 0.4769 - val_loss: 0.3728 - val_accuracy: 0.8959
Epoch 171/1000000
391/391 [==============================] - 71s 183ms/step - loss: 0.3145 - accuracy: 0.4652 - val_loss: 0.3468 - val_accuracy: 0.9044
Epoch 172/1000000
391/391 [==============================] - 72s 183ms/step - loss: 0.3134 - accuracy: 0.4679 - val_loss: 0.3376 - val_accuracy: 0.9072
Epoch 173/1000000
391/391 [==============================] - 71s 182ms/step - loss: 0.3055 - accuracy: 0.4716 - val_loss: 0.3275 - val_accuracy: 0.9075
Epoch 174/1000000
391/391 [==============================] - 71s 182ms/step - loss: 0.3098 - accuracy: 0.4645 - val_loss: 0.3510 - val_accuracy: 0.9023
Epoch 175/1000000
391/391 [==============================] - 71s 180ms/step - loss: 0.3106 - accuracy: 0.4722 - val_loss: 0.3715 - val_accuracy: 0.8992
Epoch 176/1000000
391/391 [==============================] - 70s 180ms/step - loss: 0.3092 - accuracy: 0.4795 - val_loss: 0.3102 - val_accuracy: 0.9166
Epoch 177/1000000
391/391 [==============================] - 71s 181ms/step - loss: 0.2319 - accuracy: 0.6116 - val_loss: 0.3111 - val_accuracy: 0.9212
Epoch 178/1000000
391/391 [==============================] - 71s 181ms/step - loss: 0.2150 - accuracy: 0.6546 - val_loss: 0.3281 - val_accuracy: 0.9208
Epoch 179/1000000
391/391 [==============================] - 71s 183ms/step - loss: 0.2048 - accuracy: 0.6709 - val_loss: 0.3353 - val_accuracy: 0.9215
Epoch 180/1000000
391/391 [==============================] - 71s 182ms/step - loss: 0.1969 - accuracy: 0.6900 - val_loss: 0.3463 - val_accuracy: 0.9205
Epoch 181/1000000
391/391 [==============================] - 71s 182ms/step - loss: 0.1900 - accuracy: 0.7076 - val_loss: 0.3436 - val_accuracy: 0.9247
Epoch 182/1000000
391/391 [==============================] - 71s 181ms/step - loss: 0.1831 - accuracy: 0.7326 - val_loss: 0.3467 - val_accuracy: 0.9243
Epoch 183/1000000
391/391 [==============================] - 71s 182ms/step - loss: 0.1776 - accuracy: 0.7369 - val_loss: 0.3561 - val_accuracy: 0.9219
Epoch 184/1000000
391/391 [==============================] - 71s 183ms/step - loss: 0.1726 - accuracy: 0.7188 - val_loss: 0.3588 - val_accuracy: 0.9235
Epoch 185/1000000
391/391 [==============================] - 71s 181ms/step - loss: 0.1666 - accuracy: 0.7653 - val_loss: 0.3666 - val_accuracy: 0.9230
Epoch 186/1000000
391/391 [==============================] - 71s 181ms/step - loss: 0.1617 - accuracy: 0.7649 - val_loss: 0.3744 - val_accuracy: 0.9236
Epoch 187/1000000
391/391 [==============================] - 72s 184ms/step - loss: 0.1566 - accuracy: 0.7738 - val_loss: 0.3660 - val_accuracy: 0.9236
Epoch 188/1000000
391/391 [==============================] - 72s 183ms/step - loss: 0.1522 - accuracy: 0.7683 - val_loss: 0.3683 - val_accuracy: 0.9241
Epoch 189/1000000
391/391 [==============================] - 72s 184ms/step - loss: 0.1475 - accuracy: 0.7856 - val_loss: 0.3746 - val_accuracy: 0.9220
Epoch 190/1000000
391/391 [==============================] - 71s 182ms/step - loss: 0.1445 - accuracy: 0.7368 - val_loss: 0.3701 - val_accuracy: 0.9240
Epoch 191/1000000
391/391 [==============================] - 72s 183ms/step - loss: 0.1396 - accuracy: 0.7828 - val_loss: 0.3764 - val_accuracy: 0.9217
Epoch 192/1000000
391/391 [==============================] - 71s 182ms/step - loss: 0.1364 - accuracy: 0.7546 - val_loss: 0.3777 - val_accuracy: 0.9239
Epoch 193/1000000
391/391 [==============================] - 71s 183ms/step - loss: 0.1332 - accuracy: 0.7470 - val_loss: 0.3659 - val_accuracy: 0.9261
Epoch 194/1000000
391/391 [==============================] - 71s 182ms/step - loss: 0.1286 - accuracy: 0.7663 - val_loss: 0.3680 - val_accuracy: 0.9261
Epoch 195/1000000
391/391 [==============================] - 71s 182ms/step - loss: 0.1245 - accuracy: 0.8015 - val_loss: 0.3804 - val_accuracy: 0.9231
Epoch 196/1000000
391/391 [==============================] - 72s 184ms/step - loss: 0.1223 - accuracy: 0.7552 - val_loss: 0.3861 - val_accuracy: 0.9242
Epoch 197/1000000
391/391 [==============================] - 71s 182ms/step - loss: 0.1188 - accuracy: 0.7582 - val_loss: 0.3825 - val_accuracy: 0.9254
Epoch 198/1000000
391/391 [==============================] - 71s 182ms/step - loss: 0.1164 - accuracy: 0.7417 - val_loss: 0.3916 - val_accuracy: 0.9211
Epoch 199/1000000
391/391 [==============================] - 71s 182ms/step - loss: 0.1144 - accuracy: 0.7421 - val_loss: 0.3839 - val_accuracy: 0.9217
Epoch 200/1000000
391/391 [==============================] - 71s 182ms/step - loss: 0.1112 - accuracy: 0.7242 - val_loss: 0.3593 - val_accuracy: 0.9250
Epoch 201/1000000
391/391 [==============================] - 71s 181ms/step - loss: 0.1088 - accuracy: 0.7280 - val_loss: 0.3881 - val_accuracy: 0.9186
Epoch 202/1000000
391/391 [==============================] - 71s 181ms/step - loss: 0.1054 - accuracy: 0.7476 - val_loss: 0.3717 - val_accuracy: 0.9244
Epoch 203/1000000
391/391 [==============================] - 71s 181ms/step - loss: 0.1039 - accuracy: 0.7255 - val_loss: 0.3884 - val_accuracy: 0.9214
Epoch 204/1000000
391/391 [==============================] - 71s 181ms/step - loss: 0.1011 - accuracy: 0.7229 - val_loss: 0.4099 - val_accuracy: 0.9201
Epoch 205/1000000
391/391 [==============================] - 71s 182ms/step - loss: 0.1014 - accuracy: 0.6718 - val_loss: 0.3800 - val_accuracy: 0.9215
Epoch 206/1000000
391/391 [==============================] - 72s 183ms/step - loss: 0.0992 - accuracy: 0.6658 - val_loss: 0.3910 - val_accuracy: 0.9198
Epoch 207/1000000
10000/10000 [==============================] - 3s 329us/step
Test loss: 0.467169301844
Test accuracy: 0.9213
</code></pre>

[github_is]: https://github.com/idiap/importance-sampling
[models.py]: https://github.com/idiap/importance-sampling/blob/master/importance_sampling/models.py
