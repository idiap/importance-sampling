# Examples

We have included some *Keras* examples in the [repository][github_is], modified
to use importance sampling. In this page we will compare the performance of the
networks in those examples. In all the example we change just the batch
creation method compared to the original Keras examples.

We notice that in all the examples, importance sampling improves the
performance in terms of loss minimization and accuracy but that does not always
translate into faster training. For "real" datasets experimentation with
smoothing parameters is to be expected (see [Training](training.md)).

The examples are run on the CPU and the reported time per epoch is to be taken
with a grain of salt.

## MNIST MLP

In this example we train a multi layer perceptron to classify MNIST digits. The
Keras example runs in **4s** per epoch and **10s** per epoch with importance
sampling. With importance sampling we achieve in just **3 epochs** better
training and validation loss and accuracy than it in 20 epochs with uniform
sampling.

<div class="fig col-2">
<img src="../img/mnist_mlp_train_loss.png" alt="Training Loss">
<img src="../img/mnist_mlp_test_acc.png" alt="Test accuracy">
<span>Results of training a fully connected neural network on MNIST with and
without importance sampling (orange and blue respectively).</span>
</div>

Here follows the terminal output

<pre style="height: 300px; overflow-y: scroll;"><code class="bash">$ python keras/examples/mnist_mlp.py
Using TensorFlow backend.
60000 train samples
10000 test samples
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_1 (Dense)              (None, 512)               401920    
_________________________________________________________________
dropout_1 (Dropout)          (None, 512)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 512)               262656    
_________________________________________________________________
dropout_2 (Dropout)          (None, 512)               0         
_________________________________________________________________
dense_3 (Dense)              (None, 10)                5130      
=================================================================
Total params: 669,706
Trainable params: 669,706
Non-trainable params: 0
_________________________________________________________________
Train on 60000 samples, validate on 10000 samples
Epoch 1/20
60000/60000 [==============================] - 4s - loss: 0.2425 - acc: 0.9263 - val_loss: 0.1071 - val_acc: 0.9654
Epoch 2/20
60000/60000 [==============================] - 4s - loss: 0.1044 - acc: 0.9686 - val_loss: 0.0929 - val_acc: 0.9707
Epoch 3/20
60000/60000 [==============================] - 4s - loss: 0.0755 - acc: 0.9769 - val_loss: 0.0829 - val_acc: 0.9752
Epoch 4/20
60000/60000 [==============================] - 4s - loss: 0.0607 - acc: 0.9817 - val_loss: 0.0720 - val_acc: 0.9799
Epoch 5/20
60000/60000 [==============================] - 4s - loss: 0.0507 - acc: 0.9852 - val_loss: 0.0881 - val_acc: 0.9788
Epoch 6/20
60000/60000 [==============================] - 4s - loss: 0.0447 - acc: 0.9865 - val_loss: 0.0789 - val_acc: 0.9813
Epoch 7/20
60000/60000 [==============================] - 4s - loss: 0.0386 - acc: 0.9884 - val_loss: 0.0784 - val_acc: 0.9806
Epoch 8/20
60000/60000 [==============================] - 4s - loss: 0.0348 - acc: 0.9902 - val_loss: 0.0806 - val_acc: 0.9833
Epoch 9/20
60000/60000 [==============================] - 4s - loss: 0.0328 - acc: 0.9905 - val_loss: 0.0822 - val_acc: 0.9825
Epoch 10/20
60000/60000 [==============================] - 4s - loss: 0.0294 - acc: 0.9917 - val_loss: 0.1060 - val_acc: 0.9782
Epoch 11/20
60000/60000 [==============================] - 4s - loss: 0.0265 - acc: 0.9925 - val_loss: 0.0927 - val_acc: 0.9815
Epoch 12/20
60000/60000 [==============================] - 4s - loss: 0.0253 - acc: 0.9927 - val_loss: 0.0960 - val_acc: 0.9829
Epoch 13/20
60000/60000 [==============================] - 4s - loss: 0.0244 - acc: 0.9934 - val_loss: 0.1085 - val_acc: 0.9809
Epoch 14/20
60000/60000 [==============================] - 4s - loss: 0.0244 - acc: 0.9938 - val_loss: 0.1064 - val_acc: 0.9824
Epoch 15/20
60000/60000 [==============================] - 4s - loss: 0.0221 - acc: 0.9944 - val_loss: 0.1010 - val_acc: 0.9838
Epoch 16/20
60000/60000 [==============================] - 4s - loss: 0.0202 - acc: 0.9947 - val_loss: 0.1080 - val_acc: 0.9831
Epoch 17/20
60000/60000 [==============================] - 4s - loss: 0.0208 - acc: 0.9948 - val_loss: 0.1178 - val_acc: 0.9826
Epoch 18/20
60000/60000 [==============================] - 4s - loss: 0.0208 - acc: 0.9947 - val_loss: 0.1175 - val_acc: 0.9815
Epoch 19/20
60000/60000 [==============================] - 4s - loss: 0.0188 - acc: 0.9952 - val_loss: 0.1144 - val_acc: 0.9831
Epoch 20/20
60000/60000 [==============================] - 4s - loss: 0.0190 - acc: 0.9953 - val_loss: 0.1318 - val_acc: 0.9811
Test loss: 0.131778649402
Test accuracy: 0.9811
$ python importance-sampling/examples/mnist_mlp.py
Using TensorFlow backend.
60000 train samples
10000 test samples
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_1 (Dense)              (None, 512)               401920    
_________________________________________________________________
dropout_1 (Dropout)          (None, 512)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 512)               262656    
_________________________________________________________________
dropout_2 (Dropout)          (None, 512)               0         
_________________________________________________________________
dense_3 (Dense)              (None, 10)                5130      
=================================================================
Total params: 669,706
Trainable params: 669,706
Non-trainable params: 0
_________________________________________________________________
Epoch 1/20
468/468 [==============================] - 11s - loss: 0.1888 - accuracy: 0.5446 - val_loss: 0.0761 - val_accuracy: 0.9770
Epoch 2/20
468/468 [==============================] - 10s - loss: 0.0279 - accuracy: 0.6571 - val_loss: 0.0689 - val_accuracy: 0.9807
Epoch 3/20
468/468 [==============================] - 10s - loss: 0.0093 - accuracy: 0.7260 - val_loss: 0.0675 - val_accuracy: 0.9835
Epoch 4/20
468/468 [==============================] - 10s - loss: 0.0043 - accuracy: 0.7611 - val_loss: 0.0668 - val_accuracy: 0.9858
Epoch 5/20
468/468 [==============================] - 10s - loss: 0.0026 - accuracy: 0.7917 - val_loss: 0.0673 - val_accuracy: 0.9866
Epoch 6/20
468/468 [==============================] - 10s - loss: 0.0018 - accuracy: 0.8379 - val_loss: 0.0780 - val_accuracy: 0.9851
Epoch 7/20
468/468 [==============================] - 10s - loss: 0.0013 - accuracy: 0.8774 - val_loss: 0.0778 - val_accuracy: 0.9857
Epoch 8/20
468/468 [==============================] - 10s - loss: 0.0011 - accuracy: 0.8957 - val_loss: 0.0824 - val_accuracy: 0.9857
Epoch 9/20
468/468 [==============================] - 10s - loss: 9.4248e-04 - accuracy: 0.8870 - val_loss: 0.0890 - val_accuracy: 0.9857
Epoch 10/20
468/468 [==============================] - 10s - loss: 7.0678e-04 - accuracy: 0.9169 - val_loss: 0.0932 - val_accuracy: 0.9849
Epoch 11/20
468/468 [==============================] - 10s - loss: 6.8709e-04 - accuracy: 0.9119 - val_loss: 0.0963 - val_accuracy: 0.9863
Epoch 12/20
468/468 [==============================] - 10s - loss: 5.8700e-04 - accuracy: 0.9272 - val_loss: 0.0957 - val_accuracy: 0.9864
Epoch 13/20
468/468 [==============================] - 10s - loss: 6.5915e-04 - accuracy: 0.9203 - val_loss: 0.1009 - val_accuracy: 0.9865
Epoch 14/20
468/468 [==============================] - 10s - loss: 5.5460e-04 - accuracy: 0.9382 - val_loss: 0.0997 - val_accuracy: 0.9866
Epoch 15/20
468/468 [==============================] - 10s - loss: 5.2162e-04 - accuracy: 0.9488 - val_loss: 0.1026 - val_accuracy: 0.9871
Epoch 16/20
468/468 [==============================] - 10s - loss: 5.5881e-04 - accuracy: 0.9474 - val_loss: 0.1176 - val_accuracy: 0.9848
Epoch 17/20
468/468 [==============================] - 10s - loss: 6.5865e-04 - accuracy: 0.9558 - val_loss: 0.1065 - val_accuracy: 0.9853
Epoch 18/20
468/468 [==============================] - 10s - loss: 6.1609e-04 - accuracy: 0.9596 - val_loss: 0.1180 - val_accuracy: 0.9857
Epoch 19/20
468/468 [==============================] - 10s - loss: 7.2519e-04 - accuracy: 0.9611 - val_loss: 0.1250 - val_accuracy: 0.9846
Epoch 20/20
468/468 [==============================] - 10s - loss: 8.7246e-04 - accuracy: 0.9515 - val_loss: 0.1206 - val_accuracy: 0.9852
Test loss: 0.120589451188
Test accuracy: 0.9852
</code></pre>

## MNIST CNN

This example is pretty much the same as the previous one just with a small CNN
instead of a fully connected network. Instead of plotting the loss and accuracy
with respect to the epochs the following plots show the training loss and test
accuracy with respect to seconds passed.

<div class="fig col-2">
<img src="../img/mnist_cnn_train_loss.png" alt="Training Loss">
<img src="../img/mnist_cnn_test_acc.png" alt="Test accuracy">
<span>Results of training a small CNN on MNIST with and without
importance sampling (orange and blue respectively).</span>
</div>

<pre style="height:300px; overflow-y: scroll;"><code class="bash">$ python keras/examples/mnist_cnn.py
Using TensorFlow backend.
x_train shape: (60000, 28, 28, 1)
60000 train samples
10000 test samples
Train on 60000 samples, validate on 10000 samples
Epoch 1/12
60000/60000 [==============================] - 62s - loss: 0.3292 - acc: 0.9002 - val_loss: 0.0775 - val_acc: 0.9757
Epoch 2/12
60000/60000 [==============================] - 61s - loss: 0.1101 - acc: 0.9669 - val_loss: 0.0538 - val_acc: 0.9829
Epoch 3/12
60000/60000 [==============================] - 61s - loss: 0.0866 - acc: 0.9747 - val_loss: 0.0415 - val_acc: 0.9866
Epoch 4/12
60000/60000 [==============================] - 61s - loss: 0.0705 - acc: 0.9795 - val_loss: 0.0395 - val_acc: 0.9869
Epoch 5/12
60000/60000 [==============================] - 62s - loss: 0.0626 - acc: 0.9815 - val_loss: 0.0340 - val_acc: 0.9888
Epoch 6/12
60000/60000 [==============================] - 62s - loss: 0.0556 - acc: 0.9834 - val_loss: 0.0321 - val_acc: 0.9897
Epoch 7/12
60000/60000 [==============================] - 61s - loss: 0.0501 - acc: 0.9846 - val_loss: 0.0309 - val_acc: 0.9897
Epoch 8/12
60000/60000 [==============================] - 61s - loss: 0.0458 - acc: 0.9863 - val_loss: 0.0296 - val_acc: 0.9895
Epoch 9/12
60000/60000 [==============================] - 61s - loss: 0.0443 - acc: 0.9872 - val_loss: 0.0308 - val_acc: 0.9898
Epoch 10/12
60000/60000 [==============================] - 61s - loss: 0.0407 - acc: 0.9881 - val_loss: 0.0284 - val_acc: 0.9906
Epoch 11/12
60000/60000 [==============================] - 62s - loss: 0.0394 - acc: 0.9880 - val_loss: 0.0291 - val_acc: 0.9899
Epoch 12/12
60000/60000 [==============================] - 61s - loss: 0.0391 - acc: 0.9884 - val_loss: 0.0287 - val_acc: 0.9900
Test loss: 0.0286866371772
Test accuracy: 0.99
$ python importance-sampling/examples/mnist_cnn.py
Using TensorFlow backend.
x_train shape: (60000, 28, 28, 1)
60000 train samples
10000 test samples
Epoch 1/12
468/468 [==============================] - 197s - loss: 0.2969 - accuracy: 0.5715 - val_loss: 0.0676 - val_accuracy: 0.9852
Epoch 2/12
468/468 [==============================] - 196s - loss: 0.0682 - accuracy: 0.6355 - val_loss: 0.0413 - val_accuracy: 0.9891
Epoch 3/12
468/468 [==============================] - 197s - loss: 0.0410 - accuracy: 0.6882 - val_loss: 0.0303 - val_accuracy: 0.9910
Epoch 4/12
468/468 [==============================] - 197s - loss: 0.0283 - accuracy: 0.7351 - val_loss: 0.0269 - val_accuracy: 0.9917
Epoch 5/12
468/468 [==============================] - 197s - loss: 0.0204 - accuracy: 0.7834 - val_loss: 0.0276 - val_accuracy: 0.9914
Epoch 6/12
468/468 [==============================] - 196s - loss: 0.0153 - accuracy: 0.8239 - val_loss: 0.0243 - val_accuracy: 0.9922
Epoch 7/12
468/468 [==============================] - 196s - loss: 0.0119 - accuracy: 0.8554 - val_loss: 0.0250 - val_accuracy: 0.9921
Epoch 8/12
468/468 [==============================] - 196s - loss: 0.0099 - accuracy: 0.8873 - val_loss: 0.0261 - val_accuracy: 0.9920
Epoch 9/12
468/468 [==============================] - 196s - loss: 0.0083 - accuracy: 0.9024 - val_loss: 0.0259 - val_accuracy: 0.9925
Epoch 10/12
468/468 [==============================] - 197s - loss: 0.0073 - accuracy: 0.9174 - val_loss: 0.0260 - val_accuracy: 0.9922
Epoch 11/12
468/468 [==============================] - 197s - loss: 0.0060 - accuracy: 0.9290 - val_loss: 0.0259 - val_accuracy: 0.9932
Epoch 12/12
468/468 [==============================] - 197s - loss: 0.0053 - accuracy: 0.9393 - val_loss: 0.0277 - val_accuracy: 0.9927
Test loss: 0.0277089302262
Test accuracy: 0.9927
</code></pre>

## CIFAR10 CNN

In the CIFAR-10 example we have changed the optimizer from RMSProp to Adam
since the *Keras* example diverged with the RMSProp optimizer. We deviate
further from our testing procedure and train the models with a GPU since it
takes too much time otherwise. The performance difference per epoch is more
than x2 slowdown because the network is very small and the GPU is not fully
utilized.

The following plots show that importance sampling improves both the convergence
speed and the final test score for classifying the CIFAR-10 dataset. However
for this particular combination of batch size and network importance sampling
does not perform very well and it is only in the final epochs that it
outperforms uniform sampling in wall clock time (and that only in the test
set).

<div class="fig col-2">
<img src="../img/cifar_cnn_train_loss.png" alt="Training Loss">
<img src="../img/cifar_cnn_test_loss.png" alt="Test Loss">
<span>Results of training a small CNN on CIFAR10 with and without
importance sampling (orange and blue respectively).</span>
</div>

<pre style="height:300px; overflow-y: scroll;"><code class="bash">$ python keras/examples/cifar10_cnn.py 
Using TensorFlow backend.
2017-08-03 10:59:46.320325: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.1 instructions, but these are available on your machine and could speed up CPU computations.
2017-08-03 10:59:46.320363: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
2017-08-03 10:59:46.320375: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.
2017-08-03 10:59:46.320385: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX2 instructions, but these are available on your machine and could speed up CPU computations.
2017-08-03 10:59:46.320394: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use FMA instructions, but these are available on your machine and could speed up CPU computations.
2017-08-03 10:59:48.919298: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:901] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2017-08-03 10:59:48.920086: I tensorflow/core/common_runtime/gpu/gpu_device.cc:887] Found device 0 with properties:
name: Tesla K80
major: 3 minor: 7 memoryClockRate (GHz) 0.8235
pciBusID 0000:00:04.0
Total memory: 11.17GiB
Free memory: 11.11GiB
2017-08-03 10:59:48.920111: I tensorflow/core/common_runtime/gpu/gpu_device.cc:908] DMA: 0
2017-08-03 10:59:48.920123: I tensorflow/core/common_runtime/gpu/gpu_device.cc:918] 0:   Y
2017-08-03 10:59:48.920138: I tensorflow/core/common_runtime/gpu/gpu_device.cc:977] Creating TensorFlow device (/gpu:0) -> (device: 0, name: Tesla K80, pci bus id: 0000:00:04.0)
x_train shape: (50000, 32, 32, 3)
50000 train samples
10000 test samples
Using real-time data augmentation.
Epoch 1/200
1562/1562 [==============================] - 34s - loss: 1.5801 - acc: 0.4202 - val_loss: 1.2222 - val_acc: 0.5550
Epoch 2/200
1562/1562 [==============================] - 26s - loss: 1.2324 - acc: 0.5557 - val_loss: 0.9762 - val_acc: 0.6582
Epoch 3/200
1562/1562 [==============================] - 27s - loss: 1.0865 - acc: 0.6109 - val_loss: 0.8724 - val_acc: 0.6984
Epoch 4/200
1562/1562 [==============================] - 27s - loss: 1.0017 - acc: 0.6457 - val_loss: 0.8098 - val_acc: 0.7149
Epoch 5/200
1562/1562 [==============================] - 26s - loss: 0.9374 - acc: 0.6696 - val_loss: 0.7574 - val_acc: 0.7357
Epoch 6/200
1562/1562 [==============================] - 27s - loss: 0.9122 - acc: 0.6773 - val_loss: 0.7983 - val_acc: 0.7236
Epoch 7/200
1562/1562 [==============================] - 26s - loss: 0.8799 - acc: 0.6920 - val_loss: 0.7092 - val_acc: 0.7509
Epoch 8/200
1562/1562 [==============================] - 27s - loss: 0.8567 - acc: 0.7026 - val_loss: 0.7107 - val_acc: 0.7509
Epoch 9/200
1562/1562 [==============================] - 27s - loss: 0.8390 - acc: 0.7058 - val_loss: 0.6781 - val_acc: 0.7651
Epoch 10/200
1562/1562 [==============================] - 26s - loss: 0.8199 - acc: 0.7149 - val_loss: 0.6921 - val_acc: 0.7578
Epoch 11/200
1562/1562 [==============================] - 26s - loss: 0.8045 - acc: 0.7206 - val_loss: 0.6910 - val_acc: 0.7615
Epoch 12/200
1562/1562 [==============================] - 27s - loss: 0.7945 - acc: 0.7220 - val_loss: 0.6782 - val_acc: 0.7654
Epoch 13/200
1562/1562 [==============================] - 26s - loss: 0.7851 - acc: 0.7261 - val_loss: 0.6611 - val_acc: 0.7693
Epoch 14/200
1562/1562 [==============================] - 27s - loss: 0.7746 - acc: 0.7311 - val_loss: 0.6763 - val_acc: 0.7678
Epoch 15/200
1562/1562 [==============================] - 27s - loss: 0.7673 - acc: 0.7359 - val_loss: 0.6378 - val_acc: 0.7786
Epoch 16/200
1562/1562 [==============================] - 26s - loss: 0.7541 - acc: 0.7390 - val_loss: 0.6228 - val_acc: 0.7859
Epoch 17/200
1562/1562 [==============================] - 26s - loss: 0.7533 - acc: 0.7401 - val_loss: 0.6301 - val_acc: 0.7872
Epoch 18/200
1562/1562 [==============================] - 26s - loss: 0.7459 - acc: 0.7403 - val_loss: 0.6191 - val_acc: 0.7860
Epoch 19/200
1562/1562 [==============================] - 26s - loss: 0.7408 - acc: 0.7433 - val_loss: 0.5902 - val_acc: 0.8018
Epoch 20/200
1562/1562 [==============================] - 27s - loss: 0.7346 - acc: 0.7491 - val_loss: 0.5958 - val_acc: 0.7954
Epoch 21/200
1562/1562 [==============================] - 27s - loss: 0.7391 - acc: 0.7445 - val_loss: 0.6039 - val_acc: 0.7945
Epoch 22/200
1562/1562 [==============================] - 26s - loss: 0.7271 - acc: 0.7478 - val_loss: 0.6009 - val_acc: 0.7933
Epoch 23/200
1562/1562 [==============================] - 26s - loss: 0.7215 - acc: 0.7507 - val_loss: 0.6473 - val_acc: 0.7761
Epoch 24/200
1562/1562 [==============================] - 26s - loss: 0.7182 - acc: 0.7536 - val_loss: 0.5682 - val_acc: 0.8104
Epoch 25/200
1562/1562 [==============================] - 26s - loss: 0.7186 - acc: 0.7552 - val_loss: 0.6482 - val_acc: 0.7791
Epoch 26/200
1562/1562 [==============================] - 26s - loss: 0.7111 - acc: 0.7538 - val_loss: 0.5921 - val_acc: 0.8001
Epoch 27/200
1562/1562 [==============================] - 26s - loss: 0.7075 - acc: 0.7549 - val_loss: 0.5953 - val_acc: 0.7988
Epoch 28/200
1562/1562 [==============================] - 26s - loss: 0.7059 - acc: 0.7572 - val_loss: 0.5980 - val_acc: 0.7996
Epoch 29/200
1562/1562 [==============================] - 26s - loss: 0.7049 - acc: 0.7586 - val_loss: 0.6020 - val_acc: 0.7986
Epoch 30/200
1562/1562 [==============================] - 27s - loss: 0.7040 - acc: 0.7585 - val_loss: 0.5808 - val_acc: 0.8030
Epoch 31/200
1562/1562 [==============================] - 27s - loss: 0.6983 - acc: 0.7614 - val_loss: 0.5869 - val_acc: 0.8008
Epoch 32/200
1562/1562 [==============================] - 27s - loss: 0.6931 - acc: 0.7585 - val_loss: 0.5761 - val_acc: 0.7988
Epoch 33/200
1562/1562 [==============================] - 27s - loss: 0.6933 - acc: 0.7594 - val_loss: 0.5891 - val_acc: 0.8012
Epoch 34/200
1562/1562 [==============================] - 26s - loss: 0.6980 - acc: 0.7617 - val_loss: 0.6053 - val_acc: 0.7954
Epoch 35/200
1562/1562 [==============================] - 26s - loss: 0.6898 - acc: 0.7628 - val_loss: 0.5535 - val_acc: 0.8119
Epoch 36/200
1562/1562 [==============================] - 27s - loss: 0.6880 - acc: 0.7644 - val_loss: 0.6174 - val_acc: 0.7899
Epoch 37/200
1562/1562 [==============================] - 27s - loss: 0.6833 - acc: 0.7647 - val_loss: 0.6075 - val_acc: 0.7974
Epoch 38/200
1562/1562 [==============================] - 26s - loss: 0.6831 - acc: 0.7652 - val_loss: 0.5534 - val_acc: 0.8106
Epoch 39/200
1562/1562 [==============================] - 29s - loss: 0.6848 - acc: 0.7634 - val_loss: 0.5589 - val_acc: 0.8107
Epoch 40/200
1562/1562 [==============================] - 29s - loss: 0.6847 - acc: 0.7653 - val_loss: 0.5534 - val_acc: 0.8112
Epoch 41/200
1562/1562 [==============================] - 28s - loss: 0.6829 - acc: 0.7663 - val_loss: 0.5746 - val_acc: 0.8051
Epoch 42/200
1562/1562 [==============================] - 28s - loss: 0.6783 - acc: 0.7682 - val_loss: 0.5551 - val_acc: 0.8119
Epoch 43/200
1562/1562 [==============================] - 30s - loss: 0.6733 - acc: 0.7687 - val_loss: 0.5476 - val_acc: 0.8142
Epoch 44/200
1562/1562 [==============================] - 29s - loss: 0.6782 - acc: 0.7671 - val_loss: 0.5618 - val_acc: 0.8119
Epoch 45/200
1562/1562 [==============================] - 29s - loss: 0.6758 - acc: 0.7686 - val_loss: 0.5321 - val_acc: 0.8228
Epoch 46/200
1562/1562 [==============================] - 29s - loss: 0.6732 - acc: 0.7690 - val_loss: 0.5414 - val_acc: 0.8181
Epoch 47/200
1562/1562 [==============================] - 30s - loss: 0.6690 - acc: 0.7710 - val_loss: 0.5374 - val_acc: 0.8208
Epoch 48/200
1562/1562 [==============================] - 29s - loss: 0.6705 - acc: 0.7696 - val_loss: 0.5523 - val_acc: 0.8141
Epoch 49/200
1562/1562 [==============================] - 28s - loss: 0.6764 - acc: 0.7692 - val_loss: 0.5343 - val_acc: 0.8198
Epoch 50/200
1562/1562 [==============================] - 28s - loss: 0.6631 - acc: 0.7732 - val_loss: 0.5491 - val_acc: 0.8136
Epoch 51/200
1562/1562 [==============================] - 27s - loss: 0.6694 - acc: 0.7718 - val_loss: 0.5786 - val_acc: 0.8035
Epoch 52/200
1562/1562 [==============================] - 27s - loss: 0.6639 - acc: 0.7725 - val_loss: 0.5349 - val_acc: 0.8178
Epoch 53/200
1562/1562 [==============================] - 27s - loss: 0.6693 - acc: 0.7692 - val_loss: 0.5684 - val_acc: 0.8103
Epoch 54/200
1562/1562 [==============================] - 27s - loss: 0.6622 - acc: 0.7731 - val_loss: 0.5621 - val_acc: 0.8074
Epoch 55/200
1562/1562 [==============================] - 27s - loss: 0.6565 - acc: 0.7742 - val_loss: 0.5491 - val_acc: 0.8169
Epoch 56/200
1562/1562 [==============================] - 27s - loss: 0.6583 - acc: 0.7748 - val_loss: 0.5349 - val_acc: 0.8177
Epoch 57/200
1562/1562 [==============================] - 27s - loss: 0.6658 - acc: 0.7730 - val_loss: 0.5471 - val_acc: 0.8133
Epoch 58/200
1562/1562 [==============================] - 27s - loss: 0.6578 - acc: 0.7764 - val_loss: 0.5686 - val_acc: 0.8082
Epoch 59/200
1562/1562 [==============================] - 28s - loss: 0.6596 - acc: 0.7734 - val_loss: 0.5547 - val_acc: 0.8174
Epoch 60/200
1562/1562 [==============================] - 27s - loss: 0.6587 - acc: 0.7742 - val_loss: 0.5491 - val_acc: 0.8097
Epoch 61/200
1562/1562 [==============================] - 27s - loss: 0.6585 - acc: 0.7742 - val_loss: 0.5308 - val_acc: 0.8224
Epoch 62/200
1562/1562 [==============================] - 27s - loss: 0.6612 - acc: 0.7730 - val_loss: 0.5564 - val_acc: 0.8137
Epoch 63/200
1562/1562 [==============================] - 27s - loss: 0.6598 - acc: 0.7753 - val_loss: 0.5438 - val_acc: 0.8194
Epoch 64/200
1562/1562 [==============================] - 27s - loss: 0.6537 - acc: 0.7761 - val_loss: 0.5250 - val_acc: 0.8222
Epoch 65/200
1562/1562 [==============================] - 28s - loss: 0.6503 - acc: 0.7767 - val_loss: 0.5070 - val_acc: 0.8272
Epoch 66/200
1562/1562 [==============================] - 28s - loss: 0.6531 - acc: 0.7764 - val_loss: 0.5435 - val_acc: 0.8178
Epoch 67/200
1562/1562 [==============================] - 28s - loss: 0.6546 - acc: 0.7773 - val_loss: 0.5137 - val_acc: 0.8273
Epoch 68/200
1562/1562 [==============================] - 28s - loss: 0.6498 - acc: 0.7789 - val_loss: 0.5523 - val_acc: 0.8170
Epoch 69/200
1562/1562 [==============================] - 27s - loss: 0.6602 - acc: 0.7728 - val_loss: 0.5189 - val_acc: 0.8263
Epoch 70/200
1562/1562 [==============================] - 28s - loss: 0.6539 - acc: 0.7762 - val_loss: 0.5683 - val_acc: 0.8077
Epoch 71/200
1562/1562 [==============================] - 27s - loss: 0.6518 - acc: 0.7784 - val_loss: 0.5397 - val_acc: 0.8196
Epoch 72/200
1562/1562 [==============================] - 28s - loss: 0.6509 - acc: 0.7795 - val_loss: 0.5440 - val_acc: 0.8199
Epoch 73/200
1562/1562 [==============================] - 28s - loss: 0.6493 - acc: 0.7796 - val_loss: 0.5524 - val_acc: 0.8151
Epoch 74/200
1562/1562 [==============================] - 28s - loss: 0.6548 - acc: 0.7768 - val_loss: 0.5663 - val_acc: 0.8091
Epoch 75/200
1562/1562 [==============================] - 29s - loss: 0.6503 - acc: 0.7773 - val_loss: 0.5114 - val_acc: 0.8251
Epoch 76/200
1562/1562 [==============================] - 28s - loss: 0.6385 - acc: 0.7812 - val_loss: 0.5433 - val_acc: 0.8184
Epoch 77/200
1562/1562 [==============================] - 27s - loss: 0.6375 - acc: 0.7835 - val_loss: 0.5468 - val_acc: 0.8189
Epoch 78/200
1562/1562 [==============================] - 27s - loss: 0.6434 - acc: 0.7826 - val_loss: 0.5346 - val_acc: 0.8207
Epoch 79/200
1562/1562 [==============================] - 27s - loss: 0.6386 - acc: 0.7827 - val_loss: 0.5453 - val_acc: 0.8180
Epoch 80/200
1562/1562 [==============================] - 28s - loss: 0.6442 - acc: 0.7809 - val_loss: 0.5404 - val_acc: 0.8158
Epoch 81/200
1562/1562 [==============================] - 28s - loss: 0.6425 - acc: 0.7822 - val_loss: 0.5203 - val_acc: 0.8245
Epoch 82/200
1562/1562 [==============================] - 30s - loss: 0.6498 - acc: 0.7784 - val_loss: 0.5438 - val_acc: 0.8198
Epoch 83/200
1562/1562 [==============================] - 30s - loss: 0.6448 - acc: 0.7782 - val_loss: 0.5423 - val_acc: 0.8172
Epoch 84/200
1562/1562 [==============================] - 28s - loss: 0.6405 - acc: 0.7811 - val_loss: 0.5399 - val_acc: 0.8165
Epoch 85/200
1562/1562 [==============================] - 28s - loss: 0.6436 - acc: 0.7791 - val_loss: 0.5246 - val_acc: 0.8220
Epoch 86/200
1562/1562 [==============================] - 27s - loss: 0.6411 - acc: 0.7806 - val_loss: 0.5086 - val_acc: 0.8262
Epoch 87/200
1562/1562 [==============================] - 28s - loss: 0.6371 - acc: 0.7834 - val_loss: 0.5272 - val_acc: 0.8201
Epoch 88/200
1562/1562 [==============================] - 27s - loss: 0.6404 - acc: 0.7820 - val_loss: 0.5064 - val_acc: 0.8291
Epoch 89/200
1562/1562 [==============================] - 27s - loss: 0.6423 - acc: 0.7805 - val_loss: 0.5412 - val_acc: 0.8198
Epoch 90/200
1562/1562 [==============================] - 27s - loss: 0.6431 - acc: 0.7818 - val_loss: 0.5270 - val_acc: 0.8250
Epoch 91/200
1562/1562 [==============================] - 27s - loss: 0.6356 - acc: 0.7827 - val_loss: 0.5419 - val_acc: 0.8162
Epoch 92/200
1562/1562 [==============================] - 27s - loss: 0.6422 - acc: 0.7806 - val_loss: 0.5427 - val_acc: 0.8146
Epoch 93/200
1562/1562 [==============================] - 28s - loss: 0.6415 - acc: 0.7818 - val_loss: 0.5257 - val_acc: 0.8211
Epoch 94/200
1562/1562 [==============================] - 27s - loss: 0.6420 - acc: 0.7797 - val_loss: 0.5173 - val_acc: 0.8232
Epoch 95/200
1562/1562 [==============================] - 27s - loss: 0.6408 - acc: 0.7813 - val_loss: 0.5258 - val_acc: 0.8246
Epoch 96/200
1562/1562 [==============================] - 27s - loss: 0.6381 - acc: 0.7839 - val_loss: 0.5404 - val_acc: 0.8162
Epoch 97/200
1562/1562 [==============================] - 27s - loss: 0.6364 - acc: 0.7835 - val_loss: 0.5758 - val_acc: 0.8050
Epoch 98/200
1562/1562 [==============================] - 28s - loss: 0.6491 - acc: 0.7775 - val_loss: 0.5499 - val_acc: 0.8180
Epoch 99/200
1562/1562 [==============================] - 27s - loss: 0.6437 - acc: 0.7828 - val_loss: 0.5451 - val_acc: 0.8176
Epoch 100/200
1562/1562 [==============================] - 27s - loss: 0.6389 - acc: 0.7828 - val_loss: 0.5408 - val_acc: 0.8200
Epoch 101/200
1562/1562 [==============================] - 28s - loss: 0.6433 - acc: 0.7832 - val_loss: 0.5144 - val_acc: 0.8293
Epoch 102/200
1562/1562 [==============================] - 28s - loss: 0.6369 - acc: 0.7809 - val_loss: 0.5217 - val_acc: 0.8275
Epoch 103/200
1562/1562 [==============================] - 29s - loss: 0.6325 - acc: 0.7849 - val_loss: 0.5623 - val_acc: 0.8128
Epoch 104/200
1562/1562 [==============================] - 28s - loss: 0.6414 - acc: 0.7820 - val_loss: 0.5527 - val_acc: 0.8187
Epoch 105/200
1562/1562 [==============================] - 28s - loss: 0.6325 - acc: 0.7861 - val_loss: 0.5331 - val_acc: 0.8199
Epoch 106/200
1562/1562 [==============================] - 28s - loss: 0.6384 - acc: 0.7818 - val_loss: 0.5173 - val_acc: 0.8300
Epoch 107/200
1562/1562 [==============================] - 28s - loss: 0.6306 - acc: 0.7862 - val_loss: 0.5084 - val_acc: 0.8308
Epoch 108/200
1562/1562 [==============================] - 27s - loss: 0.6295 - acc: 0.7860 - val_loss: 0.5201 - val_acc: 0.8288
Epoch 109/200
1562/1562 [==============================] - 27s - loss: 0.6344 - acc: 0.7859 - val_loss: 0.5156 - val_acc: 0.8252
Epoch 110/200
1562/1562 [==============================] - 27s - loss: 0.6373 - acc: 0.7811 - val_loss: 0.5454 - val_acc: 0.8165
Epoch 111/200
1562/1562 [==============================] - 27s - loss: 0.6415 - acc: 0.7834 - val_loss: 0.5677 - val_acc: 0.8140
Epoch 112/200
1562/1562 [==============================] - 27s - loss: 0.6368 - acc: 0.7838 - val_loss: 0.5589 - val_acc: 0.8117
Epoch 113/200
1562/1562 [==============================] - 27s - loss: 0.6330 - acc: 0.7846 - val_loss: 0.5335 - val_acc: 0.8219
Epoch 114/200
1562/1562 [==============================] - 27s - loss: 0.6320 - acc: 0.7870 - val_loss: 0.5578 - val_acc: 0.8121
Epoch 115/200
1562/1562 [==============================] - 27s - loss: 0.6347 - acc: 0.7868 - val_loss: 0.5215 - val_acc: 0.8214
Epoch 116/200
1562/1562 [==============================] - 27s - loss: 0.6315 - acc: 0.7872 - val_loss: 0.5093 - val_acc: 0.8312
Epoch 117/200
1562/1562 [==============================] - 27s - loss: 0.6345 - acc: 0.7858 - val_loss: 0.5421 - val_acc: 0.8197
Epoch 118/200
1562/1562 [==============================] - 27s - loss: 0.6368 - acc: 0.7848 - val_loss: 0.5432 - val_acc: 0.8172
Epoch 119/200
1562/1562 [==============================] - 27s - loss: 0.6346 - acc: 0.7841 - val_loss: 0.5077 - val_acc: 0.8318
Epoch 120/200
1562/1562 [==============================] - 27s - loss: 0.6385 - acc: 0.7842 - val_loss: 0.4953 - val_acc: 0.8322
Epoch 121/200
1562/1562 [==============================] - 27s - loss: 0.6286 - acc: 0.7856 - val_loss: 0.5679 - val_acc: 0.8170
Epoch 122/200
1562/1562 [==============================] - 27s - loss: 0.6290 - acc: 0.7887 - val_loss: 0.5226 - val_acc: 0.8281
Epoch 123/200
1562/1562 [==============================] - 27s - loss: 0.6341 - acc: 0.7870 - val_loss: 0.5399 - val_acc: 0.8225
Epoch 124/200
1562/1562 [==============================] - 27s - loss: 0.6270 - acc: 0.7870 - val_loss: 0.5499 - val_acc: 0.8177
Epoch 125/200
1562/1562 [==============================] - 27s - loss: 0.6315 - acc: 0.7852 - val_loss: 0.5279 - val_acc: 0.8244
Epoch 126/200
1562/1562 [==============================] - 27s - loss: 0.6320 - acc: 0.7856 - val_loss: 0.5279 - val_acc: 0.8279
Epoch 127/200
1562/1562 [==============================] - 27s - loss: 0.6267 - acc: 0.7872 - val_loss: 0.5651 - val_acc: 0.8158
Epoch 128/200
1562/1562 [==============================] - 27s - loss: 0.6284 - acc: 0.7891 - val_loss: 0.5255 - val_acc: 0.8233
Epoch 129/200
1562/1562 [==============================] - 29s - loss: 0.6342 - acc: 0.7856 - val_loss: 0.5208 - val_acc: 0.8261
Epoch 130/200
1562/1562 [==============================] - 29s - loss: 0.6328 - acc: 0.7837 - val_loss: 0.5339 - val_acc: 0.8198
Epoch 131/200
1562/1562 [==============================] - 27s - loss: 0.6272 - acc: 0.7876 - val_loss: 0.5115 - val_acc: 0.8283
Epoch 132/200
1562/1562 [==============================] - 28s - loss: 0.6341 - acc: 0.7862 - val_loss: 0.5420 - val_acc: 0.8202
Epoch 133/200
1562/1562 [==============================] - 28s - loss: 0.6377 - acc: 0.7852 - val_loss: 0.5105 - val_acc: 0.8333
Epoch 134/200
1562/1562 [==============================] - 28s - loss: 0.6288 - acc: 0.7865 - val_loss: 0.5489 - val_acc: 0.8157
Epoch 135/200
1562/1562 [==============================] - 27s - loss: 0.6226 - acc: 0.7885 - val_loss: 0.5412 - val_acc: 0.8164
Epoch 136/200
1562/1562 [==============================] - 28s - loss: 0.6262 - acc: 0.7877 - val_loss: 0.5015 - val_acc: 0.8339
Epoch 137/200
1562/1562 [==============================] - 27s - loss: 0.6272 - acc: 0.7871 - val_loss: 0.5160 - val_acc: 0.8295
Epoch 138/200
1562/1562 [==============================] - 27s - loss: 0.6309 - acc: 0.7860 - val_loss: 0.5025 - val_acc: 0.8312
Epoch 139/200
1562/1562 [==============================] - 28s - loss: 0.6276 - acc: 0.7875 - val_loss: 0.5545 - val_acc: 0.8206
Epoch 140/200
1562/1562 [==============================] - 27s - loss: 0.6227 - acc: 0.7896 - val_loss: 0.5191 - val_acc: 0.8300
Epoch 141/200
1562/1562 [==============================] - 27s - loss: 0.6274 - acc: 0.7896 - val_loss: 0.5606 - val_acc: 0.8128
Epoch 142/200
1562/1562 [==============================] - 28s - loss: 0.6301 - acc: 0.7867 - val_loss: 0.5395 - val_acc: 0.8180
Epoch 143/200
1562/1562 [==============================] - 27s - loss: 0.6241 - acc: 0.7891 - val_loss: 0.5130 - val_acc: 0.8248
Epoch 144/200
1562/1562 [==============================] - 27s - loss: 0.6305 - acc: 0.7878 - val_loss: 0.5012 - val_acc: 0.8329
Epoch 145/200
1562/1562 [==============================] - 27s - loss: 0.6339 - acc: 0.7855 - val_loss: 0.5524 - val_acc: 0.8157
Epoch 146/200
1562/1562 [==============================] - 27s - loss: 0.6215 - acc: 0.7889 - val_loss: 0.5286 - val_acc: 0.8203
Epoch 147/200
1562/1562 [==============================] - 28s - loss: 0.6249 - acc: 0.7876 - val_loss: 0.5380 - val_acc: 0.8217
Epoch 148/200
1562/1562 [==============================] - 27s - loss: 0.6386 - acc: 0.7826 - val_loss: 0.5076 - val_acc: 0.8328
Epoch 149/200
1562/1562 [==============================] - 27s - loss: 0.6240 - acc: 0.7895 - val_loss: 0.5495 - val_acc: 0.8164
Epoch 150/200
1562/1562 [==============================] - 28s - loss: 0.6290 - acc: 0.7854 - val_loss: 0.5536 - val_acc: 0.8138
Epoch 151/200
1562/1562 [==============================] - 27s - loss: 0.6285 - acc: 0.7878 - val_loss: 0.5173 - val_acc: 0.8313
Epoch 152/200
1562/1562 [==============================] - 29s - loss: 0.6295 - acc: 0.7859 - val_loss: 0.4931 - val_acc: 0.8329
Epoch 153/200
1562/1562 [==============================] - 28s - loss: 0.6250 - acc: 0.7879 - val_loss: 0.5485 - val_acc: 0.8179
Epoch 154/200
1562/1562 [==============================] - 29s - loss: 0.6235 - acc: 0.7903 - val_loss: 0.5413 - val_acc: 0.8231
Epoch 155/200
1562/1562 [==============================] - 29s - loss: 0.6313 - acc: 0.7858 - val_loss: 0.5149 - val_acc: 0.8316
Epoch 156/200
1562/1562 [==============================] - 28s - loss: 0.6190 - acc: 0.7899 - val_loss: 0.5452 - val_acc: 0.8214
Epoch 157/200
1562/1562 [==============================] - 29s - loss: 0.6245 - acc: 0.7902 - val_loss: 0.5306 - val_acc: 0.8235
Epoch 158/200
1562/1562 [==============================] - 27s - loss: 0.6273 - acc: 0.7872 - val_loss: 0.5589 - val_acc: 0.8160
Epoch 159/200
1562/1562 [==============================] - 28s - loss: 0.6271 - acc: 0.7856 - val_loss: 0.5276 - val_acc: 0.8275
Epoch 160/200
1562/1562 [==============================] - 27s - loss: 0.6197 - acc: 0.7892 - val_loss: 0.4964 - val_acc: 0.8339
Epoch 161/200
1562/1562 [==============================] - 27s - loss: 0.6234 - acc: 0.7888 - val_loss: 0.5368 - val_acc: 0.8254
Epoch 162/200
1562/1562 [==============================] - 28s - loss: 0.6224 - acc: 0.7920 - val_loss: 0.5125 - val_acc: 0.8290
Epoch 163/200
1562/1562 [==============================] - 27s - loss: 0.6220 - acc: 0.7900 - val_loss: 0.5363 - val_acc: 0.8169
Epoch 164/200
1562/1562 [==============================] - 28s - loss: 0.6266 - acc: 0.7866 - val_loss: 0.5124 - val_acc: 0.8308
Epoch 165/200
1562/1562 [==============================] - 28s - loss: 0.6257 - acc: 0.7889 - val_loss: 0.5127 - val_acc: 0.8288
Epoch 166/200
1562/1562 [==============================] - 28s - loss: 0.6243 - acc: 0.7901 - val_loss: 0.5199 - val_acc: 0.8325
Epoch 167/200
1562/1562 [==============================] - 27s - loss: 0.6292 - acc: 0.7873 - val_loss: 0.5273 - val_acc: 0.8275
Epoch 168/200
1562/1562 [==============================] - 27s - loss: 0.6207 - acc: 0.7892 - val_loss: 0.5230 - val_acc: 0.8241
Epoch 169/200
1562/1562 [==============================] - 27s - loss: 0.6173 - acc: 0.7932 - val_loss: 0.5417 - val_acc: 0.8213
Epoch 170/200
1562/1562 [==============================] - 27s - loss: 0.6322 - acc: 0.7866 - val_loss: 0.5443 - val_acc: 0.8189
Epoch 171/200
1562/1562 [==============================] - 28s - loss: 0.6196 - acc: 0.7885 - val_loss: 0.5244 - val_acc: 0.8277
Epoch 172/200
1562/1562 [==============================] - 29s - loss: 0.6202 - acc: 0.7929 - val_loss: 0.5113 - val_acc: 0.8285
Epoch 173/200
1562/1562 [==============================] - 27s - loss: 0.6212 - acc: 0.7897 - val_loss: 0.5332 - val_acc: 0.8192
Epoch 174/200
1562/1562 [==============================] - 27s - loss: 0.6225 - acc: 0.7901 - val_loss: 0.5623 - val_acc: 0.8144
Epoch 175/200
1562/1562 [==============================] - 28s - loss: 0.6119 - acc: 0.7931 - val_loss: 0.5001 - val_acc: 0.8304
Epoch 176/200
1562/1562 [==============================] - 28s - loss: 0.6240 - acc: 0.7889 - val_loss: 0.5138 - val_acc: 0.8282
Epoch 177/200
1562/1562 [==============================] - 27s - loss: 0.6210 - acc: 0.7897 - val_loss: 0.5273 - val_acc: 0.8196
Epoch 178/200
1562/1562 [==============================] - 28s - loss: 0.6196 - acc: 0.7900 - val_loss: 0.5135 - val_acc: 0.8320
Epoch 179/200
1562/1562 [==============================] - 28s - loss: 0.6211 - acc: 0.7891 - val_loss: 0.5162 - val_acc: 0.8279
Epoch 180/200
1562/1562 [==============================] - 27s - loss: 0.6163 - acc: 0.7922 - val_loss: 0.5418 - val_acc: 0.8202
Epoch 181/200
1562/1562 [==============================] - 28s - loss: 0.6269 - acc: 0.7890 - val_loss: 0.5556 - val_acc: 0.8150
Epoch 182/200
1562/1562 [==============================] - 28s - loss: 0.6097 - acc: 0.7954 - val_loss: 0.5073 - val_acc: 0.8331
Epoch 183/200
1562/1562 [==============================] - 28s - loss: 0.6191 - acc: 0.7915 - val_loss: 0.5132 - val_acc: 0.8254
Epoch 184/200
1562/1562 [==============================] - 27s - loss: 0.6226 - acc: 0.7900 - val_loss: 0.5265 - val_acc: 0.8226
Epoch 185/200
1562/1562 [==============================] - 29s - loss: 0.6180 - acc: 0.7915 - val_loss: 0.5428 - val_acc: 0.8178
Epoch 186/200
1562/1562 [==============================] - 28s - loss: 0.6215 - acc: 0.7904 - val_loss: 0.5209 - val_acc: 0.8236
Epoch 187/200
1562/1562 [==============================] - 27s - loss: 0.6194 - acc: 0.7902 - val_loss: 0.5366 - val_acc: 0.8244
Epoch 188/200
1562/1562 [==============================] - 28s - loss: 0.6240 - acc: 0.7891 - val_loss: 0.5525 - val_acc: 0.8173
Epoch 189/200
1562/1562 [==============================] - 27s - loss: 0.6196 - acc: 0.7908 - val_loss: 0.5089 - val_acc: 0.8308
Epoch 190/200
1562/1562 [==============================] - 28s - loss: 0.6299 - acc: 0.7865 - val_loss: 0.4832 - val_acc: 0.8366
Epoch 191/200
1562/1562 [==============================] - 27s - loss: 0.6186 - acc: 0.7880 - val_loss: 0.5157 - val_acc: 0.8319
Epoch 192/200
1562/1562 [==============================] - 27s - loss: 0.6244 - acc: 0.7888 - val_loss: 0.5223 - val_acc: 0.8258
Epoch 193/200
1562/1562 [==============================] - 27s - loss: 0.6230 - acc: 0.7915 - val_loss: 0.5273 - val_acc: 0.8258
Epoch 194/200
1562/1562 [==============================] - 27s - loss: 0.6296 - acc: 0.7878 - val_loss: 0.5127 - val_acc: 0.8309
Epoch 195/200
1562/1562 [==============================] - 26s - loss: 0.6250 - acc: 0.7901 - val_loss: 0.5038 - val_acc: 0.8349
Epoch 196/200
1562/1562 [==============================] - 27s - loss: 0.6247 - acc: 0.7880 - val_loss: 0.5302 - val_acc: 0.8214
Epoch 197/200
1562/1562 [==============================] - 27s - loss: 0.6212 - acc: 0.7885 - val_loss: 0.5134 - val_acc: 0.8278
Epoch 198/200
1562/1562 [==============================] - 27s - loss: 0.6216 - acc: 0.7910 - val_loss: 0.5287 - val_acc: 0.8234
Epoch 199/200
1562/1562 [==============================] - 27s - loss: 0.6260 - acc: 0.7910 - val_loss: 0.5305 - val_acc: 0.8265
Epoch 200/200
1562/1562 [==============================] - 27s - loss: 0.6240 - acc: 0.7894 - val_loss: 0.5448 - val_acc: 0.8178
$ python importance-sampling/examples/cifar10_cnn.py
Using TensorFlow backend.
2017-08-03 10:59:46.125650: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.1 instructions, but these are available on your machine and could speed up CPU computations.
2017-08-03 10:59:46.125697: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
2017-08-03 10:59:46.125707: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.
2017-08-03 10:59:46.125715: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX2 instructions, but these are available on your machine and could speed up CPU computations.
2017-08-03 10:59:46.125723: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use FMA instructions, but these are available on your machine and could speed up CPU computations.
2017-08-03 10:59:48.805821: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:901] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2017-08-03 10:59:48.806851: I tensorflow/core/common_runtime/gpu/gpu_device.cc:887] Found device 0 with properties:
name: Tesla K80
major: 3 minor: 7 memoryClockRate (GHz) 0.8235
pciBusID 0000:00:04.0
Total memory: 11.17GiB
Free memory: 11.11GiB
2017-08-03 10:59:48.806882: I tensorflow/core/common_runtime/gpu/gpu_device.cc:908] DMA: 0
2017-08-03 10:59:48.806897: I tensorflow/core/common_runtime/gpu/gpu_device.cc:918] 0:   Y
2017-08-03 10:59:48.806915: I tensorflow/core/common_runtime/gpu/gpu_device.cc:977] Creating TensorFlow device (/gpu:0) -> (device: 0, name: Tesla K80, pci bus id: 0000:00:04.0)
x_train shape: (50000, 32, 32, 3)
50000 train samples
10000 test samples
Using real-time data augmentation.
Epoch 1/200
1562/1562 [==============================] - 78s - loss: 1.6850 - accuracy: 0.2931 - val_loss: 1.2907 - val_accuracy: 0.5387
Epoch 2/200
1562/1562 [==============================] - 75s - loss: 1.3339 - accuracy: 0.3797 - val_loss: 1.0269 - val_accuracy: 0.6467
Epoch 3/200
1562/1562 [==============================] - 75s - loss: 1.1828 - accuracy: 0.4116 - val_loss: 0.9463 - val_accuracy: 0.6682
Epoch 4/200
1562/1562 [==============================] - 76s - loss: 1.0830 - accuracy: 0.4344 - val_loss: 0.8892 - val_accuracy: 0.6930
Epoch 5/200
1562/1562 [==============================] - 75s - loss: 1.0286 - accuracy: 0.4491 - val_loss: 0.7940 - val_accuracy: 0.7276
Epoch 6/200
1562/1562 [==============================] - 77s - loss: 0.9908 - accuracy: 0.4570 - val_loss: 0.7728 - val_accuracy: 0.7416
Epoch 7/200
1562/1562 [==============================] - 74s - loss: 0.9592 - accuracy: 0.4616 - val_loss: 0.7418 - val_accuracy: 0.7486
Epoch 8/200
1562/1562 [==============================] - 75s - loss: 0.9302 - accuracy: 0.4690 - val_loss: 0.7361 - val_accuracy: 0.7518
Epoch 9/200
1562/1562 [==============================] - 75s - loss: 0.9052 - accuracy: 0.4773 - val_loss: 0.7211 - val_accuracy: 0.7557
Epoch 10/200
1562/1562 [==============================] - 72s - loss: 0.8927 - accuracy: 0.4781 - val_loss: 0.7063 - val_accuracy: 0.7605
Epoch 11/200
1562/1562 [==============================] - 75s - loss: 0.8732 - accuracy: 0.4807 - val_loss: 0.6708 - val_accuracy: 0.7729
Epoch 12/200
1562/1562 [==============================] - 75s - loss: 0.8569 - accuracy: 0.4832 - val_loss: 0.6764 - val_accuracy: 0.7736
Epoch 13/200
1562/1562 [==============================] - 77s - loss: 0.8470 - accuracy: 0.4890 - val_loss: 0.6873 - val_accuracy: 0.7666
Epoch 14/200
1562/1562 [==============================] - 77s - loss: 0.8342 - accuracy: 0.4899 - val_loss: 0.6388 - val_accuracy: 0.7828
Epoch 15/200
1562/1562 [==============================] - 79s - loss: 0.8223 - accuracy: 0.4975 - val_loss: 0.6390 - val_accuracy: 0.7865
Epoch 16/200
1562/1562 [==============================] - 78s - loss: 0.8115 - accuracy: 0.4979 - val_loss: 0.6444 - val_accuracy: 0.7888
Epoch 17/200
1562/1562 [==============================] - 77s - loss: 0.8061 - accuracy: 0.4997 - val_loss: 0.6279 - val_accuracy: 0.7933
Epoch 18/200
1562/1562 [==============================] - 77s - loss: 0.8005 - accuracy: 0.5006 - val_loss: 0.6122 - val_accuracy: 0.7954
Epoch 19/200
1562/1562 [==============================] - 77s - loss: 0.7938 - accuracy: 0.5029 - val_loss: 0.6135 - val_accuracy: 0.7961
Epoch 20/200
1562/1562 [==============================] - 78s - loss: 0.7831 - accuracy: 0.5014 - val_loss: 0.6199 - val_accuracy: 0.7946
Epoch 21/200
1562/1562 [==============================] - 77s - loss: 0.7762 - accuracy: 0.5051 - val_loss: 0.6080 - val_accuracy: 0.8000
Epoch 22/200
1562/1562 [==============================] - 77s - loss: 0.7680 - accuracy: 0.5068 - val_loss: 0.6290 - val_accuracy: 0.7909
Epoch 23/200
1562/1562 [==============================] - 77s - loss: 0.7644 - accuracy: 0.5097 - val_loss: 0.5937 - val_accuracy: 0.8042
Epoch 24/200
1562/1562 [==============================] - 77s - loss: 0.7528 - accuracy: 0.5104 - val_loss: 0.6020 - val_accuracy: 0.7976
Epoch 25/200
1562/1562 [==============================] - 78s - loss: 0.7557 - accuracy: 0.5119 - val_loss: 0.5867 - val_accuracy: 0.8070
Epoch 26/200
1562/1562 [==============================] - 77s - loss: 0.7436 - accuracy: 0.5141 - val_loss: 0.5697 - val_accuracy: 0.8118
Epoch 27/200
1562/1562 [==============================] - 77s - loss: 0.7418 - accuracy: 0.5113 - val_loss: 0.5735 - val_accuracy: 0.8076
Epoch 28/200
1562/1562 [==============================] - 77s - loss: 0.7336 - accuracy: 0.5167 - val_loss: 0.5905 - val_accuracy: 0.7999
Epoch 29/200
1562/1562 [==============================] - 76s - loss: 0.7419 - accuracy: 0.5129 - val_loss: 0.5925 - val_accuracy: 0.8030
Epoch 30/200
1562/1562 [==============================] - 78s - loss: 0.7344 - accuracy: 0.5151 - val_loss: 0.5678 - val_accuracy: 0.8153
Epoch 31/200
1562/1562 [==============================] - 77s - loss: 0.7302 - accuracy: 0.5165 - val_loss: 0.5757 - val_accuracy: 0.8091
Epoch 32/200
1562/1562 [==============================] - 77s - loss: 0.7295 - accuracy: 0.5167 - val_loss: 0.6170 - val_accuracy: 0.7963
Epoch 33/200
1562/1562 [==============================] - 78s - loss: 0.7152 - accuracy: 0.5196 - val_loss: 0.5563 - val_accuracy: 0.8147
Epoch 34/200
1562/1562 [==============================] - 78s - loss: 0.7124 - accuracy: 0.5178 - val_loss: 0.5980 - val_accuracy: 0.8011
Epoch 35/200
1562/1562 [==============================] - 77s - loss: 0.7153 - accuracy: 0.5193 - val_loss: 0.5664 - val_accuracy: 0.8088
Epoch 36/200
1562/1562 [==============================] - 77s - loss: 0.7096 - accuracy: 0.5249 - val_loss: 0.5531 - val_accuracy: 0.8174
Epoch 37/200
1562/1562 [==============================] - 77s - loss: 0.7102 - accuracy: 0.5205 - val_loss: 0.5760 - val_accuracy: 0.8104
Epoch 38/200
1562/1562 [==============================] - 77s - loss: 0.7035 - accuracy: 0.5221 - val_loss: 0.5385 - val_accuracy: 0.8192
Epoch 39/200
1562/1562 [==============================] - 78s - loss: 0.7053 - accuracy: 0.5246 - val_loss: 0.5846 - val_accuracy: 0.8045
Epoch 40/200
1562/1562 [==============================] - 76s - loss: 0.6978 - accuracy: 0.5226 - val_loss: 0.5502 - val_accuracy: 0.8200
Epoch 41/200
1562/1562 [==============================] - 76s - loss: 0.6960 - accuracy: 0.5250 - val_loss: 0.5457 - val_accuracy: 0.8212
Epoch 42/200
1562/1562 [==============================] - 76s - loss: 0.6974 - accuracy: 0.5248 - val_loss: 0.5525 - val_accuracy: 0.8178
Epoch 43/200
1562/1562 [==============================] - 77s - loss: 0.6892 - accuracy: 0.5261 - val_loss: 0.5710 - val_accuracy: 0.8115
Epoch 44/200
1562/1562 [==============================] - 77s - loss: 0.6899 - accuracy: 0.5241 - val_loss: 0.5623 - val_accuracy: 0.8112
Epoch 45/200
1562/1562 [==============================] - 77s - loss: 0.6855 - accuracy: 0.5280 - val_loss: 0.5391 - val_accuracy: 0.8190
Epoch 46/200
1562/1562 [==============================] - 78s - loss: 0.6809 - accuracy: 0.5311 - val_loss: 0.5308 - val_accuracy: 0.8225
Epoch 47/200
1562/1562 [==============================] - 77s - loss: 0.6810 - accuracy: 0.5249 - val_loss: 0.5432 - val_accuracy: 0.8239
Epoch 48/200
1562/1562 [==============================] - 76s - loss: 0.6769 - accuracy: 0.5315 - val_loss: 0.5408 - val_accuracy: 0.8221
Epoch 49/200
1562/1562 [==============================] - 78s - loss: 0.6796 - accuracy: 0.5327 - val_loss: 0.5380 - val_accuracy: 0.8275
Epoch 50/200
1562/1562 [==============================] - 78s - loss: 0.6703 - accuracy: 0.5307 - val_loss: 0.5443 - val_accuracy: 0.8165
Epoch 51/200
1562/1562 [==============================] - 77s - loss: 0.6725 - accuracy: 0.5266 - val_loss: 0.5524 - val_accuracy: 0.8139
Epoch 52/200
1562/1562 [==============================] - 75s - loss: 0.6748 - accuracy: 0.5300 - val_loss: 0.5373 - val_accuracy: 0.8200
Epoch 53/200
1562/1562 [==============================] - 77s - loss: 0.6712 - accuracy: 0.5329 - val_loss: 0.5440 - val_accuracy: 0.8189
Epoch 54/200
1562/1562 [==============================] - 77s - loss: 0.6665 - accuracy: 0.5320 - val_loss: 0.5295 - val_accuracy: 0.8225
Epoch 55/200
1562/1562 [==============================] - 78s - loss: 0.6660 - accuracy: 0.5324 - val_loss: 0.5344 - val_accuracy: 0.8262
Epoch 56/200
1562/1562 [==============================] - 76s - loss: 0.6629 - accuracy: 0.5355 - val_loss: 0.5226 - val_accuracy: 0.8284
Epoch 57/200
1562/1562 [==============================] - 76s - loss: 0.6633 - accuracy: 0.5325 - val_loss: 0.5157 - val_accuracy: 0.8327
Epoch 58/200
1562/1562 [==============================] - 77s - loss: 0.6627 - accuracy: 0.5322 - val_loss: 0.5192 - val_accuracy: 0.8311
Epoch 59/200
1562/1562 [==============================] - 76s - loss: 0.6607 - accuracy: 0.5361 - val_loss: 0.5292 - val_accuracy: 0.8248
Epoch 60/200
1562/1562 [==============================] - 77s - loss: 0.6585 - accuracy: 0.5342 - val_loss: 0.5378 - val_accuracy: 0.8166
Epoch 61/200
1562/1562 [==============================] - 76s - loss: 0.6582 - accuracy: 0.5376 - val_loss: 0.5533 - val_accuracy: 0.8105
Epoch 62/200
1562/1562 [==============================] - 76s - loss: 0.6538 - accuracy: 0.5358 - val_loss: 0.5434 - val_accuracy: 0.8200
Epoch 63/200
1562/1562 [==============================] - 77s - loss: 0.6542 - accuracy: 0.5355 - val_loss: 0.5227 - val_accuracy: 0.8244
Epoch 64/200
1562/1562 [==============================] - 77s - loss: 0.6496 - accuracy: 0.5385 - val_loss: 0.5126 - val_accuracy: 0.8272
Epoch 65/200
1562/1562 [==============================] - 76s - loss: 0.6544 - accuracy: 0.5370 - val_loss: 0.5054 - val_accuracy: 0.8321
Epoch 66/200
1562/1562 [==============================] - 76s - loss: 0.6421 - accuracy: 0.5381 - val_loss: 0.5361 - val_accuracy: 0.8187
Epoch 67/200
1562/1562 [==============================] - 76s - loss: 0.6478 - accuracy: 0.5414 - val_loss: 0.5190 - val_accuracy: 0.8234
Epoch 68/200
1562/1562 [==============================] - 76s - loss: 0.6415 - accuracy: 0.5430 - val_loss: 0.5313 - val_accuracy: 0.8223
Epoch 69/200
1562/1562 [==============================] - 76s - loss: 0.6447 - accuracy: 0.5404 - val_loss: 0.5137 - val_accuracy: 0.8273
Epoch 70/200
1562/1562 [==============================] - 75s - loss: 0.6457 - accuracy: 0.5377 - val_loss: 0.5140 - val_accuracy: 0.8267
Epoch 71/200
1562/1562 [==============================] - 76s - loss: 0.6445 - accuracy: 0.5410 - val_loss: 0.5117 - val_accuracy: 0.8320
Epoch 72/200
1562/1562 [==============================] - 77s - loss: 0.6393 - accuracy: 0.5398 - val_loss: 0.5240 - val_accuracy: 0.8240
Epoch 73/200
1562/1562 [==============================] - 78s - loss: 0.6416 - accuracy: 0.5388 - val_loss: 0.5278 - val_accuracy: 0.8254
Epoch 74/200
1562/1562 [==============================] - 78s - loss: 0.6413 - accuracy: 0.5422 - val_loss: 0.5119 - val_accuracy: 0.8309
Epoch 75/200
1562/1562 [==============================] - 77s - loss: 0.6315 - accuracy: 0.5450 - val_loss: 0.5130 - val_accuracy: 0.8288
Epoch 76/200
1562/1562 [==============================] - 77s - loss: 0.6340 - accuracy: 0.5423 - val_loss: 0.5043 - val_accuracy: 0.8291
Epoch 77/200
1562/1562 [==============================] - 80s - loss: 0.6311 - accuracy: 0.5413 - val_loss: 0.5084 - val_accuracy: 0.8316
Epoch 78/200
1562/1562 [==============================] - 78s - loss: 0.6346 - accuracy: 0.5407 - val_loss: 0.5057 - val_accuracy: 0.8308
Epoch 79/200
1562/1562 [==============================] - 77s - loss: 0.6306 - accuracy: 0.5402 - val_loss: 0.4976 - val_accuracy: 0.8358
Epoch 80/200
1562/1562 [==============================] - 78s - loss: 0.6297 - accuracy: 0.5438 - val_loss: 0.4907 - val_accuracy: 0.8359
Epoch 81/200
1562/1562 [==============================] - 79s - loss: 0.6323 - accuracy: 0.5436 - val_loss: 0.5106 - val_accuracy: 0.8325
Epoch 82/200
1562/1562 [==============================] - 78s - loss: 0.6254 - accuracy: 0.5494 - val_loss: 0.5456 - val_accuracy: 0.8135
Epoch 83/200
1562/1562 [==============================] - 79s - loss: 0.6343 - accuracy: 0.5421 - val_loss: 0.4916 - val_accuracy: 0.8358
Epoch 84/200
1562/1562 [==============================] - 78s - loss: 0.6314 - accuracy: 0.5417 - val_loss: 0.4988 - val_accuracy: 0.8347
Epoch 85/200
1562/1562 [==============================] - 79s - loss: 0.6204 - accuracy: 0.5490 - val_loss: 0.5123 - val_accuracy: 0.8249
Epoch 86/200
1562/1562 [==============================] - 78s - loss: 0.6290 - accuracy: 0.5448 - val_loss: 0.4962 - val_accuracy: 0.8324
Epoch 87/200
1562/1562 [==============================] - 78s - loss: 0.6230 - accuracy: 0.5470 - val_loss: 0.5198 - val_accuracy: 0.8219
Epoch 88/200
1562/1562 [==============================] - 78s - loss: 0.6208 - accuracy: 0.5452 - val_loss: 0.4948 - val_accuracy: 0.8357
Epoch 89/200
1562/1562 [==============================] - 79s - loss: 0.6285 - accuracy: 0.5421 - val_loss: 0.5229 - val_accuracy: 0.8236
Epoch 90/200
1562/1562 [==============================] - 78s - loss: 0.6252 - accuracy: 0.5444 - val_loss: 0.5061 - val_accuracy: 0.8312
Epoch 91/200
1562/1562 [==============================] - 77s - loss: 0.6169 - accuracy: 0.5476 - val_loss: 0.5147 - val_accuracy: 0.8331
Epoch 92/200
1562/1562 [==============================] - 78s - loss: 0.6119 - accuracy: 0.5470 - val_loss: 0.4955 - val_accuracy: 0.8381
Epoch 93/200
1562/1562 [==============================] - 78s - loss: 0.6191 - accuracy: 0.5471 - val_loss: 0.4960 - val_accuracy: 0.8361
Epoch 94/200
1562/1562 [==============================] - 78s - loss: 0.6203 - accuracy: 0.5500 - val_loss: 0.5049 - val_accuracy: 0.8286
Epoch 95/200
1562/1562 [==============================] - 78s - loss: 0.6199 - accuracy: 0.5486 - val_loss: 0.4960 - val_accuracy: 0.8326
Epoch 96/200
1562/1562 [==============================] - 78s - loss: 0.6124 - accuracy: 0.5497 - val_loss: 0.5078 - val_accuracy: 0.8346
Epoch 97/200
1562/1562 [==============================] - 78s - loss: 0.6161 - accuracy: 0.5437 - val_loss: 0.5078 - val_accuracy: 0.8290
Epoch 98/200
1562/1562 [==============================] - 78s - loss: 0.6187 - accuracy: 0.5504 - val_loss: 0.4885 - val_accuracy: 0.8374
Epoch 99/200
1562/1562 [==============================] - 78s - loss: 0.6125 - accuracy: 0.5492 - val_loss: 0.5193 - val_accuracy: 0.8221
Epoch 100/200
1562/1562 [==============================] - 78s - loss: 0.6106 - accuracy: 0.5461 - val_loss: 0.4975 - val_accuracy: 0.8320
Epoch 101/200
1562/1562 [==============================] - 78s - loss: 0.6133 - accuracy: 0.5464 - val_loss: 0.5104 - val_accuracy: 0.8272
Epoch 102/200
1562/1562 [==============================] - 78s - loss: 0.6105 - accuracy: 0.5493 - val_loss: 0.4951 - val_accuracy: 0.8374
Epoch 103/200
1562/1562 [==============================] - 78s - loss: 0.6121 - accuracy: 0.5458 - val_loss: 0.4976 - val_accuracy: 0.8340
Epoch 104/200
1562/1562 [==============================] - 80s - loss: 0.6013 - accuracy: 0.5515 - val_loss: 0.4887 - val_accuracy: 0.8364
Epoch 105/200
1562/1562 [==============================] - 77s - loss: 0.6134 - accuracy: 0.5481 - val_loss: 0.4861 - val_accuracy: 0.8402
Epoch 106/200
1562/1562 [==============================] - 77s - loss: 0.6139 - accuracy: 0.5463 - val_loss: 0.5026 - val_accuracy: 0.8299
Epoch 107/200
1562/1562 [==============================] - 78s - loss: 0.6116 - accuracy: 0.5501 - val_loss: 0.5096 - val_accuracy: 0.8306
Epoch 108/200
1562/1562 [==============================] - 78s - loss: 0.6136 - accuracy: 0.5505 - val_loss: 0.5109 - val_accuracy: 0.8298
Epoch 109/200
1562/1562 [==============================] - 78s - loss: 0.6043 - accuracy: 0.5499 - val_loss: 0.5058 - val_accuracy: 0.8308
Epoch 110/200
1562/1562 [==============================] - 78s - loss: 0.6067 - accuracy: 0.5454 - val_loss: 0.4909 - val_accuracy: 0.8348
Epoch 111/200
1562/1562 [==============================] - 77s - loss: 0.6016 - accuracy: 0.5522 - val_loss: 0.4827 - val_accuracy: 0.8421
Epoch 112/200
1562/1562 [==============================] - 78s - loss: 0.6031 - accuracy: 0.5493 - val_loss: 0.4977 - val_accuracy: 0.8367
Epoch 113/200
1562/1562 [==============================] - 79s - loss: 0.6060 - accuracy: 0.5494 - val_loss: 0.4912 - val_accuracy: 0.8389
Epoch 114/200
1562/1562 [==============================] - 78s - loss: 0.6058 - accuracy: 0.5502 - val_loss: 0.4829 - val_accuracy: 0.8407
Epoch 115/200
1562/1562 [==============================] - 77s - loss: 0.6067 - accuracy: 0.5506 - val_loss: 0.4992 - val_accuracy: 0.8356
Epoch 116/200
1562/1562 [==============================] - 79s - loss: 0.6071 - accuracy: 0.5494 - val_loss: 0.4974 - val_accuracy: 0.8337
Epoch 117/200
1562/1562 [==============================] - 77s - loss: 0.6061 - accuracy: 0.5465 - val_loss: 0.4742 - val_accuracy: 0.8438
Epoch 118/200
1562/1562 [==============================] - 78s - loss: 0.6041 - accuracy: 0.5502 - val_loss: 0.4966 - val_accuracy: 0.8358
Epoch 119/200
1562/1562 [==============================] - 78s - loss: 0.6028 - accuracy: 0.5492 - val_loss: 0.4916 - val_accuracy: 0.8343
Epoch 120/200
1562/1562 [==============================] - 80s - loss: 0.5963 - accuracy: 0.5523 - val_loss: 0.5039 - val_accuracy: 0.8346
Epoch 121/200
1562/1562 [==============================] - 78s - loss: 0.5935 - accuracy: 0.5513 - val_loss: 0.4924 - val_accuracy: 0.8345
Epoch 122/200
1562/1562 [==============================] - 78s - loss: 0.5968 - accuracy: 0.5529 - val_loss: 0.5114 - val_accuracy: 0.8296
Epoch 123/200
1562/1562 [==============================] - 78s - loss: 0.6043 - accuracy: 0.5497 - val_loss: 0.5167 - val_accuracy: 0.8246
Epoch 124/200
1562/1562 [==============================] - 78s - loss: 0.5969 - accuracy: 0.5573 - val_loss: 0.5016 - val_accuracy: 0.8330
Epoch 125/200
1562/1562 [==============================] - 79s - loss: 0.5905 - accuracy: 0.5535 - val_loss: 0.4874 - val_accuracy: 0.8388
Epoch 126/200
1562/1562 [==============================] - 79s - loss: 0.5934 - accuracy: 0.5595 - val_loss: 0.4869 - val_accuracy: 0.8364
Epoch 127/200
1562/1562 [==============================] - 77s - loss: 0.6044 - accuracy: 0.5476 - val_loss: 0.4961 - val_accuracy: 0.8329
Epoch 128/200
1562/1562 [==============================] - 79s - loss: 0.5976 - accuracy: 0.5532 - val_loss: 0.4814 - val_accuracy: 0.8417
Epoch 129/200
1562/1562 [==============================] - 78s - loss: 0.5998 - accuracy: 0.5549 - val_loss: 0.4930 - val_accuracy: 0.8393
Epoch 130/200
1562/1562 [==============================] - 78s - loss: 0.5962 - accuracy: 0.5556 - val_loss: 0.4751 - val_accuracy: 0.8438
Epoch 131/200
1562/1562 [==============================] - 78s - loss: 0.5860 - accuracy: 0.5528 - val_loss: 0.4696 - val_accuracy: 0.8441
Epoch 132/200
1562/1562 [==============================] - 78s - loss: 0.5923 - accuracy: 0.5539 - val_loss: 0.4726 - val_accuracy: 0.8416
Epoch 133/200
1562/1562 [==============================] - 77s - loss: 0.5888 - accuracy: 0.5565 - val_loss: 0.4931 - val_accuracy: 0.8350
Epoch 134/200
1562/1562 [==============================] - 79s - loss: 0.5959 - accuracy: 0.5516 - val_loss: 0.5001 - val_accuracy: 0.8329
Epoch 135/200
1562/1562 [==============================] - 78s - loss: 0.5910 - accuracy: 0.5559 - val_loss: 0.4775 - val_accuracy: 0.8387
Epoch 136/200
1562/1562 [==============================] - 79s - loss: 0.5972 - accuracy: 0.5529 - val_loss: 0.4825 - val_accuracy: 0.8432
Epoch 137/200
1562/1562 [==============================] - 77s - loss: 0.5858 - accuracy: 0.5572 - val_loss: 0.4803 - val_accuracy: 0.8361
Epoch 138/200
1562/1562 [==============================] - 76s - loss: 0.5951 - accuracy: 0.5543 - val_loss: 0.4904 - val_accuracy: 0.8339
Epoch 139/200
1562/1562 [==============================] - 78s - loss: 0.5894 - accuracy: 0.5559 - val_loss: 0.5161 - val_accuracy: 0.8245
Epoch 140/200
1562/1562 [==============================] - 77s - loss: 0.5869 - accuracy: 0.5547 - val_loss: 0.4777 - val_accuracy: 0.8389
Epoch 141/200
1562/1562 [==============================] - 76s - loss: 0.5831 - accuracy: 0.5576 - val_loss: 0.4878 - val_accuracy: 0.8376
Epoch 142/200
1562/1562 [==============================] - 78s - loss: 0.5912 - accuracy: 0.5575 - val_loss: 0.4939 - val_accuracy: 0.8364
Epoch 143/200
1562/1562 [==============================] - 78s - loss: 0.5872 - accuracy: 0.5578 - val_loss: 0.4720 - val_accuracy: 0.8398
Epoch 144/200
1562/1562 [==============================] - 78s - loss: 0.5840 - accuracy: 0.5567 - val_loss: 0.4842 - val_accuracy: 0.8376
Epoch 145/200
1562/1562 [==============================] - 77s - loss: 0.5871 - accuracy: 0.5560 - val_loss: 0.4653 - val_accuracy: 0.8427
Epoch 146/200
1562/1562 [==============================] - 78s - loss: 0.5858 - accuracy: 0.5565 - val_loss: 0.4823 - val_accuracy: 0.8395
Epoch 147/200
1562/1562 [==============================] - 77s - loss: 0.5815 - accuracy: 0.5579 - val_loss: 0.4582 - val_accuracy: 0.8486
Epoch 148/200
1562/1562 [==============================] - 78s - loss: 0.5817 - accuracy: 0.5575 - val_loss: 0.4787 - val_accuracy: 0.8402
Epoch 149/200
1562/1562 [==============================] - 78s - loss: 0.5833 - accuracy: 0.5566 - val_loss: 0.4746 - val_accuracy: 0.8435
Epoch 150/200
1562/1562 [==============================] - 78s - loss: 0.5845 - accuracy: 0.5585 - val_loss: 0.4552 - val_accuracy: 0.8474
Epoch 151/200
1562/1562 [==============================] - 78s - loss: 0.5825 - accuracy: 0.5592 - val_loss: 0.4641 - val_accuracy: 0.8446
Epoch 152/200
1562/1562 [==============================] - 77s - loss: 0.5756 - accuracy: 0.5578 - val_loss: 0.4605 - val_accuracy: 0.8495
Epoch 153/200
1562/1562 [==============================] - 78s - loss: 0.5817 - accuracy: 0.5558 - val_loss: 0.4638 - val_accuracy: 0.8421
Epoch 154/200
1562/1562 [==============================] - 78s - loss: 0.5732 - accuracy: 0.5567 - val_loss: 0.4676 - val_accuracy: 0.8431
Epoch 155/200
1562/1562 [==============================] - 77s - loss: 0.5803 - accuracy: 0.5585 - val_loss: 0.4884 - val_accuracy: 0.8333
Epoch 156/200
1562/1562 [==============================] - 79s - loss: 0.5836 - accuracy: 0.5549 - val_loss: 0.4800 - val_accuracy: 0.8400
Epoch 157/200
1562/1562 [==============================] - 78s - loss: 0.5801 - accuracy: 0.5566 - val_loss: 0.4691 - val_accuracy: 0.8432
Epoch 158/200
1562/1562 [==============================] - 77s - loss: 0.5767 - accuracy: 0.5567 - val_loss: 0.5029 - val_accuracy: 0.8282
Epoch 159/200
1562/1562 [==============================] - 79s - loss: 0.5787 - accuracy: 0.5616 - val_loss: 0.4860 - val_accuracy: 0.8416
Epoch 160/200
1562/1562 [==============================] - 78s - loss: 0.5795 - accuracy: 0.5574 - val_loss: 0.4771 - val_accuracy: 0.8382
Epoch 161/200
1562/1562 [==============================] - 77s - loss: 0.5843 - accuracy: 0.5592 - val_loss: 0.4749 - val_accuracy: 0.8427
Epoch 162/200
1562/1562 [==============================] - 77s - loss: 0.5741 - accuracy: 0.5586 - val_loss: 0.4857 - val_accuracy: 0.8370
Epoch 163/200
1562/1562 [==============================] - 78s - loss: 0.5774 - accuracy: 0.5599 - val_loss: 0.4905 - val_accuracy: 0.8367
Epoch 164/200
1562/1562 [==============================] - 77s - loss: 0.5751 - accuracy: 0.5615 - val_loss: 0.4742 - val_accuracy: 0.8379
Epoch 165/200
1562/1562 [==============================] - 78s - loss: 0.5715 - accuracy: 0.5619 - val_loss: 0.4735 - val_accuracy: 0.8444
Epoch 166/200
1562/1562 [==============================] - 78s - loss: 0.5791 - accuracy: 0.5559 - val_loss: 0.4738 - val_accuracy: 0.8431
Epoch 167/200
1562/1562 [==============================] - 79s - loss: 0.5781 - accuracy: 0.5605 - val_loss: 0.4801 - val_accuracy: 0.8385
Epoch 168/200
1562/1562 [==============================] - 78s - loss: 0.5838 - accuracy: 0.5574 - val_loss: 0.4786 - val_accuracy: 0.8398
Epoch 169/200
1562/1562 [==============================] - 79s - loss: 0.5750 - accuracy: 0.5592 - val_loss: 0.4871 - val_accuracy: 0.8414
Epoch 170/200
1562/1562 [==============================] - 78s - loss: 0.5739 - accuracy: 0.5627 - val_loss: 0.4715 - val_accuracy: 0.8420
Epoch 171/200
1562/1562 [==============================] - 79s - loss: 0.5782 - accuracy: 0.5565 - val_loss: 0.4852 - val_accuracy: 0.8381
Epoch 172/200
1562/1562 [==============================] - 79s - loss: 0.5702 - accuracy: 0.5612 - val_loss: 0.4867 - val_accuracy: 0.8335
Epoch 173/200
1562/1562 [==============================] - 81s - loss: 0.5695 - accuracy: 0.5587 - val_loss: 0.4881 - val_accuracy: 0.8378
Epoch 174/200
1562/1562 [==============================] - 80s - loss: 0.5723 - accuracy: 0.5604 - val_loss: 0.4886 - val_accuracy: 0.8351
Epoch 175/200
1562/1562 [==============================] - 81s - loss: 0.5707 - accuracy: 0.5625 - val_loss: 0.4746 - val_accuracy: 0.8430
Epoch 176/200
1562/1562 [==============================] - 79s - loss: 0.5774 - accuracy: 0.5589 - val_loss: 0.4822 - val_accuracy: 0.8393
Epoch 177/200
1562/1562 [==============================] - 80s - loss: 0.5625 - accuracy: 0.5651 - val_loss: 0.4796 - val_accuracy: 0.8370
Epoch 178/200
1562/1562 [==============================] - 81s - loss: 0.5695 - accuracy: 0.5620 - val_loss: 0.4856 - val_accuracy: 0.8409
Epoch 179/200
1562/1562 [==============================] - 80s - loss: 0.5712 - accuracy: 0.5606 - val_loss: 0.5044 - val_accuracy: 0.8287
Epoch 180/200
1562/1562 [==============================] - 80s - loss: 0.5747 - accuracy: 0.5600 - val_loss: 0.4816 - val_accuracy: 0.8382
Epoch 181/200
1562/1562 [==============================] - 80s - loss: 0.5689 - accuracy: 0.5620 - val_loss: 0.4701 - val_accuracy: 0.8381
Epoch 182/200
1562/1562 [==============================] - 79s - loss: 0.5717 - accuracy: 0.5592 - val_loss: 0.4823 - val_accuracy: 0.8398
Epoch 183/200
1562/1562 [==============================] - 79s - loss: 0.5736 - accuracy: 0.5584 - val_loss: 0.4635 - val_accuracy: 0.8428
Epoch 184/200
1562/1562 [==============================] - 80s - loss: 0.5636 - accuracy: 0.5660 - val_loss: 0.4578 - val_accuracy: 0.8470
Epoch 185/200
1562/1562 [==============================] - 80s - loss: 0.5638 - accuracy: 0.5644 - val_loss: 0.4501 - val_accuracy: 0.8485
Epoch 186/200
1562/1562 [==============================] - 80s - loss: 0.5669 - accuracy: 0.5648 - val_loss: 0.4629 - val_accuracy: 0.8438
Epoch 187/200
1562/1562 [==============================] - 80s - loss: 0.5736 - accuracy: 0.5590 - val_loss: 0.4790 - val_accuracy: 0.8431
Epoch 188/200
1562/1562 [==============================] - 79s - loss: 0.5701 - accuracy: 0.5624 - val_loss: 0.5006 - val_accuracy: 0.8318
Epoch 189/200
1562/1562 [==============================] - 79s - loss: 0.5751 - accuracy: 0.5588 - val_loss: 0.4751 - val_accuracy: 0.8405
Epoch 190/200
1562/1562 [==============================] - 80s - loss: 0.5684 - accuracy: 0.5630 - val_loss: 0.4762 - val_accuracy: 0.8428
Epoch 191/200
1562/1562 [==============================] - 79s - loss: 0.5696 - accuracy: 0.5615 - val_loss: 0.4810 - val_accuracy: 0.8400
Epoch 192/200
1562/1562 [==============================] - 80s - loss: 0.5667 - accuracy: 0.5638 - val_loss: 0.4519 - val_accuracy: 0.8478
Epoch 193/200
1562/1562 [==============================] - 80s - loss: 0.5731 - accuracy: 0.5608 - val_loss: 0.4780 - val_accuracy: 0.8387
Epoch 194/200
1562/1562 [==============================] - 79s - loss: 0.5686 - accuracy: 0.5625 - val_loss: 0.4574 - val_accuracy: 0.8458
Epoch 195/200
1562/1562 [==============================] - 80s - loss: 0.5714 - accuracy: 0.5631 - val_loss: 0.4690 - val_accuracy: 0.8408
Epoch 196/200
1562/1562 [==============================] - 81s - loss: 0.5690 - accuracy: 0.5587 - val_loss: 0.4770 - val_accuracy: 0.8370
Epoch 197/200
1562/1562 [==============================] - 79s - loss: 0.5675 - accuracy: 0.5658 - val_loss: 0.4695 - val_accuracy: 0.8413
Epoch 198/200
1562/1562 [==============================] - 80s - loss: 0.5618 - accuracy: 0.5643 - val_loss: 0.4642 - val_accuracy: 0.8453
Epoch 199/200
1562/1562 [==============================] - 80s - loss: 0.5652 - accuracy: 0.5630 - val_loss: 0.4662 - val_accuracy: 0.8441
Epoch 200/200
1562/1562 [==============================] - 79s - loss: 0.5616 - accuracy: 0.5646 - val_loss: 0.4731 - val_accuracy: 0.8446
</code></pre>


## Reuters MLP

In this example we train a fully connected neural network to classify text
documents into topics. From the terminal output we can see that using
importance sampling the network learns the training set faster in terms of
gradient updates but is a lot slower with respect to wall clock time and
results in a bit of overfitting.

<pre style="height:300px; overflow-y: scroll;"><code class="bash">$ python keras/examples/reuters_mlp.py 
Using TensorFlow backend.
Loading data...
8982 train sequences
2246 test sequences
46 classes
Vectorizing sequence data...
x_train shape: (8982, 1000)
x_test shape: (2246, 1000)
Convert class vector to binary class matrix (for use with categorical_crossentropy)
y_train shape: (8982, 46)
y_test shape: (2246, 46)
Building model...
Train on 8083 samples, validate on 899 samples
Epoch 1/5
8083/8083 [==============================] - 1s - loss: 1.4294 - acc: 0.6790 - val_loss: 1.0906 - val_acc: 0.7642
Epoch 2/5
8083/8083 [==============================] - 1s - loss: 0.7894 - acc: 0.8184 - val_loss: 0.9410 - val_acc: 0.7864
Epoch 3/5
8083/8083 [==============================] - 1s - loss: 0.5506 - acc: 0.8663 - val_loss: 0.8912 - val_acc: 0.7976
Epoch 4/5
8083/8083 [==============================] - 1s - loss: 0.4155 - acc: 0.9005 - val_loss: 0.8768 - val_acc: 0.8120
Epoch 5/5
8083/8083 [==============================] - 1s - loss: 0.3257 - acc: 0.9161 - val_loss: 0.9147 - val_acc: 0.7998
1344/2246 [================>.............] - ETA: 0sTest score: 0.887472608104
Test accuracy: 0.793410507569
$ python importance-sampling/examples/reuters_mlp.py 
Using TensorFlow backend.
Loading data...
8982 train sequences
2246 test sequences
46 classes
Vectorizing sequence data...
x_train shape: (8982, 1000)
x_test shape: (2246, 1000)
Convert class vector to binary class matrix (for use with categorical_crossentropy)
y_train shape: (8982, 46)
y_test shape: (2246, 46)
Building model...
Epoch 1/5
252/252 [==============================] - 7s - loss: 1.2757 - accuracy: 0.2324 - val_loss: 0.9885 - val_accuracy: 0.7617
Epoch 2/5
252/252 [==============================] - 6s - loss: 0.6055 - accuracy: 0.2980 - val_loss: 0.8280 - val_accuracy: 0.8040
Epoch 3/5
252/252 [==============================] - 6s - loss: 0.3779 - accuracy: 0.3655 - val_loss: 0.8488 - val_accuracy: 0.8051
Epoch 4/5
252/252 [==============================] - 6s - loss: 0.2752 - accuracy: 0.4038 - val_loss: 0.8571 - val_accuracy: 0.8051
Epoch 5/5
252/252 [==============================] - 6s - loss: 0.2113 - accuracy: 0.4219 - val_loss: 0.9549 - val_accuracy: 0.7918
1216/2246 [===============>..............] - ETA: 0sTest score: 0.993414917697
Test accuracy: 0.789403383846
</code></pre>

## IMDB FastText

This example trains a FastText model on the IMDB sentiment classification task.
Importance sampling seems to improve the generalization performance but
training time is increased 6-fold. The following plot compares test accuracy
evolution with respect to epochs and seconds.

<div class="fig col-2">
<img src="../img/imdb_fasttext_test_acc_epochs.png" alt="Test accuracy w.r.t. epochs">
<img src="../img/imdb_fasttext_test_acc_time.png" alt="Test accuracy w.r.t. time">
<span>Evolution of sentiment classification accuracy in the test set with
respect to epochs (left) and seconds (right)</span>
</div>

<pre style="height:300px; overflow-y: scroll;"><code class="bash">$ python keras/examples/imdb_fasttext.py
Using TensorFlow backend.
Loading data...
25000 train sequences
25000 test sequences
Average train sequence length: 238
Average test sequence length: 230
Pad sequences (samples x time)
x_train shape: (25000, 400)
x_test shape: (25000, 400)
Build model...
Train on 25000 samples, validate on 25000 samples
Epoch 1/5
25000/25000 [==============================] - 9s - loss: 0.6102 - acc: 0.7400 - val_loss: 0.5034 - val_acc: 0.8104
Epoch 2/5
25000/25000 [==============================] - 9s - loss: 0.4019 - acc: 0.8654 - val_loss: 0.3698 - val_acc: 0.8654
Epoch 3/5
25000/25000 [==============================] - 9s - loss: 0.3025 - acc: 0.8958 - val_loss: 0.3199 - val_acc: 0.8791
Epoch 4/5
25000/25000 [==============================] - 9s - loss: 0.2521 - acc: 0.9113 - val_loss: 0.2971 - val_acc: 0.8848
Epoch 5/5
25000/25000 [==============================] - 9s - loss: 0.2181 - acc: 0.9249 - val_loss: 0.2899 - val_acc: 0.8855
$ python importance-sampling/examples/imdb_fasttext.py
Using TensorFlow backend.
Loading data...
25000 train sequences
25000 test sequences
Average train sequence length: 238
Average test sequence length: 230
Pad sequences (samples x time)
x_train shape: (25000, 400)
x_test shape: (25000, 400)
Build model...
Epoch 1/5
781/781 [==============================] - 59s - loss: 0.6115 - accuracy: 0.7119 - val_loss: 0.5149 - val_accuracy: 0.8434
Epoch 2/5
781/781 [==============================] - 59s - loss: 0.4144 - accuracy: 0.7503 - val_loss: 0.3898 - val_accuracy: 0.8755
Epoch 3/5
781/781 [==============================] - 60s - loss: 0.3120 - accuracy: 0.7322 - val_loss: 0.3381 - val_accuracy: 0.8847
Epoch 4/5
781/781 [==============================] - 60s - loss: 0.2537 - accuracy: 0.7350 - val_loss: 0.3104 - val_accuracy: 0.8880
Epoch 5/5
781/781 [==============================] - 61s - loss: 0.2109 - accuracy: 0.7481 - val_loss: 0.2970 - val_accuracy: 0.8890
</code></pre>

[github_is]: https://github.com/idiap/importance-sampling
