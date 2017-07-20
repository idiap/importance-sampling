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
<span>Results of training a fully connected neural network on MNIST with (orange) and
without (blue) importance sampling</span>
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
<span>Results of training a small CNN on MNIST with (orange) and without (blue)
importance sampling</span>
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
