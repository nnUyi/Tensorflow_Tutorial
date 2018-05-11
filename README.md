# Tensorflow_Tutorial
  - A tensorflow practice repo for tensorflow learning.(GAN:From Zero to One) In this repo, baisics of tensorflow is provided in ***Example0_Basics***. Besides, classical models like FCN, CNN, AutoEncoder, GAN, RNN, DQN etc. are provided in this repo, you can download it , configure your experimental environments and launch it. Here I strongly recommend that you use **Linux Operational System**.

  - Have a good trip in tensorflow!!!

## Tensorflow useful module
```text
0.tf
    for device choice
    | ---- tf.device
    for session creat
    | ---- tf.Session
    for session config
    | ---- tf.GPUOptions
    | ---- tf.ConfigProto
    for init variables
    | ---- tf.global_variables_initializer
    | ---- tf.local_variables_initializer
    | ---- tf.truncated_normal_initializer

1.tf.app
    | ---- tf.app.run
    | ---- tf.app.flags

2.tf.image
    for adjust data
    | ---- tf.image.adjust_contrast()
    | ---- tf.image.adjust_brightness()
    | ---- tf.image.adjust_hue()
    | ---- tf.image.adjust_saturation()
    | ---- tf.image.flip_left_right()
    | ---- tf.image.flip_up_down()
    | ---- tf.image.crop_and_resize()
    for data augment
    | ---- tf.image.random_contrast()
    | ---- tf.image.random_hue()
    | ---- tf.image.random_saturation()
    | ---- tf.image.random_brightness()
    | ---- tf.image.random_flip_left_right()
    | ---- tf.image.random_flip_up_down()
    | ---- tf.image.center_crop()
    for data metrics
    | ---- tf.image.psnr()
    | ---- tf.image.ssim()
    | ---- tf.image.ssim_multiscale()
    
3.
(1) tf.layers
    | ---- tf.layers.conv1d
    | ---- tf.layers.conv2d
    | ---- tf.layers.conv3d
    | ---- tf.layers.batch_normalization
    | ---- tf.layers.average_pooling1d
    | ---- tf.layers.average_pooling2d
    | ---- tf.layers.average_pooling3d
    | ---- tf.layers.max_pooling1d
    | ---- tf.layers.max_pooling2d
    | ---- tf.layers.max_pooling3d
    | ---- tf.layers.conv2d_transpose
    | ---- tf.layers.conv3d_transpose
    | ---- tf.layers.separable_conv1d
    | ---- tf.layers.separable_conv2d
    | ---- tf.layers.dropout
    | ---- tf.layers.dense
    | ---- tf.layers.flatten

(2) tf.contrib.slim
    | ---- slim.conv1d
    | ---- slim.conv2d
    | ---- slim.conv3d
    | ---- slim.conv2d_transpose
    | ---- slim.conv3d_transpose
    | ---- slim.separable_conv1d
    | ---- slim.separable_conv2d
    | ---- slim.conv2d_in_plane
    | ---- slim.batch_norm
    | ---- slim.layer_norm
    | ---- slim.unit_norm
    | ---- slim.avg_pool1d
    | ---- slim.avg_pool2d
    | ---- slim.avg_pool3d
    | ---- slim.max_pooling1d
    | ---- slim.max_pooling2d
    | ---- slim.max_pooling3d
    | ---- slim.dropout
    | ---- slim.flatten
    | ---- slim.fully_connected
    | ---- slim.linear
    for weights regularization
    | ---- slim.l1_regularizer
    | ---- slim.l2_regularizer
    | ---- slim.l1_l2_regularizer
    | ---- slim.arg_scope
    | ---- slim.repeat
    | ---- slim.stack

(3) tf.nn
    | ---- tf.nn.conv1d
    | ---- tf.nn.conv2d
    | ---- tf.nn.conv3d
    | ---- tf.nn.conv2d_transpose
    | ---- tf.nn.conv3d_transpose
    | ---- tf.nn.depthwise_conv2d
    | ---- tf.nn.separable_conv2d
    | ---- tf.nn.dilation2d
    | ---- tf.nn.erosion2d
    | ---- tf.nn.batch_normalization
    | ---- tf.nn.avg_pool
    | ---- tf.nn.avg_pool3d
    | ---- tf.nn.max_pool
    | ---- tf.nn.max_pool2d
    | ---- tf.nn.dropout
    | ---- tf.nn.l2_loss
    | ---- tf.nn.softmax_cross_entophy_with_logits
    | ---- tf.nn.sparse_softmax_cross_entrophy_with_logits
    | ---- tf.nn.sigmoid_cross_entropy_with_lohits
    | ---- tf.nn.relu
    | ---- tf.nn.crelu
    | ---- tf.nn.leaky_relu
    | ---- tf.nn.relu6
    | ---- tf.nn.selu
    | ---- tf.nn.sigmoid
    | ---- tf.nn.softmax
    | ---- tf.nn.softplus
    | ---- tf.nn.softsign
    | ---- tf.nn.rnn_cell
                | ---- tf.nn.rnn_cell.BasicRNNCell
                | ---- tf.nn.rnn_cell.BasicLSTMCell
                | ---- tf.nn.rnn_cell.GRUCell
                | ---- tf.nn.rnn_cell.MultiRNNCell
    | ---- tf.nn.dynamic_rnn
    | ---- tf.nn.moments

(4) tf.contrib.layers
    | ---- tf.contrib.layers.conv2d
    | ---- tf.contrib.layers.conv2d_in_plane
    | ---- tf.contrib.layers.separable_conv2d
    | ---- tf.contrib.layers.conv2d_transpose
    | ---- tf.contrib.layers.conv3d_transpose
    | ---- tf.contrib.layers.batch_norm
    | ---- tf.contrib.layers.group_norm
    | ---- tf.contrib.layers.instance_norm
    | ---- tf.contrib.layers.layer_norm
    | ---- tf.contrib.layers.unit_norm
    | ---- tf.contrib.layers.avg_pool2d
    | ---- tf.contrib.layers.avg_pool3d
    | ---- tf.contrib.layers.max_pool2d
    | ---- tf.contrib.layers.max_pool3d
    | ---- tf.contrib.layers.flatten
    | ---- tf.contrib.layers.fully_connected
    | ---- tf.contrib.layers.dropout
    | ---- tf.contrib.layers.maxout
    | ---- tf.contrib.layers.softmax
    | ---- tf.contrib.layers.one_hot_encoding
    for weights decay
    | ---- tf.contrib.layers.l1_regularizer
    | ---- tf.contrib.layers.l2_regularizer
    | ---- tf.contrib.layers.l1_l2_regularizer

4.tf.losses
    for weights decay
    | ---- tf.losses.get_regularization_loss
    | ---- tf.losses.get_regularization_losses
    for loss function
    | ---- tf.losses.absolute_difference
    | ---- tf.losses.hinge_loss
    | ---- tf.losses.huber_loss
    | ---- tf.losses.log_loss
    | ---- tf.losses.mean_squared_error
    | ---- tf.losses.softmax_cross_entophy
    | ---- tf.losses.sparse_softmax_cross_entrophy
    | ---- tf.losses.sigmoid_cross_entropy
    
5.tf.metrics
    | ---- tf.metrics.accuracy
    | ---- tf.metrics.recall
    | ---- tf.metrics.false_positives
    | ---- tf.metrics.true_positives
    | ---- tf.metrics.false_negatives
    | ---- tf.metrics.true_negatives
    | ---- tf.metrics.auc
    | ---- tf.metrics.mean_iou
    | ---- tf.metrics.mean_squared_error
    | ---- tf.metrics.root_mean_squared_error

6.tf.train
    for optimizer
    | ---- tf.train.AdamOptimizaer
    | ---- tf.train.RMSPropOptimizer
    | ---- tf.train.AdadeltaOptimizer
    | ---- tf.train.AdagradOptimizer
    | ---- tf.train.AdagradDAOptimizer
    | ---- tf.train.MomentumOptimizer
    | ---- tf.train.GradientDescentOptimizer
    for save model
    | ---- tf.train.Saver
        | ---- tf.train.Saver().save()
        | ---- tf.train.Saver().restore()

7.tf.summary
    for keeping logs
    | ---- tf.summary.scalar
    | ---- tf.summary.image
    | ---- tf.summary.histogram
    | ---- tf.summary.audio
    | ---- tf.summary.FileWriter
    | ---- tf.summary.merge
    | ---- tf.summary.merge_all
```

# Requirements
  - tensorflow
  - numpy
  - matplotlib
  - scipy
  - tqdm

# Config
## Linux
### pip install
  - pip & python2.*
  
        $ sudo apt-get install pip
        # upgrade pip
        $ sudo pip install --upgrade pip
        
  - pip3 & python3.*
  
        $ sudo apt-get install pip
        # upgrade pip
        $ sudo pip install --upgrade pip
        
### numpy install
  - python2.*
      
        $ sudo apt-get install python-numpy
        
  - python3.*
  
        $ sudo apt-get install python3-numpy

### matplotlib install
  - python2.*
      
        $ sudo apt-get install python-matplotlib
        
  - python3.*
  
        $ sudo apt-get install python3-matplotlib


### scipy install
  - python2.*
      
        $ sudo apt-get install python-scipy
        
  - python3.*
  
        $ sudo apt-get install python3-scipy
        
### tqdm install
  - python2.*
      
        $ sudo pip install tqdm
        
  - python3.*
  
        $ sudo pip3 install tqdm

### tensorflow install
  - pip install 
  
        # cpu version: 
        $ pip install tensorflow
        # gpu version: 
        $ pip install tensorflow-gpu
        # upgrade:
        $ pip install -U tensorflow


  - conda install 
  
        $ anaconda search -t conda tensorflow
        $ anaconda show [tensorflow version]
        $ conda install --channel [the show list]

# File Structure
```text
Tensorflow_Practice
|———　Example0_Basics
|     　|———　400_constant.py
|     　|———　401_variable.py	   
|     　|———　402_get_variable.py  
|　     |———　403_placeholder.py 
|　     |———　404_session.py
|　     |———　405_dataloader.py  
|  　   |———　406_optimizer.py  
|    　 |———　407_tensorboard.py  
|     　|———　408_saver.py  
|     　|———　409_simple_regression_model.py  
|———　Example1_FCN
|       |———　FCN.py
|       |———　main.py
|       |———　ops.py
|       |———　utils.py
|———　Example2_CNN
|       |———　CNN.py
|       |———　main.py
|       |———　ops.py
|       |———　utils.py
|———　Example3_AE
|       |———　AutoEncoder.py
|       |———　main.py
|       |———　ops.py
|       |———　utils.py
|———　Example4_GAN
|       |———　GAN.py
|       |———　main.py
|       |———　ops.py
|       |———　utils.py
|———　Example5_RNN
|       |———　RNN.py
|       |———　main.py
|       |———　ops.py
|       |———　utils.py
|———　Example6_DQN
|       |———　waiting for updating
```

# Usages
## Download Repo
      
      # clone repo to local
      $ git clone https://github.com/nnUyi/Tensorflow_Practice.git
      # enter root directory
      $ cd Tensorflow_Practice
      
## Example0_Basics
      
      # In Example0_Basics, each file is individual so that you can run each .py as following
      $ python [filename.py]
      
## Example1_FCN

      # In Example1_FCN, type running instruction as following:
      $ python main.py [configs according to the facts]
      # Example shows below:
      $ python main.py --batchsize=64 --is_training=True --input_height=28 --input_width=28 
        / --input_channel=1
      
## Example2_CNN

      # In Example2_CNN, type running instruction as following:
      $ python main.py [configs according to the facts]
      # Example shows below:
      $ python main.py --batchsize=64 --is_training=True --input_height=28 --input_width=28 
        / --input_channel=1

## Example3_AE

      # In Example3_AE, type running instruction as following:
      $ python main.py [configs according to the facts]
      # Example shows below:
      $ python main.py --batchsize=64 --is_training=True --input_height=28 --input_width=28 
        /--input_channel=1

## Example4_GAN

      # In Example4_GAN, type running instruction as following:
      $ python main.py [configs according to the facts]
      # Example shows below:
      $ python main.py --batchsize=64 --is_training=True --input_height=28 --input_width=28
        / --input_channel=1


## Example5_RNN

      # In Example5_RNNN, type running instruction as following:
      $ python main.py [configs according to the facts]
      # Example shows below:
      $ python main.py --batchsize=64 --is_training=True --time_steps=28 --hidden_unit_size=128
        / --hidden_layer_size=3
      
## Example6_DQN
  
  - waiting for updating
  
# Contact
  Email: computerscienceyyz@163.com
