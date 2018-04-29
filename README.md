# Tensorflow_Practice
  - A tensorflow practice repo for tensorflow learning.(GAN:From Zero to One) In this repo, baisics of tensorflow is provided in ***Example0_Basics***. Besides, classical models like FCN, CNN, AutoEncoder, GAN, RNN, DQN etc. are provided in this repo, you can download it , configure your experimental environments and launch it. Here I strongly recommend that you use **Linux Operational System**.

  - Have a good trip in tensorflow!!!

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
|       |———　waiting for updating
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
      $ python main.py --batchsize=64 --is_training=True --input_height=28 --input_width=28 --input_channel=1
      
## Example2_CNN

      # In Example2_CNN, type running instruction as following:
      $ python main.py [configs according to the facts]
      # Example shows below:
      $ python main.py --batchsize=64 --is_training=True --input_height=28 --input_width=28 --input_channel=1

## Example3_AE

      # In Example3_AE, type running instruction as following:
      $ python main.py [configs according to the facts]
      # Example shows below:
      $ python main.py --batchsize=64 --is_training=True --input_height=28 --input_width=28 --input_channel=1

## Example4_GAN

      # In Example4_GAN, type running instruction as following:
      $ python main.py [configs according to the facts]
      # Example shows below:
      $ python main.py --batchsize=64 --is_training=True --input_height=28 --input_width=28 --input_channel=1


## Example5_RNN

  - waiting for updating
      
## Example6_DQN
  
  - waiting for updating
  
# Contact
  Email: computerscienceyyz@163.com

