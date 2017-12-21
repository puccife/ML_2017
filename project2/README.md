# Readme PROJECT 2 - TEXT CLASSIFICATION
In this readme we include all the infos about the code available in this repository and further informations about the execution and the reproducibily of the results.

## Prerequisites

### Datasets + additional resources
Since we are using pretrained GloVe embeddings here is the link to obtain them, just put them in the project folder without renaming the resources: 
- GloVe pretrained word embeddings: https://drive.switch.ch/index.php/s/Nt9dT6cU8VoXTiW
We provide with this link the dataset used for training and testing, which is the one available for the competition, as above must be put in the project folder without renaming the resources:
- Twitter dataset: https://drive.switch.ch/index.php/s/SBVfQda8SFyxo0t
### Libraries
- Gensim
- NLTK
- Pandas
- Numpy
- Tensorflow, v=1.4
- dm-sonnet (pip install dm-sonnet)
- Tflearn
- Keras

## Structure
We created 4 different models:
- LSTM (Long-Short Term Memory)
- CNN (Convolutional Neural Network)
- DMN (Dynamic Memory Network)
- DNC (Differentiable Neural 

### Training
For each of these models we provide the functions to train them and to predict the results given a pretrained and saved model.
In particular:
 - train_lstm: to train LSTM network (Expected training time: ≈ 3h, with GPU)
 - train_cnn: to train CNN network (Expected training time: ≈ 6h, with CPU)
 - train_dmn: to train DMN network (Expected training time: ≈ 20h, with GPU Tesla TK80 on Amazon AWS Cluster)
 - train_dnc: to train DNC network (Expected training time: ≈ 20h, with GPU Tesla TK80 on Amazon AWS Cluster)

#### Changing Parameters
For LSTM, CNN and DNC Network is possible to modify the parameters in the json file inside the 'config' folder named rispectively:
- configuration_lstm.json
- configuration_cnn.json
- configuration.json
The specific parameters for the DMN can be modified inside the 'dmn_plus.py' file in 'dmn' folder.
### Obtaining Predictions
#### Getting the pretrained models
 To predict the results with one of the given models you have 2 alternatives:
 - Train the model and put the model in the 'weights/MODEL_weights/' folder
 - Download the pretrained models at: https://drive.switch.ch/index.php/s/J1WK5J6QKuYwqHp

#### Predicting
 Then, based on the selected models, you can test them using one of the predict functions below, all the models take < 5 minutes to create a prediction:
 - predict_lstm: to predict using LSTM network
 - predict_cnn: to predict using CNN network
 - predict_dmn: to predict using DMN network (Must be done with GPU!)
 - predict_dnc: to predict using DNC network

### Result
After the predicting phase, is possible to find the CSV prediction with the name of the model in the folder 'predictions_csv'

# About the code

All the code is modularized in classes.
The core of each network is inside the folder named 'model'. For each model there is a specific MODEL_trainer that trains the model from the beginning to the end. 
For each trainer we use different utilities function that are divided into several folders.
All the code is working and properly commented, here we explain briefly the content of each folder:
- config: files inside are used to get the configuration files and load the FLAGS for the models
- utils: used to create the submission files
- preprocessing: used to preprocess and clean the tweets
- word embedding model: used to manipulate the tweets and extract the word vector of each word

### If someone is not clear or not working contact us please:
- federico.pucci@epfl.ch
- fouad.mazen@epfl.ch
- davor.todorovski@epfl.ch

# Thanks!
