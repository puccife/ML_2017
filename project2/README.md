# Readme PROJECT 2 - TEXT CLASSIFICATION
In this readme we include all the infos about the code available in this repository and further informations about the execution and the reproducibily of the results.

## Prerequisites
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

For each of these models we provide the functions to train them and to predict the results given a pretrained and saved model.
In particular:
 - train_lstm: to train LSTM network (Expected training time: ≈ 3h, with GPU)
 - train_cnn: to train CNN network (Expected training time: ≈ 6h, with CPU)
 - train_dmn: to train DMN network (Expected training time: ≈ 20h, with GPU Tesla TK80 on Amazon AWS Cluster)
 - train_dnc: to train DNC network (Expected training time: ≈ 20h, with GPU Tesla TK80 on Amazon AWS Cluster)
 
 To predict the results with one of the given models you have 2 alternatives:
 - Train the model and put the model in the 'weights/MODEL_weights/' folder
 - Download the pretrained models at: https://drive.switch.ch/index.php/s/J1WK5J6QKuYwqHp
 
 Then, based on the selected models, you can test them using one of the predict functions below, all the models take < 5 minutes to create a prediction:
 - predict_lstm: to predict using LSTM network
 - predict_cnn: to predict using CNN network
 - predict_dmn: to predict using DMN network (Must be done with GPU!)
 - predict_dnc: to predict using DNC network
