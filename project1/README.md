# Project 1
This project counts to 10% of the final grade in the Machine Learning course [ **CS-433** ].
We are asked to do  exploratory data analysis on the HIGGS dataset to understand and explore the features.
Furthermore, feature processing and engineering is used to clean the dataset and exrtact more meaningful information.
We then use several machine learning methods to analyze the model and generate predictions that are later submitted to Kaggle platform. [ Private leaderboard ]

## Getting Started
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

## The Higgs Boson
The Higgs boson is an elementary particle in the Standard Model of physics which explains why other particles
have mass. We will apply machine learning techniques to recreate the process of discovering the Higgs particle.
#### Some physics background
Physicists at CERN smash protons into one another at high speeds to generate even smaller particles as by-products of the collisions. Rarely, these collisions can produce a Higgs boson. Since the Higgs boson decays rapidly into other particles, scientists don’t observe it directly, but rather measure its decay signature, or the products that result from its decay process. Our task here is to estimate the likelihood that a given event's signature was a result of a Higgs Boson or some other process / particle.

For more information about the dataset and the physics background please refer to [this](https://higgsml.lal.in2p3.fr/files/2014/04/documentation_v1.8.pdf).

## Prerequisites
`Python 3+`
## Installing
Use `git clone https://github.com/puccife/ML_2017.git ` to copy the repository into your local machine.
## Dataset
You will find the testing and training data in the folder named `Data`.
Please open this file and unzip both `test.csv.zip` and `train.csv.zip` to extract the csv files.
A `Sample-submission.csv` file is also there to show how your output should look like when submitting to Kaggle.
## Utilization
Inside the `utils` folder you will find several `.py` files, each of which contains functions that are used in the `run_jet` file. We will discuss them now briefly.
##### `costfunction.py`
   1.Sigmoid
   2.Mean Square Error [ **MSE** ]
   3.Mean Absolute Error [ **MAE** ] 
   4.Log likelihood
   5.Compute_loss which computes the loss using one of the methods mentioned above


## And coding style tests

## Deployment

## Authors

## Acknowledgments
