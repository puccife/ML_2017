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
  * #### costfunction.py
     * `Sigmoid`

     * `Mean Square Error [ **MSE** ]`

     * `Mean Absolute Error [ **MAE** ] `

     * `Log likelihood`

     * `Compute_loss which computes the loss using one of the methods mentioned above`

  * #### gradient.py
     * `compute_gradient`
  * #### implementations.py
     * `least_squares [ Linear regression using normal equations ]`
     * `least_squares_gd [ Linear regression using gradient descent ]`
     * `least_squares_gd [ Linear regression using stochastic gradient descent ]`
     * `ridge_regression [ Ridge regression using normal equations ] `
     * `logistic_regression [ Logistic regression method ]`
     * `reg_logistic_regression [ Regularized logistic regression ]`
     * `stochastic_gradient_descent`
     * `learning_by_gradient_descent`
     * `build_k_indices [Building k indices for k-fold cross validation.]`
     * `cross_validation`
  * #### preprocessing.py
     * `remove_features [ Removes features given column number.]`
     * `standardize [ Standardizing the Training and Testing dataset with the same parameters.]`
     * `nan_to_mean_jet [ Replaces -999 values with the mean of the feature for each jet.]`
     * `nan_to_mean [ Replaces -999 values with the mean for all the features.]`
     * `adjust_features [ Adding extra meaningful features to our training/testing sets.]`
     * `split_jets [ splits the training/testing sets to their corresponding jet number.]`
  * #### proj1_helpers
     * `load_csv_data [ Loads our train/test csv files.] `
     * `predict_labels [ Generates class predictions given weights, and a test data matrix.]`
     * `create_csv_submission [ Creates output csv file to be uploaded to Kaggle.]`
     * `sort_predictions [ Sorts the predictions using their ids`
     * `cross_validation_visualization [ Visualization of curves.]`
  * #### train_utils.py
     * `split_data [ Splits the training data given a ratio.]`
     * `build_poly [ Creates a polynomial using input features to a given degree.]`
     * `batch_iter [ Generate a minibatch iterator for a dataset.]`
     * `get_lambda [ Place your best lambdas from cross validation manually here.]`
     * `get_degree [ Place the best degree for each jet here.]`
## Running
Run the file `run_jet` to get the same results we got.
### Playing around with values
To change the degree of the features for each jet use the function `get_degree` in `train_utils.py`.
To pick your best lambda, use cross-validation first to obtain it, and then place the best values inside the `get_lambdas` function in `train_utils.py`.
You can also change the range of `lambdas` variable in the `FILL HERE` .


## Authors
  * Davor Todorovski -

  * Federico Pucci - pucci.federico@epfl.ch

  * Mazen Mahdi - fouad.mazen@epfl.ch

  
## Acknowledgments
