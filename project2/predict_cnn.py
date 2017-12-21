import numpy as np
from utils.predicting_cnn_utils import *
from keras.models import load_model


# Loading and preprocessing the testing data
x_test_text = load_data_test()
# Padding the tweets
x_test_text_padded = pad_sentences(x_test_text)
# Creating the word embeddings
embeddings_words = generate_word_embeddings()
# Create the corresponding embeddings for each tweet
x_test_embeddings = manipulate_dataset(x_test_text_padded,embeddings_words)

# Loading the model
model = load_model('./weights/CNN_weights/cnn_weights.hdf5')
# Predicting the labels
y_output = model.predict(x_test_embeddings, verbose=0)

# Getting the index of values above and below 0.5
mone_value_flags = y_output < 0.5
one_value_flags = y_output >=0.5

# Setting the labels
y_output[mone_value_flags]=-1
y_output[one_value_flags]=1

y_output_list = list(y_output.ravel())

# The ids of the tweets.
ids = np.arange(1,10001)
# Creating the submission csv file.
create_csv_submission(ids,y_output_list,'./predictions_csv/CNN_submission.csv')
