## Overview

The goal here is to enable the users to filter the food based on the categories. With more than a million food images, it is simply not efficient to manually label every individual image. Therefore, we would like to build a model, trained on a subset of the data which were manually labelled with a list of decided tags, to generate the tags for the rest of the non-tagged images.

## Problem
The problem here is that given over 2.2 million food images, and their meta information (e.g., description, food name), try 
to predict the food types (e.g., Burger, Cake_and_Bread, Chicken, Chinese, Coffee, Dessert_Sweet).
 
Note that each food item can have multiple food types and they are a total of 48 food types.

To goal is to identify food types for every food item. (Multi-label classification)



## Installation
1. Run the following commands in sequence 

    ```
    virtualenv -p python3 venv/
    source venv/bin/activate
    pip install -r requirements.txt
    ```

2. Put the extracted `glove.6B.100d.txt` in embedding directory. Please download it from [here](https://www.dropbox.com/s/pwsectjfr5i71pe/glove.6B.100d.txt.zip?dl=0).
3. Put the extracted image files (*.jpg) in `img` directory. Please download it [here](https://drive.google.com/open?id=13_fIdKmqikcmcpIhKyJOzQ0hcLJWqQS1).
4. Put the extracted document (*.csv, *.md) in `data` directory. Please download it [here](https://www.dropbox.com/s/445f4ykwsr54wox/model4.39-0.48.hdf5?dl=0).
5. Run `python3 food_prediction_train.py` to train the model.
6. After finish the training, run `python3 food_prediction_test.py` to test the trained model



## Data

The dataset for the problem can be found under data folder

Files descriptions:

- `train.csv`: Training data which consists of 9,916 items.
- `test.csv`: Test data to be predicted which consists of 2,480 items.
- `meta.csv`: Additional information of the items.
- `sample_submission.csv`: The format sample submission file.

## Method

Basically, I use deep learning model with 3 different neural network architectures and combine them, these architectures include: 

    a. CNN (inception v3) + data augmentation for image recognition  - for learning images
    
    b. Bi-directional LSTM with glove word embedding - for learning text data
    
    c. Normal neural network dense layer - for learning numerical values

The resulting model pretty much captures all the important information.
The current results are around 70-75% for the validation set. 



## Evaluation

Final results are evaluated using `micro F1-score`.

To calculate `micro F1-score`:

- Sum the true positives, true negatives, false positives and false negatives of every class.
- Calculate the `F1-score` using the sums.



