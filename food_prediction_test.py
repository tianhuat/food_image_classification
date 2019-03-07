import glob
import os
import pickle

import numpy as np
import pandas as pd
from keras.models import model_from_json
from util import *

logger = get_logger('myapp')


def evaluate_result():
    logger.info("Loading test data...")
    df_submit = pd.read_csv("data/food_tagging_test.csv")
    logger.info("Submission file size (row, col): {}".format(df_submit.shape))

    df_meta = pd.read_csv("data/meta.csv")
    df_meta.drop(columns=['photo'], inplace=True)

    df_submit = pd.merge(df_submit, df_meta, how='inner', left_on=['item_id'], right_on=['item_id'])
    add_text_field(df_submit)

    logger.info("Reading the model...")
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)

    logger.info("Reading the model weight...")
    model_file = sorted(glob.glob("*.hdf5"), reverse=True)[0]
    model.load_weights(model_file)
    result = []

    logger.info("Loading the tokenizer...")
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    for index, row in df_submit.iterrows():
        # input
        img_arr_rs = load_image(row['photo'])
        item_price_amt = row['item_price_amt']
        os.chdir(".")
        tokenized_text = vectorize_text(np.array([row['text']]), tokenizer=tokenizer)

        # calling prediction
        y_pred = model.predict([np.array([img_arr_rs]), tokenized_text, np.array([item_price_amt])])

        # postprocess of multi-label so that it is either 0 or 1
        y_pred[y_pred >= 0.5] = 1
        y_pred[y_pred < 0.5] = 0
        y_pred = y_pred.astype(str)

        # add item id
        y_pred = np.insert(y_pred, 0, row['item_id'], axis=1)
        result.append(y_pred[0])

    logger.info("Saving the CSV file...")
    df_submission = pd.DataFrame(result,
                                 columns=['item_id', 'Asian', 'Babi', 'Bakso', 'Bebek', 'Beverages', 'Boiled', 'Bubur',
                                          'Burger', 'Cake_and_Bread', 'Chicken', 'Chinese', 'Coffee', 'Dessert_Sweet',
                                          'Egg', 'Fish', 'Fries', 'Gado2', 'Gorengan', 'Grilled', 'Indonesian',
                                          'Italian',
                                          'Japanese', 'Kukus', 'Main_Course', 'Martabak', 'Middle_Eastern', 'Milk',
                                          'Noodle', 'Organic', 'Pempek', 'Personal', 'Pizza', 'Red_Meat', 'Rice',
                                          'Salty',
                                          'Sate', 'Sauce', 'Seafood', 'Set_Menu', 'Sharing', 'Siomay',
                                          'Snack_Appetizer',
                                          'Soup', 'Sour', 'Spicy', 'Tea', 'Vegetable', 'Western'])

    df_submission.to_csv("submission.csv", index=False)


if __name__ == '__main__':
    evaluate_result()
    logger.info("Evalution Done!")
