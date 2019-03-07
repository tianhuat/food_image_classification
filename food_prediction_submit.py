import os
from os import listdir
from os.path import join

import keras
import keras.backend as K
import matplotlib.image as img
import numpy as np
import pandas as pd
import tensorflow as tf
from imgaug import augmenters as iaa
from keras.applications.inception_v3 import InceptionV3
from keras.callbacks import LambdaCallback
from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler
from keras.layers import AveragePooling2D
from keras.layers import Bidirectional
from keras.layers import Dense, Input, LSTM, Embedding, Dropout
from keras.layers import Flatten
from keras.models import Model
from keras.optimizers import SGD
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.regularizers import l2
from keras.utils import multi_gpu_model
from skimage.transform import resize
from keras.models import model_from_json
import yaml
import logging


level = logging.INFO
format = '  %(message)s'
handlers = [logging.FileHandler('info.log'), logging.StreamHandler()]
logging.basicConfig(level = level, format = format, handlers = handlers)


config=yaml.load(open("config/config.yaml"))

num_of_gpus = 4

# ====Preprocessing====

train_test_split_ratio = 0.9 

# cnn
min_side = 299
nb_batches = 1 #64
img_root = 'imgs'

# lstm
max_features = 25000  # how many unique words to use (i.e num rows in embedding vector)
maxlen = 20  # max number of words in a comment to use
embedding_dim = 100

def add_text_field(df):
    df['text'] = df['name'].fillna('') + ' ' + df['menu_name'].fillna('') + ' ' + df['item_description'].fillna('')
    
nb=input("hello")

df_full_train = pd.read_csv("data/food_tagging_train.csv")
df_submit = pd.read_csv("data/food_tagging_test.csv")
df_meta = pd.read_csv("data/meta.csv")
df_meta.drop(columns=['photo'], inplace=True)

# combine the labels into a list
df_full_train["combine"] = df_full_train[df_full_train.columns.values[4:]].values.tolist()
df_full_train.drop(columns=df_full_train.columns.values[4:-1], inplace=True)

# merge information from meta.csv into training and submit test file
df_full_train = pd.merge(df_full_train, df_meta, how='inner', left_on=['item_id'], right_on=['item_id'])
df_submit = pd.merge(df_submit, df_meta, how='inner', left_on=['item_id'], right_on=['item_id'])

# we add a text field by combining information from three text fields - 'name', 'menu_name', and 'item_description'
add_text_field(df_full_train)
add_text_field(df_submit)

# train test split (assume the elements in the csvs file already done random shuffle, so we do not repeat it here)
train_idx = int(df_full_train.shape[0] * train_test_split_ratio)
df_train = df_full_train[:train_idx]
df_test = df_full_train[train_idx:].reset_index()

# tokenizer - apply on the text file
tokenizer = Tokenizer(num_words=max_features)
train_value = list(df_train["text"].values)
tokenizer.fit_on_texts(train_value)

logging.info("Train size: {}".format(df_train.shape))
logging.info("Test size: {}".format(df_test.shape))
logging.info("Submit size: {}".format(df_submit.shape))

#for image augmentation in cnn
seq = iaa.Sequential([  # apply all below
    iaa.Crop(px=(0, 16)),  # crop images from each side by 0 to 16px (randomly chosen)
    iaa.Fliplr(0.5),  # 0.5 is the probability, horizontally flip 50% of the images
    iaa.GaussianBlur(sigma=(0, 3.0)),  # blur images with a sigma of 0 to 3.0
])


image_cache = {}
def load_image(img_name):
    if not img_name in image_cache:
        try:
            img_arr = img.imread(join(img_root, img_name))[:, :, 0:3]
            #load the image and resize it accordingly
            img_arr_rs = resize(img_arr, (min_side, min_side, 3), mode='constant')
        except:
            # some images are not food image like (400,400), we use 0 for them
            img_arr_rs = np.zeros(shape=(299, 299, 3))
        image_cache[img_name] = img_arr_rs
    return image_cache[img_name]


#tokenizing text
def vectorize_text(text_arr):
    tokenized_text = tokenizer.texts_to_sequences(text_arr)
    return pad_sequences(tokenized_text, maxlen=maxlen)


# generator for yielding batch data to be consume by fit_generator
def generator(df):
    images = []
    f_combines = []
    f_prices = []
    f_text = []
    fields = []
    cnt = 1

    while True:
        imgs = listdir(img_root)
        for img_name in imgs:

            if img_name in df['photo'].values:
                #the if branch collect information
                if cnt <= nb_batches:
                    cnt += 1
                    try:
                        img_arr_rs = load_image(img_name)
                        images.append(img_arr_rs)
                        row = df[df['photo'] == img_name].iloc[0]

                        fields.append(row)
                        f_combines.append(row["combine"])
                        f_prices.append(row["item_price_amt"])
                        f_text.append(row["text"])
                    except:
                        raise
                        invalid_count += 1
                        logging.warning(invalid_count, 'images skipped')
                else:
                    #the images are augmented
                    images_aug = seq.augment_images(np.array(images))
                    tokenized_text = vectorize_text(np.array(f_text))
                    #yield 3 inputs + 1 target array
                    yield ([images_aug, tokenized_text, np.array(f_prices)], np.array(f_combines))  # , fields
                    images = []
                    f_combines = []
                    f_prices = []
                    f_text = []
                    fields = []
                    cnt = 1


# ====Training=====


K.clear_session()
K.set_image_dim_ordering('tf')


price = Input(shape=(1,))
base_model = InceptionV3(weights='imagenet', include_top=False, input_tensor=Input(shape=(min_side, min_side, 3)))

# ---CNN Layer----
cnn_layer = base_model.output
cnn_layer = AveragePooling2D(pool_size=(8, 8))(cnn_layer)
cnn_layer = Dropout(.4)(cnn_layer)
cnn_layer = Flatten()(cnn_layer)



# ---Dense Layer---
price_layer = Dense(1, input_shape=(1,), activation='relu')(price)




# copy the embedding as dictionary
embeddings_index = {}
f = open(os.path.join('embedding', 'glove.6B.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

logging.info('Found %s word vectors.' % len(embeddings_index))

word_index = tokenizer.word_index
logging.info('Found %s unique tokens.' % len(word_index))

# new embedding according to the vectorization
embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

#---LSTM Layer---
lstm_input = Input(shape=(maxlen,))
lstm_layer = Embedding(len(word_index) + 1,
                       embedding_dim,
                       weights=[embedding_matrix],
                       input_length=maxlen,
                       trainable=False)(lstm_input)
lstm_layer = Bidirectional(LSTM(100))(lstm_layer)
lstm_layer = Dropout(0.3)(lstm_layer)
lstm_layer = Dense(20, activation="sigmoid")(lstm_layer)

#--Final model
concatenated = keras.layers.concatenate([cnn_layer, lstm_layer, price_layer])
predictions = Dense(48, kernel_initializer='glorot_uniform', kernel_regularizer=l2(.0005), activation='sigmoid')(
    concatenated)

with tf.device('/cpu:0'):
    model = Model(inputs=[base_model.input, lstm_input, price], outputs=predictions)


checkpointer = ModelCheckpoint(filepath='model4.{epoch:02d}-{val_loss:.2f}.hdf5', verbose=1, save_best_only=True)
csv_logger = CSVLogger('model4.log')


def schedule(epoch):
    if epoch < 15:
        return .01
    elif epoch < 28:
        return .002
    else:
        return .0004


lr_scheduler = LearningRateScheduler(schedule)

model = multi_gpu_model(model, gpus=num_of_gpus)

#the performance metrics for the output model
def micro_f1(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

    recall = true_positives / (possible_positives + K.epsilon())
    precision = true_positives / (predicted_positives + K.epsilon())

    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


model.compile(optimizer=SGD(lr=.01, momentum=.9), loss='binary_crossentropy', metrics=[micro_f1])
# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)


def batchOutput(batch, logs):
    logging.info("Finished batch: " + str(batch))
    logging.info(logs)


# make sure we output message once a batch is finished
batchLogCallback = LambdaCallback(on_batch_end=batchOutput)

# tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
steps_per_epoch = (int)(df_train.shape[0] / nb_batches)
validation_steps = (int)(df_test.shape[0] / nb_batches)
logging.info("steps_per_epoch = {}".format(steps_per_epoch))
logging.info("validation_steps = {}".format(validation_steps))

model.fit_generator(generator(df_train),
                    validation_data=generator(df_test),
                    validation_steps=validation_steps,
                    steps_per_epoch=steps_per_epoch,
                    use_multiprocessing=True,
                    epochs=20,
                    verbose=2,
                    callbacks=[lr_scheduler, csv_logger, checkpointer, batchLogCallback])

# ----model running configuration end---

# ====Evaluation=====

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

model.load_weights("model4.39-0.48.hdf5")
result = []

for index, row in df_submit.iterrows():
    # input
    img_arr_rs = load_image(row['photo'])
    item_price_amt = row['item_price_amt']
    tokenized_text = vectorize_text(np.array([row['text']]))
    
    #calling prediction
    y_pred = model.predict([np.array([img_arr_rs]), tokenized_text, np.array([item_price_amt])])

    # postprocess of multi-label so that it is either 0 or 1
    y_pred[y_pred >= 0.5] = 1
    y_pred[y_pred < 0.5] = 0
    y_pred = y_pred.astype(str)

    # add item id
    y_pred = np.insert(y_pred, 0, row['item_id'], axis=1)
    result.append(y_pred[0])

df_submission = pd.DataFrame(result,
                             columns=['item_id', 'Asian', 'Babi', 'Bakso', 'Bebek', 'Beverages', 'Boiled', 'Bubur',
                                      'Burger', 'Cake_and_Bread', 'Chicken', 'Chinese', 'Coffee', 'Dessert_Sweet',
                                      'Egg', 'Fish', 'Fries', 'Gado2', 'Gorengan', 'Grilled', 'Indonesian', 'Italian',
                                      'Japanese', 'Kukus', 'Main_Course', 'Martabak', 'Middle_Eastern', 'Milk',
                                      'Noodle', 'Organic', 'Pempek', 'Personal', 'Pizza', 'Red_Meat', 'Rice', 'Salty',
                                      'Sate', 'Sauce', 'Seafood', 'Set_Menu', 'Sharing', 'Siomay', 'Snack_Appetizer',
                                      'Soup', 'Sour', 'Spicy', 'Tea', 'Vegetable', 'Western'])
df_submission.to_csv("submission.csv", index=False)
