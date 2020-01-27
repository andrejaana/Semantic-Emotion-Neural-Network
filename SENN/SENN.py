from keras import Input, Model, Sequential, regularizers
from keras.utils import to_categorical
from keras.utils import plot_model
from keras.layers.embeddings import Embedding
from keras.layers import Dense, Conv1D, Flatten, Reshape, MaxPool1D, concatenate, LSTM, Bidirectional, Dropout, GlobalMaxPool1D
from keras.layers.embeddings import Embedding
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.models import load_model
from gensim.models.fasttext import FastText
import os, re, csv, math, codecs
from tqdm import tqdm
import fasttext.util
import io
import numpy as np
import tensorflow as tf
from keras.optimizers import Adam

def get_attributes():
    # fasttext.util.download_model('en', if_exists='ignore')  # English
    ft = fasttext.load_model('cc.en.300.bin')
    # f = open('GoogleNews-vectors-negative300.bin').read().split('\n')
    # print(len(f))
    # print(len(f[0].split()))
    # print(len(f[1].split()))
    # print(len(f[2].split()))


def model(data, emotions):
    # result = to_categorical(np.asarray(result))
    X_train, X_test, y_train, y_test = train_test_split(data, emotions, test_size=0.2)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    build_model(X_train, X_test, y_train, y_test)

def get_coefs(word1, *arr):
    return word1, np.asarray(arr, dtype='float32')

def load_embedding(path):
    embeddings_index = {}
    f = open(path, encoding='utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:])
        embeddings_index[word] = coefs
    f.close()

def build_model(X_train, X_test, y_train, y_test):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X_train+X_test)
    X_train = np.array(tokenizer.texts_to_sequences(X_train))
    X_test = np.array(tokenizer.texts_to_sequences(X_test))
    X_train = pad_sequences(X_train, padding='post')
    tmp1, tmp2 = X_train.shape
    X_test = pad_sequences(X_test, padding='post', maxlen=tmp2)

    # print(test_pad)
    # print(test_pad.shape)
    # print(train_pad.shape)

    batch_size = 128
    sequence_length = X_train.shape[1]
    # max_features1 = 183713
    EMBEDDING_DIM = 300
    # max_features2 = 183713
    # EMBEDDING_DIM2 = 300
    MAX_NB_WORDS = 1000000




    # the semantic encoder
    print("Start building the semantic encoder")
    embeddings_index1 = {}
    f = codecs.open('/content/drive/My Drive/semantic_emotion_recognition/SENN_datasets/cc.en.300.vec',
                    encoding='utf-8')
    for line in tqdm(f):
        values = line.rstrip().rsplit(' ')
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index1[word] = coefs
    f.close()

    words_not_found = []
    nb_words1 = min(MAX_NB_WORDS, len(tokenizer.word_index))
    embedding_matrix1 = np.zeros((nb_words1, EMBEDDING_DIM))
    for word, i in tokenizer.word_index.items():
        if i >= nb_words1:
            continue
        embedding_vector1 = embeddings_index1.get(word)
        if (embedding_vector1 is not None) and len(embedding_vector1) > 0:
            # words not found in embedding index will be all-zeros.
            embedding_matrix1[i] = embedding_vector1
        else:
            words_not_found.append(word)

    inp = Input(shape=(sequence_length,))
    x = Embedding(nb_words1, EMBEDDING_DIM, weights=[embedding_matrix1])(inp)
    x = Bidirectional(LSTM(batch_size, return_sequences=True))(x)
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    model1 = Model(inputs=inp, outputs=x)




    # The emotional encoder
    embeddings_index2 = {}
    f = codecs.open('/content/drive/My Drive/semantic_emotion_recognition/SENN_datasets/ewe_uni.txt', encoding='utf-8')
    for line in tqdm(f):
        values = line.rstrip().rsplit(' ')
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index1[word] = coefs
    f.close()

    words_not_found1 = []
    nb_words2 = min(MAX_NB_WORDS, len(tokenizer.word_index))
    embedding_matrix2 = np.zeros((nb_words2, EMBEDDING_DIM))
    for word, i in tokenizer.word_index.items():
        if i >= nb_words2:
            continue
        embedding_vector2 = embeddings_index2.get(word)
        if (embedding_vector2 is not None) and len(embedding_vector2) > 0:
            # words not found in embedding index will be all-zeros.
            embedding_matrix2[i] = embedding_vector2
        else:
            words_not_found.append(word)

    num_filters = 100
    filter_sizes = [3, 4, 5]

    inp2 = Input(shape=(sequence_length,))
    embedding = Embedding(nb_words2, EMBEDDING_DIM, weights=[embedding_matrix2])(inp2)
    reshape = Reshape((sequence_length * EMBEDDING_DIM, 1))(embedding)
    conv_0 = Conv1D(filters=num_filters, kernel_size=filter_sizes[0], activation='relu',
                    kernel_regularizer=regularizers.l2(0.01))(reshape)
    conv_1 = Conv1D(filters=num_filters, kernel_size=filter_sizes[1], activation='relu',
                    kernel_regularizer=regularizers.l2(0.01))(reshape)
    conv_2 = Conv1D(filters=num_filters, kernel_size=filter_sizes[2], activation='relu',
                    kernel_regularizer=regularizers.l2(0.01))(reshape)

    maxpool_0 = MaxPool1D(sequence_length - filter_sizes[0] + 1, strides=1)(conv_0)
    maxpool_1 = MaxPool1D(sequence_length - filter_sizes[1] + 1, strides=1)(conv_1)
    maxpool_2 = MaxPool1D(sequence_length - filter_sizes[2] + 1, strides=1)(conv_2)
    y = concatenate([maxpool_0, maxpool_1, maxpool_2], axis=1)
    y = Flatten()(y)
    # reshape = Reshape((3 * num_filters,))(flatten)
    y = Dropout(0.5)(y)
    model2 = Model(inputs=inp2, outputs=y)




    # Combine both models
    combined = concatenate([model1.output, model2.output])
    # z = Dense(2, activation="relu")(combined)
    z = Dense(7, activation="softmax")(combined)
    model = Model(inputs=[model1.input, model2.input], outputs=z)
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])
    print(model.summary())
    plot_model(model, to_file='model_plot4b.png', show_shapes=True, show_layer_names=True)

    # print(y_train)
    print(y_train)
    model.fit(x=[X_train, X_train], y=y_train, batch_size=56, epochs=20)
    model.save('/content/drive/My Drive/semantic_emotion_recognition/SENN_datasets/SENN_dailydialog.h5')

def load_embedding(file):
    print("loading embedding")
