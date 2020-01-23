from keras import Input, Model, Sequential, regularizers
from keras.layers.embeddings import Embedding
from keras.layers import Dense, Conv1D, Flatten, Reshape, MaxPool1D, concatenate, LSTM, Bidirectional, Dropout, GlobalMaxPool1D
from keras.layers.embeddings import Embedding
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from gensim.models.fasttext import FastText
import os, re, csv, math, codecs
from tqdm import tqdm
import fasttext.util
import io

import numpy as np

def get_attributes():
    # fasttext.util.download_model('en', if_exists='ignore')  # English
    ft = fasttext.load_model('cc.en.300.bin')
    # f = open('GoogleNews-vectors-negative300.bin').read().split('\n')
    # print(len(f))
    # print(len(f[0].split()))
    # print(len(f[1].split()))
    # print(len(f[2].split()))


def model(data, result):
    X_train, X_test, y_train, y_test = train_test_split(data, result, test_size=0.1)
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

def build_model(X_train, X_test, Y_train, Y_test):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X_train)

    X_train = tokenizer.texts_to_sequences(X_train)
    X_test = tokenizer.texts_to_sequences(X_test)

    X_train = pad_sequences(X_train, padding='post')
    X_test = pad_sequences(X_test, padding='post')

    # print(test_pad)
    # print(test_pad.shape)
    # print(train_pad.shape)




    # the semantic encoder
    print("Start building the semantic encoder")
    max_features = 2000001
    EMBEDDING_DIM1 = 300
    batch_size = 128
    sequence_length = X_train.shape[1]
    embeddings_index = {}
    fin = io.open('cc.en.300.vec', 'r', encoding='utf-8', newline='\n', errors='ignore')
    line_counter = 0
    for line in fin:
        values = line.rstrip().rsplit(' ')
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
        line_counter+=1
    fin.close()
    # print(embeddings_index)
    all_embedding = np.stack(embeddings_index.values())
    emb_mean, emb_std = all_embedding.mean(), all_embedding.std()
    word_index = tokenizer.word_index
    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, EMBEDDING_DIM1))
    for word, i in word_index.items():
        if i >= max_features:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    inp = Input(shape=(sequence_length,))
    x = Embedding(max_features, EMBEDDING_DIM1, weights=[embedding_matrix])(inp)
    x = Bidirectional(LSTM(batch_size, return_sequences=True))(x)
    x = Dropout(0.5)(x)
    model = Model(inputs=inp, outputs=x)



    # the emotion encoder
    print("Start building the emotion encoder")
    max_features2 = 183713
    EMBEDDING_DIM2 = 300
    num_filters = 100
    filter_sizes = [3, 4, 5]
    embeddings_index2 = dict(get_coefs(*o.strip().split()) for o in open('ewe_uni.txt'))
    print(embeddings_index2)
    all_embedding2 = np.stack(embeddings_index2.values())
    emb_mean2, emb_std2 = all_embedding2.mean(), all_embedding2.std()
    word_index2 = tokenizer.word_index
    nb_words2 = min(max_features2, len(word_index2))
    embedding_matrix2 = np.random.normal(emb_mean2, emb_std2, (nb_words2, EMBEDDING_DIM2))
    for word, i in word_index2.items():
        if i >= max_features2:
            continue
        embedding_vector2 = embeddings_index2.get(word)
        if embedding_vector2 is not None:
            embedding_matrix2[i] = embedding_vector2
    inp2 = Input(shape=(sequence_length,))
    embedding = Embedding(max_features2, EMBEDDING_DIM2, weights=[embedding_matrix])(inp)
    reshape = Reshape((sequence_length, EMBEDDING_DIM2, 1))(embedding)
    conv_0 = Conv1D(num_filters, (filter_sizes[0], EMBEDDING_DIM2), activation='relu', kernel_regularizer=regularizers.l2(0.01))(reshape)
    conv_1 = Conv1D(num_filters, (filter_sizes[1], EMBEDDING_DIM2), activation='relu', kernel_regularizer=regularizers.l2(0.01))(reshape)
    conv_2 = Conv1D(num_filters, (filter_sizes[2], EMBEDDING_DIM2), activation='relu', kernel_regularizer=regularizers.l2(0.01))(reshape)

    maxpool_0 = MaxPool1D((sequence_length - filter_sizes[0] + 1, 1), strides=(1,1))(conv_0)
    maxpool_1 = MaxPool1D((sequence_length - filter_sizes[1] + 1, 1), strides=(1,1))(conv_1)
    maxpool_2 = MaxPool1D((sequence_length - filter_sizes[2] + 1, 1), strides=(1,1))(conv_2)
    y = concatenate([maxpool_0, maxpool_1, maxpool_2], axis=1)
    y = Flatten()(y)
    # reshape = Reshape((2 * num_filters,))(flatten)
    y = Dropout(0.5)(y)
    y = Model(inputs=input, outputs=y)


    # combine the output of the emotion and semantic encoder
    combined = concatenate([x.output, y.output])
    z = Dense(2, activation="relu")(combined)
    z = Dense(1, activation="softmax")(z)
    model = Model(inputs=[x.input, y.input], outputs=z)

    print("Start training the model")
    # callbacks = [EarlyStopping(monitor='acc')]
    model = model.fit(X_train, Y_train, batch_size=400, epochs=20, verbose=2)
    model.save('SENN.h5')

def load_embedding(file):
    print("loading embedding")
