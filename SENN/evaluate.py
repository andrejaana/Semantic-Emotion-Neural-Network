from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd
from keras.models import load_model
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
import pickle
from sklearn.utils import shuffle


def transform_labels(labels):
    res = []
    for item in labels:
        max = np.argmax(item)
        if max == 0:
            res.append('angry')
        elif max == 1:
            res.append('sad')
        elif max == 2:
            res.append('disgust')
        elif max == 3:
            res.append('neutral')
        elif max == 4:
            res.append('fear')
        elif max == 5:
            res.append('surprise')
        else:
            res.append('happy')
    return res

def evaluate(model, X_test, y_test):
    # model = load_model('/content/drive/My Drive/semantic_emotion_recognition/SENN_datasets/SENN_dailydialog.h5')
    print(model.evaluate([X_test, X_test], y_test))
    y_pred_label = transform_labels(model.predict([X_test, X_test]))
    y_test_label = transform_labels(y_test)

    print(confusion_matrix(y_test_label, y_pred_label))

def trans_leb(label):
    # label[3] = 0
    max = np.argmax(label)
    if max == 0:
      return 'angry'
    elif max == 1:
      return 'sad'
    elif max == 2:
      return 'disgust'
    elif max == 3:
      return 'neutral'
    elif max == 4:
      return 'fear'
    elif max==5:
      return 'surprise'
    else:
      return 'happy'

def evaluate_youtube_data():
    data = pd.read_csv('youtube_data')
    model = load_model('SENN/SENN-last-weights-improvement.hdf5')
    with open('SENN/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    emotions = data['supposed_emotion'].astype(str).tolist()

    # for i in range(0, len(emotions)):
    #   emotions[i] = map_isear_emotions(emotions[i])
    data = data['transcription_text'].astype(str).tolist()

    data, emotions = shuffle(data, emotions)
    print(emotions)

    data_seq = np.array(tokenizer.texts_to_sequences(data))
    data_seq = pad_sequences(data_seq, padding='post')

    sequence_length = 32

    pred_emos = []
    true_emos = []
    for j in range(0, len(data_seq)):
        video = []
        for i in range(sequence_length, data_seq.shape[1], sequence_length):
            tmp = data_seq[j][i - sequence_length:i].tolist()
            if len(tmp) == sequence_length and any(tmp):
                # print(tmp)
                video.append(tmp)
        video = np.array(video)
        # print(video)
        if video == []:
            print(data[j])
        if video != []:
            true_emos.append(emotions[j])
            video1 = model.predict([video, video])
            video = np.sum(model.predict([video, video]), axis=0) / len(video)
            # print(video.shape)
            word_label = trans_leb(video.tolist())
            # print(word_label)
            # print(transform_labels(video1))
            # print(emotions[j])
            pred_emos.append(word_label)
            print(data[j])
            print(video)
            print(emotions[j], word_label)

    # y_pred_label = transform_labels(model.predict([data_seq, data_seq]))
    # print(y_pred_label)
    # y_test_label = transform_labels(emotions)
    # y_test_label = emotions
    # print(y_test_label)
    # print(data)

    print(confusion_matrix(true_emos, pred_emos))
    print(classification_report(true_emos, pred_emos))
    # print(data[0])
    # print(data[1])