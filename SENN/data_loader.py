from SENN.preprocessing import preprocess_text
from SENN.mapping import *
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

def load_allDatasets():
    data_dailydialog, emotions_dailydialog = load_dailyDialogs()
    data_crowdflower, emotions_crowdflower = load_crowdFlower()
    data_tec, emotions_tec = load_TEC()
    data_talesemotions, emotions_talesemotions = load_tales_emotions()
    data_isear, emotions_isear = load_ISEAR()
    data_emoint, emotions_emoint = load_emoInt()
    data_electorialtweets, emotions_electorialtweets = load_electorialTweets()

    test_size = 0.1

    X_train_dailydialog, X_test_dailydialog, y_train_dailydialog, y_test_dailydialog \
        = train_test_split(data_dailydialog, emotions_dailydialog, test_size=test_size)
    y_train_dailydialog = np.array(y_train_dailydialog)
    y_test_dailydialog = np.array(y_test_dailydialog)

    X_train_crowdflower, X_test_crowdflower, y_train_crowdflower, y_test_crowdflower \
        = train_test_split(data_crowdflower, emotions_crowdflower, test_size=test_size)
    y_train_crowdflower = np.array(y_train_crowdflower)
    y_test_crowdflower = np.array(y_test_crowdflower)

    X_train_tec, X_test_tec, y_train_tec, y_test_tec \
        = train_test_split(data_tec, emotions_tec, test_size=test_size)
    y_train_tec = np.array(y_train_tec)
    y_test_tec = np.array(y_test_tec)

    X_train_talesemotions, X_test_talesemotions, y_train_talesemotions, y_test_talesemotions \
        = train_test_split(data_talesemotions, emotions_talesemotions, test_size=test_size)
    y_train_talesemotions = np.array(y_train_talesemotions)
    y_test_talesemotions = np.array(y_test_talesemotions)

    X_train_isear, X_test_isear, y_train_isear, y_test_isear \
        = train_test_split(data_isear, emotions_isear, test_size=test_size)
    y_train_isear = np.array(y_train_isear)
    y_test_isear = np.array(y_test_isear)

    print(len(data_emoint))
    X_train_emoint, X_test_emoint, y_train_emoint, y_test_emoint \
        = train_test_split(data_emoint, emotions_emoint, test_size=test_size)
    y_train_emoint = np.array(y_train_emoint)
    y_test_emoint = np.array(y_test_emoint)

    X_train_electorialtweets, X_test_electorialtweets, y_train_electorialtweets, y_test_electorialtweets \
        = train_test_split(data_electorialtweets, emotions_electorialtweets, test_size=test_size)
    y_train_electorialtweets = np.array(y_train_electorialtweets)
    y_test_electorialtweets = np.array(y_test_electorialtweets)

    X_train = X_train_dailydialog+X_train_crowdflower+X_train_tec+X_train_talesemotions+X_train_isear+X_train_emoint+X_train_electorialtweets
    X_test = X_test_dailydialog+X_test_crowdflower+X_test_tec+X_test_talesemotions+X_test_isear+X_test_emoint+X_test_electorialtweets
    # print(y_train_dailydialog.shape, y_train_crowdflower.shape)
    y_train = np.concatenate((y_train_dailydialog, y_train_crowdflower, y_train_tec, y_train_talesemotions, y_train_isear, y_train_emoint,y_train_electorialtweets))
    y_test = np.concatenate((y_test_dailydialog, y_test_crowdflower, y_test_tec, y_test_talesemotions, y_test_isear, y_test_emoint, y_test_electorialtweets))

    return X_train, X_test, y_train, y_test

def load_dailyDialogs():
    print("Loading daily dialogs")
    # data_rez = {}
    # emotion_rez = {}
    data_rez = []
    emotion_rez = []
    path = '/Users/andrejaanaandova/Downloads/data/electorial_tweets/datasets/dailydialog/ijcnlp_dailydialog/'
    # print(os.listdir(path))
    data = (open(path + "dialogues_text.txt", "r")).read().split('\n')
    emotion = (open(path + "dialogues_emotion.txt", "r")).read().split('\n')
    # text_type =  (open(path+"dialogues_topic.txt", "r")).read().split('\n')
    for i in range(0, len(emotion)):
        # if text_type[i]=='':
        # continue
        tmp_data = data[i].split('__eou__')
        tmp_emo = emotion[i].split(' ')
        for j in range(0, len(tmp_emo)):
            if tmp_emo[j] == '':
                continue
            tmp_emo[j] = map_dialog_emotion(tmp_emo[j])
            tmp_data[j] = preprocess_text(tmp_data[j])
            # add_element(data_rez, map_dialog_topic(text_type[i]), tmp_data)
            # add_element(emotion_rez, map_dialog_topic(text_type[i]), tmp_emo)

            data_rez.append(tmp_data[j])
            emotion_rez.append(tmp_emo[j])
        # print(i)
    return data_rez, emotion_rez

def load_crowdFlower():
    print("Loading crowd flower")
    file = '/Users/andrejaanaandova/Downloads/data/electorial_tweets/datasets/crowdflower/text_emotion.csv'
    # print (file)
    df = pd.read_csv(file)
    # print(data.head())
    # print(df.columns)
    # print(set(df['sentiment'].tolist()))
    emotions = df['sentiment']
    data = df['content']
    # print(data.shape)
    return map_crowdflower(data, emotions)

def load_TEC():
    print("Loading TEC")
    data = []
    emotions = []
    folder = '/Users/andrejaanaandova/Downloads/data/electorial_tweets/datasets/TEC/'
    file = open(folder + 'Jan9-2012-tweets-clean.txt').read().split('\n')
    set_emo = set()
    for line in file:
        line = line.split(':')
        if line == ['']:
            continue
        emotion = map_TEC_emotion(line[-1])
        text = preprocess_text(line[1])
        # print(emotion, text)
        # set_emo.add(emotion)
        data.append(text)
        emotions.append(emotion)
    # print(set_emo)
    return data, emotions

def load_tales_emotions():
    print("Loading Tales Emotions")
    data = []
    emotions = []
    author = "grimms"
    folder = '/Users/andrejaanaandova/Downloads/data/tales-emotions/Grimms/emmood/'
    for file in os.listdir(folder):
        f_tmp = open(folder + file, 'r').read().split('\n')
        for f in f_tmp:
            f = f.split('\t')
            if f[0] == '':
                continue
            text = f[3]
            emotions_tmp = f[1].split(":")
            if emotions_tmp[0] != emotions_tmp[1]:
                # print(emotions_tmp)
                continue
            data.append(preprocess_text(text))
            emotions.append(map_tales_emotions(emotions_tmp[0]))
            # add_element(data, author, [preprocess_text(text)])
            # add_element(emotions, author, [map_tales_emotions(emotions_tmp[0])])
            # print(f)

    author = "HCAndersen"
    folder = '/Users/andrejaanaandova/Downloads/data/tales-emotions/HCAndersen/emmood/'
    for file in os.listdir(folder):
        f_tmp = open(folder + file, 'r').read().split('\n')
        for f in f_tmp:
            f = f.split('\t')
            if f[0] == '':
                continue
            text = f[3]
            emotions_tmp = f[1].split(":")
            if emotions_tmp[0] != emotions_tmp[1]:
                # print(emotions_tmp)
                continue
            data.append(preprocess_text(text))
            emotions.append(map_tales_emotions(emotions_tmp[0]))
            # add_element(data, author, [preprocess_text(text)])
            # add_element(emotions, author, [map_tales_emotions(emotions_tmp[0])])
            # print(f)

    author = "Potter"
    folder = '/Users/andrejaanaandova/Downloads/data/tales-emotions/Potter/emmood/'
    for file in os.listdir(folder):
        f_tmp = open(folder + file, 'r').read().split('\n')
        for f in f_tmp:
            f = f.split('\t')
            if f[0] == '':
                continue
            text = f[3]
            emotions_tmp = f[1].split(":")
            if emotions_tmp[0] != emotions_tmp[1]:
                # print(emotions_tmp)
                continue
            data.append(preprocess_text(text))
            emotions.append(map_tales_emotions(emotions_tmp[0]))
            # add_element(data, author, [preprocess_text(text)])
            # add_element(emotions, author, [map_tales_emotions(emotions_tmp[0])])
            # print(f)

    # print(data)
    # print(emotions)
    return data, emotions

def load_ISEAR():
    print("Loading ISEAR")
    folder = '/Users/andrejaanaandova/Downloads/data/electorial_tweets/datasets/isear/'
    file = open(folder + 'isear.csv').read().split('\n')
    # print(len(file))
    # set_emos = set()
    data = []
    emotions = []
    for line in file:
        line2 = line.split('|')
        if len(line2) != 43:
            emotion = map_isear_emotions(line2[36])
            # set_emos.add(emotion)
            text = preprocess_text(" ".join(line2[40:-2]))
            # print(emotion, text)
            # print(line)
            if emotion is not None:
                data.append(text)
                emotions.append(emotion)
        else:
            text = preprocess_text(line2[40])
            emotion = map_isear_emotions(line2[36])
            # set_emos.add(emotion)
            # print(line2)
            # print(emotion, text)
            if emotion is not None:
                data.append(text)
                emotions.append(emotion)
    # print(set_emos)
    return data, emotions

def load_emoInt():
    print("Loading EmoInt")
    data = []
    emotions = []
    file = "/Users/andrejaanaandova/Downloads/data/electorial_tweets/datasets/emoint/emoint_all"
    # for file in os.listdir(folder):
    #     print(file)
    # emo = map_emoInt(file.split('-')[0])
    # key = file.split('.')[-2]
    # print(key)
    f = open(file, 'r').read().split('\n')
    for line in f:
        if len(line) == 0:
            continue
        line = line.split('\t')
        emo = map_emoInt(line[-2])
        # print(line)
        if len(line) != 1 and emo!=None:
            # print("dodano")
            text = preprocess_text(line[1])
            data.append(preprocess_text(text))
            emotions.append(emo)
            # add_element(data, key, [text])
            # add_element(emotions, key, [emo])
    return data, emotions

def load_electorialTweets():
    print("Loading electorial tweets")
    data = {}
    emotions = {}
    folder = '/Users/andrejaanaandova/Downloads/data/electorial_tweets/datasets/electoraltweets/Annotated-US2012-Election-Tweets/'
    file1 = open(folder + 'Questionnaire1/AnnotatedTweets.txt').read().split('\n')
    file2 = open(folder + 'Questionnaire2/Batch1/AnnotatedTweets.txt').read().split('\n')
    file3 = open(folder + 'Questionnaire2/Batch2/AnnotatedTweets.txt').read().split('\n')
    tweets_ids = set()
    for line in file1:
        if line[0] < '0' or line[0] > '9':
            continue
        line = line.split('\t')
        if line[10] == 'This tweet has no emotional content.':
            data[line[0]] = preprocess_text(line[14])
            emotions[line[0]] = [0, 0, 0, 1, 0, 0, 0]
            tweets_ids.add(line[0])
    for line in file2:
        if line[0] < '0' or line[0] > '9':
            continue
        line = line.split('\t')
        # print(line)
        emotion = map_electiontweets(line[15])
        if emotion == None:
            continue
        text = preprocess_text(line[13])
        # print(emotion+"   "+text)
        if line[0] in tweets_ids and emotions[line[0]] != emotion:
            emotions[line[0]] = None
            continue
        data[line[0]] = text
        emotions[line[0]] = emotion
        tweets_ids.add(line[0])
    for line in file3:
        if line[0] < '0' or line[0] > '9':
            continue
        line = line.split('\t')
        # print(line)
        emotion = map_electiontweets(line[15])
        if emotion == None:
            continue
        text = preprocess_text(line[13])
        # print(emotion+"   "+text)
        if line[0] in tweets_ids and emotions[line[0]] != emotion:
            emotions[line[0]] = None
            continue
        data[line[0]] = text
        emotions[line[0]] = emotion
        tweets_ids.add(line[0])
    # print(str(len(data))+ "    "+str(len(emotions)))
    # print(len(tweets_ids))
    # print(tweets_ids)
    res_data = []
    res_emotions = []
    for id in data:
        if emotions[id] != None:
            res_data.append(data[id])
            # print(emotion)
            res_emotions.append(emotions[id])
    return res_data, res_emotions
    # print(len(file1))
    # print(len(file2))
    # print(len(file3))

def add_element(dict, key, el):
    if el[-1] == '':
        el = el[:-1]
    # print(el)
    if key in dict:
        dict[key] = dict[key] + el
        return dict
    dict[key] = []
    dict[key] = dict[key] + el
    return dict