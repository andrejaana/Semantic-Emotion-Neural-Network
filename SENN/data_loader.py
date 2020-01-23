import os
from SENN.mapping import *
from SENN.preprocessing import preprocess_text
import pandas as pd

def load_all():
    load_dailyDialogs()

def load_dailyDialogs():
    data_rez = {}
    emotion_rez = {}
    path = '/Users/andrejaanaandova/Downloads/data/electorial_tweets/datasets/dailydialog/ijcnlp_dailydialog/'
    print(os.listdir(path))
    data = (open(path+"dialogues_text.txt", "r")).read().split('\n')
    emotion = (open(path+"dialogues_emotion.txt", "r")).read().split('\n')
    text_type =  (open(path+"dialogues_topic.txt", "r")).read().split('\n')
    for i in range(0, len(text_type)):
        if text_type[i]=='':
            continue
        tmp_data = data[i].split('__eou__')
        tmp_emo = emotion[i].split(' ')
        for j in range(0, len(tmp_emo)):
            if tmp_emo[j] == '':
                continue
            tmp_emo[j] = map_dialog_emotion(tmp_emo[j])
            tmp_data[j] = preprocess_text(tmp_data[j])
        add_element(data_rez, map_dialog_topic(text_type[i]), tmp_data)
        add_element(emotion_rez, map_dialog_topic(text_type[i]), tmp_emo)
        # print(i)
    return data_rez, emotion_rez

def load_crowdFlower():
    file = '/Users/andrejaanaandova/Downloads/data/electorial_tweets/datasets/crowdflower/text_emotion.csv'
    data = pd.read_csv(file)
    print(data.head())
    print(data.columns)
    print(set(data['sentiment'].tolist()))
    print(data.shape)
    # data = data[data['sentiment']=='anger'
    #             or data['sentiment']=='sadness'
    #             or data['sentiment']=='neutral'
    #             or data['sentiment']=='happiness'
    #             or data['sentiment']=='surprise']
    # print(set(data['sentiment'].tolist()))
    # print(data[(data['sentiment']=='anger') | (data['sentiment']=='hate')].shape)
    return data, data['sentiment']

def load_TEC():
    folder = '/Users/andrejaanaandova/Downloads/data/electorial_tweets/datasets/TEC/'
    file = open(folder+'Jan9-2012-tweets-clean.txt').read().split('\n')
    set_emo = set()
    for line in file:
        line = line.split(':')
        if line==['']:
            continue
        emotion = map_TEC_emotion(line[-1])
        text = preprocess_text(line[1])
        print(emotion, text)
        set_emo.add(emotion)
    print(set_emo)

def load_tales_emotions():
    data = {}
    emotions = {}
    author = "grimms"
    folder = '/Users/andrejaanaandova/Downloads/data/tales-emotions/Grimms/emmood/'
    for file in os.listdir(folder):
        f_tmp = open(folder+file, 'r').read().split('\n')
        for f in f_tmp:
            f = f.split('\t')
            if f[0] == '':
                continue
            text = f[3]
            emotions_tmp = f[1].split(":")
            if emotions_tmp[0]!=emotions_tmp[1]:
                # print(emotions_tmp)
                continue
            add_element(data, author, [preprocess_text(text)])
            add_element(emotions, author, [map_tales_emotions(emotions_tmp[0])])
            # print(f)


    author = "HCAndersen"
    folder = '/Users/andrejaanaandova/Downloads/data/tales-emotions/HCAndersen/emmood/'
    for file in os.listdir(folder):
        f_tmp = open(folder+file, 'r').read().split('\n')
        for f in f_tmp:
            f = f.split('\t')
            if f[0] == '':
                continue
            text = f[3]
            emotions_tmp = f[1].split(":")
            if emotions_tmp[0]!=emotions_tmp[1]:
                # print(emotions_tmp)
                continue
            add_element(data, author, [preprocess_text(text)])
            add_element(emotions, author, [map_tales_emotions(emotions_tmp[0])])
            # print(f)


    author = "Potter"
    folder = '/Users/andrejaanaandova/Downloads/data/tales-emotions/Potter/emmood/'
    for file in os.listdir(folder):
        f_tmp = open(folder+file, 'r').read().split('\n')
        for f in f_tmp:
            f = f.split('\t')
            if f[0] == '':
                continue
            text = f[3]
            emotions_tmp = f[1].split(":")
            if emotions_tmp[0]!=emotions_tmp[1]:
                # print(emotions_tmp)
                continue
            add_element(data, author, [preprocess_text(text)])
            add_element(emotions, author, [map_tales_emotions(emotions_tmp[0])])
            # print(f)

    # print(data)
    # print(emotions)
    return data, emotions

def load_ISEAR():
    folder = '/Users/andrejaanaandova/Downloads/data/electorial_tweets/datasets/isear/'
    file = open(folder+'isear.csv').read().split('\n')
    # print(len(file))
    set_emos = set()
    data = []
    emotions = []
    for line in file:
        line2 = line.split('|')
        if len(line2)!=43:
            emotion = map_isear_emotions(line2[36])
            set_emos.add(emotion)
            text = preprocess_text(" ".join(line2[40:-2]))
            # print(emotion, text)
            # print(line)
            if emotion is not None:
                data.append(text)
                emotions.append(emotion)
        else:
            text = preprocess_text(line2[40])
            emotion = map_isear_emotions(line2[36])
            set_emos.add(emotion)
            # print(line2)
            # print(emotion, text)
            if emotion is not None:
                data.append(text)
                emotions.append(emotion)
    print(set_emos)
    return data, emotions

def load_emoInt():
    data = {}
    emotions = {}
    folder = "/Users/andrejaanaandova/Downloads/data/EmoInt/"
    for file in os.listdir(folder):
        emo = map_emoInt(file.split('-')[0])
        key = file.split('.')[-2]
        # print(key)
        f = open(folder+file, 'r').read().split('\n')
        for line in f:
            line = line.split('\t')
            # print(line)
            if len(line)!=1 and line[3]=="NONE":
                text = preprocess_text(line[1])
                add_element(data, key, [text])
                add_element(emotions, key, [emo])
    return data, emotions

def load_electorialTweets():
    data = {}
    emotions = {}
    folder = '/Users/andrejaanaandova/Downloads/data/electorial_tweets/datasets/electoraltweets/Annotated-US2012-Election-Tweets/'
    file1 = open(folder + 'Questionnaire1/AnnotatedTweets.txt').read().split('\n')
    file2 = open(folder+'Questionnaire2/Batch1/AnnotatedTweets.txt').read().split('\n')
    file3 = open(folder + 'Questionnaire2/Batch2/AnnotatedTweets.txt').read().split('\n')
    tweets_ids = set()
    for line in file1:
        if line[0]<'0' or line[0]>'9':
            continue
        line = line.split('\t')
        if line[10]=='This tweet has no emotional content.':
            data[line[0]] = preprocess_text(line[14])
            emotions[line[0]] = 'neutral'
            tweets_ids.add(line[0])
    for line in file2:
        if line[0]<'0' or line[0]>'9':
            continue
        line = line.split('\t')
        # print(line)
        emotion = map_electiontweets(line[15])
        if emotion==None:
            continue
        text = preprocess_text(line[13])
        # print(emotion+"   "+text)
        if line[0] in tweets_ids and emotions[line[0]]!=emotion:
            emotions[line[0]] = None
            continue
        data[line[0]] = text
        emotions[line[0]] = emotion
        tweets_ids.add(line[0])
    for line in file3:
        if line[0]<'0' or line[0]>'9':
            continue
        line = line.split('\t')
        # print(line)
        emotion = map_electiontweets(line[15])
        if emotion==None:
            continue
        text = preprocess_text(line[13])
        # print(emotion+"   "+text)
        if line[0] in tweets_ids and emotions[line[0]]!=emotion:
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
        res_data.append(data[id])
        # print(emotion)
        res_emotions.append(emotions[id])
    return res_data, res_emotions
    # print(len(file1))
    # print(len(file2))
    # print(len(file3))

def add_element(dict, key, el):
    if el[-1]=='':
        el = el[:-1]
    # print(el)
    if key in dict:
        dict[key] = dict[key]+el
        return dict
    dict[key] = []
    dict[key] = dict[key]+el
    return dict