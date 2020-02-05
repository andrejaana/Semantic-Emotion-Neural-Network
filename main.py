# from data_loader import load_italian_data
# from classification import classify_italian, classify_english
# from youtube_downloader.file_choosing import get_youtube_data
# from text_emotion_classification.text_emotion_classification import classify_texts
# from graph_star.run_text_classification import graph_star_classification
from SENN.data_loader import *
from SENN.SENN import build_model, model
import sys
import pandas as pd
import SENN.evaluate


# data = get_youtube_data()
# transcriptions = data['transcription_text'].tolist()
# print(transcriptions)
# print(data['videoId'])
# result = classify_english(data=transcriptions)
# # result = classify_texts(transcriptions)
# data ['supposed_emotion'] = result
# print(data[['videoId', 'supposed_emotion']])


# graph_star_classification()

data_folder = '/Users/andrejaanaandova/Downloads/data/electorial_tweets/datasets/'
if len(sys.argv)>1:
    data_folder = sys.argv[1]
X_train, X_test, y_train, y_test = load_allDatasets(data_folder)
model = build_model(X_train, X_test, y_train, y_test)

# from SENN.SENN import get_attributes
# get_attributes()

# load_crowdFlower()
# load_TEC()
# load_tales_emotions()
# load_ISEAR()
# load_emoInt()
# data, emotions = load_electorialTweets()
# data, emotions = load_emotioncause(data_folder)
# model = model(data,emotions)