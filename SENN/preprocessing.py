import re

def preprocess_text(text):
    lowercase = text.lower()
    cleanText = re.sub('[^a-z\']+',' ', lowercase)
    cleanTwitterIds = re.sub('@[^\s]+', '', cleanText)
    # print(cleanTwitterIds)
    return cleanTwitterIds