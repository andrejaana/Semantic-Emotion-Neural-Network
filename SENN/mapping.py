from SENN.preprocessing import preprocess_text

def map_dialog_topic(topic):
    if topic == '1':
        return "ordinary_life"
    elif topic=='2':
        return 'school_life'
    elif topic=='3':
        return 'culture_and_education'
    elif topic=='4':
        return 'attitude_and_emotion'
    elif topic=='5':
        return 'relationships'
    elif topic=='6':
        return 'tourism'
    elif topic=='7':
        return 'health'
    elif topic=='8':
        return 'work'
    elif topic=='9':
        return 'politics'
    elif topic=='10':
        return 'finance'
    else:
        print("Error while reading topic - Topic unknown " + topic)
        exit(1)

def map_dialog_emotion(emo):
    if emo=='0':
        return [0,0,0,1,0,0,0]
    elif emo=='1':
        return [1,0,0,0,0,0,0]
    elif emo=='2':
        return [0,0,1,0,0,0,0]
    elif emo=='3':
        return [0,0,0,0,1,0,0]
    elif emo=='4':
        return [0,0,0,0,0,0,1]
    elif emo=='5':
        return [0,1,0,0,0,0,0]
    elif emo=='6':
        return [0,0,0,0,0,1,0]
    else:
        print("Unknown emotion "+ emo)
        exit(1)

def map_crowdflower(data, emotions):
    res_data = []
    res_emotions = []
    for i in range(0, len(emotions)):
        emotion = emotions[i]
        if emotion == 'neutral' or emotion=='boredom' or emotion=='relief':
            res_emotions.append([0,0,0,1,0,0,0])    # Neutral
            res_data.append(preprocess_text(data[i]))
        elif emotion=='hate' or emotion=='anger':
            res_emotions.append([1,0,0,0,0,0,0])    # Anger
            res_data.append(preprocess_text(data[i]))
        elif emotion=='happiness' or emotion=='love' or emotion=='fun' or emotion=='enthusiasm':
            res_emotions.append([0,0,0,0,0,0,1])    # Happiness
            res_data.append(preprocess_text(data[i]))
        elif emotion=='sadness':
            res_emotions.append([0,1,0,0,0,0,0])    # Sadness
            res_data.append(preprocess_text(data[i]))
        elif emotion=='worry':
            res_emotions.append([0,0,0,0,1,0,0])    # Fear
            res_data.append(preprocess_text(data[i]))
        elif emotion=='surprise':
            res_emotions.append([0,0,0,0,0,1,0])    # Surprise
            res_data.append(preprocess_text(data[i]))
        elif emotion=='empty':
            continue
    return res_data, res_emotions

def map_TEC_emotion(emo):
    if emo==' anger':
        return [1,0,0,0,0,0,0]
    elif emo==' disgust':
        return [0,0,1,0,0,0,0]
    elif emo==' fear':
        return [0,0,0,0,1,0,0]
    elif emo==' joy':
        return [0,0,0,0,0,0,1]
    elif emo==' sadness':
        return [0,1,0,0,0,0,0]
    elif emo==' surprise':
        return [0,0,0,0,0,1,0]
    else:
        print("Unknown emotion \;"+ emo+'\;')
        exit(1)

def map_tales_emotions(emo):
    if emo=='N':
        return [0,0,0,1,0,0,0]
    elif emo=='A':
        return [1,0,0,0,0,0,0]
    elif emo=='D':
        return [0,0,1,0,0,0,0]
    elif emo=='F':
        return [0,0,0,0,1,0,0]
    elif emo=='H':
        return [0,0,0,0,0,0,1]
    elif emo=='Sa':
        return [0,1,0,0,0,0,0]
    elif emo=='Su+' or emo=='Su-':
        return [0,0,0,0,0,1,0]
    else:
        print("Unknown emotion "+ emo)
        exit(1)

def map_isear_emotions(emo):
    if emo=='anger':
        return [1,0,0,0,0,0,0]
    elif emo=='fear':
        return [0,0,0,0,1,0,0]
    elif emo=='joy':
        return [0,0,0,0,0,0,1]
    elif emo=='sadness':
        return [0,1,0,0,0,0,0]
    elif emo=='disgust':
        return [0,0,1,0,0,0,0]
    elif emo=='shame' or emo=='guilt' or emo=='Field1':
        return None
    else:
        print("Unknown emotion "+ emo)
        exit(1)

def map_emoInt(emo):
    if emo=='anger':
        return [1,0,0,0,0,0,0]
    elif emo=='fear':
        return [0,0,0,0,1,0,0]
    elif emo=='joy':
        return [0,0,0,0,0,0,1]
    elif emo=='sadness':
        return [0,1,0,0,0,0,0]
    else:
        print("Unknown emotion "+ emo)
        return None

def map_electiontweets(emotion):
    emotion = emotion.split()
    if "anger" in emotion or 'vigilance' in emotion:
        return [1,0,0,0,0,0,0]
    elif 'sadness' in emotion:
        return [0,1,0,0,0,0,0]
    elif 'disgust' in emotion or 'dislike' in emotion or 'hate' in emotion or 'disappointment' in emotion:
        return [0,0,1,0,0,0,0]
    elif 'indifference' in emotion or 'calmness' in emotion:
        return [0,0,0,1,0,0,0]
    elif 'fear' in emotion:
        return [0,0,0,0,1,0,0]
    elif 'surprise' in emotion or 'amazement' in emotion or 'anticipation' in emotion or 'confusion' in emotion:
        return [0,0,0,0,0,1,0]
    elif 'joy' in emotion or 'like' in emotion or 'trust' in emotion or \
            'admiration' in emotion or 'acceptance' in emotion:
        return [0,0,0,0,0,0,1]
    else:
        print("Emotion not recognised "+" ".join(emotion))
    return None

