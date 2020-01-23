


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
        return 'neutral'
    elif emo=='1':
        return 'anger'
    elif emo=='2':
        return 'disgust'
    elif emo=='3':
        return 'fear'
    elif emo=='4':
        return 'happiness'
    elif emo=='5':
        return 'sadness'
    elif emo=='6':
        return 'surprise'
    else:
        print("Unknown emotion "+ emo)
        exit(1)

def map_TEC_emotion(emo):
    if emo==' anger':
        return 'anger'
    elif emo==' disgust':
        return 'disgust'
    elif emo==' fear':
        return 'fear'
    elif emo==' joy':
        return 'happiness'
    elif emo==' sadness':
        return 'sadness'
    elif emo==' surprise':
        return 'surprise'
    else:
        print("Unknown emotion \;"+ emo+'\;')
        exit(1)

def map_tales_emotions(emo):
    if emo=='N':
        return 'neutral'
    elif emo=='A':
        return 'anger'
    elif emo=='D':
        return 'disgust'
    elif emo=='F':
        return 'fear'
    elif emo=='H':
        return 'happiness'
    elif emo=='Sa':
        return 'sadness'
    elif emo=='Su+' or emo=='Su-':
        return 'surprise'
    else:
        print("Unknown emotion "+ emo)
        exit(1)

def map_isear_emotions(emo):
    if emo=='anger':
        return 'anger'
    elif emo=='fear':
        return 'fear'
    elif emo=='joy':
        return 'happiness'
    elif emo=='sadness':
        return 'sadness'
    elif emo=='disgust':
        return 'disgust'
    elif emo=='shame' or emo=='guilt' or emo=='Field1':
        return None
    else:
        print("Unknown emotion "+ emo)
        exit(1)

def map_emoInt(emo):
    if emo=='anger':
        return 'anger'
    elif emo=='fear':
        return 'fear'
    elif emo=='joy':
        return 'happiness'
    elif emo=='sadness':
        return 'sadness'
    else:
        print("Unknown emotion "+ emo)
        exit(1)

def map_electiontweets(emotion):
    # print(emotion)
    emotion = emotion.split()
    # print(emotion)
    if "anger" in emotion or 'vigilance' in emotion:
        return "anger"
    elif 'sadness' in emotion:
        return 'sadness'
    elif 'disgust' in emotion or 'dislike' in emotion or 'hate' in emotion or 'disappointment' in emotion:
        return 'disgust'
    elif 'indifference' in emotion or 'calmness' in emotion:
        return 'neutral'
    elif 'fear' in emotion:
        return 'fear'
    elif 'surprise' in emotion or 'amazement' in emotion or 'anticipation' in emotion or 'confusion' in emotion:
        return 'surprise'
    elif 'joy' in emotion or 'like' in emotion or 'trust' in emotion or \
            'admiration' in emotion or 'acceptance' in emotion:
        return 'happiness'
    else:
        print("Emotion not recognised "+" ".join(emotion))
    return None