from sklearn.metrics import confusion_matrix
import numpy as np


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