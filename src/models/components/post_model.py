import pandas as pd
import joblib


class PostModel():
    def __init__(self, model_path, using_label):
        if model_path is None:
            self.model = None
        else:
            self.model = joblib.load(model_path)
        self.using_label = using_label

    def __call__(self, inputs, preds):
        if self.model is None:
            return preds
        inputs = inputs[self.using_label]
        inputs = pd.concat([inputs, pd.Series(preds, name='BERTPRED')], axis=1)
        return self.model.predict(inputs)
