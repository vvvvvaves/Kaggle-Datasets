import pickle, os
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, recall_score, precision_score, confusion_matrix, accuracy_score

from preprocessor import *

lgb_params = {

    "objective": "binary",
    "learning_rate": 0.01,
    "num_threads": 10,
    "metric": "AUC",
    "seed": 42,
    "verbose": -1,
    "class_weight": "balanced",

    # regularization
    "colsample_bytree": 0.7,
    "subsample": 0.8,

    "subsample_freq": 1,
    "min_data_in_leaf": 150,

    "num_leaves": 17,

    "n_estimators": 3500

    # categorical features
    #     'cat_smooth': 5,
    #     'min_data_per_group': 2
    #     did not improve the results

}

class Predictor:
    def __init__(self):
        self.preprocessor = None
        self.train_prep = None
        self.test_prep = None

        self.model = None

        if os.path.exists('model.pickle'):
            with open('model.pickle', 'rb') as handle:
                self.model = pickle.load(handle)
        else:
            self.train_model()

            with open('model.pickle', 'wb') as handle:
                pickle.dump(self.model, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def model_presentation(self, threshold=0.5):
        self.load_data()
        test_X, test_y = self.test_prep.drop('RainTomorrow', axis=1), self.test_prep.RainTomorrow.fillna(0)
        probs = self.model.predict_proba(test_X)
        auc_score = roc_auc_score(y_true=test_y, y_score=probs[:, 1])
        preds = np.vectorize(lambda x: 1 if x >= threshold else 0)(probs[:, 1])
        conf_m = confusion_matrix(y_true=test_y, y_pred=preds, normalize='all')
        accuracy = accuracy_score(y_true=test_y, y_pred=preds)
        recall = recall_score(y_true=test_y, y_pred=preds)
        precision = precision_score(y_true=test_y, y_pred=preds)
        f1_score = 2*precision*recall/(precision+recall)
        return auc_score, conf_m, accuracy, recall, precision, f1_score

    def train_model(self):
        self.preprocessor = Preprocessor(numerical_impute_strategy='mean',
                                         categorical_impute_strategy='mode',
                                         roll_cols=standard_roll_cols,
                                         roll_period=standard_roll_period,
                                         roll_strategies=standard_roll_strategies)
        self.load_data()

        train_X, train_y = self.train_prep.drop('RainTomorrow', axis=1), self.train_prep.RainTomorrow.fillna(0)
        classifier = lgb.LGBMClassifier(**lgb_params)
        classifier.fit(train_X, train_y)

        self.model = classifier

    def load_data(self):
        test_set = None
        if self.train_prep is None and self.model is None:
            if os.path.exists('train_prep.pickle'):
                with open('train_prep.pickle', 'rb') as handle:
                    self.train_prep = pickle.load(handle)
            else:
                train_set, test_set = self.preprocessor.load_and_split()
                self.train_prep = self.preprocessor.preprocess(_data=train_set,
                                                               visualize=False,
                                                               train=True)

                with open('train_prep.pickle', 'wb') as handle:
                    pickle.dump(self.train_prep, handle, protocol=pickle.HIGHEST_PROTOCOL)

        if self.test_prep is None:
            if os.path.exists('test_prep.pickle'):
                with open('test_prep.pickle', 'rb') as handle:
                    self.test_prep = pickle.load(handle)
            else:
                if test_set is None:
                    train_set, test_set = self.preprocessor.load_and_split()

                self.test_prep = self.preprocessor.preprocess(_data=test_set,
                                                              visualize=False,
                                                              train=False)

                with open('test_prep.pickle', 'wb') as handle:
                    pickle.dump(self.test_prep, handle, protocol=pickle.HIGHEST_PROTOCOL)


