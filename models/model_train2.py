from dateutil.parser import parse
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import shuffle

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, ComplementNB

import tensorflow as tf
from tensorflow import feature_column
from tensorflow.keras import layers
from tensorflow.keras.models import load_model

from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import joblib
class model_train():
    def __init__(self):
        self.results = []
        self.models = {}
    def load_data(self):
        # Genuine Account CSV files
        path_genuineAccounts_USERS = r"E:\edge下载\cresci-2017.csv\data\genuine_accounts.csv\genuine_accounts.csv\genuine_users.csv"
        path_genuineAccounts_TWEETS = r"E:\edge下载\cresci-2017.csv\data\genuine_accounts.csv\genuine_accounts.csv\genuine_tweets.csv"

        # Spam Bot Account CSV files
        path_spamBots_USERS = r"E:\edge下载\cresci-2017.csv\data\social_spambots_1.csv\social_spambots_1.csv\social_users.csv"
        path_spamBots_TWEETS = r"E:\edge下载\cresci-2017.csv\data\social_spambots_1.csv\social_spambots_1.csv\social_tweets.csv"

        self.df_genuine_users = pd.read_csv(path_genuineAccounts_USERS, encoding='utf-8')
        self.df_genuine_tweets = pd.read_csv(path_genuineAccounts_TWEETS, encoding='utf-8')
        self.df_spambots_users = pd.read_csv(path_spamBots_USERS, encoding='utf-8')
        self.df_spambots_tweets = pd.read_csv(path_spamBots_TWEETS, encoding='utf-8')

    def concat_data(self):
        self.x_df = pd.concat([self.df_genuine_users, self.df_spambots_users])
        self.tweets_df = pd.concat([self.df_genuine_tweets, self.df_spambots_tweets])

    @staticmethod
    def change_ts(ts):
        cts = 0
        try:
            cts = parse(ts).timestamp()
        except:
            cts = int(ts[:-1]) / 1000
        return cts

    @staticmethod
    def get_intertime(df):
        tweet_timestamps = []

        for index, row in df.iterrows():
            tweet_timestamps.append(model_train.change_ts(str(row["created_at"])))

        # df["created_at"].values.tolist().map(change_ts)

        tweet_timestamps.sort()
        tts_diff = np.diff(np.array(tweet_timestamps))
        return sum(tts_diff) / (len(tts_diff) if len(tts_diff) != 0 else 1)
    def features_get(self):
        x_vector = []

        for index, row in self.x_df.iterrows():
            # current user's id
            usr = row['id']
            usr_tweets_df = self.tweets_df.loc[self.tweets_df["user_id"] == usr]
            tweet_count = len(usr_tweets_df) if len(usr_tweets_df) != 0 else 1

            # calculate features
            retweets = len(usr_tweets_df[usr_tweets_df.retweeted_status_id != 0]) / tweet_count
            replies = len(usr_tweets_df[usr_tweets_df.in_reply_to_status_id != 0]) / tweet_count
            favoriteC = row["favourites_count"] / tweet_count
            hashtag = sum(usr_tweets_df["num_hashtags"].values.tolist()) / tweet_count
            url = sum(usr_tweets_df["num_urls"].values.tolist()) / tweet_count
            mentions = sum(usr_tweets_df["num_mentions"].values.tolist()) / tweet_count
            intertime = self.get_intertime(usr_tweets_df)
            ffratio = row["friends_count"] / (row["followers_count"] if row["followers_count"] != 0 else 1)
            favorites = row["favourites_count"]
            listed = row["listed_count"]
            uniqueHashtags = -1
            uniqueMentions = -1
            uniqueURL = -1

            # usr_features = [retweets, replies, favoriteC, hashtag, url, mentions, intertime, ffratio, favorites, listed]
            usr_features = [retweets, replies, favoriteC, hashtag, url, mentions, intertime, ffratio]
            x_vector.append(usr_features)
            self.x = preprocessing.normalize(x_vector)

    def trans(self):
        # 将经过归一化处理后的数据（存储在变量 x 中）转换为一个 Pandas 数据框 df，然后设置显示选项以展示数据框的所有列，并打印出数据框的前几行数据进行查看，最后重置之前设置的显示选项
        # df = pd.DataFrame(self.x, columns=['retweets', 'replies', 'favoriteC', 'hashtag', 'url', 'mentions', 'intertime',
        #                               'ffratio', 'favorites', 'listed'])
        df = pd.DataFrame(self.x, columns=['retweets', 'replies', 'favoriteC', 'hashtag', 'url', 'mentions', 'intertime', 'ffratio'])
        pd.set_option('display.max_columns', None)
        print(df.head())
        pd.reset_option('max_columns')

        # 创建一个用于标记数据类别的标签列表 y，其中标签值根据数据来源进行设定：来自 df_genuine_users 的数据对应的标签为 0，来自 df_spamBots_users 的数据对应的标签为 1
        y = np.zeros(len(self.df_genuine_users), dtype=int).tolist() + np.ones(len(self.df_spambots_users), dtype=int).tolist()
        # 对特征数据 x 和对应的标签数据 y 进行随机打乱操作
        x, y = shuffle(self.x, y)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y)

    def SVC_linear_classifier(self):
        SVC_linear_classifier = SVC(kernel='linear', probability=True)
        SVC_linear_classifier.fit(self.x_train, self.y_train)

        y_pred = SVC_linear_classifier.predict(self.x_test)

        # print(confusion_matrix(y_test,y_pred))
        print(classification_report(self.y_test, y_pred, target_names=['Human', 'Bot']))

        tn, fp, fn, tp = confusion_matrix(self.y_test, y_pred).ravel()
        print("True Negatives: ", tn)
        print("False Positives: ", fp)
        print("False Negatives: ", fn)
        print("True Positives: ", tp)

        fpr, tpr, _ = roc_curve(self.y_test, SVC_linear_classifier.predict_proba(self.x_test)[:, 1])
        auc = roc_auc_score(self.y_test, SVC_linear_classifier.predict_proba(self.x_test)[:, 1])
        self.results.append([fpr, tpr, "SVM Linear Kernel, AUC={:.3f}".format(auc)])
        self.models['SVC_linear_classifier'] = SVC_linear_classifier

    def SVC_poly_classifier(self):
        SVC_poly_classifier = SVC(kernel='poly', probability=True)
        SVC_poly_classifier.fit(self.x_train, self.y_train)

        y_pred = SVC_poly_classifier.predict(self.x_test)

        # print(confusion_matrix(y_test,y_pred))
        print(classification_report(self.y_test, y_pred, target_names=['Human', 'Bot']))

        tn, fp, fn, tp = confusion_matrix(self.y_test, y_pred).ravel()
        print("True Negatives: ", tn)
        print("False Positives: ", fp)
        print("False Negatives: ", fn)
        print("True Positives: ", tp)

        # 基于已经训练好的采用多项式核函数的支持向量机分类器（SVC_poly_classifier），计算其接收者操作特征曲线（ROC 曲线）相关的数据，并求出曲线下面积（AUC）值，最后将这些结果添加到 results 列表中，以便后续对不同模型（如果有多个模型进行类似评估）的性能进行比较、展示或进一步分析等操作
        fpr, tpr, _ = roc_curve(self.y_test, SVC_poly_classifier.predict_proba(self.x_test)[:, 1])
        auc = roc_auc_score(self.y_test, SVC_poly_classifier.predict_proba(self.x_test)[:, 1])
        self.results.append([fpr, tpr, "SVM Polynominal Kernel, AUC={:.3f}".format(auc)])

        self.models['SVC_poly_classifier'] = SVC_poly_classifier

    def Random_Forest_Classifier(self):
        rfc_model = RandomForestClassifier()
        rfc_model.fit(self.x_train, self.y_train)
        y_pred = rfc_model.predict(self.x_test)
        print(classification_report(self.y_test, y_pred, target_names=['Human', 'Bot']))
        print(confusion_matrix(self.y_test, y_pred))
        tn, fp, fn, tp = confusion_matrix(self.y_test, y_pred).ravel()
        print(f"True Negatives: {tn}, False Positives: {fp}, False Negatives: {fn}, True Positives: {tp}")
        fpr, tpr, _ = roc_curve(self.y_test, rfc_model.predict_proba(self.x_test)[:, 1])
        auc = roc_auc_score(self.y_test, rfc_model.predict_proba(self.x_test)[:, 1])
        self.results.append([fpr, tpr, "RFC, AUC={:.3f}".format(auc)])

        self.models['rfc_model'] = rfc_model

    def Logistic_Regression(self):
        logreg_model = LogisticRegression()
        logreg_model.fit(self.x_train, self.y_train)
        y_pred = logreg_model.predict(self.x_test)
        print(classification_report(self.y_test, y_pred, target_names=['Human', 'Bot']))
        print(confusion_matrix(self.y_test, y_pred))
        tn, fp, fn, tp = confusion_matrix(self.y_test, y_pred).ravel()
        print(f"True Negatives: {tn}, False Positives: {fp}, False Negatives: {fn}, True Positives: {tp}")
        fpr, tpr, _ = roc_curve(self.y_test, logreg_model.predict_proba(self.x_test)[:, 1])
        auc = roc_auc_score(self.y_test, logreg_model.predict_proba(self.x_test)[:, 1])
        self.results.append([fpr, tpr, "Log Regression, AUC={:.3f}".format(auc)])

        self.models['logreg_model'] = logreg_model

    def n_bayes_gaussian(self):
        n_bayes = GaussianNB()
        n_bayes.fit(self.x_train, self.y_train)
        expected = self.y_test
        predicted = n_bayes.predict(self.x_test)
        print(confusion_matrix(expected, predicted))
        tn, fp, fn, tp = confusion_matrix(expected, predicted).ravel()
        print(f"True Negatives: {tn}, False Positives: {fp}, False Negatives: {fn}, True Positives: {tp}")
        fpr, tpr, _ = roc_curve(self.y_test, n_bayes.predict_proba(self.x_test)[:, 1])
        auc = roc_auc_score(self.y_test, n_bayes.predict_proba(self.x_test)[:, 1])
        self.results.append([fpr, tpr, "Gaussian NB, AUC={:.3f}".format(auc)])

        self.models['n_bayes'] = n_bayes

    def cn_bayes_complement(self):
        cn_bayes = ComplementNB()
        cn_bayes.fit(self.x_train, self.y_train)
        y_pred = cn_bayes.predict(self.x_test)
        print(classification_report(self.y_test, y_pred, target_names=['Human', 'Bot']))
        print(confusion_matrix(self.y_test, y_pred))
        tn, fp, fn, tp = confusion_matrix(self.y_test, y_pred).ravel()
        print(f"True Negatives: {tn}, False Positives: {fp}, False Negatives: {fn}, True Positives: {tp}")
        fpr, tpr, _ = roc_curve(self.y_test, cn_bayes.predict_proba(self.x_test)[:, 1])
        auc = roc_auc_score(self.y_test, cn_bayes.predict_proba(self.x_test)[:, 1])
        self.results.append([fpr, tpr, "Complement NB, AUC={:.3f}".format(auc)])

        self.models['cn_bayes'] = cn_bayes

    def neural_network(self):
        # parameters
        batch_size = 32
        epochs = 20
        model = tf.keras.Sequential([
            layers.Dense(10),
            layers.Dense(128, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dropout(.1),
            layers.Dense(1, activation='sigmoid')
        ])

        model.compile(optimizer='adam',
                      loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                      metrics=['accuracy'])

        model.fit(np.array(self.x_train),
                  np.array(self.y_train),
                  batch_size=batch_size,
                  epochs=epochs)
        loss, accuracy = model.evaluate(np.array(self.x_test), np.array(self.y_test), batch_size=batch_size)
        print("Accuracy: ", accuracy)
        y_pred = (model.predict(self.x_test) > 0.5).astype('int32')
        print(classification_report(self.y_test, y_pred, target_names=['Human', 'Bot']))
        print(confusion_matrix(self.y_test, y_pred))
        tn, fp, fn, tp = confusion_matrix(self.y_test, y_pred).ravel()
        print(f"True Negatives: {tn}, False Positives: {fp}, False Negatives: {fn}, True Positives: {tp}")
        fpr, tpr, _ = roc_curve(self.y_test, model.predict(self.x_test).ravel())
        auc = roc_auc_score(self.y_test, model.predict(self.x_test).ravel())
        self.results.append([fpr, tpr, "Neural Network, AUC={:.3f}".format(auc)])

        self.models['neural_network'] = model

    def ROC_Curves(self):
        fig = plt.figure(figsize=(8, 6))

        for result in self.results:
            plt.plot(result[0], result[1], label=result[2])

        plt.plot([0, 1], [0, 1], color='orange', linestyle='--')

        plt.xticks(np.arange(0.0, 1.1, step=0.1))
        plt.xlabel("False Positive Rate", fontsize=15)

        plt.yticks(np.arange(0.0, 1.1, step=0.1))
        plt.ylabel("True Positive Rate", fontsize=15)

        plt.title('ROC Curves', fontweight='bold', fontsize=15)
        plt.legend(prop={'size': 13}, loc='lower right')

        plt.show()

    def model_save(self):
        # 假设 models 是一个字典，包含所有训练好的模型
        for model_name, model in self.models.items():
            joblib.dump(model, f'D:/{model_name}_model.pkl')
        # 保存神经网络模型
        model.save('D:/neural_network_model.keras')

    def run(self):
        self.load_data()
        self.concat_data()
        self.features_get()
        self.trans()

        self.SVC_linear_classifier()
        self.SVC_poly_classifier()
        self.Random_Forest_Classifier()
        self.Logistic_Regression()
        self.n_bayes_gaussian()
        self.cn_bayes_complement()
        self.neural_network()

        self.ROC_Curves()
        self.model_save()

if __name__ == '__main__':
    model_train = model_train()
    model_train.run()




