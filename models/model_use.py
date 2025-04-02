import numpy as np
import pandas as pd
from dateutil.parser import parse
from sklearn import preprocessing
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.models import load_model
import joblib

class Predictor:
    def __init__(self, path, friends, followers):
        self.path = path
        self.friends = friends
        self.followers = followers
        self.models = self.load_models()

    def load_models(self):
        """
        加载所有预训练模型
        :return: 包含所有模型的字典
        """
        model_paths = {
            'svc_linear': '../resources/models/SVC_linear_classifier_model.pkl',
            'svc_poly': '../resources/models/SVC_poly_classifier_model.pkl',
            'rfc_model': '../resources/models/rfc_model_model.pkl',
            'logreg_model': '../resources/models/logreg_model_model.pkl',
            'n_bayes': '../resources/models/n_bayes_model.pkl',
            'cn_bayes': '../resources/models/cn_bayes_model.pkl',
            'neural_network': '../resources/models/neural_network_model.keras'
        }
        models = {}
        for model_name, path in model_paths.items():
            try:
                if model_name == 'neural_network':
                    model = load_model(path, compile=False)
                    model.compile(optimizer='adam',
                                  loss=BinaryCrossentropy(from_logits=True),
                                  metrics=['accuracy'])
                else:
                    model = joblib.load(path)
                models[model_name] = model
            except FileNotFoundError:
                print(f"Error: {path} not found.")
            except Exception as e:
                print(f"Error loading {model_name}: {e}")
        return models

    def convert_to_timestamp(self, ts):
        """
        将输入的时间戳相关的字符串ts转换为一个数值形式的时间戳
        :param ts: 输入的时间戳字符串
        :return: 转换后的数值形式时间戳
        """
        try:
            return parse(ts).timestamp()
        except:
            try:
                return int(ts[:-1]) / 1000
            except:
                return np.nan

    def extract_features(self):
        """
        从CSV文件中提取特征
        :return: 用户特征列表
        """
        try:
            data_tem = pd.read_csv(self.path)
            columns = ['id', 'retweeted_status', 'in_reply_to', 'favorite_count', 'full_text', 'created_at']
            data = data_tem.filter(items=columns)
            total_count = len(data)

            retweet_count = data['retweeted_status'].count()
            retweet_ratio = retweet_count / total_count

            reply_count = data['in_reply_to'].count()
            reply_ratio = reply_count / total_count

            average_favorite = data['favorite_count'].mean()

            data['num_hashtags'] = data['full_text'].str.count('#')
            average_hashtags = data['num_hashtags'].mean()

            data['num_urls'] = data['full_text'].str.count('http')
            average_urls = data['num_urls'].mean()

            data['num_mentions'] = data['full_text'].str.count('@')
            average_mentions = data['num_mentions'].mean()

            data['timestamp'] = data['created_at'].apply(self.convert_to_timestamp)
            data = data.sort_values(by='timestamp')

            time_differences = data['timestamp'].diff()
            ffratio = self.friends / self.followers if self.followers != 0 else np.nan

            average_time_interval = time_differences.mean()
            usr_features = [retweet_ratio, reply_ratio, average_favorite, average_hashtags, average_urls,
                            average_mentions, average_time_interval, ffratio]
            return usr_features
        except FileNotFoundError:
            print(f"Error: {self.path} not found.")
            return None
        except Exception as e:
            print(f"Error extracting features: {e}")
            return None

    def normalize_data(self, x_vector):
        """
        对x_vector中的数据进行归一化处理
        :param x_vector: 包含用户特征向量的列表
        :return: 归一化后的NumPy数组
        """
        if x_vector is None:
            return None
        x_vector_reshaped = np.array(x_vector).reshape(1, -1)
        return preprocessing.normalize(x_vector_reshaped)

    def get_result(self):
        """
        提取特征、归一化数据并进行模型预测
        :return: 包含所有模型预测结果的字典
        """
        usr_features = self.extract_features()
        new_x_test_normalized = self.normalize_data(usr_features)
        if new_x_test_normalized is None:
            return {}
        predictions = {}
        for model_name, model in self.models.items():
            try:
                if model_name == 'neural_network':
                    pred = (model.predict(new_x_test_normalized) > 0.5).astype(int)
                else:
                    pred = model.predict(new_x_test_normalized)
                predictions[model_name] = pred
                print(f"{model_name.capitalize().replace('_', ' ')} Predictions:", pred)
            except Exception as e:
                print(f"Error predicting with {model_name}: {e}")
        return predictions

    def get_result2(self, selected_models):
        """
        根据选中的模型列表进行预测
        :param selected_models: 选中的模型名称列表
        :return: 包含选中模型预测结果的字典
        """
        print(1,selected_models)
        selected_models2 = [i.split()[0] for i in selected_models]
        print(2,selected_models2)
        usr_features = self.extract_features()
        new_x_test_normalized = self.normalize_data(usr_features)

        if new_x_test_normalized is None:
            return {}

        predictions = {}

        # 过滤出选中的有效模型
        valid_models = {
            name: model
            for name, model in self.models.items()
            if name in selected_models2 and model is not None
        }

        if not valid_models:
            print("No valid models selected for prediction.")
            return predictions

        for model_name, model in valid_models.items():
            try:
                # 特殊处理神经网络模型
                if model_name == 'neural_network':
                    pred = (model.predict(new_x_test_normalized) > 0.5).astype(int)
                else:
                    pred = model.predict(new_x_test_normalized)

                predictions[model_name] = pred
                print(f"[Selected Model] {model_name.replace('_', ' ').title()} Prediction:", pred)

            except Exception as e:
                print(f"Error predicting with {model_name}: {str(e)}")
                predictions[model_name] = [f"Error: {str(e)}"]

        return predictions

if __name__ == '__main__':
    print("埃隆马斯克")
    predictor1 = Predictor(r"../spider/test_埃隆.csv", 1095, 220048254)
    predictor1.get_result()

    print("自由战士🪖舒少波")
    predictor2 = Predictor(r"../spider/test_自由战士.csv", 24, 148)
    predictor2.get_result()    