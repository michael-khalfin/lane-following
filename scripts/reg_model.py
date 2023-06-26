import rosbag
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error
from sklearn import preprocessing
import matplotlib.pyplot as plt

SLOPE_TOPIC = '/slope'
VEH_STEERING_TOPIC = '/vehicle/twist'

class RegModel:

    def __init__(self, bag_name, model_name=0, gen_img=False):
        self.data = self.extract_data(bag_name)

        target = 'ang_z'
        self.y = self.data[target]
        self.X = self.data.drop(columns=target)

        cutoff = int(len(self.X) * .8)
        self.X_train, self.y_train = self.X.iloc[:cutoff], self.y.iloc[:cutoff]
        self.X_test, self.y_test = self.X.iloc[cutoff:], self.y.iloc[cutoff:]

        if model_name == 0:
            self.model = self.linear_reg(gen_img=gen_img)
        elif model_name == 1:
            self.model = self.log_reg(L=3, gen_img=gen_img)

    def extract_data(self, bag_name):
        data = {
            'slope': [],
            'ang_z': []
        }

        bag = rosbag.Bag(bag_name)

        for topic, msg, t in bag.read_messages(topics=[SLOPE_TOPIC]):
            data['slope'].append(float(msg.data))

        i = 0
        for topic, msg, t in bag.read_messages(topics=[VEH_STEERING_TOPIC]):
            if i == 10:
                data['ang_z'].append(float(msg.twist.angular.z))
                i = 0
            i += 1

        data['slope'].pop()
        bag.close()

        return pd.DataFrame.from_dict(data).sample(frac = 1)

    def linear_reg(self, gen_img=False):
        model = LinearRegression()
        model.fit(self.X_train, self.y_train)
        y_pred_baseline, y_pred_training = self.generate_stats(model)

        if gen_img:
            fig, ax = plt.subplots(figsize=(15,6))
            plt.plot(self.X_train.values, y_pred_baseline, color='orange', label='Baseline Model')
            plt.plot(self.X_train.values, y_pred_training, color='red', label='Linear Model')
            plt.scatter(x = self.data['slope'], y = self.data['ang_z'])
            plt.xlabel('Slope of lines')
            plt.ylabel('Angular z twist message')
            plt.title('Slope vs Angular z')
            plt.legend()
            plt.show()

        return model

    def log_reg(self, L=3, gen_img=False):
        #print(self.data.head())
        self.y_train[self.y_train < 0] = np.nan
        y_transform = np.log((1 / self.y_train) - 1)
        y_transform = y_transform.replace([np.inf, -np.inf], np.nan, regex=True)
        y_transform = y_transform.fillna(method='ffill')
        #print(self.y_train.isna().sum())

        model = LinearRegression()
        model.fit(self.X_train, y_transform)
        y_pred_baseline, y_pred_training = self.generate_stats(model)

        if gen_img:
            fig, ax = plt.subplots(figsize=(15,6))
            plt.plot(self.X_train.values, y_pred_baseline, color='orange', label='Baseline Model')

            alpha = model.coef_[0]
            beta = model.predict(pd.DataFrame({'slope': 0}, index=[0]))[0]
            predicted = 1 / (1 + np.exp(alpha * self.X_train + beta))
            #plt.scatter(self.X_train.values, predicted.values)
            print(alpha, beta)

            plt.show()

        return model

    def generate_stats(self, model):
        y_mean = self.y_train.mean()
        y_pred_baseline = [y_mean] * len(self.y_train)
        mae_baseline = mean_absolute_error(self.y_train.fillna(method='ffill'), y_pred_baseline)

        y_pred_training = model.predict(self.X_train)
        mae_training = mean_absolute_error(self.y_train.fillna(method='ffill'), y_pred_training)
        y_pred_test = model.predict(self.X_test)
        mae_testing = mean_absolute_error(self.y_test, y_pred_test)
        print(f'mae for baseline: {mae_baseline}, training: {mae_training}, testing: {mae_testing}')
        return y_pred_baseline, y_pred_training

    def make_prediction(self, slope):
        given = {
            'slope': [slope]
        }
        y = pd.DataFrame(given, index=[0])
        prediction = self.model.predict(y).round(2)
        return prediction

if __name__ == '__main__':
    model = RegModel('2023-06-26-11-01-18.bag', model_name=1, gen_img=True)
