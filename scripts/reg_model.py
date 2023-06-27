import rosbag
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

SLOPE_TOPIC = '/slope'
VEH_STEERING_TOPIC = '/vehicle/twist'

class RegModel:

    def __init__(self, bag_name, model_name=0, gen_img=False):
        self.data = self.extract_data(bag_name)
        self.model_name = model_name

        target = 'ang_z'
        self.y = self.data[target]
        self.X = self.data.drop(columns=target)

        cutoff = int(len(self.X) * .8)
        self.X_train, self.y_train = self.X.iloc[:cutoff], self.y.iloc[:cutoff]
        self.X_test, self.y_test = self.X.iloc[cutoff:], self.y.iloc[cutoff:]

        if self.model_name == 0:
            self.model = self.linear_reg(gen_img=gen_img)
        elif self.model_name == 1:
            self.model = self.log_reg(gen_img=gen_img)
        elif self.model_name == 2:
            self.model = self.arctan_reg(gen_img=gen_img)
        elif self.model_name == 3:
            self.model = self.piecewise_reg(gen_img=gen_img)
        elif self.model_name == -1:
            pass
        else:
            raise ValueError('Invalid model name')

    def extract_data(self, bag_name):
        """
        Extracts slope and angular z data from rosbag file and returns a shuffled Pandas DataFrame.

        Args:
            bag_name (str): The name/path of the rosbag file to extract data from.

        Returns:
            pandas.DataFrame: A shuffled DataFrame containing 'slope' and 'ang_z' columns.
        """
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
        """
        Fits a Linear Regression model to the training data and generates predictions on the training set and 
        a baseline model. If gen_img is True, a plot of the slope vs angular z is generated. Returns the trained 
        linear regression model.
        :param gen_img: Optional Boolean indicating whether to generate a plot of the slope vs angular z.
        :return: Linear Regression model
        """
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
            plt.savefig('linear.png')
            plt.show()

        return model
    
    def generate_stats(self, model):
        """
        Calculates the Mean Absolute Error (MAE) for a given model and returns the y_pred_baseline and y_pred_training.
        
        :param model: A trained model object to generate predictions on X_train and X_test
        :type model: Any
        
        :return: A tuple consisting of y_pred_baseline and y_pred_training
        :rtype: Tuple
        """
        y_mean = self.y_train.mean()
        y_pred_baseline = [y_mean] * len(self.y_train)
        mae_baseline = mean_absolute_error(self.y_train.fillna(method='ffill'), y_pred_baseline)

        y_pred_training = model.predict(self.X_train)
        mae_training = mean_absolute_error(self.y_train.fillna(method='ffill'), y_pred_training)
        y_pred_test = model.predict(self.X_test)
        mae_testing = mean_absolute_error(self.y_test, y_pred_test)
        print(f'mae for baseline: {mae_baseline}, training: {mae_training}, testing: {mae_testing}')
        return y_pred_baseline, y_pred_training

    def sigmoid(self, x, L, x0, k, b):
        return L / (1 + np.exp(-k * (x - x0))) + b
    def log_reg(self, gen_img=False):
        """
        Fits the logistic regression model using the training data and returns the optimized parameters. 

        :param gen_img: (bool) Whether to generate a scatter plot of the original data and the sigmoid curve. 
            Default is False.

        :return: (list) A list of optimized parameters for the logistic regression model.
        """
        # mandatory guess
        p0 = [max(self.y), np.median(self.X), 1, min(self.y)]

        # popt stores the best fit parameters
        # pcov stores the covariance matrix
        popt, pcov = curve_fit(self.sigmoid, self.X['slope'], self.y, p0=p0, method='dogbox')

        if gen_img:
            fig, ax = plt.subplots(figsize=(15,6))
            plt.scatter(self.X.values, self.y.values, color='orange', label='Data')
            plt.scatter(self.X.values, self.sigmoid(self.X.values, *popt), color='red', label='Sigmoid')
            plt.xlabel('Slope of lines')
            plt.ylabel('Angular z twist message')
            plt.title('Slope vs Angular z')
            plt.legend()
            plt.savefig('sigmoid.png')
            plt.show()

        return popt
    
    def arctan(self, x, k, w, x0, y0):
        return k * np.arctan(w * (x - x0)) + y0
    
    def arctan_reg(self, gen_img=False):
        """
        Fits the arctan regression model using the training data and returns the optimized parameters. 

        :param gen_img: (bool) Whether to generate a scatter plot of the original data and the arctan curve. 
            Default is False.

        :return: (list) A list of optimized parameters for the arctan regression model.
        """
        p0 = [1,1]
        p0.append(.5*(max(self.X['slope']) + min(self.X['slope'])))
        p0.append(.5*(max(self.y) + min(self.y)))

        # popt stores the best fit parameters
        # pcov stores the covariance matrix
        popt, pcov = curve_fit(self.arctan, self.X['slope'], self.y, p0=p0)

        if gen_img:
            fig, ax = plt.subplots(figsize=(15,6))
            plt.scatter(self.X.values, self.y.values, color='orange', label='Data')
            plt.scatter(self.X.values, self.arctan(self.X.values, *popt), color='red', label='Arctan')
            plt.xlabel('Slope of lines')
            plt.ylabel('Angular z twist message')
            plt.title('Slope vs Angular z')
            plt.legend()
            plt.savefig('arctan.png')
            plt.show()

        return popt
    
    def f(self, x, x0, x1, k1, k2, m, b, r1=.3, r2=.45):
        x = np.array(x)
        conds = [x < r1, (x >= r1) * (x < r2), x >= r2]
        f1 = lambda x: -1/(x+x0)+k1
        f2 = lambda x: m*x + b
        f3 = lambda x: -1/(x+x1)+k2
        return np.piecewise(x, conds, [f1, f2, f3])
    def piecewise_reg(self, gen_img=False):
        """
        Fits three regression models using the training data and returns the optimized parameters. 

        :param gen_img: (bool) Whether to generate a scatter plot of the original data and the piecewise curve. 
            Default is False.

        :return: (list) A list of optimized parameters for the piecewise regression model.
        """
        p0 = [1] * 6

        # popt stores the best fit parameters
        # pcov stores the covariance matrix
        try:
            popt, pcov = curve_fit(self.f, self.X['slope'], self.y, p0=p0, method='dogbox')
        except Exception as e:
            print(e)

        if gen_img:
            fig, ax = plt.subplots(figsize=(15,6))
            plt.scatter(self.X.values, self.y.values, color='orange', label='Data')
            plt.scatter(self.X.values, self.f(self.X.values, *popt), color='red', label='Piecewise Fit')
            plt.xlabel('Slope of lines')
            plt.ylabel('Angular z twist message')
            plt.title('Slope vs Angular z')
            plt.legend()
            plt.savefig('piecewise.png')
            plt.show()

        return popt

    def make_prediction(self, slope):
        """
        Given a slope value, this function returns a prediction based on the model_name set for the instance.

        Args:
            slope (float): A float value representing the slope.

        Returns:
            prediction (float): A float value representing the prediction. The prediction is based on the model_name set for the instance.

        Raises:
            None
        """
        if self.model_name == 0:
            given = {
                'slope': [slope]
            }
            y = pd.DataFrame(given, index=[0])
            prediction = self.model.predict(y).round(2)
            return prediction
        elif self.model_name == 1:
            prediction = self.sigmoid(slope, *self.model)
            return prediction
        elif self.model_name == 2:
            prediction = self.arctan(slope, *self.model)
            return prediction

if __name__ == '__main__':
    for i in range(4):
        model = RegModel('2023-06-26-11-01-18.bag', model_name=i, gen_img=True)
