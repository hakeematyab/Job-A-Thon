#Miscellaneous imports
from datetime import datetime
from pathlib import Path
import os
import pickle

# Preprocessing imports
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Model Creation imports
#from sklearn.linear_model import LinearRegression
#from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
#from xgboost import XGBRegressor

# Model Parameters Selection imports
#from kneed import KneeLocator
from sklearn.model_selection import GridSearchCV
#from sklearn.metrics  import r2_score
#from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV, ElasticNet

#Logger class to log the operations
class Logger:
    """
        This class shall  be used to log the various operations taking place and the resulting errors

        Written By: 0QYGSL
        Version: 1.0
        Revisions: None

        """
    def __init__(self):
        pass

    #Obtain date and time in the given format and log the message in the opened file
    def log(self, file_object, log_message):
        self.now = datetime.now()
        self.date = self.now.date()
        self.current_time = self.now.strftime("%H:%M:%S")
        file_object.write(
            str(self.date) + "/" + str(self.current_time) + "\t\t" + log_message +"\n")


class PredictGreenEnergy:
    """
        This class shall  be used to perform:
                preprocessinng of the passed data,
                training of the RandomForest Regressor model,
                saving of the trained model,
                loading of the trained model,
                prediction of the passed data,
                saving of the predictions.


        Written By: 0QYGSL
        Version: 1.0
        Revisions: None

        """
    def __init__(self, file_path):

        self.file_path = file_path #Training file path
        self.log_path = os.path.join(Path(self.file_path).parent.absolute(), 'log.txt') #Log file path

        self.logger = Logger() #Logger object
        file_object = open(self.log_path, 'a+') #Opening the log file and creating the object to be passed
        log_message = 'New PredictinGreenEnergy Object Initialized' #Message to be logged
        self.logger.log(file_object, log_message) #Logging the message by calling log method of Logger class

    def preprocess(self, file_path=None):
        """
                Method Name: preprocess
                Description: This method performs preprocessing operations on the passed data
                Output: A pandas DataFrame with features columns added and categorical columns removed.
                On Failure: Raise Exception

                Written By: 0QYGSL
                Version: 1.0
                Revisions: None

        """

        file_object = open(self.log_path, 'a+')
        log_message = 'Preprocessing method called'
        self.logger.log(file_object, log_message)

        #Setting the defualt train file path if the path isn't passed when the method is called
        if file_path == None:
            file_path = self.file_path
            log_message = 'File path not provided -> Default path set'
            self.logger.log(file_object, log_message)

        try:
            df = pd.read_csv(file_path) # Reading the data
            log_message = 'File read sucessfully!'
            self.logger.log(file_object, log_message)
        except Exception as e:
            log_message = 'Unable to read file: Error' + str(e)
            self.logger.log(file_object, log_message)
            return str(e)

        # Creating feature columns: day, month, year, year
        try:
            df['datetime'] = pd.to_datetime(df['datetime']) #Conversion to datetime datatype
            df['day'] = df['datetime'].dt.day
            df['month'] = df['datetime'].dt.month
            df['year'] = df['datetime'].dt.year
            df['hours'] = df['datetime'].dt.hour

            #Removing non-contributing data
            df.drop('datetime', inplace=True, axis=1)
            df.drop('row_id', inplace=True, axis=1)

            log_message = 'Feature columns created successfully!'
            self.logger.log(file_object, log_message)
        except Exception as e:
            log_message = 'Unable to crate feature columns: Error' + str(e)
            self.logger.log(file_object, log_message)
            return str(e)

        try:
            # Imputing null values using KNN imputer
            imputer = KNNImputer(n_neighbors=3, weights='uniform', missing_values=np.nan) #KNNInputer object creation
            new_array = imputer.fit_transform(df)  # Impute the missing values
            # Convert the nd-array returned in the step above to a Dataframe
            new_data = pd.DataFrame(data=new_array, columns=df.columns)

            log_message = 'Null values imputed successfully!'
            self.logger.log(file_object, log_message)
        except Exception as e:
            log_message = 'Unable to impute null values: Error' + str(e)
            self.logger.log(file_object, log_message)
            return str(e)

        log_message = 'Preprocessing completed!'
        self.logger.log(file_object, log_message)

        return new_data

    def train(self, preprocessed_df):
        """
                Method Name: train
                Description: This method creates a RandomForestRegression model
                Output: RandomForestRegression Model

                Written By: 0QYGSL
                Version: 1.0
                Revisions: None

        """
        file_object = open(self.log_path, 'a+')
        log_message = 'Train method called'
        self.logger.log(file_object, log_message)

        try:
            X = preprocessed_df.drop('energy', axis=1) #Extracing X = Features
            y = preprocessed_df['energy'] #Extracting Y = Labels

            log_message = 'Feature and Label created successfully!'
            self.logger.log(file_object, log_message)
        except Exception as e:
            log_message = 'Unable to create feature and label: Error' + str(e)
            self.logger.log(file_object, log_message)
            return str(e)

        try:
            #Performing Scaling operation
            scalar = StandardScaler()

            x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=1 / 3, random_state=36, )
            #Converting the scaled values into dataframe
            x_train = pd.DataFrame(scalar.fit_transform(x_train), columns=x_train.columns, index=x_train.index)
            x_test = pd.DataFrame(scalar.fit_transform(x_test), columns=x_test.columns, index=x_test.index)

            log_message = 'Splitting and Scaling of data completed'
            self.logger.log(file_object, log_message)
        except Exception as e:
            log_message = 'Unable to split and/or scale data: Error' + str(e)
            self.logger.log(file_object, log_message)
            return str(e)
        #Obtaining the best hyperparameters for the model
        n_estimators,max_features,min_samples_split,bootstrap = self.obtain_best_params(x_train,y_train,x_test,y_test)

        x_train = X
        y_train = y

        try:
            log_message = 'Training of the model started...'
            self.logger.log(file_object, log_message)
            # Creating a new model with the best parameters
            rf_reg = RandomForestRegressor(n_estimators=n_estimators, max_features=max_features,min_samples_split=min_samples_split, bootstrap=bootstrap)
            # Training the mew models
            rf_reg.fit(x_train, y_train)

            log_message = 'Training completed successfully'
            self.logger.log(file_object, log_message)
        except Exception as e:
            log_message = 'Unable to train: Error' + str(e)
            self.logger.log(file_object, log_message)
            return str(e)

        return rf_reg

    def obtain_best_params(self, x_train, y_train, x_test, y_test):
        """
                Method Name: obtain_best_params
                Description: This method obtains the best hyperperameters for the randomforest model
                Output: A tuple of the following values: learning_rate, max_depth, n_estimators

                Written By: 0QYGSL
                Version: 1.0
                Revisions: None

        """
        file_object = open(self.log_path, 'a+')
        log_message = 'obtain_best_params called'
        self.logger.log(file_object, log_message)

        try:
            RandomForestReg = RandomForestRegressor()
            param_grid_Random_forest_Tree = {
                                            "n_estimators": [10,30,60,90,120],
                                            "max_features": ["auto", "sqrt", "log2"],
                                            "min_samples_split": [2,4,8],
                                            "bootstrap": [True, False]
                                                                  }

            grid = GridSearchCV(RandomForestReg, param_grid_Random_forest_Tree, verbose=3, cv=5)#Using gridsearchcv for hyperparameter tuning
            # finding the best parameters
            grid.fit(x_train, y_train)

            n_estimators = grid.best_params_['n_estimators']
            max_features = grid.best_params_['max_features']
            min_samples_split = grid.best_params_['min_samples_split']
            bootstrap = grid.best_params_['bootstrap']

            log_message = 'Best parameters obtained successfully!'
            self.logger.log(file_object, log_message)
        except Exception as e:
            log_message = 'Unable to obtain best parameters: Error' + str(e)
            self.logger.log(file_object, log_message)
            return str(e)

        return (n_estimators,max_features,min_samples_split,bootstrap)

    def predict_future(self, model, file_path):
        """
                Method Name: predict_future
                Description: This method is used to obtain the prediction values for the passed data
                Output: A pandas dataframe caontining the following columns: row_id, energy

                Written By: 0QYGSL
                Version: 1.0
                Revisions: None

        """

        file_object = open(self.log_path, 'a+')
        log_message = 'predict_future called'
        self.logger.log(file_object, log_message)

        #Preprocessing the sent data
        prepdata = self.preprocess(file_path)

        try:
            #Scaling the preprocessed data
            scalar = StandardScaler()

            X = pd.DataFrame(scalar.fit_transform(prepdata), columns=prepdata.columns, index=prepdata.index)

            log_message = 'Features extracted'
            self.logger.log(file_object, log_message)
        except Exception as e:
            log_message = 'Unable to extract features: Error' + str(e)
            self.logger.log(file_object, log_message)
            return str(e)

        try:
            log_message = 'Prediction started...'
            self.logger.log(file_object, log_message)
            prediction = model.predict(X) #Performing prediction

            log_message = 'Prediction completed!'
            self.logger.log(file_object, log_message)
        except Exception as e:
            log_message = 'Unable to predict: Error' + str(e)
            self.logger.log(file_object, log_message)
            return str(e)

        try:
            #Creating a dataframe in the specified format containing the row_id and predictions for the data
            result = pd.DataFrame()
            result['row_id'] = pd.read_csv(file_path)['row_id']
            result['energy'] = prediction

            log_message = 'Results dataframe created!'
            self.logger.log(file_object, log_message)
        except Exception as e:
            log_message = 'Unable to create results dataframe: Error' + str(e)
            self.logger.log(file_object, log_message)
            return str(e)

        return result

    def save_file(self, df):
        """
                Method Name: save_file
                Description: This method saves the prediction output in .csv format
                Output: A .csv file with file name 'future_energy.csv'

                Written By: 0QYGSL
                Version: 1.0
                Revisions: None

        """
        file_object = open(self.log_path, 'a+')
        log_message = 'save_file called'
        self.logger.log(file_object, log_message)
        try:
            df.to_csv('future_energy_lr.csv', index=False) #conversion to csv format and saving

            log_message = 'File saved successfully!'
            self.logger.log(file_object, log_message)
        except Exception as e:
            log_message = 'Unable to save file: Error' + str(e)
            self.logger.log(file_object, log_message)
            return str(e)

        return 'File Saved'

    def save_model(self, model):
        """
                Method Name: save_model
                Description: This method saves the passed model in .sav format
                Output: A .sav model file of the passed model with the name 'saved_model.sav'

                Written By: 0QYGSL
                Version: 1.0
                Revisions: None

        """
        file_object = open(self.log_path, 'a+')
        log_message = 'save_model called'
        self.logger.log(file_object, log_message)

        try:
            with open('saved_model_rf.sav', 'wb') as f:
                pickle.dump(model, f) #Saving the model in the .sav format using pickle

            log_message = 'Model saved successfully!'
            self.logger.log(file_object, log_message)
        except Exception as e:
            log_message = 'Unable to save model: Error' + str(e)
            self.logger.log(file_object, log_message)
            return str(e)

        return 'Model Saved'

    def load_model(self, model_path):
        """
                Method Name: load_model
                Description: This method loads the model present at the specified path
                Output: Trained model

                Written By: 0QYGSL
                Version: 1.0
                Revisions: None

        """
        file_object = open(self.log_path, 'a+')
        log_message = 'load_model called'
        self.logger.log(file_object, log_message)

        try:
            with open(model_path, 'rb') as f:
                loaded_model = pickle.load(f) #Loading the model using pickle
            log_message = 'Model loaded successfully!'
            self.logger.log(file_object, log_message)
        except Exception as e:
            log_message = 'Unable to load model: Error' + str(e)
            self.logger.log(file_object, log_message)
            return str(e)

        return loaded_model

if __name__ == '__main__':
    gn_obj = PredictGreenEnergy('/content/drive/MyDrive/Jobathon/train_IxoE5JN.csv') #Train file location
    preprocessed_df = gn_obj.preprocess() #Call the preprocess method for preprocessing
    model = gn_obj.train(preprocessed_df) #Call the train method to trian the model for the preprocessed data
    result = gn_obj.predict_future(model, '/content/drive/MyDrive/Jobathon/test_WudNWDM.csv') #Predicting the output for the passed data and the model
    output = gn_obj.save_file(result) #Saving the .csv output file
