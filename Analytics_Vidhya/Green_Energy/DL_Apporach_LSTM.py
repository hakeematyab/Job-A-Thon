#!pip install darts

#!pip uninstall matplotlib
#!pip install matplotlib==3.1.3

#Miscellaneous imports
from datetime import datetime
from pathlib import Path
import os
import pickle
import matplotlib.pyplot as plt

# Preprocessing imports
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from darts import TimeSeries


# Training imports
'''%load_ext autoreload
%autoreload 2
%matplotlib inline'''
import numpy as np
import pandas as pd
from tqdm import tqdm_notebook as tqdm

import matplotlib.pyplot as plt

from darts import TimeSeries, concatenate
from darts.dataprocessing.transformers import Scaler
from darts.models import TFTModel
from darts.metrics import mape
from darts.utils.statistics import check_seasonality, plot_acf
from darts.datasets import AirPassengersDataset, IceCreamHeaterDataset
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.utils.likelihood_models import QuantileRegression

import warnings

warnings.filterwarnings("ignore")
import logging

logging.disable(logging.CRITICAL)

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
                training of the Temporal Transformer Fusion model using darts,
                saving of the trained model,
                loading of the trained model,
                prediction of the passed data,
                saving of the predictions.


        Written By: 0QYGSL
        Version: 1.0
        Revisions: None

        """
    
    def __init__(self,file_path):

        self.file_path = file_path #Training file path
        self.log_path = os.path.join(Path(self.file_path).parent.absolute(),'log.txt')#Log file path

        self.logger = Logger() #Logger object
        file_object = open(self.log_path, 'a+') #Opening log file
        log_message = 'New PredictinGreenEnergy Object Initialized'
        self.logger.log(file_object,log_message) #Logging the log message
        
    
    def preprocess(self):
        """
                Method Name: preprocess
                Description: This method performs preprocessing operations on the passed data.
                Output: A Timeseries format data: train, validation, and entire dataset
                On Failure: Raise Exception

                Written By: 0QYGSL
                Version: 1.0
                Revisions: None

        """
        file_object = open(self.log_path, 'a+')
        log_message = 'Preprocessing method called'
        self.logger.log(file_object,log_message)
            
        try:    
            df = pd.read_csv(self.file_path) #Reading the .csv file

            log_message = 'File read sucessfully!'
            self.logger.log(file_object,log_message)
        except Exception as e:
            log_message = 'Unable to read file: Error' +str(e)
            self.logger.log(file_object,log_message)
            return str(e)
        
        try:
            df['datetime']=pd.to_datetime(df['datetime']) #Conversiton of datetime column to datetime format
            df.drop('row_id',inplace=True,axis=1)       #Removing row_id column
            log_message = 'Datatype converstion and column removal done'
            self.logger.log(file_object,log_message)           
        except Exception as e:
            log_message = 'Unable to crate feature columns: Error' +str(e)
            self.logger.log(file_object,log_message)            
            return str(e)   
        
        try:
            imputer=KNNImputer(n_neighbors=3, weights='uniform',missing_values=np.nan) #KNN imputer object
            new_array=imputer.fit_transform(np.array(df['energy']).reshape(-1,1)) # Impute the missing values
            # Convert the nd-array returned in the step above to a Dataframe
            new_data=pd.DataFrame(data=new_array) 
            df['energy'] = new_data[0]

            log_message = 'Null values imputed successfully!'
            self.logger.log(file_object,log_message)
        except Exception as e:
            log_message = 'Unable to impute null values: Error' +str(e)
            self.logger.log(file_object,log_message)
            return str(e)

        try:
            series = TimeSeries.from_dataframe(df, "datetime", "energy") #Converting to timeseries format
            train, val = series[:int(len(series)*.65)], series[int(len(series)*.65):] #Splitting train data into train and validation

            log_message = 'Converted ot Timeseries format and split into train and validation'
            self.logger.log(file_object,log_message)
        except Exception as e:
            log_message = 'Unable to convert into Timeseries format: Error' +str(e)
            self.logger.log(file_object,log_message)
            return str(e)

        try:
            series = series.astype(np.float32) #Type converstion
            #Obtaining the timestamp for 65% value
            year,month,day,hour = df.iloc[int(len(df)*.65)]['datetime'].year,df.iloc[int(len(df)*.65)]['datetime'].month,df.iloc[int(len(df)*.65)]['datetime'].day,df.iloc[int(len(df)*.65)]['datetime'].hour
        
            # Create training and validation sets:
            self.training_cutoff = pd.Timestamp(year=year, month=month, day=day, hour=hour) #Obtaining cutoff timestamp
            train, val = series.split_after(self.training_cutoff) #Splitting train and val data

            log_message = 'Train and validation split done'
            self.logger.log(file_object,log_message)
        except Exception as e:
            log_message = 'Unable to perform Train and validation split: Error' +str(e)
            self.logger.log(file_object,log_message)
            return str(e)


        
  
        log_message = 'Preprocessing completed!'
        self.logger.log(file_object,log_message)
                
        
        
        return train,val,series
     
    def train(self,train,val,series,all=False):
        """
                Method Name: train
                Description: This method creates a Temporal Transformer Fusion model using darts
                Output: Temporal Transformer Fusion model(and model score if data is split)

                Written By: 0QYGSL
                Version: 1.0
                Revisions: None

        """       
        file_object = open(self.log_path, 'a+')
        log_message = 'Train method called'
        self.logger.log(file_object,log_message)

        #Defining a few constants
        self.num_samples = 200

        figsize = (9, 6)

        #Quantiles
        lowest_q, low_q, high_q, highest_q = 0.01, 0.1, 0.9, 0.99
        label_q_outer = f"{int(lowest_q * 100)}-{int(highest_q * 100)}th percentiles"
        label_q_inner = f"{int(low_q * 100)}-{int(high_q * 100)}th percentiles"

        # Normalize the time series (note: we avoid fitting the transformer on the validation set)            
        try:
            if all:
              #Scaling operation if all the data is to be trained
              self.transformer = Scaler() 
              series_transformed = self.transformer.fit_transform(series)
              log_message = 'Scaling transformation completed'
              self.logger.log(file_object,log_message)
            else:
              #Scaling operation if the data is to be split into train and val
              self.transformer = Scaler()
              train_transformed = self.transformer.fit_transform(train)
              val_transformed = self.transformer.transform(val)
              series_transformed = self.transformer.transform(series)

              log_message = 'Scaling transformation completed'
              self.logger.log(file_object,log_message)
        except Exception as e:
            log_message = 'Unable to perform Scaling transformation: Error' +str(e)
            self.logger.log(file_object,log_message)
            return str(e)

        try:
            # Create year, month and integer index covariate series
            covariates = datetime_attribute_timeseries(series, attribute="year", one_hot=False)
            covariates = covariates.stack(
                datetime_attribute_timeseries(series, attribute="month", one_hot=False)
            )
            covariates = covariates.stack(
                TimeSeries.from_times_and_values(
                    times=series.time_index,
                    values=np.arange(len(series)),
                    columns=["linear_increase"],
                )
            )
            covariates = covariates.astype(np.float32)
            log_message = 'Covariates created'
            self.logger.log(file_object,log_message)
            
        except Exception as e:
            log_message = 'Unable to create covaraites: Error' +str(e)
            self.logger.log(file_object,log_message)
            return str(e)        
        
        try:
            if all:
              # transform covariates if the data isn't split
              scaler_covs = Scaler()
              scaler_covs.fit(covariates)
              self.covariates_transformed = scaler_covs.transform(covariates)
              
              log_message = 'Covariates transformed'
              self.logger.log(file_object,log_message)
            else:
              # transform covariates if the data is split
              scaler_covs = Scaler()
              cov_train, cov_val = covariates.split_after(self.training_cutoff)
              scaler_covs.fit(cov_train)
              self.covariates_transformed = scaler_covs.transform(covariates)

              log_message = 'Covariates transformed'
              self.logger.log(file_object,log_message)

        except Exception as e:
            log_message = 'Unable to transform covariates: Error' +str(e)
            self.logger.log(file_object,log_message)
            return str(e)          
        
      
        #Defining quanties to be considered during training
        quantiles = [
            0.01,
            0.05,
            0.1,
            0.15,
            0.2,
            0.25,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.75,
            0.8,
            0.85,
            0.9,
            0.95,
            0.99,
        ]

        input_chunk_length = 24
        forecast_horizon = 12

        try:#Defining model architecture
            my_model = TFTModel(
                input_chunk_length=input_chunk_length,
                output_chunk_length=forecast_horizon,
                hidden_size=64,
                lstm_layers=1,
                num_attention_heads=4,
                dropout=0.1,
                batch_size=16,
                n_epochs=1,
                add_relative_index=False,
                add_encoders=None,
                likelihood=QuantileRegression(
                    quantiles=quantiles
                ),  # QuantileRegression is set per default
                # loss_fn=MSELoss(),
                random_state=42,
            )


            log_message = 'Model architecture created'
            self.logger.log(file_object,log_message)        
        except Exception as e:
            log_message = 'Unable to create Model architecture: Error' +str(e)
            self.logger.log(file_object,log_message)
            return str(e)  

        try:
          #Training the model
            if all:
              log_message = 'Training Started'
              self.logger.log(file_object,log_message)   

              my_model.fit(series_transformed, future_covariates=self.covariates_transformed, verbose=True)  

              log_message = 'Training Completed'
              self.logger.log(file_object,log_message)   

            else:
              log_message = 'Training Started'
              self.logger.log(file_object,log_message)   

              my_model.fit(train_transformed, future_covariates=self.covariates_transformed, verbose=True)

              log_message = 'Training Completed'
              self.logger.log(file_object,log_message)   

        except Exception as e:
            log_message = 'Unable to train the model: Error' +str(e)
            self.logger.log(file_object,log_message)
            return str(e)  
  
        try:
            if not all:
              #Evaluating the model
              future_preds = 2000 #number of predictions
              log_message = 'Evaluation started'
              self.logger.log(file_object,log_message)   

              score = self.eval_model(my_model,future_preds,series_transformed, val_transformed)

              log_message = 'Evaluation completed'
              self.logger.log(file_object,log_message)   
              return my_model, score
        except Exception as e:

            log_message = 'Unable to evaluate the model: Error' +str(e)
            self.logger.log(file_object,log_message)
            return str(e)

        return my_model
    
    def eval_model(self,model, n, actual_series, val_series):
        """
                Method Name: eval_model
                Description: This method evluated the trained model
                Output: A floating point score of the model

                Written By: 0QYGSL
                Version: 1.0
                Revisions: None

        """               
        file_object = open(self.log_path, 'a+')
        log_message = 'Evaluate model method called'
        self.logger.log(file_object,log_message)
        
        try:
            log_message = 'Evluation Prediction started'
            self.logger.log(file_object,log_message)     

            pred_series = model.predict(n=n, num_samples=self.num_samples)#Performing prediction for n future datapoints

            log_message = 'Evaluatoin Prediction completed'
            self.logger.log(file_object,log_message)   

        except Exception as e:
            log_message = 'Unable to perform prediction: Error' +str(e)
            self.logger.log(file_object,log_message)
            return str(e)         
        return mape(val_series,pred_series)
        # plot actual series
        #plt.figure(figsize=figsize)
        #actual_series[: pred_series.end_time()].plot(label="actual")

        # plot prediction with quantile ranges
        #pred_series.plot(
            #low_quantile=lowest_q, high_quantile=highest_q, label=label_q_outer
        #)
        #pred_series.plot(low_quantile=low_q, high_quantile=high_q, label=label_q_inner)

        #plt.title("MAPE: {:.2f}%".format(mape(val_series, pred_series)))
        #plt.legend()

    def predict_future(self,model,file_path):
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
        self.logger.log(file_object,log_message)

        try:           
            df = pd.read_csv(file_path) #Reading the test file
            num_predictions = len(df) #Obtaining the number of the predictions

            log_message = 'File read successfully'
            self.logger.log(file_object,log_message)
        except Exception as e:
            log_message = 'Unable to read file: Error' +str(e)
            self.logger.log(file_object,log_message)
            return str(e) 

        num_predictions=1 #For testing purpose overwriting the value


        try:
            log_message = 'Prediction started...'
            self.logger.log(file_object,log_message)

            fut_cov = self.covariates_transformed.concatenate(other=self.covariates_transformed.tail(size=num_predictions),ignore_time_axis=True)
            pred_series = model.predict(n=num_predictions,future_covariates=fut_cov, num_samples=self.num_samples) #Performing predictions

            log_message = 'Prediction completed!'
            self.logger.log(file_object,log_message)
        except Exception as e:
            log_message = 'Unable to predict: Error' +str(e)
            self.logger.log(file_object,log_message)
            return str(e)
        
        try:
            quantile = 0.99

            preds = self.transformer.inverse_transform(pred_series.quantile(quantile)).pd_series()  #Obatining the predictions for 0.1 quantile
            preds = [pred for pred in preds]
            #Creating a new dataframe with row_id and energy columns
            result = pd.DataFrame()
            result['row_id'] = df['row_id']
            result['energy'] = preds

            log_message = 'Results dataframe created!'
            self.logger.log(file_object,log_message)
        except Exception as e:
            log_message = 'Unable to create results dataframe: Error' +str(e)
            self.logger.log(file_object,log_message)
            return str(e)
        
        return result
                                                    
    def save_file(self,df):
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
        self.logger.log(file_object,log_message)

        try:
            df.to_csv('future_energy.csv',index=False) #Saving the result predictions

            log_message = 'File saved successfully!'
            self.logger.log(file_object,log_message)
        except Exception as e:

            log_message = 'Unable to save file: Error' +str(e)
            self.logger.log(file_object,log_message)
            return str(e)
        
        return 'File Saved'

    def save_model(self,model,dir):
        """
                Method Name: save_model
                Description: This method saves the passed model in the specified direcory
                Output: A model file and accompanying files

                Written By: 0QYGSL
                Version: 1.0
                Revisions: None

        """
        file_object = open(self.log_path, 'a+')
        log_message = 'save_model called'
        self.logger.log(file_object,log_message)    
        
        try:
            model.save(dir) #Saving the model

            log_message = 'Model saved successfully!'
            self.logger.log(file_object,log_message)
        except Exception as e:
            log_message = 'Unable to save model: Error' +str(e)
            self.logger.log(file_object,log_message)
            return str(e) 
        
        return 'Model Saved'        
    
    def load_model(self,model_path):
        """
                Method Name: load_model
                Description: This method loads the model present at the specified path
                Output: Trained model

                Written By: 0QYGSL
                Version: 1.0
                Revisions: None

        """
        file_object
        file_object = open(self.log_path, 'a+')      
        log_message = 'load_model called'
        self.logger.log(file_object,log_message)
        
        try:
            loaded_model = TFTModel.load_model(model_path) #Loading the pretrained model

            log_message = 'Model loaded successfully!'
            self.logger.log(file_object,log_message)
        except Exception as e:
            log_message = 'Unable to load model: Error' +str(e)
            self.logger.log(file_object,log_message)
            return str(e)
        
        return loaded_model

if __name__=='__main__':
    gn_obj = PredictGreenEnergy('/content/drive/MyDrive/Jobathon/train_IxoE5JN.csv') #Train file location
    train,val,series = gn_obj.preprocess() #Call the preprocess method for preprocessing
    model,score = gn_obj.train(train,val,series) #Call the train method to trian the model for the preprocessed data
    result = gn_obj.predict_future(model,'/content/drive/MyDrive/Jobathon/test_WudNWDM.csv') #Predicting the output for the passed data and the model
    output = gn_obj.save_file(result)

    
    
