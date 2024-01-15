import pandas as pd
import numpy as np

import os
import random

from sklearn.preprocessing import LabelEncoder, StandardScaler

import torch
from torchvision import transforms

from torch.utils.data import Dataset
from PIL import Image

from scipy.stats import zscore


"""
Car price prediction project library
----------------------------------------------------------

"""





class preprocess_Ad_Table_Trim():
  """
  Class to preprocess the data of Ad_Table and Trim 
  ------------------------------------------------------------------------------------
  """

  def __init__(self, Ad_table, Trim):

    self.df = Ad_table.rename(columns={' Genmodel' : 'Genmodel',
                                       ' Genmodel_ID' : 'Genmodel_ID',
                                        'Bodytype' : 'Body_type',
                                          'Gearbox' : 'Gear_box'})
    self.df = self.clean_obj_and_NaN()
    #self.df = self.encoding()
    self.Trim = Trim
    self.merged_df = self.Merge_Ad_Trim()

  def clean_obj_and_NaN(self):

    """
    Function to fulfill Na's and transform object into numerical data
    --------

    Inputs
    --------
    Self

    Outputs
    --------
    pd.core.frame.DataFrame


    ------------------------------------------------------------------------------------
    """

    df_ = self.df

    # Step 1 : remove strings in object and convert in int or float

    if df_["Engin_size"].dtype == 'object':

        # Engine_size
        df_['Engin_size'] = pd.to_numeric(df_['Engin_size'].str.replace('L', ''), errors='coerce')

        # Speed
        df_["Top_speed"] = pd.to_numeric(df_['Top_speed'].str.replace(' mph', ''), errors='coerce')

        # Average_mpg
        df_['Average_mpg'] = pd.to_numeric(df_['Average_mpg'].str.replace(' mpg', ''), errors='coerce')

        # Runned_Miles (convert into numeric only)
        df_['Runned_Miles'] = pd.to_numeric(df_['Runned_Miles'], errors='coerce')


    # Step 2 : Fulfill Na's cells

    df_['Engine_power'] = df_.groupby(['Genmodel_ID'])['Engine_power'].transform(lambda x: x.fillna(x.mean()))

    df_['Top_speed'] = df_.groupby(['Genmodel_ID'])['Top_speed'].transform(lambda x: x.fillna(x.mean()))

    df_['Average_mpg'] = df_.groupby(['Genmodel_ID'])['Average_mpg'].transform(lambda x: x.fillna(x.mean()))

    df_['Engin_size'] = df_.groupby(['Genmodel_ID'])['Engin_size'].transform(lambda x: x.fillna(x.mean()))


    return df_


  def encoding(self, df_, one_hot_encoder):

    """
    Function to encode the categorical data
    --------

    Inputs
    --------
    Self

    Outputs
    --------
    pd.core.frame.DataFrame


    ------------------------------------------------------------------------------------
    """

    df_ = df_.copy()


    if one_hot_encoder==True : 
        # Step 3 : Encode categorical data

        # Define categorical variables to encode
        categorical_var = ['Body_type', 
                          'Gear_box', 
                          'Fuel_type', 
                          'Seat_num', 
                          'Door_num']

        for column in range(len(categorical_var)):
          label_encoder = LabelEncoder()
          df_[categorical_var[column]] = label_encoder.fit_transform(df_[categorical_var[column]])



        # One Hot encoding
        var_one_hot_encoding = [
                        'Body_type',
                        'Gear_box',
                        'Fuel_type'
                        ]
        
        df_ = pd.get_dummies(df_, columns=var_one_hot_encoding)

    else :
      categorical_var = ['Body_type', 
                          'Gear_box', 
                          'Fuel_type', 
                          'Seat_num', 
                          'Door_num',
                          'Body_type',
                        'Gear_box',
                        'Fuel_type']

      for column in range(len(categorical_var)):
        label_encoder = LabelEncoder()
        df_[categorical_var[column]] = label_encoder.fit_transform(df_[categorical_var[column]])



    return df_


  def Merge_Ad_Trim(self):

    """
    Function to merge both Ad_table and Trim tables.

    Comments 
    --------
    Key : Genmodel_ID
    Method : Left
    --------

    Inputs
    --------
    Self

    Outputs
    --------
    pd.core.frame.DataFrame


    ------------------------------------------------------------------------------------
    """


    Trim = self.Trim

    df_ = self.df

    # Step 4 : Merge data

    Trim = Trim.groupby(['Genmodel_ID'])['Gas_emission'].mean()

    df_ = df_.merge(Trim, on='Genmodel_ID', how='left')

    return df_
  

  def select_columns(self,
                     df_,
                     columns_to_drop=[],
                     columns_to_keep=[]):
    """
    Select columns to keep for the output. 

    Comments
    --------
    Run before droping Na's

    Inputs
    --------
    df_ : pd.core.DataFrame
    columns_to_drop : list
    columns_to_keep : list


    Outputs
    --------
    pd.core.frame.DataFrame


    ------------------------------------------------------------------------------------
    """
 


    if len(columns_to_drop) != 0:

      df_ = df_.drop(columns=columns_to_drop)

    elif len(columns_to_keep) != 0:

      df_ = df_[columns_to_keep]

    else:

      pass

    return df_
  

  def get_full_data(self,
                    drop_nan = False,
                    columns_to_drop = [],
                    columns_to_keep = [], 
                    one_hot_encoder=False):
    """
    Function to prepare the output as a dataframe for descriptive statistics.

    Inputs
    --------
    drop_nan : Boolean
    columns_to_drop : list
    columns_to_keep : list
    one_hot_encoder : Boolean


    Outputs
    --------
    pd.core.frame.DataFrame


    ------------------------------------------------------------------------------------
    """
    df_ = self.encoding(self.merged_df, one_hot_encoder)

    df_ = self.select_columns(df_, 
                              columns_to_drop, 
                              columns_to_keep)

    if drop_nan == True :
      df_ = df_.dropna()

    else :

      pass

    return df_


  def final_set(self,
                columns_to_drop = ['Maker',
                                  'Genmodel',
                                  'Genmodel_ID',
                                  'Adv_ID',
                                  'Adv_year',
                                  'Adv_month',
                                  'Reg_year',
                                  'Annual_Tax',
                                  'Color'],
                columns_to_keep = [],
                standardization = True,
                remove_outliers = True,
                one_hot_encoder = True):

    """
    Function to prepare the data to be trained.

    Comments
    --------
    A list of columns_to_drop is defined by default

    Inputs
    --------
    columns_to_drop : list
    columns_to_keep : list
    standardization : Boolean
    one_hot_encoder : Boolean



    Outputs
    --------
    pd.core.frame.DataFrame (x2)


    ------------------------------------------------------------------------------------
    """
 

    df_ = self.encoding(self.merged_df, one_hot_encoder)

    df_ = self.select_columns(df_, 
                              columns_to_drop)

    df_ = df_.dropna()

    if remove_outliers == True:

      percentile_99 = df_["Price"].quantile(0.99)
      df_ = df_[df_["Price"] <= percentile_99]

      z_scores = zscore(df_[['Runned_Miles', 'Engin_size', 'Price', 'Engine_power', 'Wheelbase',
                                        'Height', 'Width', 'Length', 'Average_mpg', 'Top_speed']])

      df_ = df_[(abs(z_scores) < 4).all(axis=1)]

    else : 
      pass

    y = df_['Price']

    X = df_.drop(columns = 'Price')

    if standardization==True :

      scaler = StandardScaler()
      columns_to_standardize = ["Runned_Miles", 
                                "Engin_size", 
                                "Engine_power", 
                                "Wheelbase", 
                                "Height",	
                                "Width",	
                                "Length",	
                                "Average_mpg",	
                                "Top_speed", 
                                "Gas_emission",
                                "Seat_num",	
                                "Door_num"]
      
      X[columns_to_standardize] = scaler.fit_transform(X[columns_to_standardize])

      return y, X
    
    else :
      return y, X
  





















class DVMdataset(Dataset):
    
    """
    Class to preprocess and load images
    --------

    ------------------------------------------------------------------------------------
    """
    def __init__(self, data, img_dir, transforms=True):

        df = data
        self.img_dir = img_dir

        self.brand = df['Maker'].values

        self.model_year = df['Year'].values

        self.model = df['Genmodel'].values

        self.Price = df['Entry_price'].values

        self.transform = transforms

        self.liste_images = self.unpack_all_images()

    def unpack_all_images(self):
        """
        function to unpack all images
        --------

        ------------------------------------------------------------------------------------
        """
      
        self.liste_images =[]
        self.price_images = []

        for index in range(self.brand.shape[0]):

            dir_path = os.listdir(os.path.join(self.img_dir, 
                                               self.img_dir, 
                                               self.brand[index], 
                                               str(self.model_year[index])))

            brand_name = self.brand[index]

            model_name = self.model[index]

            dico = [brand_name, model_name]
            separator = '$$'

            joindre = separator.join(dico)

            for i in range(len(dir_path)):
              joindre2 = ''
              for y in range(len(joindre)):
                joindre2 = joindre2 + dir_path[i][y]
              if joindre2 == joindre:
                lien = os.path.join(self.img_dir, self.img_dir, self.brand[index], str(self.model_year[index]), dir_path[i])
                img = Image.open(lien)
                
                self.liste_images.append(img)
                self.price_images.append(self.Price[index])
        return self.liste_images


    def __getitem__(self, index):
      """
      Function to get an image

      Comments
      --------

      Inputs
      --------
      index : int

      Outputs
      --------
      image and price

      ------------------------------------------------------------------------------------
      """
 

      if self.transform is not None : 
        image = self.transform(self.liste_images[index])
        return image, self.price_images[index]

      else :
        resize_transorm = transforms.Compose([transforms.Resize(size=(224,224))])
        image = resize_transorm(self.liste_images[index])
        return image, self.price_images[index]

    def __len__(self):
        return len(self.liste_images)