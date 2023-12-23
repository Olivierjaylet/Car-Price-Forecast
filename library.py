import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder


"""
Car price prediction project library
----------------------------------------------------------

Updates history : 
----------------------------------------------------------
1.0 : Creation of the preprocess_Ad_Table_Trim class

1.1 : [preprocess_Ad_Table_Trim] Normalize the name of some columns in the initialization

1.2 : Creation of the preprocess_Images class. Contains only a function to merge Price_table and Images_table so far

1.3 : [preprocess_Ad_Table_Trim] Creation of a function to remove the outliers (to be done)

"""





class preprocess_Ad_Table_Trim():
  """
  Class to preprocess the data of Ad_Table and Trim  
  """

  def __init__(self, Ad_table, Trim):

    self.df = Ad_table.rename(columns={' Genmodel' : 'Genmodel',
                                       ' Genmodel_ID' : 'Genmodel_ID',
                                        'Bodytype' : 'Body_type',
                                          'Gearbox' : 'Gear_box'})
    self.df = self.clean_obj_and_NaN()
    self.df = self.encoding()
    self.Trim = Trim
    self.merged_df = self.Merge_Ad_Trim()

  def clean_obj_and_NaN(self):

    """
    Function to fulfill Na's and transform object into numerical data
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


  def encoding(self):

    #df_ = self.clean_obj_and_NaN()

    df_ = self.df

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

    return df_


  def Merge_Ad_Trim(self):

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
    Select columns to keep for the output. Run before droping Na's
    """


    if len(columns_to_drop) != 0:

      df_ = df_.drop(columns=columns_to_drop)

    elif len(columns_to_keep) != 0:

      df_ = df_[columns_to_keep]

    else:

      pass

    return df_
  
  """
  to be done 
  
  def Remove_outliers(self):
    
    return df_
  """

  def get_full_data(self,
                    drop_nan = False,
                    columns_to_drop = [],
                    columns_to_keep = []):
    """
    Function to prepare the output as a dataframe for descriptive statistics.
    """

    df_ = self.select_columns(self.merged_df, 
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
                columns_to_keep = []):

    """
    Function to prepare the data to be trained.
    Can chose the Columns to drop
    """

    df_ = self.select_columns(self.merged_df, 
                              columns_to_drop)

    df_ = df_.dropna()

    y = df_['Price']

    X = df_.drop(columns = 'Price')

    return y, X
  









class preprocess_Images():
  """
  Class to preprocess the car Images
  Each image should have a price label
  """



  def __init__(self, 
               Price_table, 
               Images_table):
    
    self.Price_table = Price_table
    
    self.Images_table = Images_table[Images_table["Predicted_viewpoint"]==0]

  def merge_Price_table_to_Images_table(self):
      """
      Function to give a price to each image URL
      """

      Price_table = self.Price_table

      Images_table = self.Images_table

      # We want to create a link between both tables, so we can merge them
      # The idea is to create a common key for both tables.
      # the price will be give to an image if all those variables are the same :
          # Maker -> Genmodel -> Year -> Genmodel_ID
      
      # So the Key (or Image_link) is Maker$$Genmodel$$Year$$Genmodel

      # data to be concatenated to create the Image_link
      concatenate = ['Maker', 
                     'Genmodel', 
                     'Year', 
                     'Genmodel_ID']
      
      # Create the Image_link on Price_table
      Price_table['Image_link'] = Price_table[concatenate].apply(lambda x: '$$'.join(map(str, x)), axis=1)

      # Compute the average Entry_price for all rows with the same Image_link
      Price_table = pd.DataFrame(Price_table.groupby(["Image_link", 
                                                      'Genmodel', 
                                                      'Year', 
                                                      'Maker'])
                                                      ['Entry_price'].mean())
      
      Price_table.reset_index(inplace=True)


      # Split the Image_name of Images_table (it contains too many informations. For example, the color is unknown in Price_table)
      split_Image_name = Images_table['Image_name'].str.rsplit('$',n=8,  expand=True)
      
      # Structure the Image_link with necessary data
      Images_table['Image_link'] = split_Image_name.iloc[:,0] + '$$' + split_Image_name.iloc[:,4]


      # Merge both tables with Image_link as a key
      self.merged_images_prices = pd.merge(Images_table, 
                                           Price_table, 
                                           on='Image_link', 
                                           how='inner')
      
      # Drop useless informations for the next steps
      self.merged_images_prices.drop(columns=["Image_link", 
                                              'Quality_check', 
                                              'Image_ID', 
                                              'Predicted_viewpoint'], inplace=True)
      
      return self.merged_images_prices
  

