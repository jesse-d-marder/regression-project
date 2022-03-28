import os
import pandas as pd
from sklearn.model_selection import train_test_split
import sklearn.preprocessing
import numpy as np
from env import get_db_url

def wrangle_zillow():
    """ Acquires the Zillow housing data from the SQL database or a cached CSV file. Renames columns and outputs data as a Pandas DataFrame"""
    # Acquire data from CSV if exists
    if os.path.exists('zillow_2017.csv'):
        print("Using cached data")
        df = pd.read_csv('zillow_2017.csv')
    # Acquire data from database if CSV does not exist
    else:
        print("Acquiring data from server")
        query = """
            SELECT bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt, fips, garagecarcnt, yearbuilt, buildingqualitytypeid, regionidzip, poolcnt, lotsizesquarefeet
            FROM predictions_2017
            JOIN properties_2017
            USING (parcelid)
            JOIN propertylandusetype
            USING (propertylandusetypeid)
            WHERE transactiondate IS NOT NULL
            AND propertylandusedesc = "Single Family Residential"
            OR propertylandusedesc = "Inferred Single Family Residential"
            AND transactiondate BETWEEN '2017-01-01' and '2017-12-31';
            """
        df = pd.read_sql(query, get_db_url('zillow'))
        df.to_csv('zillow_2017.csv', index=False)
    
    # Prepare the data for exploration and modeling
    # Rename columns as needed
    df=df.rename(columns = {'bedroomcnt':'bedroom', 
                            'bathroomcnt':'bathroom', 
                            'calculatedfinishedsquarefeet':'square_feet',
                            'taxvaluedollarcnt':'tax_value',
                            'garagecarcnt':'garage',
                           'buildingqualitytypeid':'condition',
                           'regionidzip':'zip',
                           'poolcnt':'pool',
                           'lotsizesquarefeet':'lot_size'})
    df.garage = df.garage.fillna(0)
    df.pool = df.pool.fillna(0)
    
    # Make column for home age
    df["age"] = 2017 - df.yearbuilt
    
    # Make a column for living space. Average bathroom is 40 sq ft, average bedroom is 200 sq ft
    df["living_space"] = df.square_feet - (df.bathroom*40 + df.bedroom*200)
    
    # Number of rooms in the house (bath and bed)
    df["room_count"] = df.bathroom + df.bedroom
    
    # Make a column for the county based on FIPS
    df["county"] = np.select([df.fips == 6037, df.fips==6059, df.fips == 6111],["Los Angeles County", "Orange County", "Ventura County"])
    
    
    def bedroom_mapper(num):
        """ Map the number of bedrooms to a categorical label """
        if num <= 2:
            return "2_or_less"
        elif num <=3:
            return "3"
        elif num > 3:
            return "more_than_3"
    def bathroom_mapper(num):
        """ Map the number of bathrooms to a categorical label """
        if num == 2:
            return "2"
        elif num < 2:
            return "less_than_2"
        elif num > 2:
            return "more_than_2"
    
    df["bedroom_cat"] = df.bedroom.apply(lambda row: bedroom_mapper(row))
    df["bathroom_cat"] = df.bathroom.apply(lambda row: bathroom_mapper(row))


    df["has_garage"] = np.where(df.garage>0, True, False)

    df["bed_to_bath"] = (df.bedroom/df.bathroom).replace(np.inf, 0).fillna(0)

    # Condition data only available for fips 6037.0. Will be dropping nans going forward but do not want to drop every row from other fips.
    df.condition = df.condition.fillna(0)


#     # Drops the rows with Null values, representing a very small percentage of the dataset (<0.6%)
#     df = df.dropna()
    
#     # Convert year built column to integer from float
#     df.year_built = df.year_built.astype('int64')
#     df.bedroom_cnt = df.bedroom_cnt.astype('int64')


    return df

def split_data(df, train_size_vs_train_test = 0.8, train_size_vs_train_val = 0.7, random_state = 123):
    """Splits the inputted dataframe into 3 datasets for train, validate and test (in that order).
    Can specific as arguments the percentage of the train/val set vs test (default 0.8) and the percentage of the train size vs train/val (default 0.7). Default values results in following:
    Train: 0.56
    Validate: 0.24
    Test: 0.2"""
    train_val, test = train_test_split(df, train_size=train_size_vs_train_test, random_state=123)
    train, validate = train_test_split(train_val, train_size=train_size_vs_train_val, random_state=123)
    
    train_size = train_size_vs_train_test*train_size_vs_train_val
    test_size = 1 - train_size_vs_train_test
    validate_size = 1-test_size-train_size
    
    print(f"Data split as follows: Train {train_size:.2%}, Validate {validate_size:.2%}, Test {test_size:.2%}")
    
    return train, validate, test

def scale_data(train, validate, test, features_to_scale):
    """Scales data using MinMax Scaler. 
    Accepts train, validate, and test datasets as inputs as well as a list of the features to scale. 
    Returns dataframe with scaled values added on as columns"""
    
    # Fit the scaler to train data only
    scaler = sklearn.preprocessing.MinMaxScaler()
    scaler.fit(train[features_to_scale])
    
    # Generate a list of the new column names with _scaled added on
    scaled_columns = [col+"_scaled" for col in features_to_scale]
    
    # Transform the separate datasets using the scaler learned from train
    scaled_train = scaler.transform(train[features_to_scale])
    scaled_validate = scaler.transform(validate[features_to_scale])
    scaled_test = scaler.transform(test[features_to_scale])
    
    # Concatenate the scaled data to the original unscaled data
    train_scaled = pd.concat([train, pd.DataFrame(scaled_train,index=train.index, columns = scaled_columns)],axis=1).drop(columns = features_to_scale)
    validate_scaled = pd.concat([validate, pd.DataFrame(scaled_validate,index=validate.index, columns = scaled_columns)],axis=1).drop(columns = features_to_scale)
    test_scaled = pd.concat([test, pd.DataFrame(scaled_test,index=test.index, columns = scaled_columns)],axis=1).drop(columns = features_to_scale)
    
    
    return train_scaled, validate_scaled, test_scaled

def remove_outliers(df, k, col_list):
    ''' Removes outliers based on multiple of IQR. Accepts as arguments the dataframe, the k value for number of IQR to use as threshold, and the list of columns. Outputs a dataframe without the outliers.
    '''
    # Create a column that will label our rows as containing an outlier value or not
    num_obs = df.shape[0]
    df['outlier'] = False
    for col in col_list:

        q1, q3 = df[col].quantile([.25, .75])  # get quartiles
        
        iqr = q3 - q1   # calculate interquartile range
        
        upper_bound = q3 + k * iqr   # get upper bound
        lower_bound = q1 - k * iqr   # get lower bound

        # update the outlier label any time that the value is outside of boundaries
        df['outlier'] = np.where(((df[col] < lower_bound) | (df[col] > upper_bound)) & (df.outlier == False), True, df.outlier)
    
    df = df[df.outlier == False]
    df.drop(columns=['outlier'], inplace=True)
    print(f"Number of observations removed: {num_obs - df.shape[0]}")
        
    return df