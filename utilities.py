#importing necessary libraries used in the functions
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

#Helper functions
def convert_amount(df, cols):
    """
    Returns a boolean array with True if points are outliers and False 
    otherwise.

    Parameters:
    -----------
        df : date frame containing amounts as strings
        cols : list of cols having amounts as strings.
    
    Returns:
    --------
        df : Original dataframe with amount columns converted into numeric value.

    """
    #excluding dollar sign from string
    for col in cols:
        df[col] = df[col].str[1:]
        #replacing commas with empty string
        df[col] = pd.to_numeric(df[col].str.replace(",",""))

    return df

def clean_columns(column):
    column = column.replace('{', '').replace('}','').replace('[','').replace(']','').replace('_','').replace('"','').replace('/','_').replace(' ', '').replace('\'', '')
    
    return column

#function to get col as seperate columns with boolean values for each data record
#same function can be generalized to any columns having comma (or any delimitter) separated values
def seperate_col(col, df, delimitter):
    """
    Returns a boolean array with True if points are outliers and False 
    otherwise.

    Parameters:
    -----------
        col : column which contains entitites that needs to be separated as new columns
        df : dataframe
        delimitter : character which separates entities.
    
    Returns:
    --------
        df : Original dataframe with entities separated

    """
    
    #collect all distinct col in a list
    col_list = []
    
    #add unique col in the list created
    for i in range (0,df.shape[0]):
        columns = df[col][i].split(delimitter)
        for column in columns:
            column = clean_columns(column)
            if(column not in col_list):
                col_list.append(column)
            else:
                continue

    
    df['No_of_'+col] = 0 
    for column in col_list:
        df[column]=False
    
    #create a new column for each amenity with boolean values            
    for i in range (0,df.shape[0]):
        columns = df[col][i].split(delimitter)
        df['No_of_'+col][i] = len(columns)
        for j in range (0,len(columns)):
            columns[j] = clean_columns(columns[j])
        for column in col_list:
            if(column not in columns):
                df[column][i] = False
            else:
                df[column][i] = True
                
    #drop the original column as it will not be required anymore            
    df.drop(col,axis=1,inplace=True)
    return df, col_list


def fill_nan_data(df,categ_transform):
    
    """
     Returns dataframe with nan data filled with mean and categorical data transformed.

    Parameters:
    -----------
        df : date frame
        categ_transform : Boolean value if true will transform categorical values into dummies
    
    Returns:
    --------
        df : data frame with NAN values removed and categorical columns transformed.

    """
    
    #get numeric and categorical columns 
    num_cols = df.select_dtypes(include=['int64','float64']).copy().columns
    cat_cols = df.select_dtypes(include='object').copy().columns
    
    #filling NAN values with mean for numerical columns
    for col in num_cols:
        try:
            df[col].fillna(df[col].mean(), inplace=True)
        except:
            continue
            
    if(categ_transform):
        for col in cat_cols:
            try:
                df = pd.concat([df.drop(col,axis=1),pd.get_dummies(df[col],prefix=col, prefix_sep='_', 
                                                                   dummy_na=False,drop_first=True)],axis=1)
            except:
                continue
    return df

# final cleaning of data and modelling.
def clean_data_model(df, col_pred, test_size=.3, rand_state=42):
    '''
    INPUT:
    df - a dataframe holding all the variables of interest
    response_col - a string holding the name of the column
    test_size - a float between [0,1] about what proportion of data should be in the test dataset
    rand_state - an int that is provided as the random state for splitting the data into training and test

    OUTPUT:
    X - cleaned X matrix (dummy and mean imputation)
    y - cleaned response (just dropped na)
    test_score - float - r2 score on the test data
    train_score - float - r2 score on the test data
    lm_model - model object from sklearn
    X_train, X_test, y_train, y_test - output from sklearn train test split used for optimal model
    '''
    
    df = df.dropna(subset=[col_pred],axis=0)
    y = df[col_pred]
    
    df = df.drop(col_pred,axis=1)
    X=df
    
    #Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=rand_state)

    lm_model = LinearRegression(normalize=True) # Instantiate
    lm_model.fit(X_train, y_train) #Fit

    #Predict using your model
    y_test_preds = lm_model.predict(X_test)
    y_train_preds = lm_model.predict(X_train)
    
    #Score using your model
    test_score = r2_score(y_test, y_test_preds)
    train_score = r2_score(y_train, y_train_preds)

    return X, y, test_score, train_score, lm_model, X_train, X_test, y_train, y_test




