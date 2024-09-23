## How this code works:

# Subtract background radiation from each sample
# Correct for SQP efficiencies
# Divide by runtime to get spectra in CPM
# Match single spectra with similar quench (within uncertainty range)
# Add them together, keeping track of their individual activities, and append these new artificial entries
# Savgol filter
# Test/train split
# Apply PCA
# Train model

from parse import *
import sys
import os
from itertools import combinations
import random
from datetime import datetime

# Numpy, pandas, matplotlib
import pandas as pd
#pd.options.display.float_format = '{:.4f}'.format
pd.set_option('display.max_colwidth', None)
import numpy as np
#np.set_printoptions(threshold=sys.maxsize)
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# Scipy
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
from scipy.signal import savgol_filter,find_peaks

# Sklearn
import sklearn
from sklearn import tree
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer,StandardScaler,MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import multilabel_confusion_matrix,classification_report,hamming_loss,precision_recall_curve,auc,mean_squared_error
from sklearn.pipeline import Pipeline

# Tensorflow
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Softmax
from scikeras.wrappers import KerasRegressor
from keras.optimizers import Adam
from keras.utils import plot_model

# Tqdm and tabulate
from tqdm import tqdm
from tabulate import tabulate

# Ignore pandas performance warning
from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
########################################################################################################################

class SpectraPreprocessingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, isotopes=['3H', '14C', '36CL']):
        self.isotopes = isotopes

    def fit(self, X, y=None):
        return self  # No fitting needed for this transformer

    def transform(self, X):
        """ Sets negative values in all spectra to 0, corrects for SQP efficiency curve, divides by counting time to get spectra in CPM,
        separates out source activity column such that there are activity columns for each radioisotope. """
        # Copy the dataframe to avoid modifying the original one
        X_transformed = X.copy()

        # Set negative values in the spectra to 0
        spectra_columns = [*range(1,1025)]
        X_transformed[spectra_columns] = X_transformed[spectra_columns].clip(lower=0)

        # Correct for SQP efficiency (assuming correction by dividing each count by SQP efficiency)
        # This is where the AUC-based correction can be implemented but for now assume it's a simple efficiency factor.
        if 'Counting efficiency [%]' in X_transformed.columns:
            for col in spectra_columns:
                X_transformed[col] /= (X_transformed['Counting efficiency [%]']*10**-2)

        # Convert counts to counts-per-minute using the counting time
        if 'CTIME' in X_transformed.columns:
            X_transformed[spectra_columns] = X_transformed[spectra_columns].div(X_transformed['CTIME'], axis=0)

        # Replace single activity column with separate columns for each isotope
        activity_per_isotope = pd.DataFrame(0, index=X_transformed.index, columns=[f'{iso} Activity' for iso in self.isotopes])
        activity_per_isotope = activity_per_isotope.astype('object')
        for idx, row in X_transformed.iterrows():
            isotope_label = row['ISOTOPE']  # Assuming the isotope type is stored in 'isotope' column
            activity_per_isotope.loc[idx, f'{isotope_label} Activity'] = row['Activity [Bq]']

        # Drop the old 'activity' column
        X_transformed = X_transformed.drop(columns=['Activity [Bq]'])
        X_transformed = pd.concat([X_transformed, activity_per_isotope], axis=1)

        return X_transformed

class SpectraCombinationTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, isotopes=['3H', '14C', '36CL']):
        self.isotopes = isotopes

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """ Finds groups that have SQP values within each other's uncertainty ranges and adds spectra of distinct radioisotope combinations together.
        Returns a pd.DataFrame containing all original data plus new synthesised combinations """
        # Copy the dataframe to avoid modifying the original one
        X_transformed = X.copy()

        # Select spectra with similar SQPs
        similar_sqp_groups = self.group_spectra_by_sqp(X_transformed)
        #print("Similar SQP Groups: \n",similar_sqp_groups)

        # Combine spectra within each group
        combined_spectra = self.combine_spectra(similar_sqp_groups, X_transformed)

        # Return concatenated dataframe with combined spectra
        return pd.concat([X_transformed, combined_spectra], axis=0,ignore_index=True)

    def group_spectra_by_sqp(self, X):
        """Group spectra that have SQP values within each other's uncertainty ranges."""
        groups = []
        for idx, row in X.iterrows():
            # Select all spectra that fall within the uncertainty range of this sample's SQP
            sqp, sqp_uncertainty = row['SQP'], row['SQP%']
            matching_group = X[
                (X['SQP'] >= sqp - 0.01*sqp_uncertainty*sqp) & 
                (X['SQP'] <= sqp + 0.01*sqp_uncertainty*sqp)
            ]
            groups.append(matching_group.index.tolist())    # And append to groups

        # Remove single-length 'groups' in groups
        groups=[group for group in groups if len(group)>1]
        return groups

    def combine_spectra(self, groups, X):
        """Create combined spectra for each distinct radioisotope combination for each group, keeping track of isotope activities.

        Args:
            groups (list): list containing lists of rows indicies that have an SQP within each other's uncertainty range.
            X (df): df containing all LSC calibration data

        Returns:
            pd.DataFrame: df containing synthesised spectra for all combinations. df['FILENAME'] for these entries is simply the indicies of the combined spectra from the original dataset
        """
        combined_spectra = []
        spectra_columns = [*range(1,1025)]
        print("Num groups: ",len(groups))

        # groups=[group for group in groups if len(group)>34]
        # print("Length of groups trimmed: ",len(groups))

        # rand_groups = random.sample(groups, 80)
        # print("Length of rand_groups: ",len(rand_groups))

        for group in tqdm(groups,desc='Combining spectra'):

            # Get the unique isotopes in the group
            isotopes_in_group = X.loc[group, 'ISOTOPE'].unique()
            distinct_isotopes_count = len(isotopes_in_group)

            # If there are less than 2 distinct isotopes, skip this group
            if distinct_isotopes_count < 2:
                continue

            for combo_size in range(2, len(group) + 1):
                
                # If the combination size exceeds the number of distinct isotopes, break early
                if combo_size > distinct_isotopes_count:
                    break
                
                for combo in combinations(group, combo_size):
                    # Check if all spectra in the combo are from different isotopes
                    isotopes_in_combo = X.loc[list(combo), 'ISOTOPE'].unique()

                    if len(isotopes_in_combo) == combo_size: # Only combine if each isotope in the combo is unique
                        # Combine spectra
                        combined_row = X.loc[list(combo)].copy()
                        #print(combined_row)
                        combined_spectrum = combined_row[spectra_columns].sum(axis=0)
                        mean_sqp = combined_row['SQP'].mean()

                        # Combine activity for each isotope
                        combined_activity = combined_row[[f'{iso} Activity' for iso in self.isotopes]].sum(axis=0)

                        # Create new row with the combined spectrum, SQP, and activities
                        combined_entry = pd.Series(0, index=X.columns)
                        combined_entry = combined_entry.astype('object')
                        combined_entry[spectra_columns] = combined_spectrum
                        combined_entry['SQP'] = mean_sqp
                        combined_entry['FILENAME'] = str(combo)
                        isotopes_list = combined_row['ISOTOPE'].values
                        isotopes = ''
                        for i, iso in enumerate(isotopes_list):
                            if i==0:
                                isotopes += str(iso)
                                continue
                            isotopes += ', '+str(iso) 
                        combined_entry['ISOTOPE'] = isotopes
                        combined_entry[[f'{iso} Activity' for iso in self.isotopes]] = combined_activity

                        combined_spectra.append(combined_entry)

        return pd.DataFrame(combined_spectra)

class SpectraSavGolTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, savgol_window=41, savgol_polyorder=3, cols = [*range(1,1025)]):
        self.savgol_window = savgol_window
        self.savgol_polyorder = savgol_polyorder
        self.cols = cols
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """Applies Savitzky-Golay filter to X[cols].

        Args:
            X (pd.DataFrame): df containing all LSC calibration data
        """
        X_transformed = X.copy()
        
        # Apply Savitzky-Golay filter
        X_transformed[self.cols] = savgol_filter(X_transformed[self.cols], self.savgol_window, self.savgol_polyorder, axis=1)
        
        return(X_transformed)

########################################################################################################################

def data_generation(df,cols):   # Creates more data by adding random noose - but needs updating if going to use
    list_index = df.index.to_list()
    spectra = df.loc[:, cols].values # x is a 2D list of the spectra

    # Create empty array
    generated_arr = np.zeros((len(spectra),len(spectra[0])))
    for i,spectrum in enumerate(spectra):
        # Create noise array
        noise = np.random.normal(0,2,len(cols))

        # Add spectrum and noise to create new spectrum
        generated_arr[i]=abs(np.add(np.array(spectrum),noise))

    generated_list = generated_arr.tolist()
    # Convert generated data to dataframe
    generated_df = pd.DataFrame(generated_list,columns=cols,index=[x+100000 for x in list_index])
    generated_df[['File','Element']] = df[['File','Element']].values   # Copy across element/ file info
    # Concatrate the dataframes
    return(pd.concat([df, generated_df], axis=0))

def create_model(n_components,lr):
    """ Returns a compiled Sequential model of shape n_components. Optimizer learning rate lr. """
    model = Sequential([
        Dense(16,activation='relu',input_shape=(n_components,)),
        Dense(8, activation='relu'),
        Dense(3, activation='linear')  # Linear output for 3 nodes
    ])
    optimizer = Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss='mse',metrics=['mae',tf.keras.metrics.R2Score()])   # Keep track of MSE loss, MAE, and R^2
    return model

def scree_plot(x,y):
    """ PCA scree plot - x: ['PC1','PC2', ... , 'PCn'] and y is a list of explained variance per principal component. """
    cum_x = np.cumsum(y*100)
    plt.bar(x,y*100,color='black',zorder=1)
    plt.plot(x,cum_x,color='red',zorder=10,label='Cumulative',marker='.')
    plt.xlabel("Principal Components")
    plt.ylabel("Explained Variation (%)")
    plt.yticks(np.linspace(0,100,11))
    plt.legend()
    plt.grid(axis='y')
    plt.title("Scree Plot")
    plt.show()

def simplify_isotope_label(isotope,isotopes=['3H', '14C', '36CL']): 
    for iso in isotopes:
        if iso in isotope:
            return(iso)
        
def time_to_minutes(time_str):
    """Converts time string from MM:SS.ss to minutes (float).

    Args:
        time_str (str): time in format MM:SS.ss

    Returns:
        str: time in minutes (float)
    """
    # Split string into minutes and seconds
    minutes, seconds = time_str.split(':')
    
    # Convert minutes and seconds to float
    total_minutes = float(minutes) + float(seconds) / 60
    
    return total_minutes

def plot_spectrum(df,cols,activity_cols,i):
    """ Plot spectrum of ith entry in the dataframe. """
    plt.style.use('bmh')

    fig, ax = plt.subplots()
    fig.set_figheight(10)
    fig.set_figwidth(15)
    maximum = np.max(df[cols].iloc[i])
    minimum = np.min(df[cols].iloc[i])

    ticks = np.linspace(minimum,round(maximum, -1),11)

    ax.plot(cols,df[cols].iloc[i])
    ax.set_yticks(ticks)
    ax.set_title(str(df['FILENAME'].iloc[i]) +': '+ str(df['ISOTOPE'].iloc[i]))
    ax.set_xlabel("Channel")
    ax.set_ylabel("Counts per minute")
    ax.margins(x=0,y=0)

    print('\n3H: '+str(df[activity_cols].iloc[i].values[0])+'Bq\n14C: '+str(df[activity_cols].iloc[i].values[1])+'Bq\n36CL: '+str(df[activity_cols].iloc[i].values[2])+'Bq')
    plt.show()

def parse_and_process(directory):
    """Uses functions from parse.py to return dataframe containing valid LSC calibration data found in the directory"""
    df = data_from_files(directory) # get df

    cols=[*range(1,1025)]
    print("Real data cutoff: ",len(df),' at index: ',df.tail(1).index[0])


    #isotopes=['3H', '14C', '36CL']
    # Change isotope col to remove appended numbers
    df['ISOTOPE'] = df['ISOTOPE'].map(simplify_isotope_label)
    print(df['ISOTOPE'].iloc[[222]])

    # Floatify CTIME col
    df['CTIME'] = df['CTIME'].map(time_to_minutes)
    print(df['CTIME'])

    # Floatify SQP and SQP%
    df['SQP'] = df['SQP'].astype(float)
    df['SQP%'] = df['SQP%'].astype(float)
    
    # Custom transformers here
    df = SpectraPreprocessingTransformer().transform(df)
    df = SpectraCombinationTransformer().transform(df)

    activity_cols = ['3H Activity','14C Activity','36CL Activity']
    plot_spectrum(df,cols,activity_cols,6063)

    df = SpectraSavGolTransformer().transform(df)
    print("Full Df: \n",df)

    plot_spectrum(df,cols,activity_cols,6063)
    return(df)

#####################################################

def main():
    # Get dataframe
    directory = "LSC Spectra for AI proj including calibration certs/"  # Add / on end. Forward slashes not backward.
    
    # Check if spreadsheet exists
    if not os.path.isfile('transformed_data.xlsx'):
        df = parse_and_process(directory)
        # Save to excel here if neccessary
        print("Saving to .xlsx ...")
        df.to_excel('transformed_data.xlsx')
    else:
        print("transformed_data.xlsx found. Opening...")
        df = pd.read_excel('transformed_data.xlsx')
        print(df)

    ### Setting up X,y train/test and applying PCA ###

    # Cutting up dataframe to separate X and y 
    cols = [*range(1,1025)]
    cols_plus_sqp = cols.copy()
    cols_plus_sqp.append('SQP')     # Needs to have SQP as a feature as well
    X = df[cols_plus_sqp]
    print('\nX:',X)
    activity_cols = ['3H Activity','14C Activity','36CL Activity']
    y = df[activity_cols]
    print('\ny:',y)

    # Need to convert X,Y to float32 and X cols to str otherwise sklearn PCA and keras throw a fit
    X.columns = X.columns.astype(str)
    cols_str = [str(col) for col in cols]
    X = X.astype('float32')
    y = y.astype('float32')

    # Test, train split
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit PCA to X_train
    num_components = 10
    pca_cols = ['PC'+str(i+1) for i in range(num_components)]    # Get column labels
    PCA_object = PCA(n_components=num_components)
    X_train[pca_cols] = PCA_object.fit_transform(X_train[cols_str])
    X_train.drop(columns=cols_str,inplace=True)     # Drop 1024 channels

    # Plotting PCA scree
    #scree_plot(pca_cols,PCA_object.explained_variance_ratio_)
    
    # Print X_train and Y_train
    print('\nX_train:',X_train)
    print("\n",type(Y_train),Y_train.dtypes,'\n')
    
    # Use fitted PCA on X_test
    X_test[pca_cols] = PCA_object.transform(X_test[cols_str])
    X_test.drop(columns=cols_str,inplace=True)

    ### NN Training and Results ###
    hyperparameters = {'epochs':50,'batch_size':32,'verbose':1,'validation_split':0.1,'learning_rate':0.0003}

    # Neural network regressor
    model = create_model(num_components+1,hyperparameters['learning_rate']) # Arg is num of X columns (+1 for SQP) 

    # Train the model on the preprocessed data
    result = model.fit(X_train, Y_train, 
                       epochs=hyperparameters['epochs'], 
                       batch_size=hyperparameters['batch_size'], 
                       verbose=hyperparameters['verbose'],
                       validation_split=hyperparameters['validation_split'])    # Needs optimising
    
    # plot training curve for rmse
    print("Available history: ", result.history.keys())
    plt.plot(result.history['loss'])
    plt.plot(result.history['val_loss'])
    plt.title('mse')
    #plt.ylabel('mse')
    plt.xlabel('epoch')
    plt.legend(['training','validation'],loc='upper right')
    plt.show()

    # plot training curve for R^2 (beware of scale, starts very low negative)
    plt.plot(result.history['r2_score'])
    plt.plot(result.history['val_r2_score'])
    plt.title('model R^2')
    plt.ylabel('R^2')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation']) #, loc='upper right')
    plt.show()
           

    # Make predictions on the test set
    predictions = model.predict(X_test)

    # Mean squared error
    mse = mean_squared_error(Y_test, predictions)
    print("\nMean Squared Error:", mse)

    # Predictions comparison dataframe
    pred_labels = ['pred_3H', 'pred_14C','pred_36CL']
    pred_df = pd.DataFrame(data=predictions,columns=pred_labels,index=Y_test.index)
    pred_vs_act_df = pd.concat([Y_test,pred_df],axis=1)
    print(pred_vs_act_df)

    print('\nNumber of samples in training set: ',len(X_train),'\nNumber of samples in testing set: ',len(X_test),'\n')
    X_test.sort_index(inplace=True)
    Y_test.sort_index(inplace=True)

    user_input = None 
    while user_input !='y' or user_input != 'n':
        user_input = input("Save result to .xlsx? Y/N\n")
        user_input=user_input.lower()
        if user_input=='y':
            now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            filename = 'result'+str(now)+'.xlsx'
            pred_vs_act_df.to_excel('test_results/'+filename)
            break
        elif user_input=='n':
            break
        else:
            print("Invalid input.")

   
    # Visualize the model architecture
    #plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

if __name__=='__main__':
    main()