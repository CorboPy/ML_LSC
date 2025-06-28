# Alex Corbett, University of Bristol, 2025

from parse import *
import os
import pickle

# Numpy, pandas, matplotlib
import pandas as pd
#pd.options.display.float_format = '{:.4f}'.format
pd.set_option('display.max_colwidth', None)
import numpy as np
#np.set_printoptions(threshold=sys.maxsize)
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
plt.style.use('bmh')

# Scipy
from scipy.signal import savgol_filter,find_peaks
from scipy.interpolate import interp1d
from scipy.optimize import nnls
from scipy.linalg import qr

# Sklearn
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
#from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
#from sklearn.metrics import multilabel_confusion_matrix,classification_report,hamming_loss,precision_recall_curve,auc,mean_squared_error
#from sklearn.pipeline import Pipeline, make_pipeline

# Tqdm
from tqdm import tqdm

# Ignore pandas performance warning
from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)



########################### PREPROCESSING TRANSFORMERS ################################

class SpectraPreprocessingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, isotopes=['3H', '14C', '36CL','55FE','63NI','129I']):
        self.isotopes = isotopes

    def fit(self, X, y=None):
        return self  # No fitting needed for this transformer

    def transform(self, X):
        """ Sets negative values in all spectra to 0, corrects for SQP efficiency curve (now removed!), divides by counting time to get spectra in CPM,
        separates out source activity column such that there are activity columns for each radioisotope. """
        # Copy the dataframe to avoid modifying the original one
        X_transformed = X.copy()

        # Set negative values in the spectra to 0
        spectra_columns = [*range(1,1025)]
        X_transformed[spectra_columns] = X_transformed[spectra_columns].clip(lower=0)

        # Correct for SQP efficiency (assuming correction by dividing each count by SQP efficiency)
        # This is where the AUC-based correction can be implemented but for now assume it's a simple efficiency factor.
        # if 'Counting efficiency [%]' in X_transformed.columns:
        #     for col in spectra_columns:
        #         X_transformed[col] /= (X_transformed['Counting efficiency [%]']*10**-2)

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
 
class SpectrumAugmenter(BaseEstimator, TransformerMixin):
    """ Used to balance dataset by adding random noise to existing data """
    def __init__(self, category_col='Category', feature_cols=None, target_count=180, noise_level=0.05, N_neighbours=10 ,random_state=None):
        self.category_col = category_col
        self.feature_cols = feature_cols  # List of channel column names: e.g., [1, 2, ..., 1024]
        self.target_count = target_count
        self.noise_level = noise_level
        self.random_state = random_state
        self.N_neighbours = N_neighbours

        # For each index in the array, calculate the neighbors
        self.neighbors_dict = {}
        for i in range(len(feature_cols)):
            self.neighbors_dict[i] = self.get_neighbors(feature_cols, i)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        np.random.seed(self.random_state)
        df = X.copy()
        augmented_rows = []

        for category, group in tqdm(df.groupby(self.category_col),desc='Balancing dataset'):
            current_count = len(group)
            if current_count < self.target_count:
                n_to_generate = self.target_count - current_count
                for _ in range(n_to_generate):
                    row = group.sample(n=1, replace=True).copy()    # random sample
                    spect = row[self.feature_cols].to_numpy().squeeze() # squeeze from (1,1024) to (1024,)
                    #print(np.shape(spect))

                    # Iterate over each channel in the spectrum 
                    sum = np.sum(spect)
 
                    noise =  np.random.normal(loc=0, scale=(sum *self.noise_level)/len(self.feature_cols), size=len(self.feature_cols))
                    noisy_values  = spect + noise       # add the noise
                    row[self.feature_cols] = np.clip(noisy_values, a_min=0, a_max=None)  # clip to ensure no negative counts
                    # Optional: modify filename to indicate synthetic data
                    row['FILENAME'] = row['FILENAME'].values[0] + "_aug"
                    augmented_rows.append(row)

        if augmented_rows:
            augmented_df = pd.concat(augmented_rows, ignore_index=True)
            df = pd.concat([df, augmented_df], ignore_index=True)

        return df

    def get_neighbors(self, arr, index):
        # Define the number of neighbors on each side (left and right)
        left_neighbors = self.N_neighbours
        right_neighbors = self.N_neighbours

        # Find the start and end of the neighbor range
        start_idx = max(index - left_neighbors, 0)  # Ensure no negative index
        end_idx = min(index + right_neighbors, len(arr) - 1)  # Ensure no out-of-bound index
        
        # Generate the range of indices
        neighbors = list(range(start_idx, end_idx + 1))
    
        return neighbors

class SpectraSavGolTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, savgol_window=41, savgol_polyorder=3, cols = [*range(1,1025)]):
        self.savgol_window = savgol_window
        self.savgol_polyorder = savgol_polyorder
        self.cols = cols

    def fit(self, X, y=None):
        return self

    def transform(self, X, isnumpy=False):
        """Applies Savitzky-Golay filter to X[cols].

        Args:
            X (pd.DataFrame or NDArray): df or numpy array containing LSC calibration data
        """
        
        if isnumpy and isinstance(X, np.ndarray):
            #print(np.shape(X))
            X_transformed = savgol_filter(X, self.savgol_window, self.savgol_polyorder)
            return(X_transformed)
        
        
        X_transformed = X.copy()
        
        # Apply Savitzky-Golay filter
        X_transformed[self.cols] = savgol_filter(X_transformed[self.cols], self.savgol_window, self.savgol_polyorder, axis=1)
        return(X_transformed)


########################### MODEL CLASSES ################################

class IsotopeModel():
    def __init__(self, isotope, X_train, Y_train, fit_info, delta=1e-6):
        self.name = isotope
        self.delta = delta   # For zero replacement

        self.X_train = self.ensure_numpy(X_train)
        self.Y_train = self.ensure_numpy(Y_train)

        # Efficiency fit info ['Ax^2','Bx','C','R^2','FitErr']
        self.A = fit_info['Ax^2']
        self.B = fit_info['Bx']
        self.C = fit_info['C']
        self.R2 = fit_info['R^2']
        self.FitErr = fit_info['FitErr']

    def __str__(self):
        return f"IsotopeModel({self.name})"

    def ensure_numpy(self,arr):
        if not isinstance(arr, np.ndarray):
            try:
                arr = arr.to_numpy()    
            except Exception as err:
                print(err)
                err_msg = f'{self.name} found unrecognised type when converting to ndarray: {type(arr)}'
                raise TypeError(err_msg)
        return(arr)

    def fit(self):
        # Zero replacement
        self.Y_train = self.zero_replacement(self.Y_train)

        # Build ILR basis
        self.basis = self.ilr_basis(self.Y_train.shape[1])
        # Transform to ILR space
        Y_train_ilr = self.ilr_transform(self.Y_train, self.basis)  # shape: (n_samples, 1023)
        self.model = MultiOutputRegressor(RandomForestRegressor(n_estimators=50,max_depth=5))  # random forest for each channel
        self.model.fit(self.X_train.reshape(-1, 1), Y_train_ilr)

    def predict(self,X_new):    # X_new is list of SQP(s) of unseen sample(s)

        # Check X_new lies within trained SQP range
        self.check_sqp(np.array(X_new))

        X_new = self.ensure_numpy(X_new)
        Y_ilr_pred = self.model.predict(X_new.reshape(-1, 1))  # X_new: new inputs
        Y_pred = self.ilr_inverse(Y_ilr_pred, self.basis)  # back to composition space
        return(Y_pred)

    def zero_replacement(self,arr):  
        arr = np.where(arr == 0, self.delta, arr)
        arr /= arr.sum(axis=1, keepdims=True)  # re-normalize to 1
        return arr

    def ilr_basis(self,D):
        """Returns an orthonormal ILR basis using Gram-Schmidt."""
        H = np.eye(D) - np.ones((D, D)) / D
        Q, _ = qr(H[:, :-1])  # Drop last column to avoid redundancy
        return Q.T
    
    def ilr_transform(self, comp, basis):
        log_c = np.log(comp)
        return log_c @ basis.T

    def ilr_inverse(self, ilr_coords, basis):
        log_c = ilr_coords @ basis
        c = np.exp(log_c)
        return c / c.sum(axis=1, keepdims=True)

    def get_efficiency(self,X_new): # X_new is SQP of unseen sample
        """ Returns counting efficiency at unseen SQP (X_new) for this radioisotope, checking that it lies within training range """
        self.check_sqp(np.array(X_new)) # Check X_new lies within trained SQP range
        #print(self.A, self.B, self.C)
        #print(((self.A * (X_new)**2 ) + ( self.B * X_new ) + ( self.C ) )* 0.01)
        return( ((self.A * (X_new)**2 ) + ( self.B * X_new ) + ( self.C )) * 0.01 )   # 0.01 to convert to decimal from %
    
    def check_sqp(self,sqp):
        """ Checks sqp lies within the range of training data """
        # Check X_new lies within trained SQP range
        if np.any(sqp < np.min(self.X_train)):
            raise ValueError(f'{self.name}: Extrapolation! Requested SQP: {sqp} contains less than min trained SQP: {np.min(self.X_train)}')
        elif np.any(sqp > np.max(self.X_train)):
            raise ValueError(f'{self.name}: Extrapolation! Requested SQP: {sqp} contains greater than max trained SQP: {np.max(self.X_train)}')

class LSC_Model():
    def __init__(self, libraries,cols):   # (libraries = list of all the SQP+cols df for each radioisotope)
        self.libraries = libraries
        self.cols = cols
        self.names = [library.name for library in self.libraries]

        # Check efficiency fit info has been entered correctly (i.e. is the same for all entries)
        for library in libraries:
            all_same = (library[['Ax^2','Bx','C','R^2','FitErr']].nunique() == 1).all()
            assert all_same, f'Calibration efficiency fit info mismatch for {library.name}!'
        
        # might remove
        self.data_h3 = libraries[0]
        self.data_c14 = libraries[1]
        self.data_cl36 = libraries[2]
        self.data_fe55 = libraries[3]
        self.data_ni63 = libraries[4]
        self.data_i129 = libraries[5]

        self.models = {}  # where all the models will be stored

    def __str__(self):
        print_msg = 'LSC_Model('
        for name in self.names:
            print_msg+= f'{name}, '
        return f"{print_msg[:-2]})"

    def fit_models(self,test_size=0.2,random_state=42):
        test = [] # to be a list of df of test data
        iterable = tqdm(self.libraries)
        for library in iterable:
            iterable.set_description(f'Fitting {library.name}')

            # test train split
            X_train, X_test, Y_train, Y_test = train_test_split(library['SQP'], library[self.cols], test_size=test_size, random_state=random_state)

            # Extract fit info (pd series)
            fit_info = library[['Ax^2','Bx','C','R^2','FitErr']].iloc[1]

            # Create IsotopeModel for each isotope
            model = IsotopeModel(library.name,X_train,Y_train,fit_info)
            model.fit()

            test.append((X_test,Y_test))
            self.models[library.name] = model

        print(f'\nFitting complete,')
        print(f'Dictionary of models accessible using LSC_Model().models: {self.models}')
        return(test)

########################### FUNCTIONS ################################

def count_categories(df):
    """Counts the number of each unique category (i.e. 3H, 3H+14C, 3H+14C+36CL, 14C, 14C+36CL, etc)."""
    # Define categories
    def categorize(row):
        categories = []
        if row["3H Activity"] != 0:
            categories.append("3H")
        if row["14C Activity"] != 0:
            categories.append("14C")
        if row["36CL Activity"] != 0:
            categories.append("36CL")
        if row["55FE Activity"] != 0:
            categories.append("55FE")
        if row["63NI Activity"] != 0:
            categories.append("63NI")
        if row["129I Activity"] != 0:
            categories.append("129I")
        return "+".join(categories) if categories else "None"

    # Apply categorization
    df["Category"] = df.apply(categorize, axis=1)

    # Count unique category occurrences
    category_counts = df["Category"].value_counts()
    return(category_counts)

def simplify_isotope_label(isotope,isotopes=['3H', '14C', '36CL','55FE','63NI','129I']): 
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

def plot_spectrum(df,cols,i):
    """ Plot spectrum of ith entry in the dataframe. """

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

    #print('\n3H: '+str(df[activity_cols].iloc[i].values[0])+'Bq\n14C: '+str(df[activity_cols].iloc[i].values[1])+'Bq\n36CL: '+str(df[activity_cols].iloc[i].values[2])+'Bq')
    plt.show()

def parse_and_process(directory):
    """Uses functions from parse.py to return dataframe containing valid LSC calibration data found in the directory"""
    df = data_from_files(directory) # get df

    cols=[*range(1,1025)]
    #print("Real data cutoff: ",len(df),' at index: ',df.tail(1).index[0])


    #isotopes=['3H', '14C', '36CL','55FE','63NI','129I']
    # Change isotope col to remove appended numbers
    df['ISOTOPE'] = df['ISOTOPE'].map(simplify_isotope_label)

    # Floatify CTIME col
    df['CTIME'] = df['CTIME'].map(time_to_minutes)
    #print(df['CTIME'])

    # Floatify SQP and SQP%
    df['SQP'] = df['SQP'].astype(float)
    df['SQP%'] = df['SQP%'].astype(float)


    ################## Adding in extra sample to trianing data ######################
    # extra_sample = 'LSC Spectra for AI proj/Unseen/Q032301N.001' # Q-1, 2025
    # extra_spect = parse_file(extra_sample,unseen=True)
    # print(extra_sample + ' opened sucessfully')
    # extra_sqp = 725.24

    # # Need to subtract background from this sample
    # background_sample = 'LSC Spectra for AI proj/Unseen/Q022201N.001'
    # background_spect = parse_file(background_sample,unseen=True)
    # print(background_sample + ' opened sucessfully')
    # # SavGol filter background
    # plt.plot(cols,background_spect)
    # plt.show()
    # plt.close()
    # background_spect = background_spect/2  # Convert to counts per 90 mins
    # background_spect = SpectraSavGolTransformer().transform(background_spect,isnumpy=True)
    # plt.plot(cols,background_spect)
    # plt.show()
    # plt.close()
    # background_spect = np.clip(background_spect, a_min=0, a_max=None)  # Clip to ensure no negative values
    # print('Savgol filter applied to background spectrum. Zeros clipped to avoid negative values.')
    # extra_spect = extra_spect - background_spect

    # # Clip to ensure no negative values
    # extra_spect = np.clip(extra_spect, a_min=0, a_max=None)

    # extra_dict = {}
    # for i in range(1024):
    #     extra_dict[i+1] = extra_spect[i]
    # extra_dict['FILENAME'] = extra_sample
    # extra_dict['ISOTOPE'] = '129I' 
    # extra_dict['SQP'] = extra_sqp
    # extra_dict['CTIME'] = 90    # Standard counting time
    # extra_dict['Activity [Bq]'] = 0.0  # No activity known - this is just a placeholder
    # print(len(extra_dict))

    # # Append the extra sample to the dataframe
    # df = pd.concat([df, pd.DataFrame([extra_dict])], ignore_index=True)
    # print('Extra sample added to dataframe: ',extra_sample)

    # # Plot for sanity check
    # plot_spectrum(df,cols,-1)  # Plot the last entry (the extra sample)
    #################################################################################



    # Dataframe preprocessing
    df = SpectraPreprocessingTransformer().transform(df)
    # a = df[df['FILENAME'] == 'LSC Spectra for AI proj/2022/Quant 6/63NI7/Q030301N.001'].index[0]
    # plot_spectrum(df,cols,a)

    # Filter out low count / bkg spectra that snuck in
    #df = df[df[cols].mean(axis=1) >= 0.1]
    #print(len(df[df[cols].max(axis=1) >= 5]))

    category_counts = count_categories(df)
    print(category_counts)

    #df = SpectrumAugmenter(feature_cols=cols, noise_level=0.05,  N_neighbours=10 , random_state=42).transform(df)
    #plot_spectrum(df,cols,-1)

    #df = SpectraCombinationTransformer().transform(df)
    # plt.plot(cols,df[cols].iloc[-1])
    # plt.show()
    # plt.close()
    
    # Histogram of counts/min below 200
    sum_spect = df[cols].sum(axis=1)
    # plt.hist(sum_spect[sum_spect <= 200])
    # plt.show()

    #print(sum_spect[sum_spect <= 100])
    rows_with_low_max = df[sum_spect <= 100]
    print('\nLow-count spectra (< 100 counts/min):\n',rows_with_low_max)
    # for i in range(len(rows_with_low_max)):
    #     plot_spectrum(rows_with_low_max,cols,i)

    print(count_categories(df))

    df = SpectraSavGolTransformer().transform(df)
    print("Final Df: \n",df)

    # Plotting the final spectrum of the last entry again after savgol filter
    #plot_spectrum(df,cols,-1)  # Plot the last entry (the extra sample)

    # Double check no negative values
    df[cols] = df[cols].clip(lower=0)

    return(df)

def interpolate_spectrum(isotope_library, target_sqp):
    """ Used for demonstration of isotope shape vs SQP """
    cols = [*range(1,1025)]

    sqp_vals = np.array(isotope_library['SQP'].values)
    sorted_indices = np.argsort(sqp_vals)
    sqp_vals = sqp_vals[sorted_indices]

    if target_sqp < np.min(sqp_vals):
        raise ValueError(f'Extrapolation! Target SQP: {target_sqp} is less than min SQP: {np.min(sqp_vals)}')
    elif target_sqp > np.max(sqp_vals):
        raise ValueError(f'Extrapolation! Target SQP: {target_sqp} is greater than max SQP: {np.max(sqp_vals)}')
    
    spectra = isotope_library[cols].to_numpy()
    spectra = spectra[sorted_indices]
    #spectra = np.array([isotope_library[cols] for sqp in sqp_vals])

    interpolator = interp1d(sqp_vals, spectra, axis=0, kind='linear', fill_value='extrapolate')
    return interpolator(target_sqp)

def animate_sqp(isotope_library,N):
    cols = [*range(1,1025)]
    sqp_vals = np.array(isotope_library['SQP'].values)
    min_sqp = np.min(sqp_vals)
    max_sqp = np.max(sqp_vals)
    sqps_to_try = np.linspace(min_sqp,max_sqp,N)
    for i, sqp in enumerate(sqps_to_try):
        spect = interpolate_spectrum(isotope_library, sqp)
        fig, ax = plt.subplots()
        ax.plot(cols,spect)
        ax.set_ylim([0, 0.01])
        ax.set_xlim([0, 1024])
        ax.text(0.97,0.97,round(sqp,2), transform=ax.transAxes,fontsize=25,ha='right',color='r',fontweight='bold',va='top')
        fig.tight_layout()
        name = f'animation{isotope_library.name}/ani{str(i).zfill(4)}'
        fig.savefig(name, dpi=100,facecolor='white', edgecolor='none', transparent=False)
        plt.close()
    cmd = f'ffmpeg -framerate 24 -i animation{isotope_library.name}/ani%04d.png -vf "scale=iw:-1:flags=lanczos" -loop 0 -gifflags -transdiff -y animation{isotope_library.name}/output.gif'
    print(cmd)
    os.system(cmd)

def estimate_activities(measured_spectrum, sqp, models_dict):   # CURRENTLY ONLY HANDLES ONE SPECTRUM AT A TIME ! (measured_spectrum must be 1D, sqp must be float)
    """ Deconvolution of measured_spectrum into models_dict given sqp of measured_spectrum. Returns NNLS coefficients and the deconvoluted radioisotope spectra in CPM. """
    print(f'Computing deconvolution using models for {[model.name for model in models_dict.values()]}...')
    # Create the matrix of expected spectra at the measured SQP(E) from interpolate_spectrum
    isotope_shapes = [model.predict(np.array([sqp])) for model in models_dict.values()]  # shape: (n_channels, 6) where 6 is for 3h, 14c, 36cl, 55fe, 63ni, 129i in that order 
    shape_matrix = np.vstack(isotope_shapes).T 

    # Get counting efficiencies for each radioisotope from fitted polynmomial in corresponding calibration cert 
    counting_efficiencies = np.array([model.get_efficiency(sqp) for model in models_dict.values()]).flatten()
    
    # Use non-negative least squares (NNLS) to solve for activities
    # this finds the best-fit coefficients to the isotope spectra (in the matrix) such that it approximates the measured spectrum.
    activity_estimates, _ = nnls(shape_matrix, measured_spectrum)
    
    return activity_estimates, isotope_shapes, counting_efficiencies   # [H-3 coeff, C-14 coeff, Cl-36 coeff, etc]


########################### MAIN ################################

def main():
    # Get dataframe
    #directory = "LSC Spectra for AI proj including calibration certs/"  # Add / on end. Forward slashes not backward.
    directory = 'LSC Spectra for AI proj/'

    # Check if spreadsheet exists
    if not os.path.isfile('transformed_data.xlsx'):
        df = parse_and_process(directory)
        # Save to excel here if neccessary
        print("Saving to .xlsx ...")
        df.to_excel('transformed_data.xlsx',engine='openpyxl')
    else:
        print("transformed_data.xlsx found. Opening...")
        df = pd.read_excel('transformed_data.xlsx',engine='openpyxl')
        #print(df)

    ############################### 
    # For unseen you need:
    # The 001 file: SP11, or 12, etc, the SQP of the sample and which instrument (Q-1, or Q-2, etc), date and time (plus calibrations for that date), any assumptions that some isotopes are definitely not present.
    # The instrument blank file taken just before
    # The counting time of both files
    cols = [*range(1,1025)]

    #############################

    # Training for what year and what quant?
    year = 2023 # Calibrations from what year?
    machine = 6 # Q-?

    # Create directory for model
    base_dir = f'models/Q-{machine}_{year}_v'
    version = 1
    while os.path.exists(f"{base_dir}{version}"):
        version += 1
    model_dir = f"{base_dir}{version}"
    os.makedirs(model_dir)

    # Get relavent data only
    df = df[(df['YEAR'] == year) & (df['QUANT'] == machine)]     #
    #print(df.loc[378])

    # Count combiations
    print('Training for year: ',year,' and quant: ',machine)
    category_counts = count_categories(df)
    print(category_counts)

    # Normalize 
    df_copy = df.copy()
    df_copy[cols] = df_copy[cols].div(df_copy[cols].sum(axis=1), axis=0)

    pure_14c_df = df_copy[df_copy['Category'].isin(['14C'])]
    pure_3h_df = df_copy[df_copy['Category'].isin(['3H'])]
    pure_36cl_df = df_copy[df_copy['Category'].isin(['36CL'])]
    pure_55fe_df = df_copy[df_copy['Category'].isin(['55FE'])]
    pure_63ni_df = df_copy[df_copy['Category'].isin(['63NI'])]
    pure_129i_df = df_copy[df_copy['Category'].isin(['129I'])]

    #Plotting sqp vs channel 300 for 14c and 3h
    # plt.scatter(pure_14c_df['SQP'].to_numpy(),pure_14c_df[300].to_numpy(),color='r')
    # plt.xlabel('SQP')
    # plt.ylabel('Ch300')
    # plt.show()
    # plt.scatter(pure_3h_df['SQP'].to_numpy(),pure_3h_df[200].to_numpy(),color='r')
    # plt.xlabel('SQP')
    # plt.ylabel('Ch200')
    # plt.show()
    # plt.scatter(pure_55fe_df['SQP'].to_numpy(),pure_55fe_df[200].to_numpy(),color='r')
    # plt.xlabel('SQP')
    # plt.ylabel('Ch200')
    # plt.show()

    # Plotting SQP space covered by the model, which gets saved into the model directory
    plt.style.use('default')
    sqp_maxs = [np.max(pure_3h_df['SQP'].to_numpy()),np.max(pure_14c_df['SQP'].to_numpy()) ,np.max(pure_36cl_df['SQP'].to_numpy()) ,np.max(pure_55fe_df['SQP'].to_numpy()) ,np.max(pure_63ni_df['SQP'].to_numpy()) ,np.max(pure_129i_df['SQP'].to_numpy())]
    sqp_mins = [np.min(pure_3h_df['SQP'].to_numpy()),np.min(pure_14c_df['SQP'].to_numpy()) ,np.min(pure_36cl_df['SQP'].to_numpy()) ,np.min(pure_55fe_df['SQP'].to_numpy()) ,np.min(pure_63ni_df['SQP'].to_numpy()) ,np.min(pure_129i_df['SQP'].to_numpy())]
    max_model_sqp = np.min(sqp_maxs)
    min_model_sqp = np.max(sqp_mins)
    print(f'Model operating range: {min_model_sqp} to {max_model_sqp}')

    plt.plot(pure_3h_df['SQP'].to_numpy(), pure_3h_df['Counting efficiency [%]'].to_numpy(),label='3H')
    plt.plot(pure_14c_df['SQP'].to_numpy(), pure_14c_df['Counting efficiency [%]'].to_numpy(),label='14C')
    plt.plot(pure_36cl_df['SQP'].to_numpy(), pure_36cl_df['Counting efficiency [%]'].to_numpy(),label='36CL')
    plt.plot(pure_55fe_df['SQP'].to_numpy(), pure_55fe_df['Counting efficiency [%]'].to_numpy(),label='55FE')
    plt.plot(pure_63ni_df['SQP'].to_numpy(), pure_63ni_df['Counting efficiency [%]'].to_numpy(),label='63NI')
    plt.plot(pure_129i_df['SQP'].to_numpy(), pure_129i_df['Counting efficiency [%]'].to_numpy(),label='129I')
    plt.axvspan(min_model_sqp, max_model_sqp, color='gray', alpha=0.5,label='Mdl Rng')
    ax = plt.gca()
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    ax.xaxis.grid(True, which='minor')
    ax.tick_params(which='minor',color='grey')
    plt.legend()
    plt.xlabel('SQP')
    plt.ylabel('Counting Efficiency [%]')
    plt.grid(True)
    plt.savefig(f'{model_dir}/sqp_operating_range.png', dpi=300, bbox_inches='tight')
    plt.style.use('bmh')

    # Get libraries of pure radioisotopes
    cols_to_filter = cols + ['SQP','Ax^2','Bx','C','R^2','FitErr']
    library_14c = pure_14c_df[cols_to_filter] 
    library_3h = pure_3h_df[cols_to_filter]     
    library_36cl = pure_36cl_df[cols_to_filter]     
    library_55fe = pure_55fe_df[cols_to_filter]     
    library_63ni = pure_63ni_df[cols_to_filter]     
    library_129i = pure_129i_df[cols_to_filter]     

    library_14c.name = '14C'
    library_3h.name = '3H'
    library_36cl.name = '36CL'
    library_55fe.name = '55FE'
    library_63ni.name = '63NI'
    library_129i.name = '129I'

    libraries = [library_3h , library_14c , library_36cl , library_55fe , library_63ni , library_129i]

    LSCModels = LSC_Model(libraries,cols)       # one of these per quant?
    print(LSCModels)

    # Fit models
    pure_test_data = LSCModels.fit_models() #test_size=0.2,random_state=42)
    with open(f'{model_dir}/test_data.pkl', 'wb') as f:
        pickle.dump(pure_test_data, f)
    print(f'Test data saved to {model_dir}/test_data.pkl')

    # Save LSCModels to file
    with open(f'{model_dir}/LSCModels.pkl', 'wb') as f:
        pickle.dump(LSCModels, f)
    print(f'Models saved to {model_dir}/LSCModels.pkl')

    sys.exit()











    ############################ old code for testing ############################

    # Unseen test
    # unseen_file = 'LSC Spectra for AI proj/Unseen/Q022201N.001' # Q-1, 2025
    # unseen_spect = parse_file(unseen_file,unseen=True)
    # print(unseen_file + ' opened sucessfully')
    # unseen_sqp = 727.82

    plt.plot(cols,unseen_spect)
    plt.show()
    plt.close()
    # should implement check for year in order to find the most recent calibration to train model on , but this is not implemented yet
    dict_minus_these = LSCModels.models
    dict_minus_these.pop('3H')
    dict_minus_these.pop('55FE')
    dict_minus_these.pop('63NI')
    estimated_cpms, isotope_shapes, counting_efficiencies = estimate_activities(
        unseen_spect, unseen_sqp, dict_minus_these
    )  
    print(counting_efficiencies)
    print(estimated_cpms)
    dpms = estimated_cpms/counting_efficiencies
    print(pd.DataFrame([dpms],columns=['14C', '36CL', '129I']).to_string(index=False))
    names = ['14C', '36CL', '129I']

    tot= np.zeros(len(cols))
    for k, isotope_shape in enumerate(isotope_shapes):
        plt.plot(cols, isotope_shape.squeeze()*estimated_cpms[k],label=f'pred_{names[k]}',linewidth=1)
        tot += isotope_shape.squeeze()*estimated_cpms[k]

    mse = np.mean((unseen_spect - tot) ** 2)
    mae = np.mean(np.abs(unseen_spect - tot))
    print(f'MSE = {mse}')
    print(f'MAE = {mae}')

    plt.plot(cols, unseen_spect, label = 'Actual',color='dimgrey',linestyle='dotted',linewidth=0.5)
    plt.plot(cols,tot,label='Pred sum',color='black',zorder=20)
    plt.ylabel('CPM')
    plt.legend()
    plt.show()
    plt.close()
    sys.exit()

    ############################ 


    # Get fe55 test data
    pure_test_fe55 = pure_test_data[3]
    pure_test_14c = pure_test_data[1]
    print(pure_test_fe55)
    X_test_fe, Y_test = pure_test_fe55
    X_test_c, Y_test = pure_test_14c
    # print(X_test, Y_test )
    # sys.exit()
    names = LSCModels.names
    # for i,name in enumerate(names):
    #     print(name)
    #     X_test, Y_test = pure_test_data[i]
    #     Y_pred = LSCModels.models[name].predict(X_test)
    #     for j in range(len(Y_test)):
    #         plt.plot(cols,Y_test.iloc[j],label='Actual')
    #         plt.plot(cols,Y_pred[j,:],label='Pred')
    #         plt.legend()
    #         plt.show()



    #df_to_try = df[df['Category'].isin(['55FE'])]
    i = X_test_fe.index[-1]
    #i = 378
    print(i)
    #sqp_test = df_to_try['SQP'].iloc[i]
    print(df.loc[i])
    sqp_test = df['SQP'].loc[i]
    print(f'SQP= {sqp_test}')
    #test_spectrum = df_to_try[cols].iloc[i]
    test_spectrum = df[cols].loc[i]
    real_activity = df['55FE Activity'].loc[i]
    efficiency_corretion = df['Counting efficiency [%]'].loc[i] *10**-2   
    #real_activity = df_to_try['55FE Activity'].iloc[i]
    #efficiency_corretion = df_to_try['Counting efficiency [%]'].iloc[i] *10**-2       # On an unknown sample this would come from the most recent calibration cert for that radioisotope.
    # divide by ^ i.e. = 1/e

    estimated_activities, isotope_shapes = estimate_activities(
        test_spectrum, sqp_test, LSCModels.models
    )   
    for shape in isotope_shapes:
        print(np.sum(shape))

    #print('Actual: ', [dpm_3h, dpm_14c,dpm_36cl, dpm_55fe,dpm_63ni, dpm_129i ])
    print('Actual: ', [0, 0, 0, real_activity*60, 0, 0 ])
    estimated_activities[3] = estimated_activities[3] / efficiency_corretion
    print('Pred: ', estimated_activities.tolist()) # 3h, 14c, 36cl, 55fe, 63ni, 129i
    estimated_activities[3] = estimated_activities[3] * efficiency_corretion

    tot= np.zeros(len(cols))
    for k, isotope_shape in enumerate(isotope_shapes):
        plt.plot(cols, isotope_shape.squeeze()*estimated_activities[k],label=f'pred_{names[k]}',linestyle='--')
        tot += isotope_shape.squeeze()*estimated_activities[k]
    plt.plot(cols, test_spectrum, label = 'Actual',color='black',linewidth=1)
    plt.plot(cols,tot,label='Pred sum',linestyle='dotted',color='grey',zorder=20)
    plt.ylabel('CPM')
    plt.legend()
    plt.show()
    plt.close()

    ##########################

    #df_to_try = df[df['Category'].isin(['55FE'])]
    i = X_test_c.index[-1]
    #i = 378
    print(i)
    #sqp_test = df_to_try['SQP'].iloc[i]
    print(df.loc[i])
    sqp_test = df['SQP'].loc[i]
    print(f'SQP= {sqp_test}')
    #test_spectrum = df_to_try[cols].iloc[i]
    test_spectrum = df[cols].loc[i]
    real_activity = df['14C Activity'].loc[i]
    efficiency_corretion = df['Counting efficiency [%]'].loc[i] *10**-2   
    #real_activity = df_to_try['55FE Activity'].iloc[i]
    #efficiency_corretion = df_to_try['Counting efficiency [%]'].iloc[i] *10**-2       # On an unknown sample this would come from the most recent calibration cert for that radioisotope.
    # divide by ^ i.e. = 1/e

    estimated_activities, isotope_shapes = estimate_activities(
        test_spectrum, sqp_test, LSCModels.models
    )   
    for shape in isotope_shapes:
        print(np.sum(shape))

    #print('Actual: ', [dpm_3h, dpm_14c,dpm_36cl, dpm_55fe,dpm_63ni, dpm_129i ])
    print('Actual: ', [0, real_activity*60, 0, 0, 0, 0 ])
    estimated_activities[1] = estimated_activities[1] / efficiency_corretion
    print('Pred: ', estimated_activities.tolist()) # 3h, 14c, 36cl, 55fe, 63ni, 129i
    estimated_activities[1] = estimated_activities[1] * efficiency_corretion

    tot= np.zeros(len(cols))
    for k, isotope_shape in enumerate(isotope_shapes):
        plt.plot(cols, isotope_shape.squeeze()*estimated_activities[k],label=f'pred_{names[k]}',linestyle='--')
        tot += isotope_shape.squeeze()*estimated_activities[k]
    plt.plot(cols, test_spectrum, label = 'Actual',color='black',linewidth=1)
    plt.plot(cols,tot,label='Pred sum',linestyle='dotted',color='grey',zorder=20)
    plt.ylabel('CPM')
    plt.legend()
    plt.show()

if __name__=='__main__':
    main()