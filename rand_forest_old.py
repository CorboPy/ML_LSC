# Problem Files:
# C:/Users/alexc/Documents/#Uni/LSC_ML/LSC_spectra_sorted/Mixed/3H + 36Cl Q6 3H439 2023/Q072701N.001   AKA 456 (v noisy)
# C:/Users/alexc/Documents/#Uni/LSC_ML/LSC_spectra_sorted/Mixed/3H + 36Cl Q6 3H439 2023/Q092901N.001   AKA 458 (no Cl peak)

# Another idea:
# Get peaks and relative 
# (inspired by https://www.sciencedirect.com/science/article/pii/S2666389920302622)

from parse import *
import sys
import os

# Numpy, pandas, matplotlib
import pandas as pd
pd.set_option('display.max_colwidth', None)
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
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
from sklearn.metrics import multilabel_confusion_matrix,classification_report,hamming_loss,precision_recall_curve,auc

# Skmultilearn
from skmultilearn.problem_transform import BinaryRelevance
from skmultilearn.model_selection import iterative_stratification

# Tqdm and tabulate
from tqdm import tqdm
from tabulate import tabulate

# Ignore pandas performance warning
from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

########################################################################################################################

class DynamicRangeCompressor(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.1, ratio=2):
        # Initialize the compressor with specified parameters:
        # threshold: level above which compression starts
        # ratio: compression ratio
        self.threshold = threshold
        self.ratio = ratio
    
    def fit(self, X, y=None):
        # The fit method is a no-op for this transformer, as no fitting is required.
        return self
    
    def transform(self, X, y=None):
        # Apply the compression along each row (spectrum) of the input data.
        # np.apply_along_axis applies the _compress method to each row of X.
        return np.apply_along_axis(self._compress, 1, X)
    
    def _compress(self, data):
        # Helper method to apply dynamic range compression to a single spectrum.

        # Find the maximum absolute value in the spectrum for normalization.
        max_val = np.max(np.abs(data))
        if max_val == 0:
            # If the maximum value is zero, the data is already at the noise floor, so return it as is.
            return data
        # Normalize the data to have a maximum absolute value of 1.
        data = data / max_val
        
        # Initialize an array to store the compressed data.
        compressed_data = np.zeros_like(data)
        for i, sample in enumerate(data):
            if np.abs(sample) > self.threshold:
                # If the sample exceeds the threshold, apply compression.
                gain = self.threshold / np.abs(sample) * (1.0 - 1.0 / self.ratio)
            else:
                # If the sample is below the threshold, no compression is applied.
                gain = 1.0
            # Apply the gain to the sample.
            compressed_data[i] = sample * gain
        
        # Denormalize the data back to its original scale.
        compressed_data = compressed_data * max_val
        return compressed_data

###############################################################

def class_dist(df):
    """Prints the class distribution of dataframe

    Args:
        df (DataFrame): pandas DataFrame containing spectra with 'Element' information
    """
    # Getting class distribution of dataset
    class_sf = df.groupby(['Element']).size() 
    class_df =  pd.DataFrame({'Element':class_sf.index, 'Count':class_sf.values})
    print("Class distribution:\n",class_df,"\n")

def dataframe(df):
    # Getting class distribution of dataset
    print("\nFrom File DataFrame")
    class_dist(df)

    # Creating data_df (that contains 1024 columns corresponding to each channel)
    data_df = df.drop(["Year","Quant"], axis='columns')      # Remove unneccessary columns
    print("Retrieving spectra...")
    data_df['Spectra'] = data_df["File"].apply(get_data)    # Getting data from file using get_data() from parse
    data_df.dropna(inplace=True)     # Removing rows with spectra = None
    cols = [*range(1,1025)]
    data_df[cols] = pd.DataFrame(data_df.Spectra.tolist(),index=data_df.index)  # Separate out the spectra columns
    data_df.drop("Spectra", axis='columns',inplace=True)        # Remove the old column
    #print(data_df.head())
    return(data_df)

# Combines single-radioisotope spectra to form synthetic double-radioisotope spectra 
def combine(df,cols):

    print("Combining single-radioisotope spectra to form synthetic double-radioisotope spectra...\n")

    # Radioisotope-specific dfs
    df_14c = df.drop(df[df.Element !='14C'].index)
    df_36cl = df.drop(df[df.Element !='36CL'].index)
    df_3h = df.drop(df[df.Element !='3H'].index)

    # 2D list of the spectra
    spectra_14c = df_14c.loc[:, cols].values 
    spectra_36cl = df_36cl.loc[:, cols].values
    spectra_3h = df_3h.loc[:, cols].values

    # 14C + 36CL
    spectra_14c_36cl = []
    IDs = []
    quantities = []
    for i, carbon14 in enumerate(spectra_14c):
        carbon14 = (carbon14 - np.min(carbon14)) / (np.max(carbon14) - np.min(carbon14))    # Normalisation between 0 and 1
        for j, chlorine36 in enumerate(spectra_36cl):
            chlorine36 = (chlorine36 - np.min(chlorine36)) / (np.max(chlorine36) - np.min(chlorine36))    # Normalisation between 0 and 1

            r = np.random.uniform() # r is a number between 0 and 1 that will act as a multiplier and blend the distribution
            
            # if j == 100:
            #     plt.style.use('bmh')
            #     plt.plot(cols,carbon14,"-b", label="Carbon14")
            #     plt.plot(cols,chlorine36,"-r", label="Chlorine36")
            #     plt.plot(cols,np.add(carbon14,chlorine36),"-g",label="Combined")
            #     plt.legend()
            #     plt.show()

            spectra_14c_36cl.append(np.add(r*carbon14,(1-r)*chlorine36))
            IDs.append('14C: '+str(i)+', 36CL: '+str(j))
            quantities.append([r,(1-r),0])
    df_14c_36cl = pd.DataFrame(data = spectra_14c_36cl,columns=cols)
    df_14c_36cl.insert(0, 'Quantity', quantities)  # [14C, 36CL, 3H]
    df_14c_36cl.insert(0, 'Element', '14C,36CL')
    df_14c_36cl.insert(0, 'File', IDs)
    # df_14c_36cl[['Element']] == '14C,36CL'

    # 14C + 3H
    spectra_14c_3h = []
    IDs = []
    quantities = []
    for i, carbon14 in enumerate(spectra_14c):
        carbon14 = (carbon14 - np.min(carbon14)) / (np.max(carbon14) - np.min(carbon14))    # Normalisation between 0 and 1
        for j, tritium in enumerate(spectra_3h):
            tritium = (tritium - np.min(tritium)) / (np.max(tritium) - np.min(tritium))    # Normalisation between 0 and 1

            r = np.random.uniform() # r is a number between 0 and 1 that will act as a multiplier and blend the distribution

            # plt.plot(cols,tritium)
            # plt.plot(cols,carbon14)
            # plt.plot(cols,np.add(carbon14,tritium))
            # plt.show()

            spectra_14c_3h.append(np.add(r*carbon14,(1-r)*tritium))
            IDs.append('14C: '+str(i)+', 3H: '+str(j))
            quantities.append([r,0,(1-r)])
    df_14c_3h = pd.DataFrame(data = spectra_14c_3h,columns=cols)
    df_14c_3h.insert(0, 'Quantity', quantities)  # [14C, 36CL, 3H]
    df_14c_3h.insert(0, 'Element', '14C,3H')
    df_14c_3h.insert(0, 'File', IDs)
    #df_14c_3h[['Element']] == '14C,3H'

    # 3H + 36CL
    spectra_3h_36cl = []
    IDs = []
    quantities = []
    for i, tritium in enumerate(spectra_3h):
        tritium = (tritium - np.min(tritium)) / (np.max(tritium) - np.min(tritium))    # Normalisation between 0 and 1
        for j, chlorine36 in enumerate(spectra_36cl):
            chlorine36 = (chlorine36 - np.min(chlorine36)) / (np.max(chlorine36) - np.min(chlorine36))    # Normalisation between 0 and 1

            r = np.random.uniform() # r is a number between 0 and 1 that will act as a multiplier and blend the distribution
            
            # plt.plot(cols,tritium)
            # plt.plot(cols,chlorine36)
            # plt.plot(cols,np.add(tritium,chlorine36))
            # plt.show()

            spectra_3h_36cl.append(np.add((1-r)*tritium,r*chlorine36))
            IDs.append('3H: '+str(i)+', 36CL: '+str(j))
            quantities.append([0,r,(1-r)])
    df_3h_36cl = pd.DataFrame(data = spectra_3h_36cl,columns=cols)
    df_3h_36cl.insert(0, 'Quantity', quantities)  # [14C, 36CL, 3H]
    df_3h_36cl.insert(0, 'Element', '3H,36CL')
    df_3h_36cl.insert(0, 'File', IDs)
    #df_3h_36cl[['Element']] == '3H,36CL'

    # Combine into one df
    merged_spectra_df = pd.concat([df_14c_36cl,df_14c_3h,df_3h_36cl],ignore_index=True)
    #print(merged_spectra_df)

    
    return(merged_spectra_df)

# Asymmetric Least-Squares Algorithm (for baseline subtraction if neccessary)
def baseline_als(y, lam, p, niter=10):

    L = len(y)
    D = sparse.csc_matrix(np.diff(np.eye(L), 2))
    w = np.ones(L)
    for i in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = spsolve(Z, w*y)
        w = p * (y > z) + (1-p) * (y < z)
    return(z)

def standardize(arr,scaler):
    # z-score normalisation
    arr_standardized = scaler.fit_transform(arr)
    return(arr_standardized)

def compress_dynamic_range(spectrum, method='log'):
    if method == 'log':
        return np.log1p(spectrum)  # log1p is log(1 + spectrum) to handle zero values
    elif method == 'sqrt':
        return np.sqrt(abs(spectrum))
    elif method == 'cbrt':
        return([abs(x)**(1/3) for x in spectrum])
    else:
        raise ValueError("Unsupported method. Use 'log', 'cbrt' or 'sqrt'.")

def data_generation(df,cols):
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


# Cleans data - H/L pass, SG filter, standardize
def process(data_df,cols):
    data_df_new = data_df.copy() # create copy
    index = data_df.index
    data_df_new.drop(cols, axis='columns',inplace=True) # drop old data

    x = data_df.loc[:, cols].values # x is a 2D list of the spectra

    # plt.plot(cols,x[305])
    # plt.plot(cols,x[480])
    # plt.plot(cols,x[77835])
    # plt.show()

    x_processed = []
    x_num_peaks = []
    for i,spect in enumerate(tqdm(x,ascii=True,desc="Processing")):
        # High/low pass - remove the channel 0 outlier
        spect = spect[1:]

        #  Savitzky-Golay filter
        spect = savgol_filter(spect,41,3)          #41. 2nd arg must be odd

        # FIND PEAKS (remove if not used in model)
        peaks, properties = find_peaks(spect,prominence=0.05,width=60,distance=50)
        # Engineer number of peaks as a feature
        num_peaks = len(peaks)

        # Normalize by area
        # xx = [*range(len(spect))]
        # area_under = auc(xx,spect)
        # spect = [i/area_under for i in spect]

        # Compress
        #spect = compress_dynamic_range(spect, method='log')        

        # Normalisation between 0 and 1
        spect = (spect - np.min(spect)) / (np.max(spect) - np.min(spect))
    
        # plt.style.use('bmh')
        # plt.plot(x, spectrum)
        # plt.plot(x, spectrum_filter)
        # plt.scatter(peaks,spectrum_filter[peaks],color='r',marker='x',zorder=3)
        # maximum = np.max(spectrum)
        # minimum = np.min(spectrum)
        # # Make sure ticks don't look silly
        # if maximum<10:
        #     ticks = np.linspace(minimum,round(maximum, 0),11)
        # else:   
        #     ticks = np.linspace(minimum,round(maximum, -1),11)
        # ax = plt.gca()
        # fig = ax.get_figure()
        # ax.set_xlabel("Channel")
        # ax.set_ylabel("Counts")
        # plt.title("H/L Filter + Normalisation + S-G Filter\n"+isotope)
        # ax.margins(x=0,y=0)
        # fig.set_figheight(10)
        # fig.set_figwidth(15)
        # plt.show()
        x_num_peaks.append(num_peaks)
        x_processed.append(spect)

    colsnew = cols[1:] # after high pass

    # comp = DynamicRangeCompressor(threshold=0.2, ratio=6)   # DOESN'T WORK
    # x_processed = comp.fit_transform(x_processed)

    # plt.plot(colsnew,x_processed[305])
    # plt.plot(colsnew,x_processed[480])
    # plt.plot(colsnew,x_processed[77835])
    # plt.show()

    # or Standardize? (instead of normalize from above)
    # scaler = StandardScaler()
    # x_processed = standardize(x_processed,scaler)


    processed_df = pd.DataFrame(data = x_processed,columns=colsnew)   # New data df
    processed_df.index = index
    print(len(processed_df))
    
    data_df_new[colsnew] = processed_df[colsnew].values # Add new processed data

    #print(data_df_new)
    return(data_df_new,x_num_peaks,colsnew)

# Applies PCA to reduce dataset to n features
def dim_reduction(data_df,n,cols):
    print("\nFinding Principal Components...\n")
    data_df_new = data_df.copy()   # Create copy
    index = data_df.index    # Extract index
    x = data_df.loc[:, cols].values     # x is a 2D list of the spectra
    pc_cols = ['PC'+str(i+1) for i in range(n)]    # Get column labels

    pca_spect = PCA(n_components=n)
    principalComponents_spect = pca_spect.fit_transform(x)
    principal_spect_Df = pd.DataFrame(data = principalComponents_spect
             , columns = pc_cols)
    principal_spect_Df.index = index     # Reapply index
    #print(principal_spect_Df)
    
    data_df_new.drop(cols, axis='columns',inplace=True) # Remove old 1-1024 columns
    data_df_new[pc_cols] = principal_spect_Df[pc_cols]
    return(data_df_new,pca_spect)

# One-hot encoding of labels
def one_hot_encode(pc_df,df):
    mlb = MultiLabelBinarizer()
    y_one_hot = mlb.fit_transform(pc_df['y'])
    # Get the classes (unique labels)
    classes = mlb.classes_
    # Convert back to a DataFrame for better visualization 
    y_one_hot_df = pd.DataFrame(y_one_hot, columns=classes)
    y_one_hot_df.index = pc_df.index
    return(y_one_hot_df,classes)

def pca_plot(pc_df,df,comp1,comp2,comp3):
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(projection='3d')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=14)
    ax.set_xlabel('Principal Component - '+str(comp1),fontsize=20)
    ax.set_ylabel('Principal Component - '+str(comp2),fontsize=20)
    ax.set_zlabel('Principal Component - '+str(comp3),fontsize=20)
    plt.title("Principal Component Analysis of LSC Dataset",fontsize=20)
    targets = ['14C', '36CL','3H','3H,36CL']
    colors = ['r', 'g','b','orange']
    for target, color in zip(targets,colors):
        indicesToKeep = df['Element'] == target
        ax.scatter(pc_df.loc[indicesToKeep, 'PC'+str(comp1)]
                , pc_df.loc[indicesToKeep, 'PC'+str(comp2)]
                , pc_df.loc[indicesToKeep, 'PC'+str(comp3)], c = color, s = 50)

    plt.legend(targets,prop={'size': 15})
    plt.show()

def scree_plot(x,y):
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

def model_train(X_train,Y_train,n_estimators_=5):
    """Returns trained model (Binary Relevance multi-label classifier with a Random Forest base classifier)

    Args:
        X_train (list): Training split of X features
        Y_train (list): Training split of Y labels
        n_estimators_ (int, optional): Number of trees in random forest. Defaults to 5.
    """
    # Initialize the Binary Relevance multi-label classifier with a Random Forest base classifier
    rf = RandomForestClassifier(n_estimators=n_estimators_) #,random_state=42       # Look into max_depth
    model = BinaryRelevance(classifier=rf, require_dense=[False, True])

    # Train the model
    model.fit(X_train, Y_train)
    return(model)

#####################################################

def main():
    # Get dataframe
    directory = "C:/Users/alexc/Documents/#Uni/LSC_ML/LSC_spectra_sorted/"  # Add / on end. Forward slashes not backward.
    df = get_file_df(directory)

    # Getting data_df by parsing each file
    data_df = dataframe(df)
    data_df.reset_index(drop=True,inplace=True)
    cols=[*range(1,1025)]
    print("Real data cutoff: ",len(data_df),' at index: ',data_df.tail(1).index[0])

    # Getting class distribution of dataset
    print('\nAfter Cleaning')
    class_dist(data_df)

    # # Synthesise radioisotope combinations
    # merged_spectra_df = combine(data_df,cols)
    # data_df = pd.concat([data_df,merged_spectra_df],ignore_index=True)

    # # Add percentages to other entries
    # data_df.loc[data_df.Element == '14C', 'Quantity'] = str([1,0,0])    # 14C, 36CL, 3H
    # data_df.loc[data_df.Element == '36CL', 'Quantity'] = str([0,1,0])    # 14C, 36CL, 3H
    # data_df.loc[data_df.Element == '3H', 'Quantity'] = str([0,0,1])    # 14C, 36CL, 3H
    # tests = data_df[data_df.isna().any(axis=1)] # Will need processing
    # data_df.dropna(inplace=True)

    # Getting class distribution of dataset
    print('\nAfter Combination Synthesis')
    class_dist(data_df)

    # Synthesise more data using noise
    # power = 7      # Will result in 2^power x len(data_df) spectra
    # for _ in range(power):
    #     data_df = data_generation(data_df,cols)
    #print(data_df)

    # Pre-processing
    data_df_processed,num_peaks,cols = process(data_df,cols)
    
    # Principal Components
    n = 15  # Number of principal components to find
    pc_df,pca_spect = dim_reduction(data_df_processed,n,cols)   # Finding principal components

    # Adding engineered features (e.g. num_peaks)
    pc_df['Num Peaks'] = num_peaks

    # Set up one-hot encoding
    pc_df['y'] = pc_df['Element'].apply(lambda x: x.split(','))
    pc_df.drop("Element",axis='columns',inplace=True)
    pc_df.dropna(inplace=True) 

    # One-hot encode labels
    y_one_hot_df,classes = one_hot_encode(pc_df,df)

    # Append one-hot labels to pc_df 
    pc_df[classes] = y_one_hot_df[classes]
    #print('Number of samples after pre-processing:',len(pc_df))

    # Printing pc_df and explained variation
    #print("\n1024-Channel DataFrame\n",tabulate(pc_df.head(),headers='keys', tablefmt='psql'),'\n',tabulate(pc_df.tail(), tablefmt='psql'))    # Print head and tail only of pc_df
    if len(pc_df)<500:
        print("\n1024-Channel DataFrame\n\n",pc_df.to_string(),'\nLength of pc_df: ',len(pc_df))  # Print whole pc_df
    else:
        print("\n1024-Channel DataFrame\n\n",pc_df.head().to_string(),'\nLength of pc_df: ',len(pc_df))  # Print whole pc_df
    print('\nExplained variation per principal component: {}'.format(pca_spect.explained_variance_ratio_),'\n')
    
    # # Plotting PCs and scree
    # pca_plot(pc_df,df,1,2,3)
    # scree_plot(['PC'+str(i+1) for i in range(n)],pca_spect.explained_variance_ratio_)

    # Splitting into X and y
    num_components = 10  # Number of principal compnents to use with model
    cols = ['PC'+str(i+1) for i in range(num_components)]
    labels = ["14C","36CL","3H"]
    y = pc_df[labels]
    print('Y\n',y)
    X = pc_df[cols]
    #print(X,'\n')

    # Split X,y into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2,stratify=y) #,random_state=42)
    # Testing for if model can predict correctly combinations of labels without having been trained on them:
    #i = len(pc_df) - 8
    #X_train = X.iloc[:i,:]
    #X_test = pd.concat((X.iloc[i:,:],X.iloc[0:8,:],X.iloc[165:173,:],X.iloc[295:303,:]),axis=0)
    #X_test = X.iloc[i:,:]
    #Y_train = y.iloc[:i,:]
    #Y_test = pd.concat((y.iloc[i:,:],y.iloc[0:8,:],y.iloc[165:173,:],y.iloc[295:303,:]),axis=0)
    #Y_test = y.iloc[i:,:]
    #print(Y_test,'\n',Y_train)

    print('\nNumber of samples in training set: ',len(X_train),'\nNumber of samples in testing set: ',len(X_test),'\n')

    # Train model
    model = model_train(X_train,Y_train,100)

    # Predict probabilities on the test set
    y_proba = model.predict_proba(X_test)
    y_proba_arr = y_proba.toarray()
    y_proba_list = [y_proba_arr[i] for i in range(len(y_proba_arr))]
    y_proba_df = pd.DataFrame({'model probabilities': y_proba_list},index=Y_test.index)

    # Find percentage of label from probabilities
    sum_proba = y_proba_arr.sum(axis=1, keepdims=True) # Calculate the sum of probabilities for each sample
    y_perc_arr = (y_proba_arr / sum_proba) * 100 # Normalize to get percentages
    #print(y_perc_arr)
    y_perc_list = [y_perc_arr[i] for i in range(len(y_perc_arr))]
    y_perc_df = pd.DataFrame({'radioisotope_percentages': y_perc_list},index=Y_test.index)
    y_perc_df[[f'14C %',f'36CL %',f'3H %']] = pd.DataFrame(y_perc_df.radioisotope_percentages.tolist(), index= y_perc_df.index)
    #print(y_perc_df)

    # Predict the labels for the test set
    y_pred = model.predict(X_test).toarray()  # Convert predictions to array
    y_pred_list = [y_pred[i] for i in range(len(y_pred))]
    y_pred_df = pd.DataFrame({'model prediction': y_pred_list},index=Y_test.index)
    #print(y_pred_df)

    # Constructing test_vs_pred_df for review of model on test data
    test_vs_pred_df = pd.concat([Y_test['14C'],Y_test['36CL'],Y_test['3H'], y_pred_df['model prediction'],y_proba_df['model probabilities'],y_perc_df['14C %'],y_perc_df['36CL %'],y_perc_df['3H %']], axis=1)
    # Combine the three columns into a list column
    test_vs_pred_df['combined_col'] = test_vs_pred_df.apply(lambda row: np.array([row['14C'], row['36CL'], row['3H']]), axis=1)
    #print(tabulate(test_vs_pred_df,headers='keys'))
    # Compare the 'combined_col' with 'compare_col'
    test_vs_pred_df['match'] = test_vs_pred_df.apply(lambda row: np.array_equal(row['combined_col'], row['model prediction']), axis=1)
    test_vs_pred_df.drop("combined_col",axis=1,inplace=True)
    test_vs_pred_df.sort_index(inplace=True)
    if len(test_vs_pred_df)<500:
        print(tabulate(test_vs_pred_df,headers='keys'))
    else:
        print(tabulate(test_vs_pred_df.head(),headers='keys'))

    # Extract feature importances for each label's binary classifier
    print("\n")
    for i, model in enumerate(model.classifiers_):
        print(f"Label {labels[i]} - Feature Importances: {model.feature_importances_}")
    print("\n")

    # Compute the confusion matrix for each label
    conf_matrix = multilabel_confusion_matrix(Y_test, y_pred)

    # Print the confusion matrix for each label
    for i, matrix in enumerate(conf_matrix):
        print(labels[i]," Confusion Matrix:")
        print(tabulate(matrix))

    print(classification_report(Y_test, y_pred,target_names=labels))
    print("Hamming Loss:", hamming_loss(Y_test, y_pred))

    # Flatten the predicted probabilities to visualize
    plt.hist(y_proba_arr.flatten(), bins=50)
    plt.xlabel('Predicted Probability')
    plt.ylabel('Frequency')
    plt.title('Histogram of Predicted Probabilities')
    plt.show()

    # Calculate precision-recall curve
    precision, recall, thresholds = precision_recall_curve(Y_test.values.ravel(), y_pred.ravel())
    #print(thresholds)

    # Plot precision-recall curve
    plt.plot(recall, precision, marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.show()




if __name__=='__main__':
    main()