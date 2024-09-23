import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

def check(df):
    # Count the number of spectra in the .001 file using the SP marker
    markers = []
    i=0
    for value in df[0]:
        i+=1
        if value[0:2]=='SP':
            markers.append(i)
    # Length of markers will be 0 if single spectra, 1 if 2 spectra, etc
    return(markers)
def clean(df_list):
    spect_list = []
    for df in df_list:
        # Split column by space
        df = df[0].str.split(' ', expand=True)

        # Remove whitespace (if any)
        df = df.apply(lambda x: x.strip() if isinstance(x, str) else x)

        # Remove empty col
        df = df.drop(0,axis='columns')

        # Move into list
        spect = []
        for row in df.values:
            # For each row, append a list of the values from each column (row_data) to the corresponding range of values in spect
            for i in range(len(row)):
                spect.append(row.tolist()[i])

        # Remove all "None"
        spect = [x for x in spect if x is not None]

        # Convert to numerical list
        spect = [float(i) for i in spect]
        spect_list.append(np.array(spect))

    return(spect_list)

def parse_file(filename):
    """Extracts spectrum/spectra from .001 file.

    Args:
        filename (str): Relative or absolute path to .001 file

    Returns:
        tuple: (spect_list,first_num,isotope). 
            spect_list (list): Python list containing numpy array(s) of the spectrum/spectra. If only 1 spectrum in file, len(spect_list) = 1
            first_num (int): First spectra number as displayed in the file header
            #REMOVED# isotope (str): Isotope info derived from base directory. e.g. 36CL = Chlorine-36
    """
    # Load into dataframe
    #filename=r"LSC Spectra for AI\2021\Quant-1\36CL10\Q020201N.001"
    try:
        df = pd.read_table(filename,header=None)
    except FileNotFoundError as err:
        print(err,'. Ensure path has form e.g. "/dir1/dir2/dir3/36CL2/Q014101N.001"')
        sys.exit()

    # Drop header info
    df_header = df.iloc[0:2]
    first_num = int(df_header.iloc[1].values[0][4:6])
    df = df.drop([0,1])

    markers = check(df)
    #print(markers)
    if len(markers)==1:
        # Discard first spectra
        #print("2 spectra identified.")
        df1 = df.iloc[:markers[0]-1,:]
        df2 = df.iloc[markers[0]:,:]
        spect_list = clean([df1,df2])
        return(spect_list,first_num)
    elif len(markers)>1:
        print("Error: file contains more spectra than expected.")
        sys.exit()
    else:
        #print("Single spectra identified.")
        spect_list = clean([df])
        return(spect_list,first_num)

def get_file_df(dir):
    """Returns dataframe containing Path, Year, Quant, and Radioisotope infomation for each file.

    Args:
        dir (str): root directory in format:
         root
         | 14C
         |   | {file1}_q1_2021
         |   | {file2}_q2_2021
         | 3H
         |   | {file1}_q1_2021
         |   | {file2}_q2_2021        
         | 36CL
         |   | {file1}_q1_2021
         |   | {file2}_q2_2021
    """
    
    dict_list = []
    folders = [f for f in os.listdir(dir)]# if os.path.isdir(f)]
    for folder in folders:
        if folder=='Mixed':
            subfolders = [f for f in os.listdir(dir+folder+'/')]
            for subfolder in subfolders:
                elements = str(subfolder.split(',')[0]) + ',' + str(subfolder.split(',')[1]) 
                subsubfolders = [f for f in os.listdir(dir+folder+'/'+subfolder+'/')]
                for subsubfolder in subsubfolders:
                    for file in os.listdir(dir+folder+'/'+subfolder+'/'+subsubfolder+'/'):
                        if file.endswith(".001") or file.endswith(".002") or file.endswith(".003"):
                            dict_list.append({'File':dir+folder+'/'+subfolder+'/'+subsubfolder+'/'+file,'Year':'N/A','Quant':'N/A','Element':elements})

            # Skip this one out for now
            continue

        subfolders = [f for f in os.listdir(dir+folder+'/')]
        for subfolder in subfolders:
            for file in os.listdir(dir+folder+'/'+subfolder+'/'):
                if file.endswith(".001"):
                    dict_list.append({'File':dir+folder+'/'+subfolder+'/'+file,'Year':subfolder[-4:],'Quant':subfolder[-6:-5],'Element':folder})
    
    df = pd.DataFrame(dict_list)
    return(df)

def get_data(file):
    spect_list, first_num = parse_file(file)

    # Remove unwanted spectrum
    if len(spect_list)!=1:
        means = [np.average(spect) for spect in spect_list]
        spectrum = spect_list[means.index(max(means))]  #  Extract the one with the greatest signal (assuming one is always empty)

        # If it turns out both datasets are noise
        if means[means.index(max(means))]<2:
            # 3H background spectrum. Use first one
            spectrum=spect_list[0]
            return(spectrum.tolist())
    else:
        mean = np.average(spect_list[0])
        spectrum=spect_list[0]

    return(spectrum.tolist())

def quick_plot(spect_list,filename,first_num): # needs fixing
    """Quick plot spectrum/spectra.

    Args:
        spect_list (list): Python list containing numpy arrays of each spectrum
        filename (str): Relative or absolute path to .001 file
        first_num (int): First spectra number as displayed in the file header
        isotope (str): Isotope info derived from base directory. e.g. 36CL = Chlorine-36
    """
    plt.style.use('bmh')
    fig, axs = plt.subplots(len(spect_list))
    fig.suptitle(filename,fontsize='x-large',fontweight='bold')
    fig.set_figheight(10)
    fig.set_figwidth(15)

    if len(spect_list)==1:
        spect = spect_list[0]
        maximum = np.max(spect)
        minimum = np.min(spect)
        # Make sure ticks don't look silly
        if maximum<10:
            ticks = np.linspace(minimum,round(maximum, 0),11)
        else:   
            ticks = np.linspace(minimum,round(maximum, -1),11)

        axs.plot(range(len(spect)), spect)
        axs.set_yticks(ticks)
        axs.set_title('SP# '+str(first_num),fontsize='x-large')
        axs.set_xlabel("Channel")
        axs.set_ylabel("Counts")
        axs.margins(x=0,y=0)
    else:
        i = -1
        for spect in spect_list:
            i +=1

            maximum = np.max(spect)
            minimum = np.min(spect)
            if maximum<10:
                ticks = np.linspace(minimum,round(maximum, 0),11)
            else:   
                ticks = np.linspace(minimum,round(maximum, -1),11)

            axs[i].plot(range(len(spect)), spect)
            axs[i].set_yticks(ticks)
            axs[i].set_title('SP# '+str(first_num+i),fontsize='x-large')
            axs[i].set_xlabel("Channel")
            axs[i].set_ylabel("Counts")
            axs[i].margins(x=0,y=0)

    plt.show()

def list_files_recursive(path='.'):
    for entry in os.listdir(path):
        full_path = os.path.join(path, entry)
        if os.path.isdir(full_path):
            list_files_recursive(full_path)
        else:
            if full_path.endswith('.001') or full_path.endswith('.002') or full_path.endswith('.003'):
                spect_list,first_num = parse_file(full_path)
                quick_plot(spect_list,full_path,first_num)

### new stuff ###

def extract_registry(path):
    """Given path to registry.txt, extract the file names, spectrum type (e.g. BKG, CALIB, STD), isotope, CTIME, SQPs, SQP uncertainty, 

    Args:
        registry_path (str): path to registry.txt
    """
    with open(path+'/REGISTRY.txt', 'r') as file:
        data = file.read()  #.replace('\n', '')

    # Table
    table_start = data.index('\nORDER')
    table_end = data.index('\n  NUMBER OF CYCLES')
    table_str = data[table_start+1:table_end]
    lines = table_str.split("\n")
    # Extract column names from the first line
    columns = lines[0].split()

    data_rows = [line.split(maxsplit=len(columns)-1) for line in lines[1:]] # This probably won't work, need to come up with a better solution
    registry_df = pd.DataFrame(data_rows, columns=columns)

    # Printout
    printout_start = data.index('\nQ')
    printout_str = data[printout_start:]

    # Split the input data into lines
    lines = printout_str.splitlines()

    # Initialize list to hold rows of data (for dataframe)
    rows = []

    # Initialize variables to track current file info and cycle data
    line1 = None
    line2 = None
    line3 = None
    lines456 = []
    count = 0

    for line in lines:

        # Export data if counter = 3
        if count==3:
            try:
                row = line1 + line2 + line3 # Add first three lines
            except Exception as err:
                print(err,len(rows),'\n',line1,line2,line3,printout_str) #'\n',len(rows),'\n',line1,'\n',line2,'\n',line3)
                sys.exit()
            for _line in lines456:
                row += _line    # Add lines 4, 5, 6
            rows.append(row)    # Append to dataframe rows
            # Reset variables:
            line1 = None
            line2 = None
            line3 = None
            lines456 = []
            count = 0

        # Skip empty lines
        if not line:
            continue

        # Check if the line contains file info (FILENAME, DATE, TIME)
        if (line.startswith("Q") and len(line.split()) == 5):
            line1 = line.split()
            name = line1[0]
            line1[0] = path + '/' + name
            continue
        
        # Check if the line contains cycle data
        if (len(line.strip().split()) == 10) and (line.strip().split()[0].isdigit()):
            line2 = line.strip().split()
            continue
        
        # Check for line3 (if it has name of run the .split() below will =>7 )
        if (len(line.strip().split()) >= 7):
            line3 = line.strip().rsplit(maxsplit=6) #Split from the right to join up name of run as one entry
            continue

        # Check for line4-6
        if (len(line.strip().split()) == 6):
            count +=1
            lines456.append(line.strip().rsplit(maxsplit=5)) 

    # Final row
    row = line1 + line2 + line3 # Add first three lines
    for _line in lines456:
        row += _line    # Add lines 4, 5, 6
    rows.append(row)

    # Define the column headers
    columns = [
        'FILENAME', 'DAY','MONTH','YEAR', 'TIME', 'CYC', 'POS', 'REP', 'CTIME', 'DTIME1', 'DTIME2', 
        'CUCNTS', 'SQP', 'SQP%', 'STIME', 'ID', 'CPM1', 'COUNTS1', 'CPM1%', 'CPM2', 'COUNTS2', 'CPM2%',
        'CPM3', 'COUNTS3', 'CPM3%', 'CPM4', 'COUNTS4', 'CPM4%', 'CPM5', 'COUNTS5', 'CPM5%', 'CPM6', 
        'COUNTS6', 'CPM6%', 'CPM7', 'COUNTS7', 'CPM7%', 'CPM8', 'COUNTS8', 'CPM8%'
    ]
    # Convert the rows into a DataFrame
    printout_df = pd.DataFrame(rows, columns=columns)

    return(printout_df)

def dataframe(df):
    # Get data from filename column in df using get_data and parse. Returns same dataframe with spectra appended as a list in a new column 
    # Getting class distribution of dataset
    #print("\nFrom File DataFrame")
    #class_dist(df)

    # Creating data_df (that contains 1024 columns corresponding to each channel)
    #data_df = df.drop(["Year","Quant"], axis='columns')      # Remove unneccessary columns
    print("Retrieving spectra...")
    #data_df = pd.DataFrame()
    df['Spectra'] = df["FILENAME"].apply(get_data)    # Getting data from file using get_data() from parse
    #df.dropna(inplace=True)     # Removing rows with spectra = None
    #print(data_df.head())
    return(df)


def extract_spectra(folder):
    print("For folder: ",folder)
    # Get registry info for single folder
    registry_df = extract_registry(folder)
    # Dataframe containing spectra for that folder
    folder_df = dataframe(registry_df)

    # Remove rows that contain "NM" in the ID column
    folder_df = folder_df[~folder_df['ID'].str.contains('NM', case=False, na=False)]

    #print("FolderSpectra:\n",folder_df['Spectra'])

    # Get background spectrum
    bkg_row = folder_df[folder_df['ID'].str.contains('BLK|BLANK|BKG', case=False, na=False)]
    #print("Printing bkg_row (if empty, it has failed to locate bkg spectrum):\n",bkg_row)
    if bkg_row.empty:
        print("WARNING: ",folder," skipped due to unrecognisable / no bkg spectrum.")
        return(pd.DataFrame())    # SKIP ANY WHERE BACKGROUND SPECTRUM CANNOT BE FOUND (empty df)
    bkg_spectrum = bkg_row['Spectra'].values[0]
    # Subtract the background spectrum from the other spectra
    folder_df['Spectra'] = folder_df.apply(
        lambda row: [a - b for a, b in zip(row['Spectra'], bkg_spectrum)] if not any(substring in row['ID'].upper() for substring in ['BLK', 'BLANK', 'BKG']) else row['Spectra'],
        axis=1
)
    # Remove background and EMPTY spectra
    folder_df = folder_df[~folder_df['ID'].str.contains('BLK|BLANK|BKG|EMPTY', case=False, na=False)]

    # Add column containing isotope info
    #print("FOLDER DF FILENAME\n",folder_df['FILENAME'].to_string())
    try:
        folder_df['ISOTOPE'] = folder_df['FILENAME'].apply(lambda x: x.split('Quant ')[1].split('/')[1])
    except Exception as err:
        print(err)
        pd.set_option('display.max_colwidth', 1000)
        print(folder_df['FILENAME'])
        sys.exit()

    # Separate chanels
    cols = [*range(1,1025)]
    folder_df[cols] = pd.DataFrame(folder_df.copy().Spectra.tolist(),index=folder_df.index)  # Separate out the spectra columns
    folder_df.drop("Spectra", axis='columns',inplace=True)        # Remove the old column

    return(folder_df) # THIS IS A DF CONTAINING ALL SPECTRA IN ONE FOLDER. DO THIS FOR ALL FOLDERS, ADDING EACH DF TOGETHER

    
def class_dist(df): # Needs fixing
    """Prints the class distribution of dataframe

    Args:
        df (DataFrame): pandas DataFrame containing spectra with 'Element' information
    """
    # Getting class distribution of dataset
    class_sf = df.groupby(['Element']).size() 
    class_df =  pd.DataFrame({'Element':class_sf.index, 'Count':class_sf.values})
    print("Class distribution:\n",class_df,"\n")

def data_from_files(dir):
    """Iterates extract_spectra() over all valid files for training data. Returns dataframe containing header data and 1-1024 channel spectrum for each file.
    """
    
    #dict_list = []
    df = pd.DataFrame()
    folders = [f for f in os.listdir(dir)]# if os.path.isdir(f)]
    for folder in folders:
        if folder=='Unseen':
            continue
        else: #YYYY
            #subfolders = [f for f in os.listdir(dir+folder+'/')]
            subfolders = [f for f in os.listdir(dir + folder + '/') if os.path.isdir(os.path.join(dir, folder, f))] # Quant x
            for subfolder in subfolders:
                #elements = str(subfolder.split(',')[0]) + ',' + str(subfolder.split(',')[1]) 
                #subsubfolders = [f for f in os.listdir(dir+folder+'/'+subfolder+'/')]
                subsubfolders = [f for f in os.listdir(dir+folder+'/'+subfolder+'/') if os.path.isdir(os.path.join(dir, folder, subfolder, f))]
                for subsubfolder in subsubfolders:
                    df_to_append = extract_spectra(dir+folder+'/'+subfolder+'/'+subsubfolder)
                    if  df_to_append.empty:
                        print("SKIPPING...\n")
                        continue

                    # Get quant
                    quant_no = subfolder[-1]
                    df_to_append['QUANT'] = quant_no
                    if '3H' in df_to_append['ISOTOPE'].iloc[0]:
                        isotope='3H'
                    elif '14C' in df_to_append['ISOTOPE'].iloc[0]:
                        isotope='14C'
                    elif '36CL' in df_to_append['ISOTOPE'].iloc[0]:
                        isotope='Cl-36'
                    else:
                        print("Error - not able to determine isotope pattern")
                        sys.exit()

                    # Get calibration from .xls
                    for file in os.listdir(dir + folder):
                        if file.endswith(isotope+' Q-'+quant_no+'.xls') or (file.endswith(isotope+' Q-'+quant_no+'.xlsx')):
                            try:
                                calib_excel_df = pd.read_excel(dir+folder+'/'+file, sheet_name='Calculation',usecols="A:U", skiprows=24, nrows=len(df_to_append.index))
                                calib_excel_df.index = np.arange(1, len(calib_excel_df)+1)
                                print("Opening 'Calculation' worksheet in "+file)
                            except Exception as err:
                                print(err)
                                print('Error - calibration .xls(x) not found for '+dir+folder+'/'+subfolder+'/'+subsubfolder)
                                sys.exit()
                    #with pd.option_context('display.max_rows', None, 'display.max_columns', None):  
                        #print(calib_excel_df)
                    df_to_append['Activity [Bq]'] = calib_excel_df['A [Bq]'].values
                    df_to_append['Activity +/- (2σ)'] = calib_excel_df[' +/- (2σ)'].values
                    df_to_append['Counting efficiency [%]'] = calib_excel_df['Counting efficiency [%]'].values
                    df_to_append['Counting efficiency +/- (2σ)'] = calib_excel_df[' +/- (2σ).5'].values


                    # Check SQP matches
                    df_to_append_sqp = df_to_append['SQP'].copy()
                    df_to_append_sqp.index = np.arange(1, len(calib_excel_df)+1)
                    if not df_to_append_sqp.astype(np.float64).equals(calib_excel_df['SQPE']):
                        print(df_to_append['SQP'].astype(np.float64))
                        print(calib_excel_df['SQPE'])
                        print("Error - SQP in calibration does not match SQP in REGISTRY.TXT\n")
                        #sys.exit()
                        continue # Skip this folder

                    # Append to df
                    df = pd.concat([df, df_to_append], ignore_index=True)
                    print('\n')
    return(df)



def main():
    try:
        filename = sys.argv[1]
    except Exception as err:
        print('Error: missing argument. \n',err)
        sys.exit()

    
    if os.path.isdir(filename):     # Check if the path refers to a directory.
        # Print a message indicating that it is a directory.
        print("\nArg is a directory. Looping...")
        list_files_recursive(path=filename)

    elif os.path.isfile(filename):  # Check if the path refers to a regular file.
        # Print a message indicating that it is a normal file.
        print("\nArg is a file. Opening...")
        spect_list,first_num = parse_file(filename)
        quick_plot(spect_list,filename,first_num)


if __name__=='__main__':
    main()