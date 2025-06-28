# Alex Corbett, University of Bristol, 2025

import tabulate

from sklearn.metrics import r2_score
from train import *
from parse import *

def test_models(LSCModels, pure_test_data):
    """
    Test the models on the pure test data.
    """
    print('Testing models...')
    for name, model in LSCModels.models.items():
        print(f'Testing model: {name}')
       
       # needs completing!! pure_test_data is not a dataframe
       # also are we testing the random forest SQP model or the least-squares deconvolution?

def main():

    # Need the following:
    # The 001 file: SP11, or 12, etc, the SQP of the sample and which instrument (Q-1, or Q-2, etc), date and time (plus calibrations for that date), any assumptions that some isotopes are definitely not present.
    # The instrument blank file taken just before
    # The counting time of both files

    # Maybe implement a config file in the future to set these parameters, so that the user can easily change them without editing the code

    # Unseen
    #date_of_spectrum =  # neccessary for finding the correct model to use (trained on the right calibration data)
    isotope_dict = {    # Set to true/false based on whether the relavent models cover the sqp of the unseen spectrum 
        '3H': True,    # In the future, this should be done automatically in case user sets all to True and certain isotopes are not supported at that SQP
        '14C': True,  
        '36CL': True,
        '55FE': True,  
        '63NI': True,  
        '129I': True,
    }
    year = 2023
    instrument = 1
    version = 1  # Version of the model, used to differentiate between different versions of the same model
    sqp = 725.24
    unseen_file = 'LSC Spectra for AI proj/Unseen/Q032301N_725.24_Q-1_2023_pure_I129_90min.001'
    unseen_counting_time = time_to_minutes('90:00.767') # Converts to mins (float)
    unseen_SP = 11  # Spectrum number in the file, e.g. SP11, SP12, etc. This is used to parse the file correctly
    background_file = 'LSC Spectra for AI proj/Unseen/Q022201N_blk_Q-1_2023_180min.001'  # Background spectrum file
    background_counting_time = time_to_minutes('180:00.768')
    background_SP = 11

    # Parse the files
    unseen_spect = parse_file(unseen_file,unseen=True,sp=unseen_SP)
    print(unseen_file + ' opened sucessfully')
    background_spect = parse_file(background_file,unseen=True,sp=background_SP)
    print(background_file + ' opened sucessfully')

    # Background spectrum processing
    background_spect = background_spect/background_counting_time  # Convert to counts per minute
    background_spect = SpectraSavGolTransformer().transform(background_spect, isnumpy=True)
    background_spect = np.clip(background_spect, a_min=0, a_max=None)  # Clip to ensure no negative values
    print('Savgol filter applied to background spectrum. Zeros clipped to avoid negative values.')

    # Unseen spectrum processing
    unseen_spect = unseen_spect/unseen_counting_time  # Convert to counts per minute
    unseen_spect = SpectraSavGolTransformer().transform(unseen_spect, isnumpy=True)
    unseen_spect = np.clip(unseen_spect, a_min=0, a_max=None)  # Clip to ensure no negative values
    print('Savgol filter applied to unseen spectrum. Zeros clipped to avoid negative values.')

    cols = [*range(1,1025)]
    plt.plot(cols,unseen_spect)
    plt.plot(cols,background_spect,color='red',zorder=10)
    plt.title('Unseen Spectrum Before Background Subtraction')
    plt.show()
    plt.close()

    unseen_spect = unseen_spect - background_spect  # Subtract background spectrum from unseen spectrum
    unseen_spect = np.clip(unseen_spect, a_min=0, a_max=None)  # Clip to ensure no negative values

    plt.plot(cols,unseen_spect)
    plt.title('Unseen Spectrum After Background Subtraction')
    plt.show()
    plt.close()


    # Unpickle the relavent LSCModels object
    model_dir = f'models/Q-{instrument}_{year}_v{version}'
    with open(f'{model_dir}/LSCModels.pkl', 'rb') as f:
        LSCModels = pickle.load(f)
    print(f'Models loaded from {model_dir}/LSCModels.pkl')

    # Print the file and settings info
    print(f'''
File and settings info:
Unseen spectrum: {unseen_file}
Unseen counting time: {unseen_counting_time} mins
Background spectrum: {background_file}
Background counting time: {background_counting_time} mins
SQP: {sqp}
Instrument: Q-{instrument}
Year: {year}
Model version: {version}
Isotopes selected: {isotope_dict}''')   # Imlement date of spectrum here in the future

    # Remove certain isotopes from the deconvolution
    model_dict = LSCModels.models
    for isotope_name, isotope_presence in isotope_dict.items():
        if not isotope_presence:
            model_dict.pop(isotope_name)
    # Should add an extra check here to ensure that the remaining isotope models cover the SQP of the unseen spectrum


    # Run the deconvolution
    print('\nRunning deconvolution...')
    cpms, isotope_shapes, counting_efficiencies = estimate_activities(
        unseen_spect, sqp, model_dict,
    )  
    dpms = cpms/counting_efficiencies
    print('\nDeconvolution results:')

    # Prepare table data
    true_isotopes = list(model_dict.keys())
    table = [
        ['CPM'] + [f'{cpm:.4f}' for cpm in cpms],
        ['Efficiency'] + [f'{eff:.4f}' for eff in counting_efficiencies],
        ['DPM'] + [f'{dpm:.4f}' for dpm in dpms],
    ]

    print(tabulate.tabulate(table, headers=[''] + true_isotopes, tablefmt='grid'))

    # Plotting the results
    isotope_colors = {
        '3H': 'green',
        '14C': 'blue',
        '36CL': 'orange',
        '55FE': 'brown',
        '63NI': 'red',
        '129I': 'purple',
    }
    tot= np.zeros(len(cols))
    for k, isotope_shape in enumerate(isotope_shapes):
        plt.plot(cols, isotope_shape.squeeze()*cpms[k],label=f'{true_isotopes[k]}',linewidth=0.8,color=isotope_colors[true_isotopes[k]],zorder=2)
        tot += isotope_shape.squeeze()*cpms[k]

    mse = np.mean((unseen_spect - tot) ** 2)
    mae = np.mean(np.abs(unseen_spect - tot))
    r2 = r2_score(unseen_spect, tot)
    print('\nOriginal vs Summed Spectrum Metrics:')
    print(f'MSE = {mse}')
    print(f'MAE = {mae}')
    print(f'R^2 = {r2}')

    # Should also add metrics here for performance of each individual isotope. 
    # Will need comprehensive test dataset: individual CPMs / DPMs, depending on if you want to test the deconvolution only or deconvolution + calibration curve   
    # Maybe CPM is best for this as it focuses on the deconvolution performance only, not the calibration curve performance which is already tested in the calibration spreadsheet

    plt.plot(cols, unseen_spect, label = 'Original',color='black',linestyle='dashed',linewidth=0.8,alpha=0.7,zorder=3)
    plt.plot(cols,tot,label='Summed',color='black',zorder=1)
    plt.ylabel('CPM')
    plt.legend()
    plt.show()
    plt.close()

if __name__=='__main__':
    main()