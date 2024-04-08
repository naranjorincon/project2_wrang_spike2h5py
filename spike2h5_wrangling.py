import os
import glob
import h5py
import numpy as np
import scipy.io as sio

# Set working directory
data_dir = '/Volumes/gaia.tavoni/Active/naranjorincon/JC_Plexon/'

# Load spike data sessions
data_files = glob.glob(os.path.join(data_dir, 'E*.mat'))

choose_session = 14
session_file = data_files[choose_session]
session_n = sio.loadmat(session_file)

# Define cell classes
jcplexon_classes = {'0': 'untuned', '+1': 'OVA+', '-1': 'OVA-', '+2': 'OVB+', '-2': 'OBV-', 
                    '+3': 'CV+', '-3': 'CV-', '+4': 'CJA', '-4': 'CJB'}

# Extract cell classes information
cells_info = sio.loadmat(data_dir + 'E221212c_cellana.mat')
get_cell_classes = np.array([cell[0][0] for cell in cells_info['cells'][0]['cellstats'][0]['cellclass'][0]])

if len(get_cell_classes) == len(jcplexon_classes):
    print('This session has all cell classes.')
else:
    print('This session has', len(get_cell_classes), 'cell classes.')

all_cells_sesn = session_n['cellData'][0]
print(all_cells_sesn)
good_trials = session_n['goodTrials'][:, 0]

spikes_across_trials = []
for trial_idx in range(len(good_trials)):
    choose_trial = good_trials[trial_idx]

    # Initialize max_time for each trial
    max_time = 0
    
    # Iterate over each cell struct
    for cell_struct in all_cells_sesn:
        cell_data = cell_struct[1]  # Get the spike data
        cell_trial_data = cell_data[cell_data[:, 1] == choose_trial]  # Filter data for the chosen trial
        if len(cell_trial_data) > 0:
            max_time_cell = np.max(cell_trial_data[:, 0])  # Get max time for this cell on this trial
            if max_time_cell > max_time:
                max_time = max_time_cell

    # Add the max time for this trial to the list
    spikes_across_trials.append(max_time)

# Find the overall maximum time across all trials and cells
overall_max_time = max(spikes_across_trials)
print("Overall maximum time across all trials and cells:", overall_max_time)

# Split the data into train, validate, and test sets
trainRatio = 0.8
validRatio = 0.1
testRatio = 0.1

totalSize = len(spikes_across_trials)
trainSize = round(trainRatio * totalSize)
validSize = round(validRatio * totalSize)
testSize = totalSize - trainSize - validSize

randIndices = np.random.permutation(totalSize)

trainData = np.array([spikes_across_trials[i] for i in randIndices[:trainSize]])
validData = np.array([spikes_across_trials[i] for i in randIndices[trainSize:trainSize+validSize]])
testData = np.array([spikes_across_trials[i] for i in randIndices[trainSize+validSize:]])

# Save the setup to HDF5 file
filename = 'cut_full_setup_lfads_testProject2_wrang.h5'
print(filename)
h5_loc = '/Volumes/gaia.tavoni/Active/naranjorincon/temp_dynamics_VAE-RNN/autolfads-tf2/datasets'
print(h5_loc)

fileWithPath = os.path.join(h5_loc, filename)

with h5py.File(fileWithPath, 'w') as hf:
    hf.create_dataset('train_data', data=trainData)
    hf.create_dataset('train_inds', data=randIndices[:trainSize])
    hf.create_dataset('train_truth', data=trainData)
    hf.create_dataset('valid_data', data=validData)
    hf.create_dataset('valid_inds', data=randIndices[trainSize:trainSize+validSize])
    hf.create_dataset('valid_truth', data=validData)
