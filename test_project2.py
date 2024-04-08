import unittest
import h5py
import numpy as np
import scipy.io as sio
import os
import glob

class TestDataWrangling(unittest.TestCase):

    def setUp(self):
        # Load the original MATLAB data
        data_dir = '/Volumes/gaia.tavoni/Active/naranjorincon/JC_Plexon/'

        # Load spike data sessions
        data_files = glob.glob(os.path.join(data_dir, 'E*.mat'))

        choose_session = 14
        session_file = data_files[choose_session]
        mat_data = sio.loadmat(session_file)
        self.session_n = mat_data
        
        # Load the converted HDF5 data
        filename = 'cut_full_setup_lfads_testProject2_wrang.h5'
        h5_loc = '/Volumes/gaia.tavoni/Active/naranjorincon/temp_dynamics_VAE-RNN/autolfads-tf2/datasets'
        fileWithPath = os.path.join(h5_loc, filename)
        self.h5_data = h5py.File(fileWithPath, 'r')

        self.download_dir = fileWithPath
        if not os.path.exists(self.download_dir):
            os.makedirs(self.download_dir)

    def test_data_shapes(self):
        # Test if the shapes of the original data and the converted data match
        original_shapes = self.session_n['cellData'].shape
        converted_shapes = (len(self.h5_data), *self.h5_data['Train_Data'].shape)
        self.assertEqual(original_shapes, converted_shapes)


    def test_data_values(self):
        # Test if the values of the original data and the converted data match
        original_data = self.session_n.flatten()
        converted_data = [dataset['Data'][()] for dataset in self.h5_data['Datasets']]
        for original, converted in zip(original_data, converted_data):
            np.testing.assert_array_equal(original, converted)

    def test_trial_numbers(self):
        # Test if the trial numbers in the converted data match the original data
        original_trials = self.session_n['goodTrials'][:, 0]
        converted_trials = np.array([dataset['Data'][()] for dataset in self.h5_data['Datasets'] if dataset.name.endswith('_inds')]).flatten()
        np.testing.assert_array_equal(original_trials, converted_trials)

    def test_data_types(self):
        # Test if the data types of the converted data are as expected
        expected_dtype = np.int64
        for dataset in self.h5_data['Datasets']:
            if dataset.name.endswith('_inds'):
                self.assertEqual(dataset['Data'].dtype, expected_dtype)

    def test_data_range(self):
        # Test if the values of the converted data are within the expected range
        expected_min = 0
        expected_max = 1000
        for dataset in self.h5_data['Datasets']:
            if dataset.name.endswith('_data') or dataset.name.endswith('_truth'):
                data = dataset['Data'][()]
                self.assertTrue(np.all(data >= expected_min) and np.all(data <= expected_max))

    def test_dataset_names(self):
        # Test if the dataset names in the converted data are as expected
        expected_names = ['Original_Data', 'Train_Data', 'Train_Inds', 'Train_Truth', 'Valid_Data', 'Valid_Inds', 'Valid_Truth']
        actual_names = [dataset.name.split('/')[-1] for dataset in self.h5_data['Datasets']]
        self.assertListEqual(expected_names, actual_names)

    def test_download_data(self):
        # Test if downloading the data from the dataset location gives the correct data
        # Replace 'dataset_location' with the actual location of the dataset
        # This assumes that the HDF5 file is named 'converted_data.h5' in the dataset location
        dataset_location = '/Volumes/gaia.tavoni/Active/naranjorincon/temp_dynamics_VAE-RNN/autolfads-tf2/datasets'
        os.system(f'cp {dataset_location}/cut_full_setup_lfads_testProject2_wrang.h5 {self.download_dir}/')
        downloaded_data = h5py.File(f'{self.download_dir}/cut_full_setup_lfads_testProject2_wrang.h5', 'r')
        self.assertTrue(np.array_equal(self.h5_data['Datasets']['Train_Data'][()], downloaded_data['Datasets']['Train_Data'][()]))
        self.assertTrue(np.array_equal(self.h5_data['Datasets']['Valid_Data'][()], downloaded_data['Datasets']['Valid_Data'][()]))
        # Add more checks as needed for other datasets

    def test_download_data_existence(self):
        # Test if the downloaded data files exist
        dataset_location = '/Volumes/gaia.tavoni/Active/naranjorincon/temp_dynamics_VAE-RNN/autolfads-tf2/datasets'
        os.system(f'cp {dataset_location}/cut_full_setup_lfads_testProject2_wrang.h5 {self.download_dir}/')
        downloaded_files = os.listdir(self.download_dir)
        self.assertIn('cut_full_setup_lfads_testProject2_wrang.h5', downloaded_files)    

    def tearDown(self):
        self.h5_data.close()

if __name__ == '__main__':
    unittest.main()
