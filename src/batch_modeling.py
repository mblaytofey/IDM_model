import pandas as pd
import glob
import os,sys
from idm_split_data import make_dir,load_split_save
from idm_plot_CRDM_response import load_estimate_CRDM_save
from idm_plot_CDD_response import load_estimate_CDD_save

def get_user_dir():
	input_dir = input('What is the input directory, where the raw files are located?\n')
	save_dir = input('What is the output directory, where the results will be saved?\n')
	return input_dir,save_dir


def get_raw_files(input_dir):
	raw_files = glob.glob(os.path.join(input_dir,'*.csv'))
	print('I found the following files to be analyzed:')
	print(raw_files)
	return raw_files

def main():
	input_dir,save_dir = get_user_dir()
	print('I will look for raw .csv files in : {}'.format(input_dir))
	raw_files = get_raw_files(input_dir)

	print('The results will be saved under directory : {}'.format(save_dir))
	save_dir = os.path.join(save_dir,os.path.basename(input_dir))
	print('with batch located in : {}'.format(save_dir))
	make_dir(save_dir)

	load_split_save(raw_files,save_dir)
	load_estimate_CRDM_save(split_dir = save_dir)
	load_estimate_CDD_save(split_dir = save_dir)


if __name__ == "__main__":
    main()



