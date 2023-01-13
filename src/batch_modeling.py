#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
--------------------------------------------------------------------------------------

### Batch Modeling ###

This script runs the modeling estimates on a batch of .csv files. The models are the 
confidence risky decision making (CRDM) and confidence delayed discounting (CDD) from 
the Introspection and decision-making (IDM) project. The batch of files are located 
on a user-specified input_dir and the results are saved in a user-specified save_dir. 

Inputs: 
  	user-specified paths to input_dir and save_dir. The batch of raw .csv files is 
  		located in the input_dir: /input_dir/subject{1..N}.csv

Outputs: 
  	task-specific .csv files: the script splits each raw .csv file into three .csv 
  		files, one for each task: CRDM, CDD, and confidence perceptual decision making
  		(CPDM). The three .csv files are written under the save_dir using the BIDS 
  		format: /save_dir/batch_name/subject/task/subject_task.csv
	plot: the probability of choice and actual user-made decision are plotted as a 
		function of subjective value (SV, or utility) difference, defined as
		SV_delta = SV_option-SV_alternative
	modeling results: for each model, a summary .csv file is saved summarizing model
		with goodness of fit (GOF) measures, along with paths to plots

Usage: $ python batch_modeling.py

--------------------------------------------------------------------------------------
"""

# Built-in/Generic Imports
import os,sys
import glob

# Libs
import pandas as pd

# Own modules
from idm_split_data import make_dir,load_split_save
from idm_plot_CRDM_response import load_estimate_CRDM_save
from idm_plot_CDD_response import load_estimate_CDD_save


__author__ = 'Ricardo Pizarro'
__copyright__ = 'Copyright 2023, Introspection and decision-making (IDM) project'
__credits__ = ['Ricardo Pizarro, Silvia Lopez-Guzman']
__license__ = 'IDM_model 1.0'
__version__ = '0.1.0'
__maintainer__ = 'Ricardo Pizarro'
__email__ = 'ricardo.pizarro@nih.gov, silvia.lopezguzman@nih.gov'
__status__ = 'Dev'


def get_user_dir():
	input_dir = input('What is the input directory? Where are the raw files located?\n')
	save_dir = input('What is the output directory? Where will the results be saved?\n')
	return input_dir,save_dir


def get_raw_files(input_dir):
	# search under input_dir for raw .csv files
	raw_files = glob.glob(os.path.join(input_dir,'*.csv'))
	print('I found the following files to be analyzed:')
	print(raw_files)
	return raw_files

def main():
	# get paths to directories from the user
	input_dir,save_dir = get_user_dir()

	# search for .csv files under input_dir and store them as a list under raw_files
	print('I will look for raw .csv files in : {}'.format(input_dir))
	raw_files = get_raw_files(input_dir)

	# save_dir is the root directory where we will save the results
	print('The results will be saved under directory : {}'.format(save_dir))
	# save_dir gets updated to include batch name, taken from inoput_dir
	save_dir = os.path.join(save_dir,os.path.basename(input_dir))
	print('with batch located in : {}'.format(save_dir))
	# make the updated save_dir if it does not exist
	make_dir(save_dir)

	# if a raw file was found, this list will not be empty
	if not raw_files:
		# split each raw file, model each task
		load_split_save(raw_files,save_dir)
		load_estimate_CRDM_save(split_dir = save_dir)
		load_estimate_CDD_save(split_dir = save_dir)
	else:
		print('The path to batch did not have any .csv files for analysis. ')
		print('Check again : {}'.format(input_dir))


if __name__ == "__main__":
	# main will be executed after running the script
    main()



