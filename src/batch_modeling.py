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
import model_functions as mf


__author__ = 'Ricardo Pizarro'
__copyright__ = 'Copyright 2023, Introspection and decision-making (IDM) project'
__credits__ = ['Ricardo Pizarro, Silvia Lopez-Guzman']
__license__ = 'IDM_model 1.0'
__version__ = '0.1.0'
__maintainer__ = 'Ricardo Pizarro'
__email__ = 'ricardo.pizarro@nih.gov, silvia.lopezguzman@nih.gov'
__status__ = 'Dev'


def get_user_input():
	input_dir = input('What is the input directory? Where are the raw files located?\n')
	save_dir = input('What is the output directory? What is the path where we will save output files?\n')
	return input_dir,save_dir


def get_csv_files(input_dir = '/tmp/',verbose=False):
	# search under input_dir for raw .csv files
	raw_files = glob.glob(os.path.join(input_dir,'*.csv'))
	if verbose:
		print('Searched this match : {}'.format(os.path.join(input_dir,'*.csv')))
		print('I found the following files to be analyzed:')
		print(raw_files)
	return sorted(raw_files)


def check_for_files(fn='/tmp/',subj_dir='/tmp/',subject='subject1'):
	tasks = ['crdm','cdd','cpdm']
	files_there = True
	for task in tasks:
		task_dir = os.path.join(subj_dir,task)
		# this is the task specific file after splitting the raw file
		task_fn = os.path.join(task_dir,'{}_{}.csv'.format(subject,task))
		if not os.path.exists(task_fn):
			files_there = False
			return files_there
		# if 'cpdm' not in task:
		# 	# for cdd and crdm we check model fit eps figure and SV_hat.csv file
		# 	model_fit_fn = os.path.join(task_dir,'{}_{}_model_fit.eps'.format(subject,task))
		# 	SV_hat_fn = os.path.join(task_dir,'{}_{}_SV_hat.csv'.format(subject,task))
		# 	if not (os.path.exists(model_fit_fn) and os.path.exists(SV_hat_fn)):
		# 		files_there = False
		# 		return files_there
		# 	if 'cdd'in task:
		# 		# for cdd only check for model fit alpha eps figure and SV_hat_alpha.csv file
		# 		model_fit_alpha_fn = os.path.join(task_dir,'{}_{}_model_fit_alpha.eps'.format(subject,task))
		# 		SV_hat_alpha_fn = os.path.join(task_dir,'{}_{}_SV_hat_alpha.csv'.format(subject,task))
		# 		if not (os.path.exists(model_fit_alpha_fn) and os.path.exists(SV_hat_alpha_fn)):
		# 			files_there = False
		# 			return files_there
	return files_there

def already_split(raw_files = [],split_dir='/tmp/'):
	print('\n**NOTE** We found {} files in {}'.format(len(raw_files),os.path.dirname(raw_files[0])))
	print('We are checking one by one if these files have already been split\n')
	# If raw file is fully split, there should be 3 files
	not_fully_split = []
	for fn in raw_files:
		subject = mf.get_subject(fn,task='')
		subj_dir = os.path.join(split_dir,subject)
		files_there = check_for_files(fn=fn,subj_dir=subj_dir,subject=subject)
		if files_there:
			batch_name = os.path.basename(os.path.dirname(subj_dir))
			print('Subject split. To reanalyze, remove files from: /out_dir/{}/{}'.format(batch_name,subject))
		else:
			print('\n**NEW** Subject {} has not been split. We will now split.\n'.format(subject))
			not_fully_split = not_fully_split + [fn]

	return not_fully_split

def list_new_subjects(csv_files=[],task='crdm'):
	new_subjects = [os.path.basename(fn).replace('.csv','') for fn in csv_files]
	if task:
		new_subjects = [os.path.basename(fn).replace('_{}.csv'.format(task),'') for fn in csv_files]
	return new_subjects

def run_load_split_save(input_dir='/tmp/',save_dir='/tmp/'):
	# run part 1: search for .csv files under input_dir and put them in raw_files list
	print('Looking for raw .csv files in : {}'.format(input_dir))
	raw_files = get_csv_files(input_dir=input_dir)

	# save_dir is the root directory where we will save the results
	split_dir = os.path.join(save_dir,'split')
	print('The datasets will be split and stored under directory : {}'.format(split_dir))
	# make the updated split_dir if it does not exist
	make_dir(split_dir)

	# if a raw file was found, this list will not be empty
	if not raw_files:
		print('\n\n***ERROR***\nThe path to batch did not have any .csv files for analysis.\n\n')
		print('Check input path again and rerun script : {}'.format(input_dir))
		sys.exit()

	# check if any file in raw_files have already been split, saved, and modeled
	raw_files = already_split(raw_files=raw_files,split_dir=split_dir)
	new_subjects = list_new_subjects(csv_files=raw_files)

	# split each raw file and throw errors if not able to split them
	load_split_save(raw_files=raw_files,split_dir=split_dir)

	return split_dir,new_subjects



def check_model_files(task_fn='/tmp/',subj_dir='/tmp/',subject='subject1',task='crdm'):
	files_there = True
	utility_dir = os.path.join(subj_dir,task).replace('split','utility')
	# for cdd and crdm we check model fit eps figure and SV_hat.csv file
	model_fit_fn = os.path.join(utility_dir,'{}_{}_model_fit.eps'.format(subject,task))
	SV_hat_fn = os.path.join(utility_dir,'{}_{}_SV_hat.csv'.format(subject,task))
	if not (os.path.exists(model_fit_fn) and os.path.exists(SV_hat_fn)):
		files_there = False
		return files_there
	if 'cdd'in task:
		# for cdd only check for model fit alpha eps figure and SV_hat_alpha.csv file
		model_fit_alpha_fn = os.path.join(utility_dir,'{}_{}_model_fit_alpha.eps'.format(subject,task))
		SV_hat_alpha_fn = os.path.join(utility_dir,'{}_{}_SV_hat_alpha.csv'.format(subject,task))
		if not (os.path.exists(model_fit_alpha_fn) and os.path.exists(SV_hat_alpha_fn)):
			files_there = False
			return files_there
	return files_there


def already_model(task_csv_files = [],split_dir='/tmp/',task='crdm'):
	print('\n**NOTE** We found {} files in {}'.format(len(task_csv_files),
						   os.path.dirname(os.path.dirname(os.path.dirname(task_csv_files[0])))))
	print('We are checking one by one if these files have already been modeled\n')
	# If raw file is fully model, there should be 3 files
	not_fully_split = []
	for fn in task_csv_files:
		subject = mf.get_subject(fn,task=task)
		subj_dir = os.path.join(split_dir,subject)
		files_there = check_model_files(task_fn=fn,subj_dir=subj_dir,subject=subject,task=task)
		if files_there:
			batch_name = os.path.basename(os.path.dirname(subj_dir))
			print('Subject split. To reanalyze, remove files from: /out_dir/{}/{}'.format(batch_name,subject))
		else:
			print('\n**NEW** Subject {} has not been modeled. We will now model with CRDM.\n'.format(subject))
			not_fully_split = not_fully_split + [fn]

	return not_fully_split


def check_task_files(split_dir='/tmp/',task='crdm'):
	# run part 1: search for .csv files under input_dir and put them in raw_files list
	print('Looking for {} .csv files in : {}'.format(task,split_dir))
	search_dir = os.path.join(split_dir,'*/{}'.format(task))
	task_csv_files = get_csv_files(input_dir=search_dir)

	# if a raw file was found, this list will not be empty
	if not task_csv_files:
		print('\n\n***ERROR***\nThe path to batch did not have any .csv files for analysis.\n\n')
		print('Check input path again and rerun script : {}'.format(search_dir))
		sys.exit()

	# check if any file in raw_files have already been split, saved, and modeled
	task_csv_files = already_model(task_csv_files=task_csv_files,split_dir=split_dir,task='crdm')
	new_subjects = list_new_subjects(csv_files=task_csv_files)

	return split_dir,new_subjects


def run_model_CDD(split_dir='/tmp/',new_subjects=[],CRDM_counter=0):
	print('\n>>NO ALPHA<< : First step model CDD with alpha=1\n')
	load_estimate_CDD_save(split_dir=split_dir,new_subjects=new_subjects,use_alpha=False)

	if CRDM_counter==0:
		print('**WARNING** Zero CRDM files were modeled, we have no estimate for alpha. All done!')
		sys.exit()
	else:
		print('\n>>USE ALPHA<< : Second step model CDD with alpha estimated by CRDM\n')
		print('*NOTE* We will use alpha (risk parameter) for CDD estimated from the corresponding {} CRDM files'.format(CRDM_counter))
		load_estimate_CDD_save(split_dir=split_dir,new_subjects=new_subjects,use_alpha=True)


def main():

	print('\n\n===BATCH PROCESSING===\n\n')
	# get paths to directories from the user
	input_dir,save_dir = get_user_input()

	print('\nI. Raw files :: load, split, and save using BIDS format\n')
	split_dir,new_subjects = run_load_split_save(input_dir=input_dir,save_dir=save_dir)
	
	print('\nII. Model CRDM task :: estimate model, save fit plot, and save parameters \n')
	# model CRDM tasks, count how many files get modeled
	split_dir,new_subjects = check_task_files(split_dir=split_dir,task='crdm')
	CRDM_counter = load_estimate_CRDM_save(split_dir=split_dir,new_subjects=new_subjects)

	print('\nIII. Model CDD task :: estimate (with and without alpha), save fit plot, and save parameters \n')
	run_model_CDD(split_dir=split_dir,new_subjects=new_subjects,CRDM_counter=CRDM_counter)


if __name__ == "__main__":
	# main will be executed after running the script
    main()

