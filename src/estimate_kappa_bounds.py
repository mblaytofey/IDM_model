import pandas as pd
import os,sys
from idm_split_data import make_dir

def main():
	# take an arbitrary subject for example
	# '/Users/pizarror/mturk/idm_data/split/idm_2022-12-08_12h53.16.781/cdd/idm_2022-12-08_12h53.16.781_cdd.csv'
	CDD_fn = input('Please enter the path to a CDD file:\n')
	if not os.path.exists(CDD_fn):
		print('Could not find this path, please try again')
		sys.exit()

	df = pd.read_csv(CDD_fn)
	print(df)
	print(list(df))

	# indifference point for each possible set of task values, set SV_now = SV_later to find kappa
	df['kappa'] = ( df['cdd_delay_amt']/df['cdd_immed_amt'] - 1.0 ) / df['cdd_delay_wait']
	df = df.sort_values(by=['kappa'])
	df_kappa = df[['kappa','cdd_immed_amt', 'cdd_immed_wait', 'cdd_delay_amt', 'cdd_delay_wait']]
	print(df_kappa)

	# '/Users/pizarror/mturk/idm_data/kappa_values.csv'
	fn = input('Please enter the path where to write kappa estimates:\n')
	make_dir(os.path.dirname(fn))
	print('Saving kappa value to : {}'.format(fn))
	df_kappa.to_csv(fn)



if __name__ == "__main__":
	# main will be executed after running the script
    main()
