import pandas as pd



def get_user_dir():
	input_dir = input('What is the input directory, where the raw files are located?\n')
	output_dir = input('What is the output directory, where the results will be saved?\n')
	return input_dir,output_dir


def main():
	input_dir,output_dir = get_user_dir()
	print('You want me to start in : {}'.format(input_dir))
	print('and then you want me to write to : {}'.format(output_dir))

if __name__ == "__main__":
    main()



