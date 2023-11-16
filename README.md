## IDM_model : for CDD and CRDM tasks

The scripts located in the [src](src) directory, were written to sequentially do the following two items: 
1. Split the data collected in the IDM project into corresponding directories, using [BIDS](https://bids.neuroimaging.io/) format.
2. Using the files saved in the [BIDS](https://bids.neuroimaging.io/) format to analyze CRDM and CDD data by estimating the utility models. 

** *update* **

We adapted the code to separate the two items above so we could analyze the CRDM and CDD data independently. This requires that the datafiles are stored in the appropriate [BIDS](https://bids.neuroimaging.io/) format.

### Getting started

If you are unfamiliar with Python and conda environments, please refer to our [CPU_support](https://github.com/CDN-Lab/CPU_support) repository. There you will find more details to help you get started with Python.

## 1a. Clone the repository:

- Git clone IDM_model: Use the terminal and navigate to the IDM folder in the home directory and clone the IDM_model repository

    `$ cd ~/IDM`

    `$ git clone https://github.com/CDN-Lab/IDM_model `

    `$ cd IDM_model/src`

## 1b. Git pull the repository:

- If you have already cloned the repository, you should navigate to the folder and run a git pull to refresh any changes on the script. Read the output to see if any cahnges have taken place or not.

    `$ cd ~/IDM/IDM_model/`

    `$ git pull`


## 2a. Run CRDM or CDD independently:

- For CRDM, execute python script [idm_plot_CRDM_response.py](src/idm_plot_CRDM_response.py)

    `$ python idm_plot_CRDM_response.py`

- For CDD, execute python script [idm_plot_CDD_response.py](src/idm_plot_CDD_response.py)

    `$ python idm_plot_CDD_response.py`

> Enter the path (or directory) where the split files are located that you would like to analyze (no quotes).  

The script will output where the files get saved, then you can open the files in Finder or directly from Terminal, if you're feeling adventurous. 

## 2b. Run batch modeling:

**NOTE** :: Batch modeling was written to analyze the IDM project recorded online. If you have the CRDM and CDD task files already saved as separate spreadsheets (.csv) use the two functions in Step 2a above.

- Execute python script batch_modeling.py. Run the script from the Terminal and follow the steps 

    `$ python batch_modeling.py`

> Enter the path (or directory) where the raw files are located that you would like to analyze (no quotes).  

> Enter the path (or directory) where the output files will be saved (or written) 

The script will output where the files get saved, then you can open the files in Finder or directly from Terminal, if you're feeling adventurous. 

 

