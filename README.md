## IDM_model : for CDD and CRDM tasks

The scripts located in the [src](src) directory, were written to sequentially do the following two items: 
1. Split the data collected in the IDM project into corresponding directories, using [BIDS](https://bids.neuroimaging.io/) format.
2. Using the files saved in the [BIDS](https://bids.neuroimaging.io/) format to analyze CRDM and CDD data by estimating the utility models. 

** *update* **

We adapted the code to separate the two items above so we could analyze the CRDM and CDD data independently. This requires that the datafiles are stored in the appropriate [BIDS](https://bids.neuroimaging.io/) format.

### Getting started

If you are unfamiliar with Python and conda environments, please refer to our [CPU_support](https://github.com/CDN-Lab/CPU_support) repository. There you will find more details to help you get started with Python.

## 1. Clone the repository:

- Git clone IDM_model: Inside the IDM_project directory, clone the IDM_model repository 

    `$ git clone https://github.com/CDN-Lab/IDM_model `

    `$ cd IDM_model/src`

## 2. Run batch modeling:

- Execute python script batch_modeling.py. Run the script from the Terminal and follow the steps 

    `$ python batch_modeling.py`

> Enter the path (or directory) where the raw files are located that you would like to analyze (no quotes).  

> Enter the path (or directory) where the output files will be saved (or written) 

The script will output where the files get saved, then you can open the files in Finder or directly from Terminal, if you're feeling adventurous. 

 
## 3. Run CRDM or CDD independently:

- For CRDM, execute python script [idm_plot_CRDM_response.py](src/idm_plot_CRDM_response.py)

    `$ python idm_plot_CRDM_response.py`

- For CDD, execute python script [idm_plot_CDD_response.py](src/idm_plot_CDD_response.py)

    `$ python idm_plot_CDD_response.py`

> Enter the path (or directory) where the split files are located that you would like to analyze (no quotes).  

The script will output where the files get saved, then you can open the files in Finder or directly from Terminal, if you're feeling adventurous. 
