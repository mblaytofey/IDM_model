import pandas as pd
import os,sys
import glob

def get_header():
    header = ['participant','date','expName','psychopyVersion','OS','frameRate','idm_task','item']
    return header

def get_by_task(df,task):
    # select rows if idm_task is one of the following tasks: 'crdm','cdd','cpdm'
    idf = df[df['idm_task'].str.contains(task)]
    # select columns with task name in it, such as cdd_trial_type, etc.
    header = get_header()
    cols = header + [c for c in list(df) if task in c]
    try:
        idf = idf[idf.columns.intersection(cols)]
    except Exception as err :
        print('Unexpected {err=}, type(err)=')
    return idf

def split_by_task(df):
    # drop empty rows intentionally structured to separate tasks
    df = df[df['idm_task'].notna()]
    # print(list(df))
    crdm_df = get_by_task(df,'crdm')
    cdd_df = get_by_task(df,'cdd')
    cpdm_df = get_by_task(df,'cpdm')
    return crdm_df,cdd_df,cpdm_df

def make_dir(this_dir,verbose=False):
    if not os.path.exists(this_dir):
        if verbose:
            print('Creating: {}'.format(this_dir))
        os.makedirs(this_dir)

def save_df(save_dir,fn_OG,df,task='crdm',verbose=False):
    idm_subj = os.path.basename(fn_OG).replace('.csv','')
    subj_dir = os.path.join(save_dir,idm_subj)
    make_dir(subj_dir)
    task_dir = os.path.join(subj_dir,task)
    make_dir(task_dir)
    
    if df.empty:
        print('**NullData** Selection returned null, check data and try again')
    elif df.shape[0] < 10:
        print('**InsufficientData** Number of rows less than 10, data for {} task is not usable'.format(task.upper()))
    else:
        fn = os.path.join(task_dir,'{}_{}.csv'.format(idm_subj,task))
        if verbose:
            print('Saving to: {}'.format(fn))
        df.to_csv(fn)


def load_split_save(raw_files = [],split_dir = '/tmp/'):
    counter,index = 0,0
    for index, fn in enumerate(raw_files):
        if os.path.exists(fn):
            print('We will split and save the following csv file : \n{}'.format(fn))
        else:
            print('**FileNotFound** Will move on as we could not find: {}'.format(fn))
            continue
        try:
            df = pd.read_csv(fn,index_col=0)
        except:
            print('**WARNING** Some error continued reading file ...  will move on')
            continue

        if ('23_IDM' in fn) and ((df.shape[0]!=1044) or (df.shape[1] not in [75,76,77])):
            print('**DataShape** Not right number of columns, shape {}'.format(df.shape))
            continue

        crdm_df,cdd_df,cpdm_df = split_by_task(df)
        
        save_df(split_dir,fn,crdm_df,task='crdm')
        save_df(split_dir,fn,cdd_df,task='cdd')
        save_df(split_dir,fn,cpdm_df,task='cpdm')

        counter+=1

    total_split=True

    if counter<index:
        # We did not split all raw files, have to check
        print('**WARNING** For some reason we only split {} of {} files, please check the log files'.format(counter,index))
        total_split=False
    elif counter+index==0:
        print('\n**NOTE** We did not split any new subjects, all subjects have been split and stored\n')
    elif counter==0:
        print('\n\n***ERROR***\nSomehow we could not split the csv files, inspect the files and try again.\n\n')
        sys.exit()
        

def main():
    # Manually define directories here, then call load_split_save()
    data_dir = '/Users/pizarror/mturk/idm_data/'

    raw_files = glob.glob(os.path.join(data_dir,'raw','*.csv'))
    save_dir = os.path.join(data_dir,'split')

    load_split_save(raw_files,save_dir) 



if __name__ == "__main__":
    main()









