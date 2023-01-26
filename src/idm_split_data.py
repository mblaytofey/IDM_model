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
    idf = idf[idf.columns.intersection(cols)]
    return idf

def split_by_task(df):
    df = df[df['idm_task'].notna()]
    # print(list(df))
    crdm_df = get_by_task(df,'crdm')
    cdd_df = get_by_task(df,'cdd')
    cpdm_df = get_by_task(df,'cpdm')
    return crdm_df,cdd_df,cpdm_df

def make_dir(this_dir):
    if not os.path.exists(this_dir):
        print('Creating: {}'.format(this_dir))
        os.makedirs(this_dir)

def save_df(save_dir,fn_OG,df,task):
    idm_subj = os.path.basename(fn_OG).replace('.csv','')
    subj_dir = os.path.join(save_dir,idm_subj)
    make_dir(subj_dir)
    task_dir = os.path.join(subj_dir,task)
    make_dir(task_dir)
    
    if df.empty:
        print('Selection returned null, check data and try again')
    else:
        fn = os.path.join(task_dir,'{}_{}.csv'.format(idm_subj,task))
        print('Saving to: {}'.format(fn))
        df.to_csv(fn)


def load_split_save(raw_files = [],save_dir = '/tmp/'):
    counter = 0
    for index, fn in enumerate(raw_files):
        if os.path.exists(fn):
            print(fn)
        else:
            print('Will move on as we could not find: {}'.format(fn))
            continue
        try:
            df = pd.read_csv(fn,index_col=0)
        except:
            print('Some error continued reading file ...  will move on')
            continue

        if df.shape[1]<100:
            print('Not right number of columns, shape {}'.format(df.shape))
            continue
        crdm_df,cdd_df,cpdm_df = split_by_task(df)
        
        save_df(save_dir,fn,crdm_df,'crdm')
        save_df(save_dir,fn,cdd_df,'cdd')
        save_df(save_dir,fn,cpdm_df,'cpdm')

        counter+=1

    success=True
    if counter<index:
        # We did not split all raw files, have to check  
        success=False
    return success,counter,index
        

def main():
    # Manually define directories here, then call load_split_save()
    data_dir = '/Users/pizarror/mturk/idm_data/'

    raw_files = glob.glob(os.path.join(data_dir,'raw','*.csv'))
    save_dir = os.path.join(data_dir,'split')

    load_split_save(raw_files,save_dir) 



if __name__ == "__main__":
    main()









