"""
This script creates the Ctrl/*_cores.csv and Ctrl/*_samples.csv files for a cohort's Control TMA sample
"""

#imports
from asyncio import coroutines
import pathlib
from argparse import ArgumentParser
from ...shared.samplemetadata import ControlTMASampleDef
from ...utilities.tableio import writetable

import glob 
import os
import pandas as pd
import numpy as np
import re # for regular expressions
from datetime import datetime

#constants
DEFAULT_TMA_TISSUE_TYPES_FILE = '//bki04/astropath_processing/batch_correction/TMA_tissuetypes.xlsx'
OUTDIR = 0

# 
def getProjectRoot(projectNumber):
    # Gives the folder path of the project
    # ------------------------------------- # 

    dataLoc = '//BKI04'
    cohortsProgressFile = 'astropath_processing/AstropathCohortsProgress.csv'
    projectsList = pd.read_csv(os.path.join(dataLoc,cohortsProgressFile))
    
    if (projectNumber==-1):
        print('cohortsProgressfile: '+dataLoc+'/'+cohortsProgressFile)
        projectData = projectsList.copy()
        projectRoot = ''
    else:
        # Description of the project:
        projectData = projectsList.loc[projectsList.Project==projectNumber,:].reset_index(drop=True)

        # TMA data location of the project:
        projectRoot = os.path.join('\\\\',projectData.Dpath[0], projectData.Dname[0])

    return projectRoot, projectData

#
def getControlTMApath(projectRoot,CtrlNo,controlSamples):
    # Get .tif and .im3 paths of a specific control TMA
    # ------------------------------------- #

    folders = controlSamples[controlSamples['Ctrl']==CtrlNo][['SlideID','Scan']]

    folderTIFF = os.path.join(projectRoot,folders.iloc[0]['SlideID'],'inform_data\Component_Tiffs')
    folderIM3 = os.path.join(projectRoot,folders.iloc[0]['SlideID'],'im3',folders.iloc[0]['Scan'],'MSI')

    return folderTIFF, folderIM3

#
def getFile_ControlTMAinfo(projectNumber, outputpath):
    # TMAinfo -- create: 
    # Control_TMA_info for Tonsil and Spleen only and
    # Control_TMA_info_all for all tissue types
    # stored in a dictionary
    # ------------------------------------- # 

    projectFolder, projectDescription = getProjectRoot(projectNumber)
    controlTMAcollectioFile = DEFAULT_TMA_TISSUE_TYPES_FILE
    tmaDictionary = {}
    
    # Write the Ctrl files either under the project folder, 
    # or under a given optput folder:
    if outputpath !=0 :
        ctrlFolder = os.path.join(outputpath,'Ctrl')
    else:
        ctrlFolder = os.path.join(projectFolder,'Ctrl')

    # Check if Ctrl folder exists:
    if os.path.exists(ctrlFolder) is False:
        os.mkdir(ctrlFolder)
        print('Ctrl folder crerated for '+projectFolder +
            '\nin folder '+str(ctrlFolder))
        #return 

    fBegin = 'Control_TMA_'
    projectFolder = projectFolder.replace('\\','/')+'/'
    
    # list of control TMA folders (= SlideID)
    llist = pd.DataFrame( {'SlideID':glob.glob( projectFolder+fBegin+'*', recursive=False )} )

    if llist.shape[0]==0:
        print('Control TMAs not available for '+projectFolder)
        # If not exist, create it:
        return 

    llist.SlideID = llist.SlideID.apply(lambda ttext: ttext.replace('\\','/') )
    llist.SlideID = llist.SlideID.apply(lambda ttext: ttext.replace(projectFolder,'') )

    TMA = pd.DataFrame({'TMA':llist.SlideID.apply(lambda ttext: 
        re.search(fBegin+'(\d{4})_', ttext).group(1) ) })
    TMA = TMA.TMA.unique()

    # Store TMA data in dictionary, and write out afterwards
    # TMA in different file! (easier to follow if exist or not) 
    # print(TMA)

    # Loop to check if all TMA type has its _info.xlsx
    for tt in range(0,len(TMA),1):
    
        # If Control_TMA_info file is not available, read the core map from file:
        if ( (os.path.isfile(os.path.join(ctrlFolder,'Control_TMA'+TMA[tt]+'_info.xlsx')) is False) 
            or (os.path.isfile(os.path.join(ctrlFolder,'Control_TMA'+TMA[tt]+'_all_info.xlsx')) is False) ):
            datalist = pd.read_excel(controlTMAcollectioFile,sheet_name='TMA'+TMA[tt])
            
        if os.path.isfile(os.path.join(ctrlFolder,'Control_TMA'+TMA[tt]+'_info.xlsx')) is False:
            tmaDictionary['TMA'+TMA[tt]] = datalist[['Tonsil','Spleen']]

        if os.path.isfile(os.path.join(ctrlFolder,'Control_TMA'+TMA[tt]+'_all_info.xlsx')) is False:
            tmaDictionary['TMA'+TMA[tt]+'_all'] = datalist

    return tmaDictionary, ctrlFolder

#
def getList_ctrlsamples(projectNumber):
    # create content of *ProjectFolder* \ Ctrl \ *_ctrlsamples.csv
    # Uses *ProjectFolder* \ Control_TMA_* \ im3 \ Scan* \ BatchID.txt
    # ------------------------------------- # 

    projectFolder, projectDescription = getProjectRoot(projectNumber)
    
    fBegin = 'Control_TMA_'
    projectFolder = projectFolder.replace('\\','/')+'/'
    
    # list of control TMA folders (= SlideID)
    llist = pd.DataFrame( {'SlideID':glob.glob( projectFolder+fBegin+'*', recursive=False )} )
    
    if llist.shape[0]==0:
        ctrlsamples_all = pd.DataFrame( {'SlideID':glob.glob( projectFolder+'Control_'+'*', recursive=False )} )
        ctrlsamples = pd.DataFrame({})
        
        return ctrlsamples, ctrlsamples_all

    llist.SlideID = llist.SlideID.apply(lambda ttext: ttext.replace('\\','/') )
    llist.SlideID = llist.SlideID.apply(lambda ttext: ttext.replace(projectFolder,'') )
    
    ctrlsamples_obj = [] 
    ctrlsamples = pd.DataFrame({})
    ctrlsamples['Project'] = np.arange(0,len(llist),1)*0 + projectDescription.Project[0]
    ctrlsamples['Cohort'] = np.arange(0,len(llist),1)*0 + projectDescription.Cohort[0]
    ctrlsamples['CtrlID'] = np.arange(1,len(llist)+1,1)
        # Matlab, getCtrlInfo() -- 'CtrlID' -- cid = (1:numel(tma));
    
    ctrlsamples['TMA'] = llist.SlideID.apply(lambda ttext: 
        re.search(fBegin+'(\d{4})_', ttext).group(1) )
    ctrlsamples['Ctrl'] = llist.SlideID
    ctrlsamples['Ctrl'] = ctrlsamples.apply(lambda df: 
        re.search(fBegin+df['TMA']+'_(\d{1,3})_', df['Ctrl']).group(1), axis=1 )
    ctrlsamples['Date'] = llist.SlideID
    ctrlsamples['Date'] = ctrlsamples.apply(lambda df: 
        df['Date'].replace(fBegin+df['TMA']+'_'+df['Ctrl']+'_', '' ), axis=1 )
    
    ctrlsamples['TMA'] = ctrlsamples['TMA'].apply(lambda ttext: int(ttext) )
    ctrlsamples['Ctrl'] = ctrlsamples['Ctrl'].apply(lambda ttext: int(ttext) )

    ctrlsamples['BatchID'] = np.arange(0,len(llist),1)*0 +-1
    ctrlsamples['Scan'] = ['']*ctrlsamples.shape[0]
    
    ctrlsamples['SlideID'] = llist.SlideID
    ctrlsamples['TMA_imageType'] = ['']*ctrlsamples.shape[0]
    ctrlsamples['ComponentTIFF_No'] = ['']*ctrlsamples.shape[0]
    
    for ii in range(0,ctrlsamples.shape[0],1):

        # Identify Scan number
        # Matlab, getScan()
        path = projectFolder+ctrlsamples['SlideID'][ii]+'/im3/'
        
        llist = []
        llist = pd.DataFrame( {'Scan':glob.glob(path+'Scan*')} )
        
        llist.Scan = llist.Scan+'replace'
        llist.Scan = llist.Scan.apply(lambda ttext: re.search('\\Scan(\d{1,2})replace', ttext).group(1) )
        llist.Scan = llist.Scan.apply(lambda ttext: int(ttext) )
            #ctrlsamples.iloc['Scan'][ii] = 'Scan'+ str( np.max(np.array(llist)) )
            #ctrlsamples['Scan'].iloc[ii] = 'Scan'+ str(llist.Scan.max())#.copy()
        ctrlsamples.loc[ii,['Scan']] = 'Scan'+ str(llist.Scan.max())#.copy()

        # Get BatchID:
        # Matlab, getBatchId() within getCtrlInfo.m
        if os.path.isfile(os.path.join(path,ctrlsamples['Scan'].iloc[ii],'BatchID.txt')):
            # If the file is available:
            BatchID = pd.read_csv(os.path.join(path,ctrlsamples['Scan'].iloc[ii],
                'BatchID.txt'), header=None)
                #ctrlsamples['BatchID'].iloc[ii] = np.array(BatchID.loc[0,0])#BatchID.loc[0,0].copy()
            ctrlsamples.loc[ii,['BatchID']] = np.array(BatchID.loc[0,0])

        folderTIFF, folderIM3 = getControlTMApath(projectFolder,ctrlsamples['Ctrl'].iloc[ii],ctrlsamples)
        
        # Check if TMAs are mosaic imaces or HPFs
        if len(glob.glob( os.path.join(folderIM3,'*Core*.im3') ))==0:
            ctrlsamples['TMA_imageType'] = 'HPFs'
        else:
            ctrlsamples['TMA_imageType'] = 'mosaic'

        # Check if there are .tif files
        ctrlsamples.loc[ii,['ComponentTIFF_No']] = len( glob.glob( os.path.join(folderTIFF,'*.tif') ) )

        # Create “dataclass” object, eg:
        # result2writeout = ControlTMASampleDef(project,cohort,…) -- attributes given in order
        # the information, as data object, is stored in a list,
        # that will be written out in a .csv file using utilities.tableio.writetable() as writetable(filepath,list_of_objects)
        ControlTMASampleDef_obj = ControlTMASampleDef(
                    ctrlsamples['Project'].iloc[ii],# Project
                    ctrlsamples['Cohort'].iloc[ii], # Cohort
                    ctrlsamples['CtrlID'].iloc[ii], # CtrlID
                    ctrlsamples['TMA'].iloc[ii],    # TMA
                    ctrlsamples['Ctrl'].iloc[ii],   # Ctrl
                    datetime.strptime(ctrlsamples['Date'].iloc[ii], "%m.%d.%Y"),   # Date
                    ctrlsamples['BatchID'].iloc[ii],# BatchID
                    int(ctrlsamples['Scan'].iloc[ii].replace('Scan','')),   # Scan
                    ctrlsamples['SlideID'].iloc[ii] # SlideID
                    )
        print(ControlTMASampleDef_obj)
        ctrlsamples_obj.append(ControlTMASampleDef_obj)
    
    ctrlsamples_all = pd.DataFrame( {'SlideID':glob.glob( projectFolder+'Control_'+'*', recursive=False )} )

    return ctrlsamples, ctrlsamples_all, ctrlsamples_obj

# 
def getList_ctrlcores(projectNumber,outputpath,TMAinfo):
    # create content of *ProjectFolder* \ Ctrl \ *_ctrlcores.csv
    # Uses only Ctrl\Control_TMA_info.xlsx
    # Matlab getCoreInfo()
    # ------------------------------------- # 

    # TMAinfo -- use: 
    # Control_TMA_info for Tonsil and Spleen only or
    # Control_TMA_info_all for all tissue types
    # ------------------------------------- # 

    projectFolder, projectDescription = getProjectRoot(projectNumber)

    # Search for Ctrl files either under the project folder, 
    # or under a given optput folder:
    if outputpath !=0 :
        ctrlFolder = os.path.join(outputpath,'Ctrl')
    else:
        ctrlFolder = os.path.join(projectFolder,'Ctrl')

    # Get list of Control_TMS_info files:
    llist = []
    llist = pd.DataFrame( {'folders':glob.glob(ctrlFolder+'/'+'Control_TMA*') })
    llist['flag'] = llist.loc[:,'folders'].apply(lambda x: 1 if re.search('_all',x)!=None else -1)

    if TMAinfo == '':
        # read only Tonsil and Spleen
        llist = llist.loc[llist.flag==-1,['folders']]# .to_string(header=False,index=False)
    elif TMAinfo == '_all':
        # read all control cores:
        llist = llist.loc[llist.flag!=-1,['folders']]# .to_string(header=False,index=False)
    
    # Read Control_TMS_info files:
    cTMA = pd.DataFrame({})
    for ffile in llist.folders:
        TMA = re.search('(\d{4})_', ffile).group(1)
        cc = pd.read_excel(ffile)
        cc.columns = cc.columns+'_'+TMA
        cTMA = pd.concat([cTMA,cc],axis=1)

    # Create _controlcores file
    ctrlcores = pd.DataFrame({})
    for ii in range(0,len(cTMA.columns),1):
        ttext = cTMA.columns[ii]
        ccTMA = cTMA[ttext].dropna()
        ctrlcores = pd.concat([ctrlcores,
            pd.DataFrame({
                'ncore': np.arange(0,ccTMA.shape[0],1)*0 +-1,
                'project': np.arange(0,ccTMA.shape[0],1)*0 + projectDescription.Project[0],
                'cohort': np.arange(0,ccTMA.shape[0],1)*0 + projectDescription.Cohort[0],
                'TMA': [re.search('_(\d{4})', ttext).group(1)]*ccTMA.shape[0],
                'cx': ccTMA,
                'cy': ccTMA,
                'Core': ccTMA,
                'Tissue': [re.search('(.+?)_\d{4}', ttext).group(1)]*ccTMA.shape[0]
                })
            ], axis=0)

    ctrlcores['Core'] = '[1,'+ctrlcores['Core']+']'
    ctrlcores['cx'] = ctrlcores['cx'].apply(lambda ttext: ttext.split(',')[0])
    ctrlcores['cy'] = ctrlcores['cy'].apply(lambda ttext: ttext.split(',')[1])
    ctrlcores = ctrlcores.sort_values(by=['TMA','cx','cy']).reset_index(drop=True)
    ctrlcores['ncore'] = np.arange(1,ctrlcores.shape[0]+1,1)

    return ctrlcores

def main() :
    #take in command line arguments
    parser = ArgumentParser()
    parser.add_argument('project_number',type=int,help='The project number whose csv files should be created')
    parser.add_argument('--tissue_types_file',type=pathlib.Path,default=DEFAULT_TMA_TISSUE_TYPES_FILE,
                        help=f'The path to the TMA tissue types file (default = {DEFAULT_TMA_TISSUE_TYPES_FILE})')
    parser.add_argument('--outdir',type=pathlib.Path,default=OUTDIR,
                        help='Path to the directory that should hold the output *_cores.csv and *_samples.csv files')
    
    args = parser.parse_args()

    # ------------------------------------------------------------------- #
    # Create Control_TMA_info files for the different TMA types:
    tmaDictionary, ctrlFolder = getFile_ControlTMAinfo(args.project_number, args.outdir);

    for kkeys in tmaDictionary.keys():
        with pd.ExcelWriter(os.path.join(ctrlFolder,'Control_'+kkeys+'_info.xlsx')) as writer:
            tmaDictionary[kkeys].fillna('').to_excel(writer, index=False)
        print(kkeys)

    # ------------------------------------------------------------------- # 
    # Create _ctrlsamples file:
    ctrlsamples, ctrlsamples_all, ctrlsamples_obj = getList_ctrlsamples(args.project_number);

    writetable(os.path.join(ctrlFolder,'Project'+str(args.project_number)+'_ctrlsamples.csv'),
        ctrlsamples_obj) 
        # utilities.tableio.writetable()
    with pd.ExcelWriter(os.path.join(ctrlFolder,'Project'+str(args.project_number)+'_ctrlsamples_ext.xlsx')) as writer:
        ctrlsamples.to_excel(writer, index=False, sheet_name='ctrlsamples')
        ctrlsamples_all.to_excel(writer, index=False, sheet_name='ctrlsamplesList')

    # ------------------------------------------------------------------- #
    # Create _ctrlcores file:
    TMAinfo = ''
        # TMAinfo = '' - use only Tonsil and Spleen cores
        # TMAinfo = '_all' - use all control cores
    ctrlcores = getList_ctrlcores(args.project_number, args.outdir, TMAinfo);

    ctrlcores.to_csv(os.path.join(ctrlFolder,'Project'+str(args.project_number)+'_ctrlcores'+TMAinfo+'.csv'), 
        sep=';', index=False)
        # with pd.ExcelWriter(os.path.join(ctrlFolder,'Project'+str(args.project_number)+'_ctrlcores.xlsx')) as writer:
        #     ctrlcores.to_excel(writer, index=False, sheet_name='ctrlcores')

if __name__=='__main__' :
    main()
