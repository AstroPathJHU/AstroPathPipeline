#!/usr/bin/env python3

########################################################################################################################
#   ASTgen
#   Created by Sigfredo Soto-Diaz

#   This code reads in the location of AstropathPaths.csv and Astrodef.csv <mpath>
#   and generates SlideIDs for all specimens within the scope of AstropathPaths.csv.
#   This should be run before the Transfer Demon script and should be run within <spath> server.
########################################################################################################################
import os
import pathlib
import csv
import re
import sys
import numpy
import time
import argparse
from operator import itemgetter
import shared_tools.shared_tools as st


#
# Reads source csv and takes directories
#
def update_source_csv(csv_file):
    #
    # Open and read file
    #
    attempts = 0
    try:
        lines = st.read_csv(csv_file)
        #
        # Get and return relevant strings and convert filepath format to something Jenkins can read
        #
        regex = '/|\\\\'
        proj = [i.split(',')[0] for i in lines[1:]]
        dpath = ['/'.join(re.split(regex, i.split(',')[1])) for i in lines[1:]]
        dname = [i.split(',')[2] for i in lines[1:]]
        spath = ['/'.join(re.split(regex, i.split(',')[3])) for i in lines[1:]]
        return dname, dpath, spath, proj
    except OSError:
        attempts = attempts + 1
        time.sleep(10)
        if attempts == 5:
            descriptor = "Cannot access/parse: {0}".format(csv_file)
            log_string = "WARNING: " + descriptor + " after " \
                         + str(attempts) + " attempts."
            print(log_string)
            return [], [], [], []


#
# Create record keeping folders if they do not already exist
#
def create_folders(dname, dpath):
    folders = ["upkeep_and_progress", "Flatfield", "logfiles", "Batch", "Clinical",
               "Ctrl", "dbload", "tmp_inform_data", "reject"]
    for f in range(len(folders)):
        if not os.path.exists(dpath + '/' + dname + '/' + folders[f]):
            os.mkdir(dpath + '/' + dname + '/' + folders[f])
    return dpath + '/' + dname + '/' + folders[0]


def ast_gen(mpath, csv_file, mastro_csv):
    dname, dpath, spath, proj = update_source_csv(csv_file)
    if not dname:
        return
    for pos in range(0, len(dname)):
        if not os.path.exists(spath[pos] + '/' + dname[pos]):
            continue
        if not os.path.exists(dpath[pos] + '/' + dname[pos]):
            continue
        uppath = create_folders(dname[pos], dpath[pos])
        patient, batch_id, cohort = extract_data(dname[pos], spath[pos], mpath)
        if not patient:
            continue
        slide_id_csv(patient, batch_id, mastro_csv, uppath, proj[pos], cohort)


#
# Extract the patient#s and BatchIDs for each specimen
#
def extract_data(dname, spath, mpath):
    specimen_path = spath + '/' + dname + '/' + "Specimen_Table.xlsx"
    cohort_path = str(mpath) + '/' + 'AstropathCohortsProgress.csv'
    if not os.path.exists(specimen_path)\
            or not os.path.exists(cohort_path):
        return [], [], []
    attempts = 0
    try:
        #
        # get patient#s and batch ids
        #
        data_mat = st.extract_specimens(specimen_path, ['Patient #', 'Batch ID'])
        #
        # opening cohorts progress file and getting cohort information
        #
        cohorts = st.read_csv(cohort_path)
        projects = [i.split(',')[0] for i in cohorts[1:]]
        cohort = [i.split(',')[1] for i in cohorts[1:]]
        cohort_list = [projects, cohort]
        return data_mat[0], data_mat[1], cohort_list
    except OSError:
        attempts = attempts + 1
        time.sleep(10)
        if attempts == 5:
            descriptor = "Cannot open/parse either: {0} or {1}".format(cohort_path, specimen_path)
            log_string = "WARNING: " + descriptor + " after " \
                                     + str(attempts) + " attempts."
            print(log_string)
            return [], [], []


#
# Generate SlideIDs for all slides starting from highest SlideID in master Astrodef.csv
#
def next_slide_id(mastro_csv, local_string, st_patient, st_batch_id, proj):
    tags = ['SlideID', 'SampleName', 'Project', 'Cohort', 'BatchID']
    if not os.path.exists(local_string):
        with open(local_string, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=tags)
            writer.writeheader()
    #
    # Read in master csv and determine starting point of new SlideIDs
    #
    if os.path.exists(mastro_csv):
        lines = st.read_csv(mastro_csv)
        slide_id_list = [i.split(',')[0] for i in lines[1:]]
        slide_id_list = [s.replace('AP', '') for s in slide_id_list]
        slide_id_list = [idx[3:7] for idx in slide_id_list if idx[0:3].lower() == proj]
        if not slide_id_list:
            slide_id = 1
            new_patient = st_patient
        else:
            at_patient = [i1.split(',')[1] for i1 in lines[1:]]
            slide_id = max(map(int, slide_id_list)) + 1
            new_patient = numpy.setdiff1d(st_patient, at_patient, assume_unique=True)
    else:
        with open(mastro_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=tags)
            writer.writeheader()
        slide_id = 1
        new_patient = st_patient
    if len(new_patient) == 0:
        return [], [], []
    new_batch_id = [st_batch_id[st_patient.index(new_patient[b])] for b in range(len(new_patient))]
    new_slide_id = ["AP" + proj + str(n + slide_id).zfill(4) for n in range(len(new_patient))]
    return new_slide_id, new_patient, new_batch_id


#
# Update Astrodef_<project>.csv and create or update to master Astrodef.csv
#
def slide_id_csv(st_patient, st_batch_id, mastro_csv, uppath, proj, cohort):
    local_string = uppath + '/' + 'AstropathAPIDdef_' + str(proj) + '.csv'
    attempts = 0
    new_slide_id, new_patient, new_batch_id = [], [], []
    try:
        new_slide_id, new_patient, new_batch_id = next_slide_id(
            mastro_csv, local_string, st_patient, st_batch_id,
            str(proj).zfill(3)
        )
    except OSError:
        attempts = attempts + 1
        time.sleep(10)
        if attempts == 5:
            descriptor = "Cannot open/parse: {0}".format(mastro_csv)
            log_string = "WARNING: " + descriptor + " after " \
                                     + str(attempts) + " attempts."
            print(log_string)
            return
    if new_slide_id:
        proj_list = [proj] * len(new_patient)
        cohort_list = [cohort[1][cohort[0].index(proj)]] * len(new_patient)
        if len(new_slide_id) == 1:
            new_data = list(zip(new_slide_id, new_patient, proj_list, cohort_list,
                                new_batch_id))
        else:
            id_index = [i for i, val in enumerate(new_patient) if val in set(st_patient)]
            new_data = list(zip(new_slide_id, list(itemgetter(*id_index)(new_patient)),
                                proj_list, cohort_list,
                                list(itemgetter(*id_index)(new_batch_id))))
        #
        # Append Astrodef_<project>.csv for working specimen
        #
        # Create shared tools function that opens csv and writes infromation with given headers
        # Inputs: filename, cols, cols_data
        # Check that you can open file and that it exists
        # If DNE, write file with cols
        # take out cols_data
        # If file can't open return error
        #
        attempts = 0
        try:
            with open(mastro_csv, 'a', newline='') as f:
                writer = csv.writer(f, lineterminator='\r\n')
                writer.writerows(new_data)
        except OSError:
            attempts = attempts + 1
            time.sleep(10)
            if attempts == 5:
                descriptor = "Cannot write to: {0}".format(mastro_csv)
                log_string = "WARNING: " + descriptor + " after " \
                             + str(attempts) + " attempts."
                print(log_string)
                return
    # with open(mastro_csv, newline='') as csvfile:
    #     astroidreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    #     for row in astroidreader:
    #         print(', '.join(row))
    #
    # Get APID data for local file
    #
    lines = st.read_csv(mastro_csv)
    slide_id_list = [i.split(',')[0] for i in lines[1:]]
    slide_id_list = [s.replace('AP', '') for s in slide_id_list]
    indeces = [slide_id_list.index(idx) for idx in slide_id_list if int(idx[0:3]) == int(proj)]
    new_data = []
    for index in indeces:
        new_data.append(lines[index + 1].replace('\n', '').split(','))
    #
    # Check if the new list doesn't match the old file data
    #
    old_data = []
    lines = st.read_csv(local_string)
    slide_id_list = [i.split(',')[0] for i in lines[1:]]
    for i1 in range(len(slide_id_list)):
        old_data.append(lines[i1 + 1].replace('\n', '').split(','))
    if old_data == new_data:
        return
    #
    try:
        with open(local_string, 'w', newline='') as f:
            writer = csv.writer(f, lineterminator='\r\n')
            writer.writerows(new_data)
    except OSError:
        attempts = attempts + 1
        time.sleep(10)
        if attempts == 5:
            descriptor = "Cannot write to: {0}".format(local_string)
            log_string = "WARNING: " + descriptor + " after " \
                                     + str(attempts) + " attempts."
            print(log_string)
            return
    # with open(local_string, newline='') as csvfile:
    #     astroidreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    #     for row in astroidreader:
    #         print(', '.join(row))
    #
    # Append master Astrodef.csv
    #


def apid_argparser():
    version = '0.01.0001'
    parser = argparse.ArgumentParser(
        prog="ASTgen",
        description='creates APIDs for clincal specimen slides in the Astropath pipeline'
    )
    parser.add_argument('--version', action='version', version='%(prog)s ' + version)
    parser.add_argument('mpath', type=str, nargs='?',
                        help='directory for astropath processing documents')
    args, unknown = parser.parse_known_args()
    return args


#
# Main function, reads in csv
#
def start_gen():
    #
    # Process user input for the csv file path
    #
    cwd = '/'.join(os.getcwd().replace('\\', '/').split('/')[:-1])
    print(cwd)
    for root, dirs, files in os.walk(cwd, topdown=True):
        if "shared_tools" in dirs:
            os.chdir(root)
            break
    cwd = '/'.join(os.getcwd().replace('\\', '/').split('/'))
    print(cwd)
    print("Inputs: " + str(sys.argv))
    arg = apid_argparser()
    mpath = pathlib.Path(arg.mpath)
    csv_file = mpath/"AstropathPaths.csv"
    mastro_csv = mpath/"AstropathAPIDdef.csv"
    if not os.path.exists(csv_file):
        print("No AstropathPaths.csv found in " + str(mpath))
        return
    contents = os.listdir(mpath)
    print(contents)
    #
    # Perform the SlideID generation
    #
    for t in range(2):
        ast_gen(mpath, csv_file, mastro_csv)
        minutes = 0.1
        print("ALL AVAILABLE IDS GENERATED. SLEEP FOR " + str(minutes) + " MINUTES...")
        wait_time = 60 * minutes
        time.sleep(wait_time)


if __name__ == '__main__':
    start_gen()
