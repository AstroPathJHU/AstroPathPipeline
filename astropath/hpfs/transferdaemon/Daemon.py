#!/usr/bin/env python

import os
import time
import re
import sys
import hashlib
import shutil
import smtplib
import pathlib
import traceback
import subprocess
import pandas
import argparse
import numpy as np
from ...shared import shared_tools as st
import lxml.etree as et
from pathlib import Path
from datetime import datetime
from joblib import Parallel, delayed


#
# Main running function
#
def check_ready_files(paths_data, config_data, cohort_data, arg):
    sor_string = paths_data[0]
    des_string = paths_data[1]
    comp_string = paths_data[2]
    paths_proj = paths_data[3]
    #
    # loop for each directory in path input document
    #
    directory_waiting = []
    directory_roots = []
    for i1 in cohort_data[0]:
        try:
            cohort_data_index = cohort_data[0].index(i1)
            config_data_index = config_data[0].index(i1)
            paths_data_index = paths_data[3].index(i1)
        except ValueError:
            continue
        specimen_path = sor_string[paths_data_index] + '/Specimen_Table.xlsx'
        #
        if not os.path.exists(sor_string[paths_data_index]) \
                or not os.path.exists(des_string[paths_data_index]) \
                or not os.path.exists(comp_string[paths_data_index]) \
                or not os.path.exists(specimen_path):
            continue
        #
        # build directory_waiting data. Uses project # from paths and
        # finds coresponding project # in config and cohort to determine
        # relevant cohort #, space allocation, and delete protocol
        #
        delete_i = config_data[1][config_data_index]
        space_i = config_data[2][config_data_index]
        cohort_i = cohort_data[1][cohort_data_index]
        #
        specimen_table = pandas.read_excel(specimen_path, engine='openpyxl')
        st_slide_ids = specimen_table['Patient #'].tolist()
        st_batch_ids = specimen_table['Batch ID'].tolist()
        #
        total_size = 0
        astro_ids = get_astro_id(des_string[paths_data_index], paths_proj[paths_data_index])
        if not astro_ids[0]:
            continue
        #
        # get the paths with batchIDs and not in directory waiting
        #
        for root, dirs, files in os.walk(sor_string[paths_data_index], topdown=False):
            if "Scan" in root and os.path.exists(root + "/BatchID.txt") and \
                    root not in directory_roots:
                #
                regex = '/|\\\\'
                root = '/'.join(re.split(regex, root))
                slide_id = str(root.split('/')[-2])
                #
                # SlideID is not in AstroID_def
                #
                if slide_id in astro_ids[1]:
                    astro_id = astro_ids[0][astro_ids[1].index(slide_id)]
                    string_list = [i1, cohort_i, astro_id]
                elif "Control" in slide_id:
                    astro_id = slide_id
                    string_list = [i1, cohort_i, '_'.join(slide_id.split('_')[:-1])]
                else:
                    continue
                #
                log_base = ';'.join(string_list)
                #
                file = open(root + "/BatchID.txt", 'r')
                batch_id = str(file.read())
                if slide_id in st_slide_ids:
                    st_batch_id = str(st_batch_ids[st_slide_ids.index(slide_id)])
                else:
                    st_batch_id = ''
                if batch_id != st_batch_id and "Control" not in slide_id:
                    log_string = log_base + ";ERROR: BatchID.txt does not match " \
                                            "BatchID in Specimen Table. Skipping transfer"
                    st.print_to_log(log_string, des_string[paths_data_index], arg.v, arg.q, astro_id, "master")
                    continue
                #
                row = [comp_string[paths_data_index], des_string[paths_data_index], delete_i, root,
                       i1, space_i, cohort_i, astro_id, log_base]
                #
                # If the directory to be transferred is larger than the space avaliable
                # in the destination directory as given by AstropathConfig.csv, then
                # the directory is not transferred. Updated for each new specimen
                #
                total_size = total_size + st.get_size(root)
                if total_size > space_i * 10 ** 12:
                    log_string = log_base + ";ERROR: Insufficient space. Skipping transfer"
                    st.print_to_log(log_string, des_string[paths_data_index], arg.v, arg.q, astro_id, "master")
                    continue
                #
                # If automatic has been chosen the DoNotDelete texts files are ignored
                # and deleted. Otherwise only add directory waiting queue if the text
                # file does not exist
                #
                dnd_path = root + "/DoNotDelete.txt"
                delete_i = delete_i.upper()
                if arg.delete_type == "automatic":
                    if delete_i == "YES" and os.path.exists(dnd_path):
                        os.remove(dnd_path)
                else:
                    temp_des = os.path.join(des_string[paths_data_index], root.split('/')[-2])
                    if os.path.exists(dnd_path) and os.path.exists(temp_des):
                        continue
                #
                directory_waiting.append(row)
                directory_roots.append(root)
    return directory_waiting


#
# transfer the directories from directory_waiting
#
def transfer_loop(directory_waiting, arg, zip_path):
    #
    for direct in directory_waiting:
        transfer_one_sample(direct, arg, zip_path)
        #
        # log file is to be saved in each specimen folder. To avoid conflicts with
        # md5 checks, log file will be moved after all transfer processes have finished
        #


#
# Needed to manage the log files if the destination needs to be deleted.
# Want to keep what has been logged in before
#
def merge_logs(des, log_path):
    new_log = des + '/transfer.log'
    prev_log = open(log_path + '/transfer.log', 'a+')
    with open(new_log) as f:
        for line in f:
            prev_log.write(line)
    prev_log.close()


#
# transfer a single sample and delete based off of corresponding settings
#
def transfer_one_sample(direct, arg, zip_path):
    astro_id = direct[7]
    log_base = direct[8]
    des_string = direct[1]
    #
    # set up transfer strings
    #
    current_sor_string = direct[3]
    current_des_string = des_string + "/" + astro_id + "/im3" + "/" \
                         + str(direct[3].split('/')[-1])
    current_compress_string = direct[0] + "/" + astro_id + "/im3" + "/" \
                              + str(direct[3].split('/')[-1])
    if arg.delete_type == 'manual':
        del_string = 'YES'
    else:
        del_string = direct[2]
    #
    # If the destination path already exists, compare the files and determine if
    # an error occurred in the transfer process. If it did then delete destination
    # and compression then continue with this sample. If no error occurred then
    # remove the source directory and continue to the next sample.
    #
    if os.path.exists(current_des_string):
        compare = compare_file_names(current_sor_string, direct[1], current_des_string,
                                     current_compress_string, del_string, arg,
                                     log_base=log_base, astro_id=astro_id)
        if compare == 1:
            return
    #
    # transfer process
    #
    i2 = 2
    while i2 == 2:
        err, result, result2 = error_check("TRANSFER", direct[1], arg, current_sor_string,
                                           current_des_string, astro=astro_id,
                                           log_base=log_base)
        if err:
            return
        if not arg.no_compress:
            err, result, result2 = error_check("COMPRESS", direct[1], arg, current_sor_string,
                                               current_des_string, current_compress_string,
                                               log_base=log_base, astro=astro_id, zip_path=zip_path)
            if err:
                return
            log_string = log_base + ";Compression finished"
            st.print_to_log(log_string, direct[1], arg.v, arg.q, astro_id, "master")
        #
        # MD5 calculation and file comparison
        #
        err = compare_file_names(current_sor_string, direct[1], current_des_string,
                                 current_compress_string, del_string, arg, 1,
                                 log_base, astro_id=astro_id)
        if err == 2:
            continue
        elif err == 1:
            return
        else:
            break


#
# Get the AstroIDs from U/P folder. If no file exists, push a warning and move to
# next specimen
#
def get_astro_id(des_dir, proj):
    astro_id_csv = str(des_dir) + '/upkeep_and_progress/AstropathAPIDdef_' + \
                   str(proj) + '.csv'
    if not os.path.exists(astro_id_csv):
        return ['', '']
    lines = st.read_csv(astro_id_csv)
    astro_ids = [i.split(',')[0] for i in lines[1:]]
    slide_ids = [i.split(',')[1] for i in lines[1:]]
    return [astro_ids, slide_ids]


#
# Resolve directory already in the destination directory
# Returning 1 means that the source directory was deleted and the program should
# continue to next directory.  Returning 2 means that the transfer process should
# be re-initiated
#
def compare_file_names(current_sor_string, main_des_string, current_des_string,
                       current_compress_string, del_string, arg, post_transfer=0,
                       log_base="", astro_id=""):
    slide_id = str(current_sor_string.split('/')[-2])
    #
    annotation_file = astro_id + "_" + current_sor_string.split('/')[-1] \
                      + "_annotations.xml"
    #
    if post_transfer == 0:
        log_string = log_base + ";Slide ID is in source and destination on source directory recheck comparing files"
        st.print_to_log(log_string, main_des_string, arg.v, arg.q, astro_id)
    #
    # Compute the number of files in each directory
    # Only compare those files which are transferred from source
    #
    hash_path = [current_des_string, current_sor_string]
    file_array = []
    for x in [0, 1]:
        x1 = []
        p = hash_path[x]
        for root, dirs, files in os.walk(p):
            x1 += files
        if 'CheckSums.txt' in x1:
            x1.remove('CheckSums.txt')
        if 'transfer.log' in x1:
            x1.remove('transfer.log')
        if annotation_file in x1 and slide_id != astro_id:
            x1.remove(annotation_file)
        file_array.append(x1)
    #
    if len(file_array[1]) < len(file_array[0]):
        if post_transfer == 1:
            log_string = log_base + ";Source lost files after transfer"
            st.print_to_log(log_string, main_des_string, arg.v, arg.q, astro_id, "master")
            log_string = log_base + ";Error sent. Next slide"
            st.print_to_log(log_string, main_des_string, arg.v, arg.q, astro_id)
            mail_string = "Source directory has less files after transfer."
            st.send_email(arg.email, mail_string, debug=arg.d)
            #
            return 1
        #
        # delete source path if there are missing files in it
        #
        log_string = log_base + ";Source directory missing files"
        st.print_to_log(log_string, main_des_string, arg.v, arg.q, astro_id)
        #
        if not os.path.exists(current_sor_string + "/DoNotDelete.txt") \
                and del_string == "YES":
            to_delete = str(Path(current_sor_string).parents[0])
            shutil.rmtree(to_delete, ignore_errors=True)
            #
            log_string = log_base + ";Deleted source directory"
            st.print_to_log(log_string, main_des_string, arg.v, arg.q, astro_id)
        #
        # return 1 to continue to next specimen
        #
        return 1
        #
    elif len(file_array[1]) > len(file_array[0]):
        #
        # delete destination path if there are missing files in it
        #
        log_string = log_base + ";Destination directory missing files"
        st.print_to_log(log_string, main_des_string, arg.v, arg.q, astro_id)
        #
        delete_destination(main_des_string, current_des_string, current_compress_string,
                           log_base, astro_id, arg)
        #
        return 2
    #
    # For the situation where the number of files is equal between the source
    # directory and the destination directory, the following compares the hash values
    # of each file.
    #
    elif len(file_array[1]) == len(file_array[0]):
        #
        # compute the new hash files
        #
        #
        location_string = ['DEST', 'SOURCE']
        hash_list = []
        #
        for x in [0, 1]:
            #
            # if old check sum file exists delete it
            #
            c_hash_path = hash_path[x] + '/CheckSums.txt'
            if os.path.exists(c_hash_path):
                os.remove(c_hash_path)
            #
            # compute hash values and store them in the array hash_array
            #
            error_string = "COMPUTE " + location_string[x] + " MD5"
            err, hash_value, sums_value = error_check(error_string, main_des_string, arg,
                                                      current_sor_string, current_des_string,
                                                      log_base=log_base, astro=astro_id)
            if err:
                return err
            hash_list.append(hash_value)
        #
        log_string = log_base + ";MD5 calculations finished"
        st.print_to_log(log_string, main_des_string, arg.v, arg.q, astro_id)
        #
        # If all the values in the arrays match between source and destination files,
        # the data was transferred successfully and the source directory should be
        # removed according to the user input.
        #
        if not list(set(hash_list[0]) - set(hash_list[1])):
            #
            # if do not delete text file does not exist and protocol allows deletion
            # then delete the source directory
            #
            if not os.path.exists(current_sor_string + "/DoNotDelete.txt") \
                    and del_string == "YES":
                log_string = log_base + ";Source and destination match"
                st.print_to_log(log_string, main_des_string, arg.v, arg.q, astro_id)
                to_delete = str(Path(current_sor_string).parents[0])
                shutil.rmtree(to_delete, ignore_errors=True)
                log_string = log_base + ";Deleted source directory"
                st.print_to_log(log_string, main_des_string, arg.v, arg.q, astro_id)
            #
            elif not os.path.exists(current_sor_string + "/DoNotDelete.txt") \
                    or not os.path.exists(current_des_string + "/DoNotDelete.txt"):
                st.print_to_log(log_base + ";Source and destination match", main_des_string, arg.v, arg.q, astro_id)
                create_delete_txt(current_sor_string)
                create_delete_txt(current_des_string)
                st.print_to_log(log_base + ";Created DoNotDelete file", main_des_string, arg.v, arg.q, astro_id)
            st.print_to_log(log_base + ";Processing finished", main_des_string, arg.v, arg.q, astro_id)
            #
            # return 1 to continue to next specimen if we are before the transfer
            #
            if post_transfer == 1:
                return 0
            else:
                return 1
        #
        # If a given file has different hash values, something was corrupted
        # in the data during transfer and the process is re-initiated.
        #
        else:
            log_string = log_base + ";Destination and source inconsistency"
            st.print_to_log(log_string, main_des_string, arg.v, arg.q, astro_id)
            delete_destination(main_des_string, current_des_string, current_compress_string,
                               log_base, astro_id, arg)
    return 0


#
# function that deletes the destination when necessary
#
def delete_destination(main_des_string, current_des_string, current_compress_string,
                       log_base, astro_id, arg):
    to_delete = str(Path(current_des_string).parents[0])
    shutil.rmtree(to_delete, ignore_errors=True)
    #
    log_string = log_base + ";Deleted destination directory"
    st.print_to_log(log_string, main_des_string, arg.v, arg.q, astro_id)
    #
    # also delete the compressed path if it exists
    #
    if os.path.exists(current_compress_string):
        to_delete = str(Path(current_compress_string).parents[0])
        shutil.rmtree(to_delete, ignore_errors=True)
        log_string = log_base + ";Deleted compression directory"
        st.print_to_log(log_string, main_des_string, arg.v, arg.q, astro_id)
    #
    log_string = log_base + ";Re-initiating transfer process."
    st.print_to_log(log_string, main_des_string, arg.v, arg.q, astro_id)


#
# evaluate the functions while checking for errors
#
def error_check(action, main_des_string, arg, current_sor_string="", current_des_string="",
                comp="", astro="", log_base="", zip_path=""):
    slide_id = str(current_sor_string.split('/')[-2])
    attempts = 2
    err = 0
    warning = ""
    mins = 0.1
    result = []
    result2 = []
    while err < attempts:
        try:
            if action == "TRANSFER":
                transfer_directory(current_sor_string, main_des_string, current_des_string,
                                   astro, arg, log_base=log_base)
                st.print_to_log(log_base + ";Transfer finished", main_des_string, astro, arg, "master")
            elif action == "COMPUTE SOURCE MD5":
                result, result2 = compute_md5(current_sor_string, main_des_string, "SOURCE", arg,
                                              log_base=log_base, slide_id=slide_id,
                                              astro_id=astro)
            elif action == "COMPUTE DEST MD5":
                result, result2 = compute_md5(current_des_string, main_des_string,
                                              "DESTINATION", arg, log_base=log_base,
                                              slide_id=slide_id, astro_id=astro)
            elif action == "COMPRESS":
                compress_directory(current_sor_string, main_des_string, comp, astro, arg,
                                   zip_path, log_base=log_base)
            elif action == "ANNOTATE":
                xmlfile = main_des_string + '/' + astro + '/im3/' \
                          + current_sor_string.split('/')[-1] \
                          + '/' + astro + '_' + current_sor_string.split('/')[-1] \
                          + '_annotations.xml'
                warning = annotation_handler(xmlfile, str(current_sor_string.split('/')[-2]), astro)
            if err > 0:
                log_string = log_base + ";Warning: " + action.lower() + " passed with " + \
                             str(err) + " error(s)"
                st.print_to_log(log_string, main_des_string, astro, arg, "master")
            if warning:
                log_string = log_base + warning
                st.print_to_log(log_string, main_des_string, astro, arg)
            err = attempts
        except OSError:
            #
            # increase count and check if it is greater than number of allowed attempts
            #
            err = err + 1
            #
            # send error message to log
            #
            error_msg = traceback.format_exc().splitlines()[-1].split(':')[0]
            descriptor = traceback.format_exc().splitlines()[-1].split(']')[-1]
            log_string = log_base + ";WARNING: attempt " + str(err) + " failed for " \
                         + action.lower()
            st.print_to_log(log_string, main_des_string, astro, arg)
            if err < attempts:
                #
                # if we have not met the allowed count wait <mins> minutes and try again
                #
                log_string = log_base + ";Attempting to " + action.lower() \
                             + " again after " + str(mins) + " minutes"
                st.print_to_log(log_string, main_des_string, astro, arg)
                time.sleep(mins * 60)
                continue
                #
            else:
                #
                # if we have met the allowed count something else must be wrong.
                # Email, return positive err value
                #
                log_string = log_base + ";ERROR: " + error_msg + descriptor
                st.print_to_log(log_string, main_des_string, astro, arg, "master")
                error = traceback.format_exc()
                st.send_email(arg.email, error, err=err, error_check_dec=True, debug=arg.d)
                err = 1
                return err, result, result2
        except et.ParseError as what:
            #
            # Annotation handler error catch if parsing error occurs
            #
            err = err + 1
            error_msg = traceback.format_exc().splitlines()[-1].split(':')[0]
            log_string = log_base + ";ERROR: " + error_msg + " - " + str(what)
            st.print_to_log(log_string, main_des_string, astro, arg, "master")
            error = traceback.format_exc()
            st.send_email(arg.email, error, err=err, error_check_dec=True, debug=arg.d)
            return err, result, result2
    err = 0
    return err, result, result2


#
# Generates Hash Values and CheckSums.txt files
#
def compute_md5(current_directory, main_des_string, location_string, arg, log_base="",
                slide_id="", astro_id=""):
    #
    # print starting strings to log
    #
    log_string = log_base + ";MD5 computation started"
    st.print_to_log(log_string, main_des_string, astro_id, arg)
    log_string = log_base + ";Computing " + location_string.lower() + " MD5 check sums"
    st.print_to_log(log_string, main_des_string, astro_id, arg)
    #
    # create the md5 hash values in parallel for each file in the current directory
    # put the strings for the check sums file and the hash values into separate arrays
    # Only compare those files that were transferred
    #
    start = time.time()
    num = 0
    sums_array = []
    hash_array = []
    for root, dirs, files in os.walk(current_directory):
        results = Parallel(n_jobs=4, backend="loky")(
            delayed(md5)(root + '/' + file_1) for file_1 in files)
        for item in results:
            if ("annotations.xml" in item[0] and "xml.lock" not in item[0].lower()
                and location_string != "SOURCE" and astro_id != slide_id) \
                    or "transfer.log" in item[0]:
                continue
            num += 1
            sums_array.append(item[0])
            hash_array.append(item[1])
    #
    # write sums to checksum file
    #
    sums_file = open(current_directory + "/CheckSums.txt", "w")
    sums_file.writelines(["%s\n" % item for item in sums_array])
    sums_file.close()
    #
    end = time.time()
    log_string = log_base + ";Completed " + str(num) + " files in " \
                 + str(round(end - start, 2)) + " seconds"
    st.print_to_log(log_string, main_des_string, astro_id, arg)
    #
    return hash_array, sums_array


#
# Outputs CheckSums.txt lines and Hash values as 2 object array
#
def md5(item):
    hash_md5 = hashlib.md5()
    with open(item, "rb") as f:
        for chunk in iter(lambda: f.read(104857600), b""):
            hash_md5.update(chunk)
    return [item + '\t' + hash_md5.hexdigest(), hash_md5.hexdigest()]


#
# creates DoNotDelete.txt
#
def create_delete_txt(current_directory):
    text_file = open(current_directory + "/DoNotDelete.txt", "w")
    text_file.write("Do not delete me unless this folder is going to be removed.")
    text_file.close()


#
# Performs TransferItem() on every file in the source directory
# For rename process, create a duplicate of the annotation file with "-original".
# This will have the astroID in the filename but nothing changed inside.
# The other version will have the .im3 portion inside changed to the AstroID
#
def transfer_directory(current_sor_string, main_des_string, current_des_string,
                       astro_id, arg, log_base=""):
    slide_id = str(current_sor_string.split('/')[-2])
    #
    # get the number of files and bytes in the source directory
    #
    n_sor_files, n_sor_bytes = 0, 0
    M_files = []
    all_files = []
    #
    # Check for duplicate files. Remove all but latest version of each duplicate
    #
    for root, dirs, files in os.walk(current_sor_string):
        for f in sorted(files):
            if ".im3" in f:
                all_files.append(f)
            if "]_M" in f:
                M_files.append(f)
    if M_files:
        st.print_to_log(log_base + ";Duplicate files found", main_des_string, astro_id, arg)
        st.M_file_handler(current_sor_string, all_files, M_files)
        st.print_to_log(log_base + ";Duplicate files handled", main_des_string, astro_id, arg)
    #
    names = [""] * 2
    names[1] = astro_id
    for root, dirs, files in os.walk(current_sor_string):
        if not names[0]:
            names[0] = str(root.split('/')[-2])
        for f in files:
            f = os.path.join(root, f)
            n_sor_bytes += os.path.getsize(f)
        n_sor_files += len(files)
    #
    log_string = log_base + ";Transfer process started"
    st.print_to_log(log_string, main_des_string, astro_id, arg, "master")
    log_string = (log_base + ";Source Contains " + str(n_sor_files) +
                  " File(s) " + str(n_sor_bytes) + " bytes")
    st.print_to_log(log_string, main_des_string, astro_id, arg)
    #
    pathlib.Path(current_des_string).mkdir(parents=True, exist_ok=True)
    #
    for item in os.listdir(current_sor_string):
        transfer_item(item, current_sor_string, current_des_string, names)
    #
    # get files and bytes from destination directory
    #
    n_des_files, n_des_bytes = 0, 0
    #
    for root, dirs, files in os.walk(current_des_string):
        for f in files:
            f = os.path.join(root, f)
            n_des_bytes += os.path.getsize(f)
        n_des_files += len(files)
    #
    # Once transfer process is finished, duplicate and edit annotations
    # folder to match new naming convention
    #
    if slide_id != astro_id:
        error_check("ANNOTATE", main_des_string, arg, current_sor_string, astro=astro_id,
                    log_base=log_base)
    log_string = (log_base + ";Transferred " + str(n_des_files) +
                  " File(s) " + str(n_des_bytes) + " bytes")
    st.print_to_log(log_string, main_des_string, astro_id, arg)


#
# Duplicates existing annotations file and edits the version not labeled "-original"
# to match the new naming convention
#
def annotation_handler(xmlfile, slide_id, astro_id):
    if not os.path.exists(xmlfile):
        return ";WARNING: " + xmlfile + " does not exist"
    newfile = xmlfile.replace('.xml', '-original.xml')
    shutil.copy(xmlfile, newfile)
    with open(xmlfile, 'rb+') as f:
        tree = et.parse(f)
        root = tree.getroot()
        for elem in root.getiterator():
            if elem.text:
                elem.text = elem.text.replace(slide_id, astro_id)
            if elem.tail:
                elem.tail = elem.tail.replace(slide_id, astro_id)
        f.seek(0)
        f.write(et.tostring(tree, encoding='UTF-8', xml_declaration=True))
        f.truncate()
    return ""


#
# Transfers an individual file from source to directory
#
def transfer_item(item, src, dst, names):
    s = os.path.join(src, item)
    d = os.path.join(dst, item)
    if os.path.isdir(s):
        if not os.path.exists(d):
            pathlib.Path(d).mkdir(parents=True, exist_ok=True)
        Parallel(n_jobs=4, backend="loky")(
            delayed(transfer_one)(item, s, d, names)
            for item in os.listdir(s))
    else:
        transfer_one(item, src, dst, names)


#
# transfers items and changes SlideID in filenames to AstroID
#
def transfer_one(item, src, dst, names):
    s = os.path.join(src, item)
    d = os.path.join(dst, item)
    if names[0] in d:
        d = d.replace(names[0], names[1])
    with open(s, 'rb') as f_src:
        with open(d, 'wb') as f_dst:
            shutil.copyfileobj(f_src, f_dst, length=16 * 1024 * 1024)
    shutil.copystat(s, d)


#
# Runs Compress() on each file in the working directory
#
def compress_directory(current_sor_string, main_des_string, compress, astro_id, arg,
                       zip_path, log_base=""):
    #
    # get the number of files and bytes in the source directory
    #
    n_sor_files, n_sor_bytes = 0, 0
    xmlfile = astro_id + '_' + current_sor_string.split('/')[-1] \
                       + '_annotations.xml'
    xml_list = [xmlfile, xmlfile.replace('.xml', '-original.xml')]
    #
    names = [""] * 2
    names[1] = astro_id
    for root, dirs, files in os.walk(current_sor_string):
        if not names[0]:
            names[0] = str(root.split('/')[-2])
        for f in files:
            if "annotations.xml.lock" in f.lower() or "annotations.xml" not in f \
                    or names[0] == names[1]:
                f = os.path.join(root, f)
                n_sor_bytes += os.path.getsize(f)
        n_sor_files += len(files)
    n_sor_files = n_sor_files
    #
    log_string = log_base + ";Compression started"
    st.print_to_log(log_string, main_des_string, astro_id, arg, "master")
    #
    # do the compression one file at a time
    #
    if not os.path.exists(compress):
        pathlib.Path(compress).mkdir(parents=True, exist_ok=True)
    #
    for item in os.listdir(current_sor_string):
        compress_item(item, current_sor_string, compress, names, zip_path)
    annotation_path = main_des_string + '/' + astro_id + '/im3/' \
                      + current_sor_string.split('/')[-1] + '/'
    for item in xml_list:
        if os.path.exists(annotation_path + item) and names[0] != names[1]:
            compress_item(item, annotation_path, compress, names, zip_path, ann=True)
    #
    # get files and bytes from destination directory
    #
    n_des_files, n_des_bytes = 0, 0
    #
    for root, dirs, files in os.walk(compress):
        for f in files:
            f = os.path.join(root, f)
            n_des_bytes += os.path.getsize(f)
        n_des_files += len(files)
    #
    log_string = (log_base + ";Compressing " + str(n_sor_files) +
                  " file(s) and " + str(n_sor_bytes) + " bytes from source")
    st.print_to_log(log_string, main_des_string, astro_id, arg)
    log_string = (log_base + ";Compressed " + str(n_des_files) +
                  " total file(s) " + str(n_des_bytes) + " total bytes")
    st.print_to_log(log_string, main_des_string, astro_id, arg)


#
# Compresses individual files
#
def compress_item(item, sor, des, names, zip_path, ann=False):
    s = os.path.join(sor, item)
    d = os.path.join(des, item)
    d = d.replace(names[0], names[1])
    if os.path.isdir(s):
        if not os.path.exists(d):
            pathlib.Path(d).mkdir(parents=True, exist_ok=True)
        Parallel(n_jobs=4, backend="loky")(
            delayed(compress_item)(item, s, d, names, zip_path)
            for item in os.listdir(s))
    if (not ann or names[0] not in item) and not os.path.isdir(s):
        subprocess.check_output([zip_path + '/7z.exe', 'a', d + ".7z", '-mx1', s])
    elif names[0] == names[1] and not os.path.isdir(s):
        subprocess.check_output([zip_path + '/7z.exe', 'a', d + ".7z", '-mx1', s])
    elif "annotations.xml.lock" in item.lower() or "annotations.xml" not in item \
            and not os.path.isdir(s):
        pre_string = str(s.split('/')[-2]) + '/' + names[0]
        post_string = str(s.split('/')[-2]) + '/' + names[1]
        temp_s = s.replace(pre_string, post_string)
        shutil.copy(s, temp_s)
        subprocess.check_output([zip_path + '/7z.exe', 'a', d + ".7z", '-mx1', temp_s])
        os.remove(temp_s)


#
# Reads source csv and takes directories
#
def update_source_csv(arg):
    if arg.d:
        leading = ''
    else:
        leading = '//'
    path = arg.mpath + '/AstropathPaths.csv'
    config = arg.mpath + '/AstropathConfig.csv'
    cohort_csv = arg.mpath + '/AstropathCohortsProgress.csv'
    #
    # Catches and alerts user to which cource files weren't found
    #
    all_files = [path, config, cohort_csv]
    a_exist = [f for f in all_files if os.path.isfile(f)]
    a_non_exist = list(set(a_exist) ^ set(all_files))
    if a_non_exist:
        return [], a_non_exist, []
    #
    # open and read files
    #
    paths = st.read_csv(path)
    configs = st.read_csv(config)
    cohorts = st.read_csv(cohort_csv)
    #
    # get and return relevant strings
    #
    c_proj = [i.split(',')[0] for i in cohorts[1:]]
    cohort = [i.split(',')[1] for i in cohorts[1:]]
    #
    d_proj = [i.split(',')[0] for i in configs[1:]]
    delete = [i.split(',')[3] for i in configs[1:]]
    space = [float(i.split(',')[4]) for i in configs[1:]]
    #
    proj = [i.split(',')[0] for i in paths[1:]]
    dpath = [leading + i.split(',')[1] for i in paths[1:]]
    dname = [i.split(',')[2] for i in paths[1:]]
    spath = [leading + i.split(',')[3] for i in paths[1:]]
    cpath = [leading + i.split(',')[4] for i in paths[1:]]
    #
    sor_string = [''] * len(dname)
    des_string = [''] * len(dname)
    comp_string = [''] * len(dname)
    #
    # Convert filepath format to something Jenkins can read
    #
    regex = '/|\\\\'
    for i1 in range(0, len(dpath)):
        sor_string[i1] = '/'.join(re.split(regex, spath[i1])) + '/' + dname[i1]
        des_string[i1] = '/'.join(re.split(regex, dpath[i1])) + '/' + dname[i1]
        comp_string[i1] = '/'.join(re.split(regex, cpath[i1])) + '/' + dname[i1]
    #
    paths_data = [sor_string, des_string, comp_string, proj]
    config_data = [d_proj, delete, space]
    cohort_data = [c_proj, cohort]
    return paths_data, config_data, cohort_data


#
# Create and edit local transfer.log
# create a log folder and save master file in there
# <console> is a hard coded method of showing log entries
# pending user input method
#
def print_to_log(log_string, des_string, astro_id, arg, loc=""):
    #
    # Make a check for starting and ending lines for version number entries
    #
    pathlib.Path(des_string + '/' + astro_id + '/logfiles').mkdir(parents=True, exist_ok=True)
    if loc == "master":
        if not os.path.exists(des_string + '/logfiles'):
            os.mkdir(des_string + '/logfiles')
        logfile = open(des_string + '/logfiles' + r"\transfer.log", 'ab')
        now = datetime.now()
        str1 = "{0}-{1};{2}\r\n".format(log_string, arg.v, now.strftime("%Y-%m-%d %H:%M:%S"))
        strb = bytes(str1, 'utf-8')
        logfile.write(strb)
        logfile.close()
    logfile = open(des_string + '/' + astro_id + '/logfiles' + r"\transfer.log", 'ab')
    now = datetime.now()
    str1 = "{0}-{1};{2}\r\n".format(log_string, arg.v, now.strftime("%Y-%m-%d %H:%M:%S"))
    strb = bytes(str1, 'utf-8')
    logfile.write(strb)
    logfile.close()
    if not arg.q:
        print(log_string)


def apid_argparser():
    version = '0.01.0001'
    parser = argparse.ArgumentParser(
        prog="Daemon",
        description='launches transfer for clincal specimen slides in the Astropath pipeline'
    )
    parser.add_argument('--version', action='version', version='%(prog)s ' + version)
    parser.add_argument('mpath', type=str, nargs='?',
                        help='directory for astropath processing documents')
    parser.add_argument('email', type=str, nargs='?',
                        help='defines person to email in case of errors')
    parser.add_argument('delete_type', type=str, nargs='?',
                        choices=["hybrid", "automatic", "manual"],
                        default='hybrid',
                        help='sets delete type protocol defined in readme')
    parser.add_argument('-no_compress', action='store_true',
                        help='do not compress transferred files')
    parser.add_argument('-q', action='store_true',
                        help='runs the function quietly')
    parser.add_argument('-v', type=str, nargs='?',
                        default=parser.prog + ' ' + version,
                        help='used for transmitting version to log')
    parser.add_argument('-d', action='store_true',
                        help='runs debug mode')
    args, unknown = parser.parse_known_args()
    return args


#
# main function, reads in input arguments, opens source file, and begins the checking function
#
def launch_transfer():
    #
    # User input for the csv file path with all the transfer protocols.
    #
    print(sys.argv)
    arg = apid_argparser()
    if not arg.mpath:
        print("No mpath")
    if not arg.email:
        print("No email")
    # cwd = '/'.join(os.getcwd().replace('\\', '/').split('/')[:-1])
    # print(cwd)
    # for root, dirs, files in os.walk(cwd, topdown=True):
    #     if "shared_tools" in dirs:
    #         os.chdir(root)
    #         break
    # cwd = '/'.join(os.getcwd().replace('\\', '/').split('/'))
    # print(cwd)
    #
    # run the file checking and transfer algorithms in an infinite loop
    #
    print("Starting Server Demon for Clinical Specimen...")
    try:
        for ii in range(3):
            paths_data, config_data, cohort_data = update_source_csv(arg)
            if not paths_data:
                sys.exit()
            cwd = '/'.join(os.getcwd().replace('\\', '/').split('/')[:-1])
            print(cwd)
            zip_path = ""
            for root, dirs, files in os.walk(cwd, topdown=False):
                if "7-Zip" in dirs:
                    zip_path = os.path.join(root, "7-Zip")
                    break
            if not zip_path and not arg.no_compress:
                sys.exit()
            directory_waiting = check_ready_files(paths_data, config_data, cohort_data, arg)
            print("DIRECTORIES CHECKED. FOUND " + str(len(directory_waiting)) +
                  " POTENTIAL SAMPLES TO TRANSFER...")
            transfer_loop(directory_waiting, arg, zip_path)
            minutes = 0.1
            print("ALL DIRECTORIES CHECKED SLEEP FOR " + str(minutes) + " MINUTES...")
            wait_time = 60 * minutes
            time.sleep(wait_time)
            print("RECHECKING TRANSFER DIRECTORY")
    except OSError:
        if arg.d:
            return
        error = traceback.format_exc()
        st.send_email(arg.email, error, debug=arg.d)
        traceback.print_exc()
    except SystemExit:
        if arg.d:
            return
        error = "ERROR: Missing source csv files.\n"
        for file in config_data:
            error = error + file + '\n'
        st.send_email(arg.email, error, debug=arg.d)
        traceback.print_exc()


#
# call the function
#
if __name__ == "__main__":
    launch_transfer()
