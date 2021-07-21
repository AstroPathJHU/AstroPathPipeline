#!/usr/bin/env python

import os
import time
import re
import sys
import hashlib
import shutil
import pathlib
import traceback
import subprocess
import pandas
import argparse
from ...shared.logging import getlogger
import logging
from ...shared import shared_tools as st
import lxml.etree as et
from pathlib import Path
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
        specimen_path = '{0}/Specimen_Table.xlsx'.format(sor_string[paths_data_index])
        #
        if not os.path.exists(sor_string[paths_data_index]) \
                or not os.path.exists(des_string[paths_data_index]) \
                or not os.path.exists(comp_string[paths_data_index]) \
                or not os.path.exists(specimen_path):
            continue
        #
        # build directory_waiting data. Uses project # from paths and
        # finds corresponding project # in config and cohort to determine
        # relevant cohort #, space allocation, and delete protocol
        #
        delete_i = config_data[1][config_data_index]
        space_i = config_data[2][config_data_index]
        cohort_i = cohort_data[1][cohort_data_index]
        #
        try:
            specimen_table = pandas.read_excel(specimen_path, engine='openpyxl')
            st_slide_ids = specimen_table['Patient #'].tolist()
            st_batch_ids = specimen_table['Batch ID'].tolist()
        except KeyError:
            error_msg = traceback.format_exc().splitlines()[-1].split(':')[0]
            missing_key = traceback.format_exc().splitlines()[-1].split(':')[1]
            log_string = "WARNING: {0}:{1} missing in {2}." \
                         "\nMake sure 'Patient #' and 'Batch ID' are " \
                         "correctly labeled.".format(error_msg, missing_key, specimen_path)
            print(log_string)
            continue
        #
        # If no AstroIDs have been generated for the current specimen, move on to next specimen
        #
        total_size = 0
        astro_ids = get_astro_id(des_string[paths_data_index], paths_proj[paths_data_index])
        if not astro_ids[0]:
            continue
        #
        # get the paths with batchIDs
        #
        for root, dirs, files in os.walk(sor_string[paths_data_index], topdown=False):
            if "Scan" in root and os.path.exists("{0}/BatchID.txt".format(root)) and \
                    root not in directory_roots:
                #
                regex = '/|\\\\'
                root = '/'.join(re.split(regex, root))
                slide_id = str(root.split('/')[-2])
                #
                # Check for valid SlideIDs and set new AstroID moving forward
                #
                logger_keys = [None, None]
                if slide_id in astro_ids[1]:
                    astro_id = astro_ids[0][astro_ids[1].index(slide_id)]
                elif "Control" in slide_id:
                    logger_keys = [i1, cohort_i]
                    astro_id = slide_id
                else:
                    continue
                #
                # check on whether to skip specimen based on wrong BatchID or insufficient space
                #
                skip = ["", ""]
                #
                file = open("{0}/BatchID.txt".format(root), 'r')
                batch_id = str(file.read()).lstrip("0")
                file.close()
                if slide_id in st_slide_ids:
                    st_batch_id = str(st_batch_ids[st_slide_ids.index(slide_id)])
                else:
                    st_batch_id = ''
                if batch_id != st_batch_id and "Control" not in slide_id:
                    skip[0] = "BatchID"
                #
                # If the directory to be transferred is larger than the space avaliable
                # in the destination directory as given by AstropathConfig.csv, then
                # the directory is not transferred. Updated for each new specimen
                #
                total_size = total_size + st.get_size(root)
                if total_size > space_i * 10 ** 12:
                    skip[1] = "Space"
                #
                row = [comp_string[paths_data_index], des_string[paths_data_index], delete_i, root,
                       astro_id, skip, logger_keys]
                #
                # If automatic has been chosen the DoNotDelete texts files are ignored
                # and deleted. Otherwise only add directory waiting queue if the text
                # file does not exist
                #
                dnd_path = "{0}/DoNotDelete.txt".format(root)
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
        des_dir = direct[1]
        astro_id = direct[4]
        skip = direct[5]
        Project, Cohort = direct[6][0], direct[6][1]
        #
        apidfile = pathlib.Path(des_dir)
        apidfile = apidfile / "upkeep_and_progress" / "AstropathAPIDdef_15.csv"
        #
        # set whether log entries print to console
        #
        if arg.q:
            printthreshold = logging.CRITICAL+1
        else:
            printthreshold = logging.DEBUG
        #
        with getlogger(module="transfer", root=des_dir, samp=astro_id, uselogfiles=True,
                       printthreshold=printthreshold, apidfile=apidfile, Project=Project, Cohort=Cohort) as logger:
            if skip[0]:
                logger.error("BatchID.txt does not match BatchID in Specimen Table")
            if skip[1]:
                logger.error("Insufficient space")
            if skip == ["", ""]:
                transfer_one_sample(direct, arg, zip_path, logger)


#
# transfer a single sample and delete based off of corresponding settings
#
def transfer_one_sample(direct, arg, zip_path, logger):
    comp_dir = direct[0]
    des_dir = direct[1]
    if arg.delete_type == 'manual':
        del_string = 'YES'
    else:
        del_string = direct[2]
    full_sor_dir = direct[3]
    astro_id = direct[4]
    #
    # build full transfer and compression directories
    #
    full_des_dir = "{0}/{1}/im3/{2}".format(des_dir, astro_id, str(full_sor_dir.split('/')[-1]))
    full_comp_dir = "{0}/{1}/im3/{2}".format(comp_dir, astro_id, str(full_sor_dir.split('/')[-1]))
    #
    if os.path.exists(full_des_dir):
        compare = compare_file_names(full_sor_dir, des_dir, full_des_dir,
                                     full_comp_dir, del_string, arg, logger,
                                     astro_id=astro_id)
        if compare == 1:
            return
    #
    # transfer process
    #
    i2 = 2
    while i2 == 2:
        err, result, result2 = error_check("TRANSFER", des_dir, arg, logger, full_sor_dir,
                                           full_des_dir, astro=astro_id)
        if err:
            return
        if not arg.no_compress:
            err, result, result2 = error_check("COMPRESS", des_dir, arg, logger, full_sor_dir,
                                               full_des_dir, full_comp_dir,
                                               astro=astro_id, zip_path=zip_path)
            if err:
                return
            logger.critical("Compression finished")
        #
        # MD5 calculation and file comparison
        #
        err = compare_file_names(full_sor_dir, des_dir, full_des_dir,
                                 full_comp_dir, del_string, arg, logger, 1,
                                 astro_id=astro_id)
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
    astro_id_csv = "{0}/upkeep_and_progress/AstropathAPIDdef_{1}.csv".format(str(des_dir), str(proj))
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
                       current_compress_string, del_string, arg, logger,
                       post_transfer=0, astro_id=""):
    slide_id = str(current_sor_string.split('/')[-2])
    #
    annotation_file = "{0}_{1}_annotations.xml".format(astro_id, current_sor_string.split('/')[-1])
    #
    if post_transfer == 0:
        logger.info("Slide ID is in source and destination on source directory recheck. Comparing files")
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
            logger.critical("Source lost files after transfer")
            logger.info("Error sent. Next slide")
            mail_string = "Source directory has less files after transfer."
            st.send_email(arg.email, mail_string, debug=arg.d)
            #
            return 1
        #
        # delete source path if there are missing files in it
        #
        logger.info("Source directory missing files")
        #
        if not os.path.exists("{0}/DoNotDelete.txt".format(current_sor_string)) \
                and del_string == "YES":
            to_delete = str(Path(current_sor_string).parents[0])
            shutil.rmtree(to_delete, ignore_errors=True)
            #
            logger.info("Deleted source directory")
        #
        # return 1 to continue to next specimen
        #
        return 1
        #
    elif len(file_array[1]) > len(file_array[0]):
        logger.info("Destination directory missing files")
        #
        delete_destination(current_des_string, current_compress_string, logger)
        #
        return 2
    #
    elif len(file_array[1]) == len(file_array[0]):
        location_string = ['DEST', 'SOURCE']
        hash_list = []
        #
        logger.info("MD5 calculations started")
        for x in [0, 1]:
            #
            # if old check sum file exists delete it
            #
            c_hash_path = '{0}/CheckSums.txt'.format(hash_path[x])
            if os.path.exists(c_hash_path):
                os.remove(c_hash_path)
            #
            # compute hash values and store them in hash_array
            #
            error_string = "COMPUTE {0} MD5".format(location_string[x])
            err, hash_value, sums_value = error_check(error_string, main_des_string, arg, logger,
                                                      current_sor_string, current_des_string,
                                                      astro=astro_id)
            if err:
                return err
            hash_list.append(hash_value)
        #
        # remove check sum files
        #
        for x in [0, 1]:
            c_hash_path = '{0}/CheckSums.txt'.format(hash_path[x])
            if os.path.exists(c_hash_path):
                os.remove(c_hash_path)
        logger.info("MD5 calculations finished")
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
            if not os.path.exists("{0}/DoNotDelete.txt".format(current_sor_string)) \
                    and del_string == "YES":
                logger.info("Source and destination match")
                to_delete = str(Path(current_sor_string).parents[0])
                shutil.rmtree(to_delete, ignore_errors=True)
                logger.info("Deleted source directory")
            #
            elif not os.path.exists("{0}/DoNotDelete.txt".format(current_sor_string)) \
                    or not os.path.exists("{0}/DoNotDelete.txt".format(current_des_string)):
                logger.info("Source and destination match")
                create_delete_txt(current_sor_string)
                create_delete_txt(current_des_string)
                logger.info("Created DoNotDelete file")
            logger.info("Processing finished")
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
            logger.info("Destination and source inconsistency")
            delete_destination(current_des_string, current_compress_string, logger)
    return 0


#
# function that deletes the destination when necessary
#
def delete_destination(current_des_string, current_compress_string,
                       logger):
    to_delete = str(Path(current_des_string).parents[0])
    shutil.rmtree(to_delete, ignore_errors=True)
    #
    logger.info("Deleted destination directory")
    #
    # also delete the compressed path if it exists
    #
    if os.path.exists(current_compress_string):
        to_delete = str(Path(current_compress_string).parents[0])
        shutil.rmtree(to_delete, ignore_errors=True)
        logger.info("Deleted compression directory")
    #
    logger.info("Re-initiating transfer process.")


#
# evaluate the functions while checking for errors
#
def error_check(action, main_des_string, arg, logger, current_sor_string="", current_des_string="",
                comp="", astro="", zip_path=""):
    slide_id = str(current_sor_string.split('/')[-2])
    attempts = 1
    err = 0
    warning = ""
    mins = 5
    result = []
    result2 = []
    while err < attempts:
        try:
            if action == "TRANSFER":
                transfer_directory(current_sor_string, main_des_string, current_des_string,
                                   astro, arg, logger)
                logger.critical("Transfer finished")
            elif action == "COMPUTE SOURCE MD5":
                result, result2 = compute_md5(current_sor_string, "SOURCE", logger, slide_id=slide_id,
                                              astro_id=astro)
            elif action == "COMPUTE DEST MD5":
                result, result2 = compute_md5(current_des_string, "DESTINATION", logger,
                                              slide_id=slide_id, astro_id=astro)
            elif action == "COMPRESS":
                compress_directory(current_sor_string, main_des_string, comp, astro,
                                   zip_path, logger)
            elif action == "ANNOTATE":
                xml = [main_des_string, astro, current_sor_string.split('/')[-1],
                       current_sor_string.split('/')[-1]]
                xmlfile = "{0}/{1}/im3/{2}/{3}_{4}_annotations.xml".format(xml[0], xml[1], xml[2], xml[1], xml[3])
                warning = annotation_handler(xmlfile, str(current_sor_string.split('/')[-2]), astro)
            if err > 0:
                logger.warningglobal("{0} passed with {1} error(s)".format(action.lower(), str(err)))
            if warning:
                logger.warning(warning)
            err = attempts
        except OSError:
            err = err + 1
            error_msg = traceback.format_exc().splitlines()[-1].split(':')[0]
            descriptor = traceback.format_exc().splitlines()[-1].split(':')[1]
            logger.warning("attempt {0} failed for {1}".format(str(err), action.lower()))
            #
            if err < attempts:
                logger.info("Attempting to {0} again after {1} minutes".format(action.lower(), str(mins)))
                time.sleep(mins * 60)
                continue
            else:
                #
                # if we have met the allowed count something else must be wrong.
                # Email, return positive err value
                #
                time.sleep(30)
                logger.error(error_msg + descriptor)
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
            logger.error(error_msg + " - " + str(what))
            error = traceback.format_exc()
            st.send_email(arg.email, error, err=err, error_check_dec=True, debug=arg.d)
            return err, result, result2
    err = 0
    return err, result, result2


#
# Generates Hash Values and CheckSums.txt files
#
def compute_md5(current_directory, location_string, logger, slide_id="", astro_id=""):
    #
    # print starting strings to log
    #
    logger.info("Computing {0} MD5 check sums".format(location_string.lower()))
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
            delayed(md5)("{0}/{1}".format(root, file_1)) for file_1 in files)
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
    sums_file = open("{0}/CheckSums.txt".format(current_directory), "w")
    sums_file.writelines(["%s\n" % item for item in sums_array])
    sums_file.close()
    #
    end = time.time()
    logger.info("Completed {0} files in {1} seconds".format(str(num), str(round(end - start, 2))))
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
    text_file = open("{0}/DoNotDelete.txt".format(current_directory), "w")
    text_file.write("Do not delete me unless this folder is going to be removed.")
    text_file.close()


#
# Performs TransferItem() on every file in the source directory
#
def transfer_directory(current_sor_string, main_des_string, current_des_string,
                       astro_id, arg, logger):
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
        logger.info("Duplicate files found")
        st.M_file_handler(current_sor_string, all_files, M_files)
        logger.info("Duplicate files handled")
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
    logger.critical("Transfer process started")
    logger.info("Source Contains {0} File(s) {1} bytes".format(str(n_sor_files), str(n_sor_bytes)))
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
        error_check("ANNOTATE", main_des_string, arg, logger, current_sor_string, astro=astro_id)
    logger.info("Transferred {0} File(s) {1} bytes".format(str(n_des_files), str(n_des_bytes)))


#
# Duplicates existing annotations file and edits the version not labeled "-original"
# to match the new naming convention
#
def annotation_handler(xmlfile, slide_id, astro_id):
    if not os.path.exists(xmlfile):
        return "{0} does not exist".format(str(xmlfile))
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
def compress_directory(current_sor_string, main_des_string, compress, astro_id,
                       zip_path, logger):
    #
    # get the number of files and bytes in the source directory
    #
    logger.critical("Compression started")
    n_sor_files, n_sor_bytes = 0, 0
    xmlfile = "{0}_{1}_annotations.xml".format(astro_id, current_sor_string.split('/')[-1])
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
    logger.info("Compressing {0} file(s) and {1} bytes from source".format(str(n_sor_files), str(n_sor_bytes)))
    #
    # do the compression one file at a time
    #
    if not os.path.exists(compress):
        pathlib.Path(compress).mkdir(parents=True, exist_ok=True)
    #
    for item in os.listdir(current_sor_string):
        compress_item(item, current_sor_string, compress, names, zip_path)
    annotation_path = "{0}/{1}/im3/{2}/".format(main_des_string, astro_id, current_sor_string.split('/')[-1])
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
    logger.info("Compressed {0} total file(s) {1} total bytes".format(str(n_des_files), str(n_des_bytes)))


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
    path = "{0}/AstropathPaths.csv".format(arg.mpath)
    config = "{0}/AstropathConfig.csv".format(arg.mpath)
    cohort_csv = "{0}/AstropathCohortsProgress.csv".format(arg.mpath)
    #
    # Catches and alerts user to which cource files weren't found
    #
    all_files = [path, config, cohort_csv]
    a_exist = [f for f in all_files if os.path.isfile(f)]
    a_non_exist = list(set(a_exist) ^ set(all_files))
    if a_non_exist:
        return [], a_non_exist, []
    #
    paths = st.read_csv(path)
    configs = st.read_csv(config)
    cohorts = st.read_csv(cohort_csv)
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
    print(sys.argv)
    arg = apid_argparser()
    if not arg.mpath:
        print("No mpath")
        sys.exit()
    if not arg.email:
        print("No email")
        sys.exit()
    #
    checker = []
    print("Starting Server Demon for Clinical Specimen...")
    try:
        while True:
            paths_data, config_data, cohort_data = update_source_csv(arg)
            if not paths_data:
                checker = config_data
                sys.exit()
            cwd = '/'.join(os.getcwd().replace('\\', '/').split('/')[:-1])
            print(cwd)
            zip_path = ""
            for root, dirs, files in os.walk(cwd, topdown=False):
                if "7-Zip" in dirs:
                    zip_path = os.path.join(root, "7-Zip")
                    break
            if not zip_path and not arg.no_compress:
                checker = ["7zip"]
                sys.exit()
            directory_waiting = check_ready_files(paths_data, config_data, cohort_data, arg)
            print("DIRECTORIES CHECKED. FOUND " + str(len(directory_waiting)) +
                  " POTENTIAL SAMPLES TO TRANSFER...")
            transfer_loop(directory_waiting, arg, zip_path)
            minutes = 30
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
        error = ""
        for file in checker:
            error = "{0}ERROR: Missing  \n{1}\n".format(error, file)
        print(error)
        st.send_email(arg.email, error, debug=arg.d)
        traceback.print_exc()
    except KeyboardInterrupt:
        print("Transfer Daemon Closed")


#
# call the function
#
if __name__ == "__main__":
    launch_transfer()
