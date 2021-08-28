import openpyxl
assert openpyxl #make pyflakes happy
import os
import pathlib
import pandas
import smtplib
import numpy as np
from datetime import datetime


#
# calculate size in bytes of a given directory
#
def get_size(filepath):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(filepath):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size


#
# check all _M files and remove duplicates while keeping latest iteration
# current_sor_string is the location of im3 files
# requires a list of filenames, not filepaths, for all files and a list for all files with an "_M#" extension
#
def M_file_handler(current_sor_string, all_files, m_files):
    filepath = current_sor_string + "/MSI/"
    base_names = []
    ext = ".im3"
    #
    # Find which files have multiple instances
    #
    for i1 in range(len(m_files)):
        base = m_files[i1].split(']_M')[0] + ']'
        if base not in base_names:
            base_names.append(base)
    for i2 in range(len(base_names)):
        indices = [f for f, x in enumerate(all_files) if base_names[i2] in x]
        working_files = list(np.array(all_files)[indices])
        if "]_M" not in working_files[0]:
            #
            # if working_file[0] does not contain ']_M' then add it
            #
            working_files[0] = working_files[0].replace("].", "]_M0.")
        #
        # Get number values of all duplicate files and find highest value.
        #
        m_numbers = [int(n[n.find("]_M") + 3:n.find(".")]) for n in working_files]
        keeper = working_files[m_numbers.index(max(m_numbers))]
        #
        # Remove all lower number duplicates and rename highest value as base name
        #
        for i3 in range(len(indices)):
            if os.path.exists(filepath + all_files[indices[i3]]) \
                    and keeper not in all_files[indices[i3]]:
                os.remove(filepath + all_files[indices[i3]])
        os.rename(filepath + keeper,
                  filepath + base_names[i2] + ext)


#
# sends an email when an error occurs
# name is the name of the running script
# err and error_check are for putting out number of attempts and error description
#
def send_email(person_to_email, error_msg, name='Daemon', err=1, error_check_dec=False, debug=False):
    email_user = 'demonemailer@gmail.com'
    email_password = 'Taubelab1'
    sent_from = email_user
    to = [person_to_email]
    subject = name + ' Failed'
    if error_check_dec:
        body = "ERROR AS FOLLOWS: \n" + str(error_msg) + "\n \n" \
               + str(err) + " Attempts were made to overcome this error."
    else:
        body = "ERROR AS FOLLOWS: \n" + str(error_msg)
    email_text = """
    From: %s
    To: %s
    Subject: %s
    %s
    """ % (sent_from, ", ".join(to), subject, body)
    if not debug:
        server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        server.ehlo()
        server.login(email_user, email_password)
        server.sendmail(sent_from, to, email_text)
        server.close()
        print('Email sent!')
    else:
        print(body)


#
# Extract the patient#s and BatchIDs for each specimen
#
def extract_specimens(specimen_path, data_headers):
    specimen_table = pandas.read_excel(specimen_path, engine='openpyxl')
    data_matrix = []
    for i in range(len(data_headers)):
        data_matrix.append(specimen_table[data_headers[i]].tolist())
    return data_matrix


def read_csv(filepath):
    f = open(filepath, 'r')
    lines = f.readlines()
    f.close()
    return lines


def main():
    print("Main")


if __name__ == "__main__":
    main()
