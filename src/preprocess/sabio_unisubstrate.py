#!/usr/bin/python
# coding: utf-8

# Author: Xiao He
# Date: 2023-08-08


import os
import csv
import tqdm

out = "../../data/KM_sabio_4_unisubstrate.tsv"
outfile = open(out, "wt")

tsv_writer = csv.writer(outfile, delimiter="\t")
tsv_writer.writerow([
    "EntryID", #0
    "Type", #8
    "ECNumber", #6
    "Substrate", #1
    "EnzymeType", #2
    "PubMedID", #3
    "Organism", #4
    "UniprotID", #5
    "Value", #10
    "Unit" #13
])

input_folder = "../../data/sabio_download"

filenames = os.listdir(input_folder)

i = 0
j=0
print("Building unisubstrate.")
for filename in tqdm.tqdm(filenames) :
    if filename != '.DS_Store' :
        with open("%s/%s" % (input_folder, filename), 'r', encoding="utf-8") as file :
            lines = file.readlines()
        for line in lines[1:] :
            data = line.strip().split('\t')
            # print(data)
            try :
                if data[7] == 'Km' :  #and data[9] 
                    i += 1
                    entryID = data[0]
                    for line in lines[1:] :
                        data2 = line.strip().split('\t')
                        if data2[0] == entryID and data2[7] == 'Km': 
                            j += 1
                            tsv_writer.writerow([j, data[7], data[6], data2[1], data[2], data[3], data[4], data[5], data[10], data[-1]])
            except :
                continue
outfile.close()
print("Finished building unisubstrate.")