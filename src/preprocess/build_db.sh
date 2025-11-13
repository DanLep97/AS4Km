#!/bin/bash

# Use this script to build the main .csv file database to replicate results.
# To use this script, first download the last version of enzyme.dat file here: https://ftp.expasy.org/databases/enzyme/
# then put the path to downloaded enzyme.dat in this variable:
SABIO_ENZYME_DAT="../../data/enzyme.dat"

# Then, navigate to https://www.brenda-enzymes.org/download.php and download the JSON database, extract it and put the path here:
BRENDA_JSON_DB="../../data/brenda_2023_1.json"

# independant HXKm dataset (supplementary material of GraphKM paper) 
# it's available as excel format, download it and convert to .csv
HXKM="../../data/hxkm.csv"

# BUILD SABIO DATABASE:
echo "BUILDING SABIO DATABASE"
python sabio_download.py --dat $SABIO_ENZYME_DAT
python sabio_unisubstrate.py
python sabio_clean_unisubstrate.py

# BUILD BRENDA DATABASE:
echo "BUILDING BRENDA DATABASE"
python brenda_from_json.py --json-file $BRENDA_JSON_DB

# COMBINE INTO ONE DATABASE:
echo "COMBINING SABIO AND BRENDA"
python combine.py 

# GATHER PROTEIN, MOLECULE INFORMATION AND REMOVE REDUNDANCIES WITH HXKM:
echo "GATHERING FEATURE INFO"
python preprocess.py --hxkm $HXKM

# RUN p2rank:
echo "RUNNING P2RANK"
ls ../../data/structures/*.pdb > structures.txt
../../tankbind_stuff/p2rank_2.3/prank predict structures.txt\
    -threads 12 \
    -o ../../data/p2rank > p2rank_out.txt

# USE p2rank OUTPUT FOR TANKBIND:
echo "RUNNING TANKBIND"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tankbind # use your conda env for tankbind here

# # tankbind on train data
python build_conditioned_bs.py 
# # tankbind on test data
python build_conditioned_bs.py \
    --csv-input ../../data/csv/HXKm_dataset_final_new.csv

# build ablation datasets:
conda activate tankbind_py38
python modify_binding_feature.py \
    --db ../../data/csv/train_dataset_hxkm_complex_conditioned_bs.csv
python modify_binding_feature.py \
    --db ../../data/csv/HXKm_dataset_final_new_conditioned_bs.csv