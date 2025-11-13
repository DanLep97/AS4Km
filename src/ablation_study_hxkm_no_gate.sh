#!/bin/bash

# Run this script to replicate raw results for the ablation study.

FOLDS=20
RUNS=10
EPOCHS=100

# Train databases:
CONDITIONED_BS_DB=../data/csv/train_dataset_hxkm_complex_conditioned_bs.csv
UNCONDITIONED_BS_DB=../data/csv/train_dataset_hxkm_complex_unconditioned_bs.csv
DESCRIPTOR_FREE_DB=../data/csv/train_dataset_hxkm_complex_descriptors_free.csv
FINGERPRINT_FREE_DB=../data/csv/train_dataset_hxkm_complex_fingerprint_free.csv
BS_FREE_DB=../data/csv/train_dataset_hxkm_complex_bs_free.csv
AA_ID_FREE_DB=../data/csv/train_dataset_hxkm_complex_aa_id_free.csv
PROTEIN_FREE_DB=../data/csv/train_dataset_hxkm_complex_protein_free.csv
MOLECULE_FREE_DB=../data/csv/train_dataset_hxkm_complex_molecule_free.csv

# Test databases:
CONDITIONED_BS_TEST_DB=../data/csv/HXKm_dataset_final_new_conditioned_bs.csv
UNCONDITIONED_BS_TEST_DB=../data/csv/HXKm_dataset_final_new_unconditioned_bs.csv
DESCRIPTOR_FREE_TEST_DB=../data/csv/HXKm_dataset_final_new_descriptors_free.csv
FINGERPRINT_FREE_TEST_DB=../data/csv/HXKm_dataset_final_new_fingerprint_free.csv
BS_FREE_TEST_DB=../data/csv/HXKm_dataset_final_new_bs_free.csv
AA_ID_FREE_TEST_DB=../data/csv/HXKm_dataset_final_new_aa_id_free.csv
PROTEIN_FREE_TEST_DB=../data/csv/HXKm_dataset_final_new_protein_free.csv
MOLECULE_FREE_TEST_DB=../data/csv/HXKm_dataset_final_new_molecule_free.csv

INFERENCES_CSV=../data/csv/inferences_no_gates.csv
INFERENCES_CONDITIONED_CSV=../data/csv/inferences_conditioned_bs_no_gates.csv

# clean run:
rm $INFERENCES_CSV
rm $INFERENCES_CONDITIONED_CSV
rm -r ../data/models/conditioned_bs_no_gates
rm -r ../data/models/unconditioned_bs_no_gates
rm -r ../data/models/descriptor_free_no_gates
rm -r ../data/models/fingerprint_free_no_gates
rm -r ../data/models/bs_free_no_gates
rm -r ../data/models/aa_id_free_no_gates
rm -r ../data/models/protein_free_no_gates
rm -r ../data/models/molecule_free_no_gates

# Train models:
python trainer.py\
    --name conditioned_bs_no_gates\
    --folds $FOLDS\
    --runs $RUNS\
    --epochs $EPOCHS\
    --db $CONDITIONED_BS_DB\
    --with-seqid\
    > ../data/logs/conditioned_bs_no_gates_$(date +%Y%m%d_%H%M%S).log 2>&1 &

python trainer.py\
    --name unconditioned_bs_no_gates\
    --folds $FOLDS\
    --runs $RUNS\
    --epochs $EPOCHS\
    --db $UNCONDITIONED_BS_DB\
    --with-seqid\
    > ../data/logs/unconditioned_bs_no_gates_$(date +%Y%m%d_%H%M%S).log 2>&1 &

python trainer.py\
    --name descriptor_free_no_gates\
    --folds $FOLDS\
    --runs $RUNS\
    --epochs $EPOCHS\
    --db $DESCRIPTOR_FREE_DB\
    --with-seqid\
    > ../data/logs/descriptor_free_no_gates_$(date +%Y%m%d_%H%M%S).log 2>&1 &

python trainer.py\
    --name fingerprint_free_no_gates\
    --folds $FOLDS\
    --runs $RUNS\
    --epochs $EPOCHS\
    --db $FINGERPRINT_FREE_DB\
    --with-seqid\
    > ../data/logs/fingerprint_free_no_gates_$(date +%Y%m%d_%H%M%S).log 2>&1 &

python trainer.py\
    --name bs_free_no_gates\
    --folds $FOLDS\
    --runs $RUNS\
    --epochs $EPOCHS\
    --db $BS_FREE_DB\
    --with-seqid\
    > ../data/logs/bs_free_seqid_no_gates_$(date +%Y%m%d_%H%M%S).log 2>&1 &

python trainer.py\
    --name aa_id_free_no_gates\
    --folds $FOLDS\
    --runs $RUNS\
    --epochs $EPOCHS\
    --db $AA_ID_FREE_DB\
    --with-seqid\
    > ../data/logs/aa_id_free_no_gates_$(date +%Y%m%d_%H%M%S).log 2>&1 &

python trainer.py\
    --name protein_free_no_gates\
    --folds $FOLDS\
    --runs $RUNS\
    --epochs $EPOCHS\
    --db $PROTEIN_FREE_DB\
    --with-seqid\
    > ../data/logs/protein_free_no_gates_$(date +%Y%m%d_%H%M%S).log 2>&1 &

python trainer.py\
    --name molecule_free_no_gates\
    --folds $FOLDS\
    --runs $RUNS\
    --epochs $EPOCHS\
    --db $MOLECULE_FREE_DB\
    --with-seqid\
    > ../data/logs/molecule_free_no_gates_$(date +%Y%m%d_%H%M%S).log 2>&1 &

echo "waiting for training to be done"
wait

# Test models on trained model:
python tester.py\
    --db $CONDITIONED_BS_TEST_DB\
    --model conditioned_bs_no_gates\
    --csv-output $INFERENCES_CSV\
    --name conditioned_bs_test

python tester.py\
    --db $UNCONDITIONED_BS_TEST_DB\
    --model unconditioned_bs_no_gates\
    --csv-output $INFERENCES_CSV\
    --name unconditioned_bs_test

python tester.py\
    --db $DESCRIPTOR_FREE_TEST_DB\
    --model descriptor_free_no_gates\
    --csv-output $INFERENCES_CSV\
    --name descriptor_free_test

python tester.py\
    --db $FINGERPRINT_FREE_TEST_DB\
    --model fingerprint_free_no_gates\
    --csv-output $INFERENCES_CSV\
    --name fingerprint_free_test

python tester.py\
    --db $BS_FREE_TEST_DB\
    --model bs_free_no_gates\
    --csv-output $INFERENCES_CSV\
    --name bs_free_test

python tester.py\
    --db $AA_ID_FREE_TEST_DB\
    --model aa_id_free_no_gates\
    --csv-output $INFERENCES_CSV\
    --name aa_id_free_test

python tester.py\
    --db $PROTEIN_FREE_TEST_DB\
    --model protein_free_no_gates\
    --csv-output $INFERENCES_CSV\
    --name protein_free_test

python tester.py\
    --db $MOLECULE_FREE_TEST_DB\
    --model molecule_free_no_gates\
    --csv-output $INFERENCES_CSV\
    --name molecule_free_test

# Test models on conditioned:
python tester.py\
    --db $CONDITIONED_BS_TEST_DB\
    --model conditioned_bs_no_gates\
    --csv-output $INFERENCES_CONDITIONED_CSV\
    --name conditioned_bs_test

python tester.py\
    --db $UNCONDITIONED_BS_TEST_DB\
    --model conditioned_bs_no_gates\
    --csv-output $INFERENCES_CONDITIONED_CSV\
    --name unconditioned_bs_test

python tester.py\
    --db $DESCRIPTOR_FREE_TEST_DB\
    --model conditioned_bs_no_gates\
    --csv-output $INFERENCES_CONDITIONED_CSV\
    --name descriptor_free_test

python tester.py\
    --db $FINGERPRINT_FREE_TEST_DB\
    --model conditioned_bs_no_gates\
    --csv-output $INFERENCES_CONDITIONED_CSV\
    --name fingerprint_free_test

python tester.py\
    --db $BS_FREE_TEST_DB\
    --model conditioned_bs_no_gates\
    --csv-output $INFERENCES_CONDITIONED_CSV\
    --name bs_free_test

python tester.py\
    --db $AA_ID_FREE_TEST_DB\
    --model conditioned_bs_no_gates\
    --csv-output $INFERENCES_CONDITIONED_CSV\
    --name aa_id_free_test

python tester.py\
    --db $PROTEIN_FREE_TEST_DB\
    --model conditioned_bs_no_gates\
    --csv-output $INFERENCES_CONDITIONED_CSV\
    --name protein_free_test

python tester.py\
    --db $MOLECULE_FREE_TEST_DB\
    --model conditioned_bs_no_gates\
    --csv-output $INFERENCES_CONDITIONED_CSV\
    --name molecule_free_test