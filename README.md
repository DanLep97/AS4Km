# AS4Km



[![DOI](https://zenodo.org/badge/460836351.svg)](https://doi.org/10.5281/zenodo.13820275)

![image](./TOC.png)

Repository containing the code to replicate the training and ablation study of the AS4Km paper.

## Replicate training and ablation study
### Requirements
- python 3.10
- torch installed.
- rdkit package version 2024.3.5
- TankBind installed (https://github.com/luwei0917/TankBind) in `tankbind_stuff/TankBind`.
- p2rank installed (https://github.com/rdk/p2rank) in `tankbind_stuff/p2rank_2.3`. The version can be the latest (2.5) but renamed as `p2rank_2.3`. 
### Build the database
- Navigate to https://ftp.expasy.org/databases/enzyme/ and download the latest version of `enzyme.dat` file, put it in the `/data/` folder.
- Navigate to https://www.brenda-enzymes.org/download.php and download the `.json` database, put it in the `/data/` folder under the name `brenda_2023_1.json`.
- Download the HXKm dataset from the GraphKM paper (https://doi.org/10.1186/s12859-024-05746-1), convert to `csv` and save it in `data/hxkm.csv`.
- Run the `src/preprocess/build_db.sh` script to build the database, features, predict the binding sites.

### Train and test AS4Km, replicate the ablation study
- Once the database is ready, go ahead and run `src/replicate_ablation.sh`.
- Figures are replicated by running `src/explore.ipynb` and `src/brenda_viz.ipynb`.
