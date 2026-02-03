import glob
import pandas
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
from scipy import stats
import random
from dataset import KmClass
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats
from torcheval.metrics.functional import r2_score
from torchmetrics import PearsonCorrCoef
from model import Network
from hyperparameters import hyperparameters
from rdkit import DataStructs
import numpy
import numpy as np
from scipy import stats

def plot_prot_lig_clustered(test_name):
    pcc = PearsonCorrCoef()
    hxkm = pandas.read_csv("../data/hxkm.csv")
    df_train = pandas.read_csv("../data/csv/train_dataset_hxkm_complex_conditioned_bs.csv")
    train_db = KmClass(df_train).dataframe
    df_test = pandas.read_csv(f"../data/csv/HXKm_dataset_final_new_{test_name}.csv")
    # df_test = pandas.read_csv("../data/csv/HXKm_dataset_final_new_bs_free.csv")
    test_db = KmClass(df_test).dataframe
    clusters = pandas.read_csv("../data/enzyme_test_vs_train.tsv", sep="\t")
    columns = "query,target,pident,evalue,qstart,qend,qlen,tstart,tend,tlen".split(",")
    clusters.columns = columns 
    clusters.reset_index(drop=True, inplace=True)

    # enzyme thresholds per unique query:
    query_pidents = {
        query: clusters.loc[clusters["query"] == query].pident
        for query in clusters["query"].unique()
    }

    # substrate thresholds per unique smiles:
    def bitvect(bit_array, n_bits=2048):
        fp = DataStructs.ExplicitBitVect(n_bits)
        
        # Set the bits that are 1
        indices = numpy.where(bit_array == 1)[0]
        for idx in indices:
            fp.SetBit(int(idx))
        
        return fp

    train_fingerprints = [bitvect(row.values) for _,row in train_db.iloc[:,-2049:-1].iterrows()]
    test_fingerprints = [bitvect(row.values) for _,row in test_db.iloc[:,-2049:-1].iterrows()]
    tanimoto_matrix = torch.tensor([
        [DataStructs.TanimotoSimilarity(test_fp, train_fp) for train_fp in train_fingerprints]
        for test_fp in test_fingerprints
    ]) # each row are similarities to one given test entry

    output_files = glob.glob(f"../data/models/conditioned_bs_full_features/*_{test_name}_test_outputs.pkl")

    def compute_significance(ref, test):
        _, p_value = stats.ttest_rel(ref, test)
        significance = ""
        if p_value < 0.001:
            significance = "***"
        elif p_value < 0.01:
           significance =  "**"
        elif p_value < 0.05:
           significance =  "*"
        return significance

    # threshold data:
    enz_clusters = {
        100: {
            "pcc": [],
            "r2": [],
            "n": [],
        },
        99: {
            "pcc": [],
            "r2": [],
            "n": [],
            "significance": None,
        },
        80: {
            "pcc": [],
            "r2": [],
            "n": [],
            "significance": None,
        },
        60: {
            "pcc": [],
            "r2": [],
            "n": [],
            "significance": None,
        },
        40: {
            "pcc": [],
            "r2": [],
            "n": [],
            "significance": None,
        }
    }
    lig_clusters = {
        100: {
            "pcc": [],
            "r2": [],
            "n": [],
        },
        99: {
            "pcc": [],
            "r2": [],
            "n": [],
            "significance": None,
        },
        80: {
            "pcc": [],
            "r2": [],
            "n": [],
            "significance": None,
        },
        60: {
            "pcc": [],
            "r2": [],
            "n": [],
            "significance": None,
        },
        40: {
            "pcc": [],
            "r2": [],
            "n": [],
            "significance": None,
        }
    }

    for file in output_files:
        outputs = pickle.load(open(file, "rb"))
        y = torch.tensor(outputs["y_scaled"])
        preds = torch.tensor(outputs["preds_scaled"])
        indices = outputs["all_idx"].flatten()
        test_entries = test_db.iloc[indices]

        uniprot_keys = test_entries.apply(
            lambda x: hxkm.loc[hxkm.sequence == x.sequence].uniprot_key.values[0],
            axis=1
        )
        uniprot_keys.reset_index(drop=True, inplace=True)
        for threshold in enz_clusters.keys():
            # get below threshold uniprot ids:
            threshold_queries = set([
                q for q,pidents in query_pidents.items()
                if all((pidents <= threshold).tolist())
            ])
            # compute scores and save them:
            threshold_output_idx = uniprot_keys[
                uniprot_keys.isin(threshold_queries)
            ].index.tolist()
            threshold_preds = preds[threshold_output_idx]
            threshold_y = y[threshold_output_idx]
            r2 = r2_score(threshold_preds, threshold_y).item()
            pearson = pcc(threshold_preds, threshold_y).item()
            enz_clusters[threshold]["r2"].append(r2)
            enz_clusters[threshold]["pcc"].append(pearson)
            enz_clusters[threshold]["n"].append(len(threshold_output_idx))
        

        # scores based on substrate similarity:
        for threshold in lig_clusters:
            t = threshold/100
            similarity_matrix_nonzero = (tanimoto_matrix <= t).nonzero()[:,0]
            threshold_output_idx = (
                similarity_matrix_nonzero.bincount() == tanimoto_matrix.shape[1]
            ).nonzero()[:,0].tolist()
            # compute scores and save them:
            threshold_preds = preds[threshold_output_idx]
            threshold_y = y[threshold_output_idx]
            r2 = r2_score(threshold_preds, threshold_y).item()
            pearson = pcc(threshold_preds, threshold_y).item()
            lig_clusters[threshold]["r2"].append(r2)
            lig_clusters[threshold]["pcc"].append(pearson)
            lig_clusters[threshold]["n"].append(len(threshold_output_idx))
    # add significance:
    r2_ref = enz_clusters[100]["r2"]
    for threshold in enz_clusters.keys():
        if "significance" not in enz_clusters[threshold].keys(): # skip the ref
            continue
        r2 = enz_clusters[threshold]["r2"]
        significance = compute_significance(r2_ref, r2)
        enz_clusters[threshold]["significance"] = significance

    r2_ref = lig_clusters[100]["r2"]
    for threshold in lig_clusters.keys():
        if "significance" not in lig_clusters[threshold].keys(): # skip the ref
            continue
        r2 = lig_clusters[threshold]["r2"]
        significance = compute_significance(r2_ref, r2)
        lig_clusters[threshold]["significance"] = significance

    # Assuming enz_clusters and lig_clusters are defined with your data
    # Calculate means for each threshold
    thresholds = sorted(enz_clusters.keys())
    enz_pcc_means = [numpy.mean(enz_clusters[t]['pcc']) for t in thresholds]
    enz_r2_means = [numpy.mean(enz_clusters[t]['r2']) for t in thresholds]
    lig_pcc_means = [numpy.mean(lig_clusters[t]['pcc']) for t in thresholds]
    lig_r2_means = [numpy.mean(lig_clusters[t]['r2']) for t in thresholds]

    # Calculate standard deviations for error bars
    enz_pcc_stds = [numpy.std(enz_clusters[t]['pcc']) for t in thresholds]
    enz_r2_stds = [numpy.std(enz_clusters[t]['r2']) for t in thresholds]
    lig_pcc_stds = [numpy.std(lig_clusters[t]['pcc']) for t in thresholds]
    lig_r2_stds = [numpy.std(lig_clusters[t]['r2']) for t in thresholds]

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3))

    # Bar width and x positions
    x = numpy.arange(len(thresholds))
    width = 0.35

    # Plot 1: Enzyme-based clustering
    bars1 = ax1.bar(x - width/2, enz_r2_means, width, 
                    label='R²', color='orange', alpha=0.8,
                    yerr=enz_r2_stds, capsize=5,
                    error_kw={'elinewidth': 1.5, 'ecolor': 'black'})
    bars2 = ax1.bar(x + width/2, enz_pcc_means, width, 
                    label='Pearson', color='purple', alpha=0.8,
                    yerr=enz_pcc_stds, capsize=5,
                    error_kw={'elinewidth': 1.5, 'ecolor': 'black'})

    ax1.set_xlabel('Clustering Threshold (%)', fontsize=12)
    ax1.set_ylabel('Score Value', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{t}%' for t in thresholds])
    ax1.tick_params(axis="both", labelsize=12)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.text(-.15, 0.95, "A", fontsize=20, fontweight='bold', transform=ax1.transAxes)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for i, bar in enumerate(bars):
            height = bar.get_height()
            txt = enz_clusters[thresholds[i]]["n"][0]
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'n={txt}', ha='center', va='bottom', fontsize=12)
    # add significances:
    for i, bar in enumerate(bars1):
        height = bar.get_height() + 0.3
        if "significance" in enz_clusters[thresholds[i]].keys():
            txt = enz_clusters[thresholds[i]]["significance"]
        else:
            txt = ""
        ax1.text(
            bar.get_x() + bar.get_width()/2., 
            height, txt, ha='center', va='bottom', fontsize=13, color="red",
        )

    # Plot 2: Ligand-based clustering  
    bars3 = ax2.bar(x - width/2, lig_r2_means, width,
                    label='R²', color='orange', alpha=0.8,
                    yerr=lig_r2_stds, capsize=5,
                    error_kw={'elinewidth': 1.5, 'ecolor': 'black'})
    bars4 = ax2.bar(x + width/2, lig_pcc_means, width,
                    label='Pearson', color='purple', alpha=0.8,
                    yerr=lig_pcc_stds, capsize=5,
                    error_kw={'elinewidth': 1.5, 'ecolor': 'black'})

    ax2.set_xlabel('Clustering Threshold (%)', fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'{t}%' for t in thresholds])
    ax2.legend(fontsize=12, loc="lower right")
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.text(-.15, 0.95, "B", fontsize=20, fontweight='bold', transform=ax2.transAxes)
    ax2.tick_params(axis="both", labelsize=12)

    # Add value labels on bars
    for bars in [bars3, bars4]:
        for i, bar in enumerate(bars):
            height = bar.get_height()
            txt = lig_clusters[thresholds[i]]["n"][0]
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'n={txt}', ha='center', va='bottom', fontsize=12)

    # add significances:
    for i, bar in enumerate(bars3):
        height = bar.get_height() + 0.3
        if "significance" in lig_clusters[thresholds[i]].keys():
            txt = lig_clusters[thresholds[i]]["significance"]
        else:
            txt = ""
        ax2.text(
            bar.get_x() + bar.get_width()/2., 
            height, txt, ha='center', va='bottom', fontsize=13, color="red",
        )

    # Adjust layout
    plt.tight_layout()
    plt.savefig(f"../figures/scores_on_clustered_{test_name}.jpg", dpi=600, bbox_inches='tight')
    plt.savefig(f"../figures/scores_on_clustered_{test_name}.tiff", dpi=600, bbox_inches='tight')
    plt.show()


def plot_gating_weights():
    """Plot gating weight contributions from neural network"""
    
    # Set up device and network
    device = ("cpu", "cuda")[torch.cuda.is_available()]
    net = Network(
        hidden_dim1=hyperparameters["hidden_dim1"], 
        hidden_dim2=hyperparameters["hidden_dim2"], 
        hidden_dim3=hyperparameters["hidden_dim3"], 
        dropout1=hyperparameters["dropout1"], 
        dropout2=hyperparameters["dropout2"]
    ).to(device)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(12, 3))
    gs = fig.add_gridspec(1, 3, width_ratios=[2, 0.1, 1], wspace=0.1)
    
    # ====================================================================
    # PANEL A: Scatter plot of feature importance
    # ====================================================================
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Load model and get gating weights
    params = torch.load("../data/models/conditioned_bs_full_features/1_fold_model.pth")["model_state_dict"]
    net.load_state_dict(params)
    gated_layer = net.net[0].gate.weight.data.sum(dim=0).softmax(dim=0).detach().cpu()
    
    # Plot enzyme features
    res_feats = 1024 * 3
    enzyme_x = list(range(res_feats))
    enzyme_y = gated_layer[:res_feats].numpy()
    
    # Plot substrate features
    substrate_x = list(range(res_feats, gated_layer.shape[0]))
    substrate_y = gated_layer[res_feats:].numpy()
    
    ax1.plot(enzyme_x, enzyme_y, color='red', linewidth=1.5, label='Enzyme')
    ax1.plot(substrate_x, substrate_y, color='blue', linewidth=1.5, label='Substrate')
    
    ax1.set_xlabel('Feature Index', fontsize=16)
    ax1.set_ylabel('Importance Score', fontsize=16)
    ax1.legend(fontsize=14, loc='upper left', ncols=2)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='both', labelsize=14)
    ax1.set_yscale('log')  # For scientific notation
    ax1.set_ylim(0, 0.0035)
    
    # Add annotation A
    ax1.text(-0.12, .95, 'A', transform=ax1.transAxes, 
             fontsize=20, fontweight='bold', va='bottom', ha='right')
    
    # ====================================================================
    # PANEL B: Bar plot of main weight contributions
    # ====================================================================
    ax2 = fig.add_subplot(gs[0, 2])
    
    # Process all models
    bs_weights, aa_weights, seqid_weights = [], [], []
    protein_weights, descriptors_weights, fingerprint_weights = [], [], []
    molecule_weights = []
    
    model_files = glob.glob("../data/models/conditioned_bs_full_features/*_fold_model.pth")
    for model_file in model_files:
        params = torch.load(model_file)["model_state_dict"]
        net.load_state_dict(params)
        gated_layer = net.net[0].gate.weight.data.sum(dim=0).softmax(dim=0).detach().cpu()
        
        # Calculate weight sums
        bs_weight = gated_layer[1:res_feats:3].sum()
        aa_weight = gated_layer[0:res_feats:3].sum()
        seqid_weight = gated_layer[2:res_feats:3].sum()
        descriptors_weight = gated_layer[res_feats:res_feats+196].sum()
        fingerprint_weight = gated_layer[-2048:].sum()
        
        bs_weights.append(bs_weight)
        aa_weights.append(aa_weight)
        seqid_weights.append(seqid_weight)
        protein_weights.append(bs_weight + aa_weight + seqid_weight)
        descriptors_weights.append(descriptors_weight)
        fingerprint_weights.append(fingerprint_weight)
        molecule_weights.append(descriptors_weight + fingerprint_weight)
    
    # Convert to tensors
    bs_weights = torch.tensor(bs_weights)
    aa_weights = torch.tensor(aa_weights)
    seqid_weights = torch.tensor(seqid_weights)
    protein_weights = torch.tensor(protein_weights)
    descriptors_weights = torch.tensor(descriptors_weights)
    fingerprint_weights = torch.tensor(fingerprint_weights)
    molecule_weights = torch.tensor(molecule_weights)
    
    # Calculate means and stds for main plot
    weights_mean_main = [
        protein_weights.mean().item(),
        molecule_weights.mean().item()
    ]
    weights_std_main = [
        protein_weights.std().item(),
        molecule_weights.std().item()
    ]
    
    # Create bar plot
    categories = ["Enzyme", "Substrate"]
    colors = ["red", "blue"]
    
    bars = ax2.bar(categories, weights_mean_main, color=colors, 
                   edgecolor='black', linewidth=1.5, width=0.6)
    
    # Add error bars
    for bar, std in zip(bars, weights_std_main):
        x_pos = bar.get_x() + bar.get_width() / 2
        y_pos = bar.get_height()
        ax2.errorbar(x_pos, y_pos, yerr=std, fmt='none', 
                    color='black', capsize=5, capthick=1.5, elinewidth=1.5)
    
    ax2.set_ylabel('Importance Score', fontsize=16)
    ax2.tick_params(axis='x', labelsize=14, rotation=0)
    ax2.tick_params(axis='y', labelsize=14)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add annotation B
    ax2.text(-0.15, 0.95, 'B', transform=ax2.transAxes, 
             fontsize=20, fontweight='bold', va='bottom', ha='right')
    
    # Sanity check
    print(f"Sanity check - Protein+Molecule sum: {(protein_weights + molecule_weights).mean():.6f}")
    
    # ====================================================================
    # Create supplementary figure (optional separate plot)
    # ====================================================================
    def create_supplementary_figure():
        """Create supplementary figure with detailed breakdown"""
        fig_supp, ax_supp = plt.subplots(figsize=(10, 4))
        
        # Calculate means and stds for supplementary plot
        weights_mean_supp = [
            bs_weights.mean().item(),
            aa_weights.mean().item(),
            seqid_weights.mean().item(),
            descriptors_weights.mean().item(),
            fingerprint_weights.mean().item()
        ]
        weights_std_supp = [
            bs_weights.std().item(),
            aa_weights.std().item(),
            seqid_weights.std().item(),
            descriptors_weights.std().item(),
            fingerprint_weights.std().item()
        ]
        
        categories_supp = ["Active Site", "AA Identity", "Positional Encoding", 
                          "Descriptors", "Fingerprints"]
        colors_supp = ["red", "red", "red", "blue", "blue"]
        
        bars_supp = ax_supp.bar(categories_supp, weights_mean_supp, 
                                color=colors_supp, edgecolor='black', 
                                linewidth=1.5, width=0.7)
        
        # Add error bars
        for bar, std in zip(bars_supp, weights_std_supp):
            x_pos = bar.get_x() + bar.get_width() / 2
            y_pos = bar.get_height()
            ax_supp.errorbar(x_pos, y_pos, yerr=std, fmt='none',
                           color='black', capsize=5, capthick=1.5, elinewidth=1.5)
        
        
        ax_supp.set_ylabel('Importance Score', fontsize=16)
        ax_supp.tick_params(axis='x', labelsize=16, rotation=20)
        ax_supp.tick_params(axis='y', labelsize=16)
        ax_supp.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig("../figures/supp_fig_4.jpg", dpi=600, bbox_inches='tight')
        plt.savefig("../figures/supp_fig_4.tiff", dpi=600, bbox_inches='tight')
        plt.show()
        return fig_supp
    
    # Save and show main figure
    plt.tight_layout()
    plt.savefig("../figures/figure_6.jpg", dpi=600, bbox_inches='tight')
    plt.savefig("../figures/figure_6.tiff", dpi=600, bbox_inches='tight')
    plt.show()
    create_supplementary_figure()

def plot_all_ablation(
        experiment_filenames,
        experiment_conditions
    ):
    zipped = zip(experiment_filenames, experiment_conditions)
    labels = {
        "protein_free_test": "Enzyme free",
        "aa_id_free_test": "AA identity free",
        "bs_free_test": "AS free",
        "unconditioned_bs_test": "Unconditioned AS",
        "conditioned_bs_test": "Conditioned AS",
        "molecule_free_test": "Substrate free",
        "descriptor_free_test": "Descriptor free",
        "fingerprint_free_test": "Fingerprint free",
    }
    fig_data = []
    for csv, cond_name in zipped:
        inferences = pandas.read_csv(csv)
        test_metrics = inferences.loc[~inferences.name.str.contains("_train")]
        grouped = test_metrics.groupby(["name"], as_index=False).agg(
            r2_mean=pandas.NamedAgg(column="r2", aggfunc="mean"),
            r2_std=pandas.NamedAgg(column="r2", aggfunc="std"),
            p_mean=pandas.NamedAgg(column="pearson", aggfunc="mean"), 
            p_std=pandas.NamedAgg(column="pearson", aggfunc="std"),
            mse_mean=pandas.NamedAgg(column="mse", aggfunc="mean"), 
            mse_std=pandas.NamedAgg(column="mse", aggfunc="std")
        )
        plot_order = [grouped[grouped.name == "conditioned_bs_test"].index[0]] + \
            [grouped[grouped.name == n].index[0] for n in labels.keys() if n != "conditioned_bs_test"]
        grouped = grouped.reindex(plot_order)
        fig_data.extend([
            go.Bar(
                name=cond_name, x=grouped.name.values, y=grouped.r2_mean.values, 
                error_y={
                    "type":"data",
                    "array": grouped.r2_std.values,
                    "visible": True
                },
            ),
        ])

    fig = go.Figure(data=fig_data)
    fig.update_xaxes(labelalias=labels)
    fig.update_yaxes(title_text="R²")
    fig.update_layout(
        margin=dict(t=5, b=5, r=0, l=50),
        font=dict(size=19),
        width=1000,
        height=400,
        legend={
            "font": {"size":14}
        }
    )
    fig.add_annotation(
        x=-0.08,
        y=.95,
        showarrow=False,
        yref="paper",
        xref="paper",
        text="B",
        align="center",
        font={"weight": 800}
    )
    fig.write_image("../figures/all_ablations.jpg", width=1000, height=400)
    fig.show()

def plot_ablation(inferences_file, fig_title, file_name, with_table=False, with_title=False):
    inferences = pd.read_csv(inferences_file)
    test_metrics = inferences.loc[~inferences.name.str.contains("_train")]

    # compute statistics:
    model_r2 = {group: state.r2.values for group, state in test_metrics.groupby("name")}
    ref_sample = model_r2.pop("conditioned_bs_test")
    p_values = {}
    significances = {}
    for model, r2 in model_r2.items():
        _, p_value = stats.ttest_rel(ref_sample, r2)
        p_values[model] = p_value
        if p_value < 0.001:
            significances[model] = "***"
        elif p_value < 0.01:
            significances[model] = "**"
        elif p_value < 0.05:
            significances[model] = "*"

    # rank by r2
    ranked_r2 = {g: r2.mean() for g, r2 in model_r2.items()}
    ranked_r2 = dict(sorted(ranked_r2.items(), key=lambda i: i[1], reverse=True))

    grouped = test_metrics.groupby(["name"], as_index=False).agg(
        r2_mean=pd.NamedAgg(column="r2", aggfunc="mean"),
        r2_std=pd.NamedAgg(column="r2", aggfunc="std"),
        p_mean=pd.NamedAgg(column="pearson", aggfunc="mean"), 
        p_std=pd.NamedAgg(column="pearson", aggfunc="std"),
        mse_mean=pd.NamedAgg(column="mse", aggfunc="mean"), 
        mse_std=pd.NamedAgg(column="mse", aggfunc="std")
    )

    # Get the count for each model
    model_counts = test_metrics.groupby("name").size()
    grouped['count'] = grouped['name'].map(model_counts)

    # Define ESM models and regular models
    esm_models = ["esm_test", "esm_as_test", "esm_enz_test", "conditioned_bs_test"]
    regular_models = [
        n for n in ranked_r2.keys()
        if n not in esm_models
    ]

    # Create two separate groups
    regular_grouped = grouped[grouped['name'].isin(regular_models + ["conditioned_bs_test"])].copy()
    esm_grouped = grouped[grouped['name'].isin(esm_models)].copy()

    # Order regular models
    regular_plot_order = [regular_grouped[regular_grouped.name == "conditioned_bs_test"].index[0]] + \
                        [regular_grouped[regular_grouped.name == n].index[0] for n in regular_models if n != "conditioned_bs_test" and n in regular_grouped['name'].values]
    regular_grouped = regular_grouped.reindex(regular_plot_order)
    regular_grouped.reset_index(inplace=True, drop=True)

    # Order ESM models with Conditioned AS first
    esm_plot_order = [esm_grouped[esm_grouped.name == "conditioned_bs_test"].index[0]] + \
                    [esm_grouped[esm_grouped.name == n].index[0] for n in esm_models if n != "conditioned_bs_test" and n in esm_grouped['name'].values]
    esm_grouped = esm_grouped.reindex(esm_plot_order)
    esm_grouped.reset_index(inplace=True, drop=True)

    labels = {
        "conditioned_bs_test": "Conditioned AS",
        "unconditioned_bs_test": "Unconditioned AS\n(AS4Km)",
        "aa_id_free_test": "AA identity free",
        "bs_free_test": "AS free",
        "protein_free_test": "Enzyme free",
        "descriptor_free_test": "Descriptor free",
        "fingerprint_free_test": "Fingerprint free",
        "molecule_free_test": "Substrate free",
        "esm_test": "ESM (enzyme + AS)",
        "esm_as_test": "AS ESM",
        "esm_enz_test": "Enzyme ESM"
    }

    # Function to create a bar plot
    def create_bar_plot(fig_size, grouped_data, show_legend=False, letter="A"):
        fig, ax = plt.subplots(figsize=fig_size)
        
        x = np.arange(len(grouped_data))
        width = 0.40
        
        # Get x labels with aliases
        x_labels = [labels.get(name, name) for name in grouped_data['name'].values]
        
        # Plot R2 bars with error bars
        r2_bars = ax.bar(x - width/2, grouped_data.r2_mean.values, width, 
                        yerr=grouped_data.r2_std.values, 
                        capsize=5, error_kw={'elinewidth': 2, 'ecolor': 'red'},
                        label='R$^2$', color='orange', edgecolor='black', linewidth=1)
        
        # Plot Pearson bars with error bars
        p_bars = ax.bar(x + width/2, grouped_data.p_mean.values, width, 
                        yerr=grouped_data.p_std.values,
                        capsize=5, error_kw={'elinewidth': 2, 'ecolor': 'red'},
                        label='Pearson', color='purple', edgecolor='black', linewidth=1)
        
        # Add significance annotations (only for models being compared to Conditioned AS)
        for model, significance in significances.items():
            if model in grouped_data['name'].values and model != "conditioned_bs_test":
                idx = grouped_data[grouped_data['name'] == model].index[0]
                ax.text(idx, 0.61, significance, ha='center', va='bottom', 
                        fontsize=16, fontweight='bold', color='red')
        
        # Add value labels on bars
        def add_value_labels(bars, ax, values, color, decimals=3):
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height/3,
                        f'{value:.{decimals}f}', ha='center', va='center', 
                        fontsize=11, fontweight='bold', color=color)
        
        add_value_labels(r2_bars, ax, grouped_data.r2_mean.values, "black")
        add_value_labels(p_bars, ax, grouped_data.p_mean.values, "white")
        
        # Customize the plot
        ax.set_ylabel('Score', fontsize=16)
        if with_title:
            ax.set_title(fig_title, fontsize=16, pad=15)
        
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=14, rotation=20, ha='right')
        ax.tick_params(axis='y', labelsize=16)
        
        if show_legend:
            ax.legend(fontsize=14, loc='lower left')
        
        ax.grid(True, alpha=0.25, axis='y')
        
        # Adjust y-axis
        y_max = max(grouped_data.r2_mean.max(), grouped_data.p_mean.max())
        if show_legend:
            ax.set_ylim([-.3, 0.7])
        else:
            ax.set_ylim([0, 0.73])
        ax.text(-.07, 0.95, letter, fontsize=24, fontweight='bold', transform=ax.transAxes)
        
        plt.tight_layout()
        return fig

    # Create first figure (regular models)
    fig1 = create_bar_plot((11, 5), regular_grouped, show_legend=True, letter="A")
    file_name1 = file_name.replace('.jpg', '_regular.jpg').replace('.tiff', '_regular.tiff')
    fig1.savefig(file_name1, dpi=600, bbox_inches='tight')
    fig1.savefig(file_name1.replace(".jpg", ".tiff"), dpi=600, bbox_inches='tight')
    plt.show()
    plt.close(fig1)

    # Create second figure (ESM models)
    fig2 = create_bar_plot((8, 4), esm_grouped, show_legend=False, letter="B")
    file_name2 = file_name.replace('.jpg', '_esm.jpg').replace('.tiff', '_esm.tiff')
    fig2.savefig(file_name2, dpi=600, bbox_inches='tight')
    fig2.savefig(file_name2.replace(".jpg", ".tiff"), dpi=600, bbox_inches='tight')
    plt.show()
    plt.close(fig2)
    
    # Print table if requested
    cols_str = ""
    if with_table:
        for exp_name, exp_label in labels.items():
            if exp_name in grouped['name'].values:
                gr = grouped.loc[grouped.name == exp_name]
                n = gr['count'].values[0] if 'count' in gr.columns else test_metrics.loc[test_metrics.name == exp_name].shape[0]
                avg = [gr[c].values[0] for c in ["r2_mean", "p_mean", "mse_mean"]]
                std = [gr[c].values[0] for c in ["r2_std", "p_std", "mse_std"]]
                p_value = f"{p_values[exp_name]:.3e}" if exp_name != "conditioned_bs_test" else "ref"

                avg_std_str = "& ".join([f"{a:.3f} $\pm$ {s:.3f}" for a,s in zip(avg, std)])
                col_str = exp_label + "& " + avg_std_str + f"& {n}" + f"& {p_value}" + "\\\\"
                cols_str += "\hline \n" + col_str + "\n"
        print(cols_str)
    
    return grouped

def plot_learning_curve_fig(y_train, y_val):
    # build learning curve figs:
    assert len(y_train) == len(y_val), "y_train and y_val don't have the same length in plot_learning_curv_fig"
    fig = go.Figure()

    for train, val in zip(y_train, y_val):
        # random color generator:
        bits = list(range(255))
        r = random.sample(bits, 1)[0]
        g = random.sample(bits, 1)[0]
        b = random.sample(bits, 1)[0]

        x = list(range(len(train)))
        fig.add_trace(go.Scatter(
            x=x, y=train,
            line_color=f'rgb({r},{g},{b})',
            name="Train",
            showlegend=False,
            line_dash="dot"
        ))
        fig.add_trace(go.Scatter(
            x=x, y=val,
            line_color=f'rgb({r},{g},{b})',
            name="Validation",
            showlegend=False
        ))
    fig.update_traces(mode="lines")
    fig.update_layout(height=800, width=800)
    return fig

def learning_curves(
    experiments, experiment_titles
):
    figure_data = []
    zipped = zip(experiments, experiment_titles)
    for i, (experiment, title) in enumerate(zipped):
        figure_data.append(
            plot_val_train_metrics(experiment, title)
        )
    y_axes = [y for f_data in figure_data for y in f_data["titles"]["y_axis"]] 
    figures = [f for f_data in figure_data for f in f_data["figs"]] 
    # build final fig:
    rows = 8
    cols = 3
    big_fig = make_subplots(
        rows=rows, cols=cols,
        vertical_spacing=0.02
    )
    for i in range(22):
        col = i % cols + 1
        row = i // cols + 1 
        for trace in figures[i]["data"]:
            big_fig.add_trace(trace, row=row, col=col)
        big_fig.update_yaxes(title_text=y_axes[i], row=row, col=col)
        if i % 2 != 0:
            big_fig.update_yaxes(range=[0, 1], col=col, row=row)
        else:
            big_fig.update_yaxes(range=[0, 0.030], col=col, row=row)


    big_fig.update_layout(
        margin=dict(t=0,b=0,l=0,r=0),
        font=dict(size=18),
    )
    big_fig.write_image("../figures/supp_fig_2.jpg", height=1300, width=1800)
    return big_fig


def plot_val_train_metrics(
        model, title
):
    # load train metrics
    folder_model = f"../data/models/{model}"
    train_val_metrics_files = glob.glob(f"{folder_model}/*metrics.pth")
    train_val_metrics_fold = [f.split("/")[-1].split("_")[0] for f in train_val_metrics_files]
    train_val_metrics_data = [torch.load(f) for f in train_val_metrics_files]
    folds = [int(j) for i,m in enumerate(train_val_metrics_data) \
        for j in [train_val_metrics_fold[i]]*len(m["train_losses"])
    ]
    df = pandas.DataFrame({
        "train_losses": [tl for m in train_val_metrics_data for tl in m["train_losses"]],
        "train_r2": [tl for m in train_val_metrics_data for tl in m["train_r2"]],
        "train_pearson": [tl for m in train_val_metrics_data for tl in m["train_pearson"]],
        "val_losses": [tl for m in train_val_metrics_data for tl in m["val_losses"]],
        "val_r2": [tl for m in train_val_metrics_data for tl in m["val_r2"]],
        "val_pearson": [tl for m in train_val_metrics_data for tl in m["val_pearson"]],
        "fold": folds
    })
    df_grouped = df.groupby("fold")

    train_losses = []
    train_r2 = []
    train_p = []

    val_losses = []
    val_r2 = []
    val_p = []

    for _, state in df_grouped:
        train_losses.append(state.train_losses.values)
        train_r2.append(state.train_r2.values)
        train_p.append(state.train_pearson.values)

        val_losses.append(state.val_losses.values)
        val_r2.append(state.val_r2.values)
        val_p.append(state.val_pearson.values)

    fig_data = {
        "figs": [],
        "titles": {
            "x_axis": [],
            "y_axis": [],
            "figure": []
        },
    }
    fig_data["figs"].append(plot_learning_curve_fig(train_losses, val_losses))
    fig_data["titles"]["y_axis"].append(f"{title}: MSE")

    fig_data["figs"].append(plot_learning_curve_fig(train_r2, val_r2))
    fig_data["titles"]["y_axis"].append(f"{title}: R²")
    return fig_data


def results_plot(model_name, test_name, grouped_data, save_path):
    """
    Create a 2x3 subplot figure with scatter plot, histogram, and bar chart
    
    Parameters:
    -----------
    model_name : str
        Name of the model directory (e.g., "conditioned_bs_full_features")
    test_name : str
        Test set name (e.g., "unconditioned_bs_test")
    grouped_data : pandas.DataFrame
        DataFrame with mean and std metrics
    title : str
        Main title for the scatter plot
    save_path : str
        Path to save the figure
    """
    
    # Create figure with GridSpec layout
    fig = plt.figure(figsize=(15, 7))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # ====================================================================
    # PANEL A: Scatter plot with error bars (spanning 2 rows, 2 columns)
    # ====================================================================
    ax1 = fig.add_subplot(gs[:, 0:2])
    
    # Load and process data for scatter plot
    outputs_files = glob.glob(f"../data/models/{model_name}/*_{test_name}_outputs.pkl")
    all_y = []
    all_preds = []
    
    # Load data for genetic construct annotation
    hxkm = pd.read_csv("../data/hxkm.csv")
    df_test = pd.read_csv("../data/csv/HXKm_dataset_final_new_conditioned_bs.csv")
    genetic_constructs = None
    test_db = KmClass(df_test).dataframe
    
    for i, file in enumerate(outputs_files):
        outputs = pickle.load(open(file, "rb"))
        y = torch.tensor(outputs["y_scaled"])
        preds = torch.tensor(outputs["preds_scaled"])
        indices = outputs["all_idx"].flatten()
        
        if i == 0:
            # Protein info for coloring
            test_entries = test_db.iloc[indices]
            enzyme_types = test_entries.apply(
                lambda x: hxkm.loc[hxkm.sequence == x.sequence].protein_type.values[0], 
                axis=1
            )
            enzyme_types.reset_index(drop=True, inplace=True)
            mutant_indices = enzyme_types[enzyme_types == "mutant"].index.tolist()
            
            # Create color array
            colors = pandas.Series(["green"]*indices.shape[0])
            colors[mutant_indices] = 'purple'
            genetic_constructs = colors
        
        all_y.append(y.flatten())
        all_preds.append(preds.flatten())
    
    all_y = torch.stack(all_y)
    all_preds = torch.stack(all_preds)
    
    # Calculate metrics
    r2_mean = grouped_data[grouped_data.name == test_name].r2_mean.item()
    r2_std = grouped_data[grouped_data.name == test_name].r2_std.item()
    p_mean = grouped_data[grouped_data.name == test_name].p_mean.item()
    p_std = grouped_data[grouped_data.name == test_name].p_std.item()
    
    true_means = all_y.mean(dim=0).numpy()
    pred_means = all_preds.mean(dim=0).numpy()
    pred_stds = all_preds.std(dim=0).numpy()
    
    # Create scatter plot with error bars
    for i in range(len(true_means)):
        ax1.errorbar(true_means[i], pred_means[i], 
                     yerr=pred_stds[i], 
                     fmt='o', 
                     color=genetic_constructs[i],
                     alpha=0.7,
                     markersize=6,
                     capsize=3,
                     capthick=1,
                     elinewidth=2)
    
    # Add ideal prediction line
    min_val = min(true_means.min(), pred_means.min())
    max_val = max(true_means.max(), pred_means.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 
             'k--', alpha=0.5, linewidth=2, label='Ideal prediction')
    
    # Add metrics annotation
    ax1.text(0.55, 0.06, f'R²: {r2_mean:.3f} ± {r2_std:.3f} \nPearson: {p_mean:.3f} ± {p_std:.3f}',
             transform=ax1.transAxes, fontsize=18,
             verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Create custom legend for scatter plot
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', edgecolor='black', label='Wild Type'),
        Patch(facecolor='purple', edgecolor='black', label='Mutant'),
        plt.Line2D([0], [0], color='k', linestyle='--', linewidth=2, label='Ideal prediction')
    ]
    ax1.legend(handles=legend_elements, loc='upper left', fontsize=16)
    
    ax1.set_xlabel('Experimental Km normalized value', fontsize=18)
    ax1.set_ylabel('Predicted Km normalized value', fontsize=18)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='both', labelsize=16)
    
    # Add annotation A
    ax1.text(-0.07, 0.95, 'A', transform=ax1.transAxes, 
             fontsize=24, fontweight='bold', va='bottom', ha='right')
    
    # ====================================================================
    # PANEL B: Histogram (top right)
    # ====================================================================
    ax2 = fig.add_subplot(gs[0, 2])
    
    # Create histogram data
    ax2.hist(true_means, bins=30, alpha=0.7, label='Experimental', 
             color='orange', edgecolor='black', linewidth=0.5)
    ax2.hist(pred_means, bins=30, alpha=0.7, label='Predicted', 
             color='blue', edgecolor='black', linewidth=0.5)
    
    ax2.set_xlabel('Km normalized value', fontsize=18)
    ax2.set_ylabel('Count', fontsize=18)
    ax2.legend(fontsize=13, loc="upper left", ncols=1)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='both', labelsize=16)
    
    # Add annotation B
    ax2.text(-0.15, 0.90, 'B', transform=ax2.transAxes, 
             fontsize=24, fontweight='bold', va='bottom', ha='right')
    
    # ====================================================================
    # PANEL C: Bar chart for mutants vs wild type (bottom right)
    # ====================================================================
    ax3 = fig.add_subplot(gs[1, 2])
    
    # Load data for mutant vs wild type analysis
    outputs_files = glob.glob(f"../data/models/conditioned_bs_full_features/*_fold_unconditioned_bs_test_outputs.pkl")
    r2_mutants = []
    r2_wildtypes = []
    pcc = PearsonCorrCoef()
    pcc_mutants = []
    pcc_wildtypes = []

    hxkm = pandas.read_csv("../data/hxkm.csv")
    df_test = pandas.read_csv("../data/csv/HXKm_dataset_final_new_conditioned_bs.csv")
    test_db = KmClass(df_test).dataframe
    for file in outputs_files:
        outputs = pickle.load(open(file, "rb"))

        y = torch.tensor(outputs["y_unscaled"])
        preds = torch.tensor(outputs["preds_unscaled"])
        indices = outputs["all_idx"].flatten()

        # get protein type info:
        test_entries = test_db.iloc[indices]
        enzyme_types = test_entries.apply(
            lambda x: hxkm.loc[
                hxkm.sequence == x.sequence
            ].protein_type.values[0]
        , axis=1)
        enzyme_types.reset_index(drop=True, inplace=True) # now in the same order as preds
        mutant_indices = enzyme_types[enzyme_types=="mutant"].index.tolist()
        wildtype_indices = enzyme_types[enzyme_types=="wildtype"].index.tolist()

        # compute r2 for wildtype and mutants:
        mutant_targets = y[mutant_indices]
        mutant_preds = preds[mutant_indices]
        r2_mutants.append(r2_score(mutant_preds, mutant_targets))
        pcc_mutants.append(pcc(mutant_preds, mutant_targets))
        
        wildtype_targets = y[wildtype_indices]
        wildtype_preds = preds[wildtype_indices]
        r2_wildtypes.append(r2_score(wildtype_preds, wildtype_targets))
        pcc_wildtypes.append(pcc(wildtype_preds, wildtype_targets))

    _, p_value = stats.ttest_rel(pcc_mutants, pcc_wildtypes)
    if p_value < 0.001:
        significance = "***"
    elif p_value < 0.01:
        significance = "**"
    elif p_value < 0.05:
        significance = "*"

    r2_mutants = torch.tensor(r2_mutants)
    pcc_mutants = torch.tensor(pcc_mutants)
    r2_wildtypes = torch.tensor(r2_wildtypes)
    pcc_wildtypes = torch.tensor(pcc_wildtypes)

    r2_means = [r2_wildtypes.mean().item(), r2_mutants.mean().item()]
    r2_stds = [r2_wildtypes.std().item(), r2_mutants.std().item()]
    pcc_means = [pcc_wildtypes.mean().item(), pcc_mutants.mean().item()]
    pcc_stds = [pcc_wildtypes.std().item(), pcc_mutants.std().item()]

    categories = ["Wild Type", "Mutant"]
    x = torch.arange(len(categories))
    width = 0.35
    
    # Create grouped bar chart
    r2_bars = ax3.bar(x - width/2, r2_means, width,
                        yerr=r2_stds, capsize=5,
                        error_kw={'elinewidth': 2, 'ecolor': 'red'},
                        label='R²', color='orange', 
                        edgecolor='black', linewidth=1)
    
    pcc_bars = ax3.bar(x + width/2, pcc_means, width,
                        yerr=pcc_stds, capsize=5,
                        error_kw={'elinewidth': 2, 'ecolor': 'red'},
                        label='Pearson', color='purple',
                        edgecolor='black', linewidth=1)
    
    # Add value labels
    for i, (bars, values) in enumerate(zip([r2_bars, pcc_bars], [r2_means, pcc_means])):
        for bar, value in zip(bars, values):
            height = bar.get_height()
            color = "white"
            if i == 0:
                color = "black"
            ax3.text(bar.get_x() + bar.get_width()/2., height/2,
                    f'{value:.3f}', ha='center', va='center',
                    fontsize=14, fontweight='bold', color=color)
    
    # Add significance marker
    if significance != "ns":
        ax3.text(1, max(max(r2_means), max(pcc_means)*0.93), 
                significance, ha='center', va='bottom',
                fontsize=16, fontweight='bold', color='red')
    
    ax3.set_ylabel('R² and Pearson', fontsize=18)
    ax3.set_xticks(x)
    ax3.set_xticklabels(categories, fontsize=16)
    ax3.legend(fontsize=14, loc='upper left', ncols=2)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.tick_params(axis='y', labelsize=16)
    
    # Set y-axis limits
    y_max = max(max(r2_means), max(pcc_means))
    ax3.set_ylim([0, y_max * 1.35])
    
    # Add annotation C
    ax3.text(-0.16, 0.92, 'C', transform=ax3.transAxes, 
             fontsize=24, fontweight='bold', va='bottom', ha='right')
    
    # Save and show
    # plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.savefig(save_path.replace(".jpg", ".tiff"), dpi=600, bbox_inches='tight')
    plt.show()
    
    return fig, r2_mean, p_mean