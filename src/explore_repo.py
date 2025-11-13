import glob
import pandas
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
from scipy import stats
import random
from dataset import KmClass

def plot_all_ablation(
        experiment_filenames,
        experiment_conditions
    ):
    zipped = zip(experiment_filenames, experiment_conditions)
    labels = {
        "protein_free_test": "Protein free",
        "aa_id_free_test": "AA identity free",
        "bs_free_test": "AS free",
        "unconditioned_bs_test": "Unconditioned AS",
        "conditioned_bs_test": "Conditioned AS",
        "molecule_free_test": "Molecule free",
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
        x=-0.05,
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
    inferences = pandas.read_csv(inferences_file)
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
    ranked_r2 = {g:r2.mean() for g, r2 in model_r2.items()}
    ranked_r2 = dict(sorted(ranked_r2.items(), key=lambda i: i[1], reverse=True))

    grouped = test_metrics.groupby(["name"], as_index=False).agg(
        r2_mean=pandas.NamedAgg(column="r2", aggfunc="mean"),
        r2_std=pandas.NamedAgg(column="r2", aggfunc="std"),
        p_mean=pandas.NamedAgg(column="pearson", aggfunc="mean"), 
        p_std=pandas.NamedAgg(column="pearson", aggfunc="std"),
        mse_mean=pandas.NamedAgg(column="mse", aggfunc="mean"), 
        mse_std=pandas.NamedAgg(column="mse", aggfunc="std")
    )
    plot_order = [grouped[grouped.name == "conditioned_bs_test"].index[0]] + [grouped[grouped.name == n].index[0] for n in ranked_r2.keys()]
    grouped = grouped.reindex(plot_order)
    labels = {
        "protein_free_test": "Protein free",
        "aa_id_free_test": "AA identity free",
        "molecule_free_test": "Molecule free",
        "bs_free_test": "AS free",
        "conditioned_bs_test": "Conditioned AS",
        "descriptor_free_test": "Descriptor free",
        "fingerprint_free_test": "Fingerprint free",
        "unconditioned_bs_test": "Unconditioned AS"
    }
    fig = go.Figure(data=[
        go.Bar(
            name="R2", x=grouped.name.values, y=grouped.r2_mean.values, 
            error_y={
                "type":"data",
                "array": grouped.r2_std.values,
                "visible": True,
                "color": "red",
            },
            text = [f"{t:.3f}" for t in grouped.r2_mean.values],
            textfont=dict(weight=900),
            marker_color="orange"
        ),
        go.Bar(
            name="Pearson", x=grouped.name.values, y=grouped.p_mean.values, 
            text = [f"{t:.3f}" for t in grouped.p_mean.values],
            error_y={
                "type":"data",
                "array": grouped.p_std.values,
                "visible": True,
                "color": "red",
            },
            textfont=dict(weight=900),
            marker_color="purple"
        )
    ])
    for model, significance in significances.items():
        fig.add_annotation(
            x=model,
            y=0.7,
            text=significance,
            showarrow=False,
            font={"size":25, "color": "red"},
            yref="y"
        )
    fig.update_layout(
        width=1000, height=500, 
        uniformtext_minsize=18, uniformtext_mode="hide",
        margin=dict(t=0, b=0, l=0, r=0),  # Tighter margins
        font={"size": 20}
    )
    if with_title:
        fig.update_layout(
            title_text=fig_title,
            margin=dict(t=50, b=0, l=0, r=0),  # Tighter margins
        )
    fig.update_xaxes(labelalias=labels)
    fig.update_yaxes(title={"text": "Score"})
    fig.update_traces(textposition="inside", insidetextanchor="middle")
    fig.write_image(
        file_name,
        width=1000, height=500
    )
    cols_str = ""
    if with_table:
        for exp_name, exp_label in labels.items():
            gr = grouped.loc[grouped.name == exp_name]
            n = test_metrics.loc[test_metrics.name == exp_name].n.values[0]
            avg = [gr[c].item() for c in ["r2_mean", "p_mean", "mse_mean"]]
            std = [gr[c].item() for c in ["r2_std", "p_std", "mse_std"]]
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
    print(len(figures))
    # build final fig:
    rows = 6
    cols = 3
    big_fig = make_subplots(
        rows=rows, cols=cols,
        vertical_spacing=0.02
        # subplot_titles=[
        #     "MSE over epochs",
        #     "R2 over epochs",
         #]
    )
    for i in range(16):
        col = i % cols + 1
        row = i // cols + 1 
        for trace in figures[i]["data"]:
            big_fig.add_trace(trace, row=row, col=col)
        big_fig.update_yaxes(title_text=y_axes[i], row=row, col=col)

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
    fig_data["titles"]["y_axis"].append(f"{title}: R2")
    return fig_data


def plot_scatter(model, test_name, grouped, title, filename):
    outputs_files = glob.glob(f"../data/models/{model}/*_{test_name}_outputs.pkl")
    all_y = []
    all_preds = []

    # for genetic construct annotation:
    hxkm = pandas.read_csv("../data/hxkm.csv")
    df_test = pandas.read_csv("../data/csv/HXKm_dataset_final_new_conditioned_bs.csv")
    genetic_constructs = None
    test_db = KmClass(df_test).dataframe
    for i,file in enumerate(outputs_files):
        outputs = pickle.load(open(file, "rb"))
        y = torch.tensor(outputs["y_scaled"])
        preds = torch.tensor(outputs["preds_scaled"])
        indices = outputs["all_idx"].flatten()

        if i == 0:
            # protein info:
            test_entries = test_db.iloc[indices]
            enzyme_types = test_entries.apply(
                lambda x: hxkm.loc[
                    hxkm.sequence == x.sequence
                ].protein_type.values[0]
            , axis=1)
            enzyme_types.reset_index(drop=True, inplace=True) # now in the same order as preds
            mutant_indices = enzyme_types[enzyme_types=="mutant"].index.tolist()
            genetic_constructs = pandas.Series(["green"]*indices.shape[0])
            genetic_constructs[mutant_indices] = "purple"

        all_y.append(y.flatten())
        all_preds.append(preds.flatten())
    all_y = torch.stack(all_y)
    all_preds = torch.stack(all_preds)
    r2 = grouped[grouped.name == test_name].r2_mean.item()
    p = grouped[grouped.name == test_name].p_mean.item()

    fig_data = []
    bits = list(range(255))
    # colors = [f"rgb({random.sample(bits, 1)[0]},{random.sample(bits, 1)[0]},{random.sample(bits, 1)[0]})" for i in range(all_preds.shape[1])]
    colors = genetic_constructs
    true_means = all_y.mean(dim=0)
    pred_means = all_preds.mean(dim=0)
    pred_stds = all_preds.std(dim=0)

    # make the histogram:
    hist_fig = go.Figure(data=[
        go.Histogram(x=pred_means.tolist(), name="Predicted distribution"),
        go.Histogram(x=true_means.tolist(), name="True distribution")
    ])

    for i in range(all_y.shape[1]):
        fig_data.append(
            go.Scatter(
                x=[true_means[i].item()], y=[pred_means[i].item()],mode="markers", showlegend=False,
                error_y={
                    "type": "data",
                    "array": [pred_stds[i]],
                    "visible": True,
                    "color": colors[i],
                },
                marker=dict(
                    color=colors[i], #set color equal to a variable
                )
            )
        )
    fig_data.append(go.Scatter(
        x=[all_y.mean(dim=0).min(), all_y.mean(dim=0).max()], 
        y=[all_y.mean(dim=0).min(), all_y.mean(dim=0).max()], 
        name="Ideal prediction",
        line={
            "dash": "dash",
            "color": "grey"
        },
    ))
    fig = go.Figure(data=fig_data)

    fig.add_annotation(
        x=.9, y=0.25,
        xref="paper",
        text=f"Average R2: {r2:.3f}",
        showarrow=False,
        font={
            "size": 23,
        }
    )
    fig.add_annotation(
        x=.9, y=0.15,
        xref="paper",
        text=f"Average Pearson: {p:.3f}",
        showarrow=False,
        font={
            "size": 23,
        }
    )
    fig.update_layout(
        width=900,
        height=600,
        title={
            "text": title,
            "font": {"size": 23}
        },
        xaxis={
            "title": "True",
            "title_font": {"size": 21}
        },
        yaxis={
            "title":"Predicted",
            "title_font": {"size": 21}
        },
        legend={
            "font": {"size": 18}
        }
    )
    fig.write_image(filename, width=900, height=600)
    return fig, r2, p, hist_fig
