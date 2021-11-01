import os
from copy import copy
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from math import sqrt
import pickle

# for stats tests
import pandas as pd
from statsmodels.stats.anova import AnovaRM
import statsmodels.stats.multicomp as multi
from sklearn.metrics import f1_score, accuracy_score

# for plotting
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Circle, RegularPolygon, Ellipse
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
from matplotlib.colors import to_rgba

# for terminal logging
from explainable_kge.logger.terminal_utils import logout, load_config


import pdb


###########################################################
# TensorBoard Visualizations during training
###########################################################
class AbstractProcessorViz:
    def __init__(self, args):
        log_name = str(args["logging"]["tag"]) + "__"
        log_name += str(args["dataset"]["name"]) + "_"
        # log_name += "mt" + str(args["model"]["name"]) + "_"
        log_name += "clm" + str(args["continual"]["cl_method"]) + "_"
        log_name += "ln" + str(args["logging"]["log_num"])
        log_dir = os.path.abspath(os.path.dirname(__file__)) + "/logs/"
        self.log_fp = log_dir + log_name


class ProcessorViz(AbstractProcessorViz):
    def __init__(self, args):
        super(ProcessorViz, self).__init__(args)
        if os.path.isdir(self.log_fp):  # overwrites existing events log
            files = os.listdir(self.log_fp)
            for filename in files:
                if "events" in filename:
                    os.remove(self.log_fp+"/"+filename)
                # rmtree(self.log_fp)
        self._writer = SummaryWriter(self.log_fp)
        self.timestamp = 0
        self.gruvae_timestamp = 0

    def add_tr_sample(self, sess, sample):
        loss = sample
        self._writer.add_scalar("Loss/TrainSess_"+str(sess), loss, self.timestamp)
        self.timestamp += 1

    def add_de_sample(self, sample):
        hits_avg = 0.0
        mrr_avg = 0.0
        for sess in range(sample.shape[0]):
            hits, mrr = sample[sess,:]
            self._writer.add_scalar("HITS/DevSess_"+str(sess), hits, self.timestamp)
            self._writer.add_scalar("MRR/DevSess_"+str(sess), mrr, self.timestamp)
            hits_avg += hits
            mrr_avg += mrr
        hits_avg = hits_avg / float(sample.shape[0])
        mrr_avg = mrr_avg / float(sample.shape[0])
        self._writer.add_scalar("HITS/DevAvg", hits_avg, self.timestamp)
        self._writer.add_scalar("MRR/DevAvg", mrr_avg, self.timestamp)

    def add_gruvae_tr_sample(self, sample):
        total_loss, rc_loss, kl_loss, kl_weight = sample
        self._writer.add_scalar("GRUVAE/Loss", total_loss, self.gruvae_timestamp)
        self._writer.add_scalar("GRUVAE/RCLoss", rc_loss, self.gruvae_timestamp)
        self._writer.add_scalar("GRUVAE/KLWeight", kl_weight, self.gruvae_timestamp)
        self._writer.add_scalar("GRUVAE/KLLoss", kl_loss, self.gruvae_timestamp)
        self.gruvae_timestamp += 1

    def add_gruvae_de_sample(self, sample):
        precision, u_precision, coverage = sample[0]
        self._writer.add_scalar("GRUVAE/Precision", precision, self.gruvae_timestamp)
        self._writer.add_scalar("GRUVAE/UPrecision", u_precision, self.gruvae_timestamp)
        self._writer.add_scalar("GRUVAE/Coverage", coverage, self.gruvae_timestamp)

###########################################################
# Generic plotting
###########################################################
def plot_bar(values, names, colors=None, ylabel=None, title=None, ylim=None, yerr=None):
    fig, ax = plt.subplots(1, 1)
    bar = ax.bar(x=range(len(values)), height=values, color=colors, yerr=yerr)
    ax.get_xaxis().set_visible(False)
    ax.legend(bar, names,
              loc='lower center', bbox_to_anchor=(0.5, -0.12),
              ncol=4, fancybox=True, shadow=True)

    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(ylim)

    return fig


def plot_mbar(values, names, colors, hatches, ylabel=None, titles=None,
              top_title=None, ylim=None, yerr=None):
    """

    :param values: num groups x num methods data
    :param names:
    :param colors:
    :param hatches:
    :param ylabel:
    :param titles:
    :param top_title:
    :param ylim:
    :param yerr:
    :return:
    """
    fig, ax = plt.subplots(1, values.shape[0])
    for i in range(values.shape[0]):
        bars = ax[i].bar(x=range(len(values[i])), height=values[i],
                        color=colors[i] if type(colors[0]) == list else colors,
                        alpha=.99,
                        yerr=yerr[i] if yerr is not None else None)
        for bar, hatch in zip(bars, hatches):
            bar.set_hatch(hatch)
        ax[i].get_xaxis().set_visible(False)
        if i == round(float(len(values)) / 2.0):
            ax[i].legend(bars, names[i] if type(names[0]) == list else names,
                         loc='lower center', bbox_to_anchor=(0.5, -0.17),
                         ncol=4, fancybox=True, shadow=True)

        if ylim is not None:
            ax[i].set_ylim(ylim)
        if i == 0 and ylabel is not None:
            ax[i].set_ylabel(ylabel)
        if i != 0:
            ax[i].get_yaxis().set_visible(False)
        if titles is not None:
            ax[i].set_title(titles[i])

    if top_title is not None:
        fig.suptitle(top_title)

    return fig


def plot_mbar_stacked(values1, values2, names, colors, hatches, ylabel=None, titles=None,
              top_title=None, ylim=None, yerr1=None, yerr2=None):
    """

    :param values: num groups x num methods data
    :param names:
    :param colors:
    :param hatches:
    :param ylabel:
    :param titles:
    :param top_title:
    :param ylim:
    :param yerr:
    :return:
    """
    fig, ax = plt.subplots(1, values1.shape[0])
    for i in range(values1.shape[0]):
        bars = ax[i].bar(x=range(len(values1[i])), height=values1[i],
                         color=colors[i] if type(colors[0]) == list else colors,
                         alpha=.99,
                         yerr=yerr1[i] if yerr1 is not None else None)
        for bar, hatch in zip(bars, hatches):
            bar.set_hatch(hatch)
        ax[i].get_xaxis().set_visible(False)
        if i == round(float(len(values1)) / 2.0):
            ax[i].legend(bars, names[i] if type(names[0]) == list else names,
                         loc='lower center', bbox_to_anchor=(0.5, -0.17),
                         ncol=4, fancybox=True, shadow=True)
        # stacked bars
        bars = ax[i].bar(x=range(len(values1[i])), height=values2[i]-values1[i],
                         bottom=values1[i],
                         color=colors[i] if type(colors[0]) == list else colors,
                         alpha=.30,
                         yerr=yerr2[i] if yerr2 is not None else None)
        for bar, hatch in zip(bars, hatches):
            bar.set_hatch(hatch)

        if ylim is not None:
            ax[i].set_ylim(ylim)
        if i == 0 and ylabel is not None:
            ax[i].set_ylabel(ylabel)
        if i != 0:
            ax[i].get_yaxis().set_visible(False)
        if titles is not None:
            ax[i].set_title(titles[i])

    if top_title is not None:
        fig.suptitle(top_title)

    return fig


def plot_line(xvalues, yvalues, names, colors, linestyles,
              ylabel=None, titles=None, ylim=None, yerr=None,
              xticks=None, top_title=None):

    num_lines = yvalues.shape[0]

    fig = plt.figure(figsize=(4.25, 4))

    ax = fig.add_subplot(1, 1, 1)
    lines = []
    for j in range(num_lines):
        line, = ax.plot(xvalues, yvalues[j], color=colors[j], linestyle=linestyles[j])
        if yerr is not None:
            ax.fill_between(xvalues, yvalues[j] - yerr[j], yvalues[j] + yerr[j],
                            color=colors[j], alpha=0.2)
        lines.append(line)

    ax.legend(lines, names,
              loc='best',
              ncol=1, fancybox=True, shadow=True)

    if ylim is not None:
        ax.set_ylim(ylim)

    if xticks is not None:
        ax.set_xlim([xticks[0][0], xticks[0][-1]])
        ax.set_xticks(xticks[0])
        ax.set_xticklabels(xticks[1])

    if ylabel is not None:
        ax.set_ylabel(ylabel)

    if top_title is not None:
        fig.suptitle(top_title, x=0.5, y=0.99)

    return fig


def plot_mline(xvalues, yvalues, names, colors, linestyles,
               ylabel=None, titles=None, ylim=None, yerr=None,
               xticks=None, top_title=None):
    num_plots = xvalues.shape[0]
    num_lines = []
    for i in range(yvalues.shape[0]):
        num_lines.append(yvalues[i].shape[0])

    fig = plt.figure(figsize=(10, 6))

    if ylabel is not None:
        ax = fig.add_subplot(1, 1, 1)
        ax.set_ylabel(ylabel)
        ax.spines['top'].set_color('none')
        ax.spines['bottom'].set_color('none')
        ax.spines['left'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)

    ax = []
    for i in range(num_plots):
        ax.append(fig.add_subplot(num_plots, 1, i+1))
        lines = []
        for j in range(num_lines[i]):
            line, = ax[i].plot(xvalues[i], yvalues[i,j], color=colors[j], linestyle=linestyles[j])
            if yerr is not None:
                ax[i].fill_between(xvalues[i], yvalues[i, j] - yerr[i, j], yvalues[i, j] + yerr[i, j],
                                   color=colors[j], alpha=0.2)
            lines.append(line)


        if i == 0:
            ax[i].legend(lines, names,
                      loc='upper center', bbox_to_anchor=(0.5, 1.64),
                      ncol=4, fancybox=True)

        if titles is not None:
            ax[i].set_ylabel(titles[i])
            ax[i].yaxis.set_label_position("right")

        if i == num_plots-1:
            ax[i].get_xaxis().set_visible(True)
        else:
            ax[i].get_xaxis().set_visible(False)

        if ylim is not None:
            ax[i].set_ylim(ylim)

        if xticks is not None:
            ax[i].set_xlim([xticks[0][0], xticks[0][-1]])
            ax[i].set_xticks(xticks[0])
            ax[i].set_xticklabels(xticks[1])

    if top_title is not None:
        fig.suptitle(top_title, x=0.5, y=0.99)

    fig.subplots_adjust(hspace=0.07)

    return fig


def plot_table(stats, row_labels, col_labels, title=None):
    fig = plt.figure(figsize=(10, 6))
    axs = fig.add_subplot(1, 1, 1)
    fig.patch.set_visible(False)
    axs.axis('off')
    axs.axis('tight')
    plt.grid('off')

    format_stats = copy(stats).astype(str)
    for i in range(format_stats.shape[0]):
        for j in range(format_stats.shape[1]):
            format_stats[i,j] = "{:.4f}".format(stats[i,j])

    the_table = axs.table(cellText=format_stats, rowLabels=row_labels, colLabels=col_labels, loc='center')
    fig.tight_layout()
    if title is not None:
        axs.set_title(title, weight='bold', size='medium',
                      horizontalalignment='center', verticalalignment='center')
    return fig


def radar_factory(num_vars, frame='circle'):
    """Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle' | 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarAxes(PolarAxes):

        name = 'radar'
        # use 1 line segment to connect specified points
        RESOLUTION = 1

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.concatenate((x, [x[0]]))
                y = np.concatenate((y, [y[0]]))
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta


def plot_radar(stats, colors, linestyles, metric_labels, method_labels, title):
    N = len(metric_labels)
    theta = radar_factory(N, frame='circle')

    spoke_labels = metric_labels

    fig, ax = plt.subplots(figsize=(4, 4), nrows=1, ncols=1,
                           subplot_kw=dict(projection='radar'))
    fig.subplots_adjust(wspace=0.25, hspace=0.20, top=0.85, bottom=0.05)

    ax.set_rgrids([0.2, 0.4, 0.6, 0.8], angle=95)
    ax.set_title(title, weight='bold', size='medium', position=(0.5, 1.2),
                 horizontalalignment='center', verticalalignment='center')

    for idx in range(stats.shape[0]):
        ax.plot(theta, stats[idx, :], color=colors[idx], linestyle=linestyles[idx])
        ax.fill(theta, stats[idx, :], facecolor=colors[idx], alpha=0.25)
    ax.set_varlabels(spoke_labels)

    legend = ax.legend(method_labels, loc=(0.9, .95),
                       labelspacing=0.1, fontsize='small',
                       fancybox=True, shadow=True)
    return fig


def plot_scatter(xvalues, yvalues, names, colors, linestyles,
                 xlabel=None, ylabel=None,
                 xerr=None, yerr=None, top_title=None):
    ells = [Ellipse((xvalues[i], yvalues[i]),
                    width=xerr[0, i] if xerr is not None else 0.03,
                    height=yerr[0, i] if yerr is not None else 0.03,
                    angle=0) for i in range(len(xvalues))]

    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')

    for i in range(len(ells)):
        ells[i].set_clip_box(ax.bbox)
        ells[i].set_facecolor(to_rgba(colors[i], 0.3))
        ells[i].set_edgecolor(to_rgba(colors[i], 1.0))
        ells[i].set_linestyle(linestyles[i])
        ells[i].set_linewidth(1.5)
        ax.add_artist(ells[i])
        ax.scatter(xvalues[i], yvalues[i], c=to_rgba(colors[i], 1.0), s=1.0)

    ax.legend(ells, names,
              loc='center right', bbox_to_anchor=(1.27, 0.5),
              ncol=1, fancybox=True, shadow=True)


    ax.set_xlim([0.0, np.max(xvalues)+0.05])
    ax.set_ylim([0.0, np.max(yvalues)+0.05])

    if ylabel is not None:
        ax.set_ylabel(ylabel)

    if xlabel is not None:
        ax.set_xlabel(xlabel)

    if top_title is not None:
        ax.set_title(top_title)

    return fig


def plot_hist(values, bins=None, color=None, ylabel=None, title=None, ylim=None):
    fig, ax = plt.subplots(1, 1)
    hist = ax.hist(x=values, bins=bins, facecolor=color, alpha=0.75)
    plt.grid(True)

    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(ylim)

    return fig


def figs2pdf(figs, filepath):
    pdf = PdfPages(filepath)
    for fig in figs:
        pdf.savefig(fig, bbox_inches="tight")
    pdf.close()


def format_method_names(methods):
    method_names = []
    method2name = {
        "logit": "Logistic Classification",
    }
    for method in methods:
        method_names.append(method2name[method])
    return method_names


def format_method_colors(methods):
    method_colors = []
    method2color = {
        "logit": "m",
    }
    for method in methods:
        method_colors.append(method2color[method])
    return method_colors


def format_method_linestyles(methods):
    method_markers = []
    method2marker = {
        "logit": ":",
    }
    for method in methods:
        method_markers.append(method2marker[method])
    return method_markers


def format_method_hatches(methods):
    method_markers = []
    method2marker = {
        "logit": "//",
    }
    for method in methods:
        method_markers.append(method2marker[method])
    return method_markers

###########################################################
# Generic stats tests
###########################################################
def run_stats_test_all_sessions(data, methods, num_exp, num_sess, test_label, log_file):
    for i in range(num_sess):
        run_stats_test(data[:, i, :], methods, num_exp, test_label + " in session " + str(i), log_file)


def run_stats_test(data, methods, num_exp, test_label, log_file):
    df = pd.DataFrame(columns=["exp", "method", "value"])
    for exp_num in range(num_exp):
        for method_num in range(len(methods)):
            df = df.append(pd.DataFrame([[exp_num, methods[method_num], data[exp_num, method_num]]],
                                        columns=["exp", "method", "value"]), ignore_index=True)
    aovrm = AnovaRM(df, 'value', 'exp', within=['method'])
    res = aovrm.fit()
    mcDate = multi.MultiComparison(df["value"], df["method"])
    res2 = mcDate.tukeyhsd()
    with open(log_file, "a") as f:
        f.write(test_label + "\n" + str(res) + "\n" + str(res2))


def load_data(data_fp):
    with open(data_fp, "rb") as f:
        results = pickle.load(f)
    return results


def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values-average)**2, weights=weights)
    return (average, sqrt(variance))


def get_summary(results):
    summary = pd.DataFrame(columns=["Relation", "Coverage", "Fidelity", "F1-Fidelity", "Weight"], dtype=float)
    rels = np.unique(results["rel"].to_numpy())
    for rel in rels:
        rel_results = results.loc[results["rel"].isin([rel]),:]
        weight = rel_results.shape[0]
        coverage = float(rel_results.shape[0] - rel_results["predict"].isin([0]).sum()) / float(rel_results.shape[0])
        rel_pred_results = rel_results.loc[rel_results["predict"].isin([1,-1]),:]
        fidelity = accuracy_score(rel_pred_results["label"].to_numpy(np.int), rel_pred_results["predict"].to_numpy(np.int))
        f1_fidelity = f1_score(rel_pred_results["label"].to_numpy(np.int), rel_pred_results["predict"].to_numpy(np.int))
        summary = summary.append({"Relation": rel, "Coverage": coverage, "Fidelity": fidelity, "F1-Fidelity": f1_fidelity, "Weight": weight}, ignore_index=True)
    cov_avg, cov_std = weighted_avg_and_std(summary["Coverage"].values, summary["Weight"].values)
    fid_avg, fid_std = weighted_avg_and_std(summary["Fidelity"].values, summary["Weight"].values)
    f1_avg, f1_std = weighted_avg_and_std(summary["F1-Fidelity"].values, summary["Weight"].values)
    summary = summary.append({"Relation": "Mean", "Coverage": cov_avg, "Fidelity": fid_avg, "F1-Fidelity": f1_avg, "Weight": 0}, ignore_index=True)
    summary = summary.append({"Relation": "Std", "Coverage": cov_std, "Fidelity": fid_std, "F1-Fidelity": f1_std, "Weight": 0}, ignore_index=True)
    logout("\n" + str(summary), "s")
    return summary


def plot_params(results):
    l1s = []
    alphas = []
    losses = []
    loss2id = {"log":1, "modified_huber":2, "perceptron":3}
    for idx, result in results.iterrows():
        if type(result["params"]) != str: continue
        params = eval(result["params"])
        l1s.append(params["l1_ratio"])
        alphas.append(params["alpha"])
        losses.append(loss2id[params["loss"]])
    bins = [0] + list(np.unique(l1s) + 0.0001)
    l1_fig = plot_hist(l1s, bins=bins, color="g", title="Best L1 ratio values")
    bins = [0] + list(np.unique(alphas) + 0.0001)
    alpha_fig = plot_hist(alphas, bins=bins, color="b", title="Best alpha values")
    bins = [0] + list(np.unique(losses) + 0.0001)
    losses_fig = plot_hist(losses, bins=bins, color="r", title="Best loss types")
    return [l1_fig, alpha_fig, losses_fig]


def locality_plot(root_fp, args, locality=[2,3,4,5,6,7,8,9,10,12,14,16,18,20,22,24,26,28,30,40,50,100,150,200,250,300,400,500,1000,1500,2000,2500]):
    summary_means = pd.DataFrame(columns=["k","Coverage","Fidelity","F1-Fidelity"])
    summary_stds = pd.DataFrame(columns=["k","Coverage","Fidelity","F1-Fidelity"])
    for k in locality:
        result_name = "{}.pkl".format(args["explain"]["xmodel"] + "_local3_" + str(k))
        results_fp = os.path.join(root_fp, "results", result_name)
        result = load_data(results_fp)
        summary = get_summary(result)
        summary_means = summary_means.append(summary.iloc[-2], ignore_index=True)
        summary_stds = summary_stds.append(summary.iloc[-1], ignore_index=True)
    names = format_method_names([args["explain"]["xmodel"]])
    colors = format_method_colors([args["explain"]["xmodel"]])
    linestyles = format_method_linestyles([args["explain"]["xmodel"]])
    yvals = np.expand_dims(summary_means["F1-Fidelity"].values,0) * 100
    plot1 = plot_line(locality, yvals, names, colors, linestyles,
                     ylabel="F1-Fidelity %", titles=None, ylim=None, yerr=None,
                     xticks=None, top_title=None)
    yvals = np.expand_dims(summary_means["Coverage"].values,0) * 100
    plot2 = plot_line(locality, yvals, names, colors, linestyles,
                     ylabel="Coverage %", titles=None, ylim=None, yerr=None,
                     xticks=None, top_title=None)
    return [plot1, plot2]


if __name__ == "__main__":
    exp_config = load_config("Experiment Visualizations")
    plt.rcParams.update({'font.weight': 'bold'})
    figs = []
    main_fp = os.path.join("explainable_kge/logger/logs", exp_config["dataset"]["name"] + "_" + exp_config["model"]["name"] + "_" + str(exp_config["logging"]["log_num"]))
    
    figs += locality_plot(main_fp, exp_config)

    results_fp = os.path.join(main_fp, "results", "{}.pkl".format(exp_config["explain"]["xmodel"] + "_" + exp_config["explain"]["locality"] + "_" + str(exp_config["explain"]["locality_k"])))
    if not os.path.exists(results_fp):
        logout("Experiment results pickle does not exist: " + str(results_fp), "f")
    results = load_data(results_fp)
    result_summary = get_summary(results)
    pdb.set_trace()
    
    title_str = "Student: " + exp_config["explain"]["xmodel"] + ", Locality: " + exp_config["explain"]["locality"]
    fig = plot_table(stats=result_summary[result_summary.columns[1:]].to_numpy(dtype=float),
                     row_labels=result_summary["Relation"].to_numpy(str),
                     col_labels=result_summary.columns[1:].to_numpy(str),
                     title=title_str)
    figs.append(fig)
    figs += plot_params(results)
    figs_fp = os.path.join(main_fp, "results", "{}.pdf".format(exp_config["explain"]["xmodel"] + "_" + exp_config["explain"]["locality"] + "_" + str(exp_config["explain"]["locality_k"])))
    figs2pdf(figs, figs_fp)

