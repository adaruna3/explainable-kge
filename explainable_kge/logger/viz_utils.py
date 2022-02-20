from argparse import Action
import os
import json
from copy import copy

from matplotlib import colors
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from math import log, sqrt
import pickle

# for stats tests
import pandas as pd
from statsmodels.stats.anova import AnovaRM
import statsmodels.stats.multicomp as multi
from sklearn.metrics import f1_score, accuracy_score, cohen_kappa_score
from scipy import stats
import probscale
import seaborn
import pingouin as ping

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
from explainable_kge.models import model_utils


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
              ylabel=None, ylim=None, yerr=None, xlim=None,
              xticks=None, top_title=None, legend_loc="best", anchor=(0,0)):

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

    ax.legend(lines, names, bbox_to_anchor=anchor,
              loc=legend_loc,
              ncol=1, fancybox=True, shadow=True)

    if ylim is not None:
        ax.set_ylim(ylim)

    if xlim is not None:
        ax.set_xlim(xlim)

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


def plot_table(stats, row_labels, col_labels, title=None, fmt_string="{:.4f}", figure_size=(10,6)):
    fig = plt.figure(figsize=figure_size)
    axs = fig.add_subplot(1, 1, 1)
    fig.patch.set_visible(False)
    axs.axis('off')
    axs.axis('tight')
    plt.grid('off')

    format_stats = copy(stats).astype(str)
    for i in range(format_stats.shape[0]):
        for j in range(format_stats.shape[1]):
            format_stats[i,j] = fmt_string.format(stats[i,j])

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


def format_names(methods):
    method_names = []
    method2name = {
        "logit": "Logistic Classification",
        "decision_tree": "Decision Tree Classification",
        "HasEffect": "(Action) Has Effect (State)",
        "InverseActionOf": "(Action) Inverse Action Of (Action)",
        "InverseStateOf": "(State) Inverse State Of (State)",
        "LocInRoom": "(Location) In Room (Room)",
        "ObjCanBe": "(Object) Can Be (Action)",
        "ObjInLoc": "(Object) In Location (Location)",
        "ObjInRoom": "(Object) In Room (Room)",
        "ObjOnLoc": "(Object) On Location (Location)",
        "ObjUsedTo": "(Object) Used To (Action)",
        "ObjhasState": "(Object) Has State (State)",
        "OperatesOn": "(Object) Operates On (Object)",
    }
    for method in methods:
        method_names.append(method2name[method])
    return method_names


def format_colors(methods):
    method_colors = []
    method2color = {
        "logit": "m",
        "decision_tree": "b",
        "HasEffect": "b",
        "InverseActionOf": "m",
        "InverseStateOf": "c",
        "LocInRoom": "g",
        "ObjCanBe": "r",
        "ObjInLoc": "y",
        "ObjInRoom": "k",
        "ObjOnLoc": "violet",
        "ObjUsedTo": "orangered",
        "ObjhasState": "pink",
        "OperatesOn": "yellowgreen",
    }
    for method in methods:
        method_colors.append(method2color[method])
    return method_colors


def format_linestyles(methods):
    method_markers = []
    method2marker = {
        "logit": "solid",
        "decision_tree": "solid",
        "HasEffect": "solid",
        "InverseActionOf": "solid",
        "InverseStateOf": "solid",
        "LocInRoom": "solid",
        "ObjCanBe": "solid",
        "ObjInLoc": "solid",
        "ObjInRoom": "solid",
        "ObjOnLoc": "solid",
        "ObjUsedTo": "solid",
        "ObjhasState": "solid",
        "OperatesOn": "solid",
    }
    for method in methods:
        method_markers.append(method2marker[method])
    return method_markers


def format_hatches(methods):
    method_markers = []
    method2marker = {
        "logit": "//",
        "decision_tree": "//",
        "HasEffect": "//",
        "InverseActionOf": "//",
        "InverseStateOf": "//",
        "LocInRoom": "//",
        "ObjCanBe": "//",
        "ObjInLoc": "//",
        "ObjInRoom": "//",
        "ObjOnLoc": "//",
        "ObjUsedTo": "//",
        "ObjhasState": "//",
        "OperatesOn": "//",
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


def locality_plot(args, locality=[2,3,4,5,6,7,8,9,10,12,14,16,18,20,22,24,26,28,30,40,50,100,150,200,250,300,400,500,1000]): #,1500,2000,2500]):
    root_fp = os.path.join("explainable_kge/logger/logs", exp_config["dataset"]["name"] + "_" + exp_config["model"]["name"])
    # load all the data
    summarys = pd.DataFrame(columns=["log_num","k","Relation","Coverage","Fidelity","F1-Fidelity", "Weight"])
    for log_num in range(1,6):
        folder_path = root_fp + "_" + str(log_num)
        for k in locality:
            result_name = "{}.pkl".format(args["explain"]["xmodel"] + "_local3_" + str(k))
            results_fp = os.path.join(folder_path, "results", result_name)
            result = load_data(results_fp)
            summary = get_summary(result)
            summary["k"] = k
            summary["log_num"] = log_num
            summarys = summarys.append(summary, ignore_index=True)
    # average across log_num
    rel_avg_summary = pd.DataFrame(columns=["k","Relation","F1-Fidelity","Weight"])
    rel_std_summary = pd.DataFrame(columns=["k","Relation","F1-Fidelity","Weight"])
    relations = np.unique(summary["Relation"].to_numpy(dtype=str))
    relations = [rel for rel in relations if rel != "Mean" and rel != "Std"]
    for rel in relations:
        for k in locality:
            rel_summ = summarys.loc[summarys["Relation"].isin([rel]),:]
            k_rel_summ = rel_summ.loc[rel_summ["k"].isin([k]),:]
            wavg, wstd = weighted_avg_and_std(k_rel_summ["F1-Fidelity"].values, k_rel_summ["Weight"].values)
            rel_avg_summary = rel_avg_summary.append({"k":k,"Relation":rel,"F1-Fidelity":wavg,"Weight":np.sum(k_rel_summ["Weight"].values)}, ignore_index=True)
            rel_std_summary = rel_std_summary.append({"k":k,"Relation":rel,"F1-Fidelity":wstd,"Weight":np.sum(k_rel_summ["Weight"].values)}, ignore_index=True)
    # prepare numpy arrays
    line_avg = np.zeros(shape=(0,len(locality)))
    line_std = np.zeros(shape=(0,len(locality)))
    for rel in relations:
        avg_rel_summ = rel_avg_summary.loc[rel_avg_summary["Relation"].isin([rel]),:]
        line_avg = np.append(line_avg, [avg_rel_summ["F1-Fidelity"].to_numpy(dtype=float)], axis=0)
        std_rel_summ = rel_std_summary.loc[rel_std_summary["Relation"].isin([rel]),:]
        line_std = np.append(line_std, [std_rel_summ["F1-Fidelity"].to_numpy(dtype=float)], axis=0)
    # plot each relation type
    names = format_names(relations)
    colors = format_colors(relations)
    linestyles = format_linestyles(relations)
    plots = []
    for rel_id in range(len(relations)):
        for right_xlim in [100,300,1000]:
            line = np.expand_dims(line_avg[rel_id,:]*100.0, axis=0)
            line_err = np.expand_dims(line_std[rel_id,:]*100.0, axis=0)
            lp = plot_line(locality, line, [names[rel_id]], [colors[rel_id]], [linestyles[rel_id]],
                            ylabel="F1-Fidelity %", ylim=None, yerr=line_err, xlim=(2,right_xlim),
                            top_title="Effect of Locality on {} F1-Fidelity".format(names[rel_id]),
                            legend_loc="center", anchor=(0.5,-0.14))
            plots.append(lp)
    # average across log_num and rel
    avg_summary = pd.DataFrame(columns=["k","avg","std"])
    for k in locality:
        k_summ = rel_avg_summary.loc[rel_avg_summary["k"].isin([k]),:]
        wavg, _ = weighted_avg_and_std(k_summ["F1-Fidelity"].values, k_summ["Weight"].values)
        k_summ = rel_std_summary.loc[rel_std_summary["k"].isin([k]),:]
        wstd, _ = weighted_avg_and_std(k_summ["F1-Fidelity"].values, k_summ["Weight"].values)
        avg_summary = avg_summary.append({"k":k,"avg":wavg,"std":wstd}, ignore_index=True)
    # plot average across relations
    names = format_names([args["explain"]["xmodel"]])
    colors = format_colors([args["explain"]["xmodel"]])
    linestyles = format_linestyles([args["explain"]["xmodel"]])
    for right_xlim in [100,300,1000]:
        line = np.expand_dims(avg_summary["avg"].values*100.0, axis=0)
        line_err = np.expand_dims(avg_summary["std"].values*100.0, axis=0)
        lp = plot_line(locality, line, names, colors, linestyles,
                       ylabel="F1-Fidelity %", yerr=line_err, xlim=(2,right_xlim),
                       top_title="Effect of Locality on {} F1-Fidelity".format(args["explain"]["xmodel"]),
                       legend_loc="center", anchor=(0.5,-0.14))
        plots.append(lp)
    return plots

def best_locality(args, locality=[2,3,4,5,6,7,8,9,10,12,14,16,18,20,22,24,26,28,30,40,50,100,150,200,250,300,400,500,1000]):#,1500,2000,2500]):
    root_fp = os.path.join("explainable_kge/logger/logs", exp_config["dataset"]["name"] + "_" + exp_config["model"]["name"])
    # load all the data
    summarys = pd.DataFrame(columns=["log_num","k","Relation","Coverage","Fidelity","F1-Fidelity", "Weight"])
    for log_num in range(1,6):
        folder_path = root_fp + "_" + str(log_num)
        for k in locality:
            result_name = "{}.pkl".format(args["explain"]["xmodel"] + "_local3_" + str(k))
            results_fp = os.path.join(folder_path, "results", result_name)
            result = load_data(results_fp)
            summary = get_summary(result)
            summary["k"] = k
            summary["log_num"] = log_num
            summarys = summarys.append(summary, ignore_index=True)
    # average across log_num
    rel_summary = pd.DataFrame(columns=["k","Relation","F1-Fidelity","Weight"])
    relations = np.unique(summary["Relation"].to_numpy(dtype=str))
    relations = [rel for rel in relations if rel != "Mean" and rel != "Std"]
    for rel in relations:
        for k in locality:
            rel_summ = summarys.loc[summarys["Relation"].isin([rel]),:]
            k_rel_summ = rel_summ.loc[rel_summ["k"].isin([k]),:]
            wavg, wstd = weighted_avg_and_std(k_rel_summ["F1-Fidelity"].values, k_rel_summ["Weight"].values)
            rel_summary = rel_summary.append({"k":k,"Relation":rel,"Avg F1-Fidelity":wavg,"Std F1-Fidelity":wstd,"Weight":np.sum(k_rel_summ["Weight"].values)}, ignore_index=True)
    # select the best locality result for each relation
    best_summary = pd.DataFrame(columns=["k","Relation","Avg F1-Fidelity","Std F1-Fidelity","Weight"])
    for rel in relations:
        rel_summ = rel_summary.loc[rel_summary["Relation"].isin([rel]),:]
        best_summary = best_summary.append(rel_summ.loc[rel_summ["Avg F1-Fidelity"].idxmax()], ignore_index=True)
    wavg, _ = weighted_avg_and_std(best_summary["Avg F1-Fidelity"].values, best_summary["Weight"].values)
    wstd, _ = weighted_avg_and_std(best_summary["Std F1-Fidelity"].values, best_summary["Weight"].values)
    best_summary = best_summary.append({"Relation":"Overall","Avg F1-Fidelity":wavg,"Std F1-Fidelity":wstd}, ignore_index=True)
    best_summary = best_summary[["Relation","Avg F1-Fidelity","Std F1-Fidelity","Weight","k"]]
    # plot table
    title_str = "Student: " + exp_config["explain"]["xmodel"] + ", Locality: Best"
    fig = plot_table(stats=best_summary[best_summary.columns[1:]].to_numpy(dtype=float),
                     row_labels=best_summary["Relation"].to_numpy(str),
                     col_labels=best_summary.columns[1:].to_numpy(str),
                     title=title_str)
    return [fig]

def global_plot(args):
    # load the data
    root_fp = os.path.join("explainable_kge/logger/logs", exp_config["dataset"]["name"] + "_" + exp_config["model"]["name"])
    summarys = pd.DataFrame(columns=["log_num","Relation","Coverage","Fidelity","F1-Fidelity","Weight"])
    for log_num in range(1,6):
        folder_path = root_fp + "_" + str(log_num)
        result_name = "{}.pkl".format(args["explain"]["xmodel"] + "_global_" + str(exp_config["explain"]["locality_k"]))
        results_fp = os.path.join(folder_path, "results", result_name)
        if not os.path.exists(results_fp):
            logout("Experiment results pickle does not exist: " + str(results_fp), "f")
            exit()
        results = load_data(results_fp)
        result_summary = get_summary(results)
        result_summary["log_num"] = log_num
        summarys = summarys.append(result_summary)
    # average across log_num
    rel_summary = pd.DataFrame(columns=["Relation","Avg F1-Fidelity","Std F1-Fidelity","Weight"])
    relations = np.unique(summarys["Relation"].to_numpy(dtype=str))
    relations = [rel for rel in relations if rel != "Mean" and rel != "Std"]
    for rel in relations:
        rel_summ = summarys.loc[summarys["Relation"].isin([rel]),:]
        wavg, wstd = weighted_avg_and_std(rel_summ["F1-Fidelity"].values, rel_summ["Weight"].values)
        rel_summary = rel_summary.append({"Relation":rel,"Avg F1-Fidelity":wavg, "Std F1-Fidelity":wstd,"Weight":np.sum(rel_summ["Weight"].values)}, ignore_index=True)
    # average across rels
    wavg, _ = weighted_avg_and_std(rel_summary["Avg F1-Fidelity"].values, rel_summary["Weight"].values)
    wstd, _ = weighted_avg_and_std(rel_summary["Std F1-Fidelity"].values, rel_summary["Weight"].values)
    rel_summary = rel_summary.append({"Relation":"Overall","Avg F1-Fidelity":wavg,"Std F1-Fidelity":wstd}, ignore_index=True)
    # plot table
    title_str = "Student: " + exp_config["explain"]["xmodel"] + ", Locality: Global"
    fig = plot_table(stats=rel_summary[rel_summary.columns[1:]].to_numpy(dtype=float),
                     row_labels=rel_summary["Relation"].to_numpy(str),
                     col_labels=rel_summary.columns[1:].to_numpy(str),
                     title=title_str)
    return [fig]


class TurkerResult:
    def __init__(self, amt_json, amt_id):
        self.user_id = amt_id
        self.prac_data = amt_json["examples"]
        self.gold_data = amt_json["test"][-1:]
        self.test_id = int(amt_json["testId"])
        self.test_data = amt_json["test"][:-1]
        self.survey_data = amt_json["questionnaireData"]
        self.prac_score = 0.0
        self.gold_score = 0.0
        self.test_score = 0.0

    def eval_practice(self, answers):
        self.prac_score = eval_questions_v1(answers, self.prac_data)

    def eval_test_v1(self, answers):
        self.test_score = eval_questions_v1(answers, self.test_data)

    def eval_test_v2(self, answers):
        self.test_score = eval_questions_v2(answers, self.test_data)

    def eval_gold(self, answers):
        self.gold_score = eval_questions_v1(answers, self.gold_data)


class TurkerResultRQ1:
    def __init__(self, amt_json, amt_json_fp):
        self.user_id = amt_json["turkId"]
        self.test_id = int(amt_json["testId"])
        self.exp = amt_json_fp.split("/")[-1].split("_")[1]
        if self.exp == "exp0":
            self.test_data = amt_json["test"][0:2] + amt_json["test"][3:12] + amt_json["test"][13:]
            self.gold_data = [amt_json["test"][2]] + [amt_json["test"][12]]
        else:
            self.test_data = amt_json["test"][0:1] + amt_json["test"][2:5] + amt_json["test"][6:9] + amt_json["test"][10:13] + amt_json["test"][13:]
            self.gold_data = [amt_json["test"][1]] + [amt_json["test"][5]] + [amt_json["test"][9]] + [amt_json["test"][13]]
        self.survey_data = amt_json["questionnaireData"]
        self.gold_score = 0.0
        self.q1_score = 0.0
        self.q2_score = 0.0

    def eval_test(self):
        self.q1_score, self.q2_score = self.eval_questions(self.test_data)

    def eval_gold(self):
        self.gold_score = self.eval_gold_questions(self.gold_data)

    def eval_questions(self, data):
        num_q1 = 0
        num_q2 = 0
        correct_q1 = 0
        correct_q2 = 0
        for i, sample in enumerate(data):
            q_id, q_type, ans_id, ans_order = sample
            if q_type:
                if ans_id == ans_order.index("x"): correct_q1 += 1
                num_q1 += 1
            else:
                if ans_id == ans_order.index("c"): correct_q2 += 1
                num_q2 += 1
        return float(correct_q1) / float(num_q1), float(correct_q2) / float(num_q2)

    def eval_gold_questions(self, data):
        num_q1 = 0
        correct_q1 = 0
        for i, sample in enumerate(data):
            _, q_type, ans_id, ans_order = sample
            if self.exp == "exp0":
                if ans_id == ans_order.index("x"): correct_q1 += 1
                num_q1 += 1
            else:
                if q_type:
                    if ans_id == ans_order.index("x"): correct_q1 += 1
                    num_q1 += 1
                else:
                    if ans_id == ans_order.index("c"): correct_q1 += 1
                    num_q1 += 1
        return float(correct_q1) / float(num_q1)


def get_cont_tables(samples):
    ans_type2id = {"x":0,"c":1}
    chi_tables = np.zeros(shape=(16,2,2))
    for user_test in samples:
        for user_answer in user_test.test_data:
            q_id, q_type, ans_id, ans_order = user_answer
            chi_tables[q_id,ans_type2id[ans_order[ans_id]],int(not q_type)] += 1
            chi_tables[-1,ans_type2id[ans_order[ans_id]],int(not q_type)] += 1
    return chi_tables


def fishers_significance(tables):
    sigs = []
    for table in tables:
        # performs two Fisher's exact method tests of independence for 2x2
        _, pval, _, _ = stats.chi2_contingency(table)
        _, pval1 = stats.fisher_exact(table)
        _, pval2 = stats.fisher_exact(np.rot90(table,k=2))
        # gets Simes corrected alpha values
        if pval1 < pval2:
            alpha1 = 0.05 / 2
            alpha2 = 0.05
        else:
            alpha1 = 0.05
            alpha2 = 0.05 / 2
        sigs.append([[pval,pval],[pval1,pval2],[alpha1,alpha2],])
    return np.asarray(sigs)


def plot_cont_tables(tables, sigs):
    figs = []
    for i, table in enumerate(tables):
        table_ = table.copy()
        table_ = np.append(table_, sigs[i], axis=0)
        title = "Contingency Table for"
        title += " Question " + str(i+1) if i+1 < len(tables) else " All Questions"
        fig = plot_table(stats=table_, row_labels=["XKGE", "Causal", "Chi-P", "Fisher-P", "AdjA"], col_labels=["XKGE Q","Causal Q"], figure_size=(5,6), title=title)
        figs.append(fig)
    return figs


class AmtTest:
    def __init__(self, test_id, q_json):
        self.test_id = test_id
        self.qs, self.qas = self.parse_q_json(q_json)
        self.turker_ids = []
        self.turker_results = []

    def parse_q_json(self, q_json):
        qs = []
        qas = []
        for q in q_json:
            q_str = []
            qa_str = []
            for q_part in q["parts"]:
                options = [q_part["str_list"]]
                options += q_part["corrections"] + [["None","of","above"]]
                q_str.append(options)
                qa_str.append([0,0,0,0,0])
            possible_fact = copy(q["str_list"])
            possible_fact[2] += " is False."
            options = [possible_fact]
            possible_fact = copy(q["str_list"])
            possible_fact[2] += " is True."
            options += [possible_fact]
            q_str.append(options)
            qa_str.append([0,0])
            qs.append(q_str)
            qas.append(qa_str)
        return qs, qas
    
    def add_turker(self, turker_id, answers):
        self.turker_ids.append(turker_id)
        for i, q in enumerate(answers):
            for j, answer in enumerate(q):
                if j+1 == len(q):
                    self.qas[i][j][answer] += 1
                else:
                    self.qas[i][j][answer+1] += 1


class AmtTestRQ1:
    def __init__(self, test_id, q_json):
        self.test_id = test_id
        self.qs, self.qas = self.parse_q_json(q_json)
        self.turker_ids = []
        self.turker_results = []

    def parse_q_json(self, q_json):
        qs = []
        qas = []
        for q in q_json:
            options = []
            for response in q["responses"]:
                _, ex_str = response
                options += [ex_str]
            qs.append(options)
            qas.append([[0,0],[0,0]])
        return qs, qas
    
    def add_turker(self, turker_id, answers):
        ans_type2id = {"x":0,"c":1}
        self.turker_ids.append(turker_id)
        for i, q in enumerate(answers):
            q_id, q_type, ans_id, ans_order = q
            ans_type = ans_order[ans_id]
            self.qas[q_id][ans_type2id[ans_type]][not q_type] += 1


def eval_questions_v1(answers, data):
    assert len(answers) == len(data)
    num_q = 0
    correct = 0
    for i in range(len(answers)):
        try:
            assert len(answers[i])+1 == len(data[i])
        except:
            pdb.set_trace()
        correct += np.count_nonzero(np.asarray(answers[i]) == np.asarray(data[i][:-1]))
        if (np.all(np.asarray(answers[i])==-1) and data[i][-1]) or (np.any(np.asarray(answers[i])!=-1) and not data[i][-1]):
            correct += 1
        num_q += len(data[i])
    return float(correct) / float(num_q)


def eval_questions_v2(answers, data):
    assert len(answers) == len(data)
    num_q = 0
    correct = 0
    for i in range(len(answers)): # i loops on examples
        try:
            assert len(answers[i])+1 == len(data[i])
        except:
            pdb.set_trace()
        for j in range(len(answers[i])): # j loop on example parts
            if data[i][j] in answers[i][j]:
                correct += 1
            num_q += 1
    return float(correct) / float(num_q)


def plot_answer_tally(amt_tally):
    figs = []
    for test_id, test_tally in amt_tally.items():
        for i, q in enumerate(test_tally.qs):
            labels = []
            vals = []
            for j, q_part in enumerate(q):
                for k, q_part_option in enumerate(q_part):
                    labels.append(" ".join(q_part_option))
                    vals.append(test_tally.qas[i][j][k])
                labels.append(" ")
                vals.append(0)
            fig_title = "Practice Question" if test_id == -1 else "Test Question"
            fig = plot_table(stats=np.asarray([vals]).T,
                             row_labels=labels,
                             col_labels=["Count"],
                             fmt_string="{:.0f}",
                             figure_size=(5, 6),
                             title=fig_title)
            figs.append(fig)
    return figs


def plot_answer_tally_rq1(amt_tally):
    figs = []
    for test_id, test_tally in amt_tally.items():
        for i, q in enumerate(test_tally.qs):
            labels = []
            vals = []
            for j, q_option in enumerate(q):
                labels.append(q_option)
                vals.append(test_tally.qas[i][j])
            fig_title = "Practice Question" if test_id == -1 else "Test Question"
            fig = plot_table(stats=np.asarray(vals),
                             row_labels=labels,
                             col_labels=["XKGE Q Count","Causal Q Count"],
                             fmt_string="{:.0f}",
                             figure_size=(5, 6),
                             title=fig_title)
            figs.append(fig)
    return figs


def get_majority_score(args, tally, prac_json, test_json, gt, e2i, r2i, toggle):
    m_answers = []
    for i, q in enumerate(tally[-1].qas):
        m_answer = []
        for j, q_part in enumerate(q):
            if j+1 == len(q):
                m_answer.append(np.argmax(q_part))
            else:
                m_answer.append(np.argmax(q_part)-1)
        m_answers.append(m_answer)
    prac_answers = [[part["correct_id"] for part in question['parts']] for question in prac_json]
    prac_score = eval_questions_v1(prac_answers, m_answers)

    m_answers = []
    tally_keys = sorted(list(tally.keys()))
    del tally_keys[tally_keys.index(-1)]
    for test_id in tally_keys:
        for i, q in enumerate(tally[test_id].qas):
            m_answer = []
            for j, q_part in enumerate(q):
                if j+1 == len(q):
                    m_answer.append(np.argmax(q_part))
                else:
                    m_answer.append(np.argmax(q_part)-1)
            m_answers.append(m_answer)
    stop = int(args["plotting"]["num_examples"]) * (max(tally_keys)+1)
    if toggle:
        test_answers = [[part["correct_id"] for part in question['parts']] for question in test_json[:stop]]
        test_score = eval_questions_v1(test_answers, m_answers)
    else:
        test_answers = load_test_answers(gt, e2i, r2i, test_json[:stop])
        test_scores = []
        for i in range(int(len(m_answers)/int(args["plotting"]["num_examples"]))):
            test_scores.append(eval_questions_v2(test_answers[int(args["plotting"]["num_examples"])*i:int(args["plotting"]["num_examples"])*i+int(args["plotting"]["num_examples"])],
                                                 m_answers[int(args["plotting"]["num_examples"])*i:int(args["plotting"]["num_examples"])*i+int(args["plotting"]["num_examples"])]))
    
    return prac_score, np.asarray(test_scores)


def get_best_turker_score(args, tally, amt_data, test_json, gt, e2i, r2i, toggle):
    best_prac_score = max([amt_result.prac_score for amt_result in amt_data])

    b_answers = []
    tally_keys = sorted(list(tally.keys()))
    del tally_keys[tally_keys.index(-1)]
    for test_id in tally_keys:
        test_turker_ids = tally[test_id].turker_ids
        test_turkers = [amt_result for amt_result in amt_data if amt_result.user_id in test_turker_ids]
        #assert len(test_turkers) == 3
        test_turker_practice_scores = [test_turker.prac_score for test_turker in test_turkers]
        best_turker = test_turkers[np.argmax(test_turker_practice_scores)]
        b_answers += best_turker.test_data
    stop = int(args["plotting"]["num_examples"]) * (max(tally_keys)+1)
    if toggle:
        test_answers = [[part["correct_id"] for part in question['parts']] for question in test_json[:stop]]
        best_test_score = eval_questions_v1(test_answers, b_answers)
    else:
        test_answers = load_test_answers(gt, e2i, r2i, test_json[:stop])
        test_scores = []
        for i in range(int(len(b_answers)/int(args["plotting"]["num_examples"]))):
            test_scores.append(eval_questions_v2(test_answers[int(args["plotting"]["num_examples"])*i:int(args["plotting"]["num_examples"])*i+int(args["plotting"]["num_examples"])],
                                                 b_answers[int(args["plotting"]["num_examples"])*i:int(args["plotting"]["num_examples"])*i+int(args["plotting"]["num_examples"])]))
    return best_prac_score, (np.mean(test_scores),np.std(test_scores))


def get_cohen_kappa(tally, amt_data):
    tally_keys = sorted(list(tally.keys()))
    del tally_keys[tally_keys.index(-1)]

    test_answers = {}
    for test_id in tally_keys:
        test_turker_ids = tally[test_id].turker_ids
        test_turkers = [amt_result for amt_result in amt_data if amt_result.user_id in test_turker_ids]
        assert len(test_turkers) == 3
        for turker in test_turkers:
            if test_id in test_answers:
                flat_answers = [answer for question in turker.test_data for answer in question]
                test_answers[test_id].append(flat_answers)
            else:
                flat_answers = [answer for question in turker.test_data for answer in question]
                test_answers[test_id] = [flat_answers]
    cohen_kappas = []
    for test_id in test_answers:
        for i in range(len(test_answers[test_id])):
            for j in range(i+1,len(test_answers[test_id])):
                cohen_kappas.append(cohen_kappa_score(test_answers[test_id][i],test_answers[test_id][j],labels=[-1,0,1,2,3]))
    cohen_kappas = np.asarray(cohen_kappas)
    return np.mean(cohen_kappas), np.std(cohen_kappas)


def get_cronbach_alpha(tally, amt_data):
    tally_keys = sorted(list(tally.keys()))
    del tally_keys[tally_keys.index(-1)]

    test_answers = {}
    for test_id in tally_keys:
        test_turker_ids = tally[test_id].turker_ids
        test_turkers = [amt_result for amt_result in amt_data if amt_result.user_id in test_turker_ids]
        assert len(test_turkers) == 3
        for turker in test_turkers:
            if test_id in test_answers:
                flat_answers = [answer for question in turker.test_data for answer in question]
                test_answers[test_id].append(flat_answers)
            else:
                flat_answers = [answer for question in turker.test_data for answer in question]
                test_answers[test_id] = [flat_answers]
    alphas = []
    for test_id in test_answers:
        for i in range(len(test_answers[test_id])):
            for j in range(i+1,len(test_answers[test_id])):
                alphas.append(cronbach_alpha_score([test_answers[test_id][i],test_answers[test_id][j]]))
    alphas = np.asarray(alphas)
    return np.mean(alphas), np.std(alphas)


def cronbach_alpha_score(itemscores):
    itemscores = np.asarray(itemscores)
    itemvars = itemscores.var(axis=1, ddof=1)
    tscores = itemscores.sum(axis=0)
    nitems = len(itemscores)

    return nitems / (nitems-1.) * (1 - itemvars.sum() / tscores.var(ddof=1))


def get_chance_stats(args, tally, test_json, gt, e2i, r2i, toggle):
    tally_keys = sorted(list(tally.keys()))
    del tally_keys[tally_keys.index(-1)]
    stop = int(args["plotting"]["num_examples"]) * (max(tally_keys)+1)
    if toggle:
        test_answers = [[part["correct_id"] for part in question['parts']] for question in test_json[:stop]]
    else:
        test_answers = load_test_answers(gt, e2i, r2i, test_json[:stop])
        num_answers = [len(part) for question in test_answers for part in question]
        u, counts = np.unique(num_answers, return_counts=True)
        u_list = u.tolist()
        correct_counts = [0] * len(u)
        total_counts = [0] * len(u)
        for j, q in enumerate(test_answers):
            for k, part_answers in enumerate(q):
                part_tally = tally[j//15].qas[j%15][k]
                for tally_idx, tally_count in enumerate(part_tally):
                    if not tally_count: continue
                    if tally_idx-1 in part_answers:
                        correct_counts[u_list.index(len(part_answers))] += tally_count
                    total_counts[u_list.index(len(part_answers))] += tally_count
        accs = np.asarray(correct_counts) / np.asarray(total_counts)
    return u, counts, accs


def plot_exit_survey(amt_data):
    ages = ["18-24","25-34","35-44","45-54","55+"]
    ages_ = [0,0,0,0,0]
    genders = ["Male","Female","Other"]
    genders_ = [0,0,0]
    exposures = ["Never","Rarely","Sometimes","Often"]
    exposures_ = [0,0,0,0]
    feedback = []
    for amt_result in amt_data:
        ages_[ages.index(amt_result.survey_data["age"])] += 1
        genders_[genders.index(amt_result.survey_data["gender"])] += 1
        exp = int(amt_result.survey_data["exposure"])-1 if type(amt_result.survey_data["exposure"]) == str else 0
        exposures_[exp] += 1
        feedback.append(amt_result.survey_data["feedback"])
    figs = []
    figs.append(plot_bar(ages_, ages, title="Turker Ages", colors=["r","b","g","c","m"]))
    figs.append(plot_bar(genders_, genders, title="Turker Genders", colors=["r","b","g"]))
    figs.append(plot_bar(exposures_, exposures, title="Turker Exposure", colors=["r","b","g","c"]))
    figs.append(plot_table(np.asarray([np.arange(len(feedback))]).T, feedback, ["Feedback"]))
    return figs


def load_dataset(args):
    ds = model_utils.load_dataset(args)
    ds.load_bernouli_sampling_stats()
    if args["model"]["name"] == "tucker":
        ds.reload_er_vocab()
    ds.load_current_ents_rels()
    ds.load_current_ents_rels()
    ds.model_name = None  # makes __getitem__ only retrieve triples instead of triple pairs
    return ds


def load_gt_triples(args):
    dirty_ds_name = copy(args["dataset"]["name"])
    clean_ds_name = dirty_ds_name.split("_")[0] + "_CLEAN_" + dirty_ds_name.split("_")[-1]
    clean_train_args = copy(args)
    clean_train_args["dataset"]["name"] = clean_ds_name
    clean_train_args["dataset"]["set_name"] = "0_train2id"
    clean_train_args["continual"]["session"] = 0
    clean_tr_dataset = load_dataset(clean_train_args)
    clean_gt_triples = clean_tr_dataset.load_triples(["0_gt2id.txt"])
    return np.asarray(clean_gt_triples), clean_tr_dataset.e2i, clean_tr_dataset.r2i


def in_filter_triples(triples, filter_triples):
    filtered_triples = []
    for triple in triples:
        if any(np.equal(triple, filter_triples).all(1)):
            filtered_triples.append(triple.tolist())
    return np.asarray(filtered_triples)


def load_test_answers(gt, e2i, r2i, json_file):
    answers = []
    for example in json_file:
        ex_answers = []
        for part in example['parts']:
            part_answer = []
            # checks if default (-1) correct
            split_fact = part["fact"].split(",")
            r = split_fact[1]
            if r[0] == "_":
                h = split_fact[-1]
                t = split_fact[0]
            else:
                h = split_fact[0]
                t = split_fact[-1]
            hi = e2i[h]
            ri = r2i[r.replace("_","")]
            ti = e2i[t]
            if in_filter_triples(np.asarray([[hi,ri,ti]]),gt).shape[0]:
                part_answer.append(-1)
            # checks if any corrections correct
            for i, triple_str in enumerate(part["correction_triples"]):
                split_fact = triple_str.split(",")
                r = split_fact[1]
                if r[0] == "_":
                    h = split_fact[-1]
                    t = split_fact[0]
                else:
                    h = split_fact[0]
                    t = split_fact[-1]
                hi = e2i[h]
                ri = r2i[r.replace("_","")]
                ti = e2i[t]
                if in_filter_triples(np.asarray([[hi,ri,ti]]),gt).shape[0]:
                    part_answer.append(i)
            if not len(part_answer):
                part_answer.append(3)
            ex_answers.append(part_answer)
        answers.append(ex_answers)
    return answers


def test_normality(data):
    if len(data) > 8:
        k1, p1 = stats.normaltest(data)
        logout("Skew/Kurtosis P-Value: " + str(p1))
    k2, p2 = stats.shapiro(data)
    logout("Shapiro P-Value: " + str(p2))
    if len(data) > 8:
        normal = p1 > 5e-2 and p2 > 5e-2
    else:
        normal = p2 > 5e-2

    fig, ax = plt.subplots(figsize=(8, 4))
    fig = probscale.probplot(data, ax=ax, bestfit=True, estimate_ci=True,
                            line_kws={'label': 'BF Line', 'color': 'b'},
                            scatter_kws={'label': 'Observations'},
                            problabel='Probability (%)')
    ax.legend(loc='lower right')
    seaborn.despine(fig)

    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # stats.probplot(data, plot=ax)
    # ax.set_title("Probplot for Turker feedback accuracy")
    return normal, fig


def correlation_ratio(categories, measurements):
    fcat, _ = pd.factorize(categories)
    cat_num = np.max(fcat)+1
    y_avg_array = np.zeros(cat_num)
    n_array = np.zeros(cat_num)
    for i in range(0,cat_num):
        cat_measures = measurements[np.argwhere(fcat == i).flatten()]
        n_array[i] = len(cat_measures)
        y_avg_array[i] = np.average(cat_measures)
    y_total_avg = np.sum(np.multiply(y_avg_array,n_array))/np.sum(n_array)
    numerator = np.sum(np.multiply(n_array,np.power(np.subtract(y_avg_array,y_total_avg),2)))
    denominator = np.sum(np.power(np.subtract(measurements,y_total_avg),2))
    if numerator == 0:
        eta = 0.0
    else:
        eta = numerator/denominator
    return eta


def check_variances(amt_tests):
    scores = {}
    for amt_test in amt_tests:
        if amt_test.test_id not in scores:
            scores[amt_test.test_id] = [amt_test.test_score]
        else:
            scores[amt_test.test_id].append(amt_test.test_score)
    scores_ordered = []
    for i in range(len(scores)):
        scores_ordered.append(scores[i])
    scores_ordered = tuple(scores_ordered)
    fval, pval = stats.f_oneway(*scores_ordered)
    scores_ordered = []
    labels_ordered = []
    for i in range(len(scores)):
        scores_ordered.append(scores[i])
        labels_ordered.append([i,i,i])
    scores_ordered = np.asarray(scores_ordered)
    labels_ordered = np.asarray(labels_ordered)
    eta_squared = correlation_ratio(labels_ordered.flatten(), scores_ordered.flatten())
    num_samples = ping.power_anova(eta_squared, 5, None, 0.8, 0.05)
    pdb.set_trace()


def amt_plot_rq2(args):
    fp = os.path.join("explainable_kge/logger/logs", args["dataset"]["name"] + "_" + args["model"]["name"] + "_" + str(args["logging"]["log_num"]))
    results_fp = os.path.join(os.path.abspath(fp), "results")
    locality_str = str(args["explain"]["locality_k"]) if type(args["explain"]["locality_k"]) == int else "best"
    # load the testing json
    json_name = "explanations_" + args["explain"]["xmodel"] + "_" + args["explain"]["locality"] + "_" + locality_str + "_clean_filtered"
    json_fp = os.path.join(results_fp, json_name + '.json')
    with open(json_fp, "r") as f:
        test_json = json.load(f)
    # load the practice json
    json_name = "examples"
    json_fp = os.path.join(results_fp, json_name + '.json')
    with open(json_fp, "r") as f:
        prac_json = json.load(f)
    prac_answers = [[part["correct_id"] for part in question['parts']] for question in prac_json]
    gold_answers = [[part["correct_id"] for part in question['parts']] for question in prac_json[1:2]]
    # load the gt triples
    gt_triples, gt_e2i, gt_r2i = load_gt_triples(args)
    # load the amt results, tally answers, and score each turker individually
    amt_data = []
    amt_tally = {}
    amt_folder = "amt_" + args["explain"]["xmodel"] + "_" + args["explain"]["locality"] + "_" + locality_str
    amt_fp = os.path.join(results_fp, amt_folder)
    amt_results_fps = [os.path.join(amt_fp, file) for file in os.listdir(amt_fp) if file.endswith(".json")]
    for i, amt_results_fp in enumerate(amt_results_fps):
        with open(amt_results_fp, "r") as f:
            turker_data = TurkerResult(json.load(f), amt_results_fp)
        turker_data.eval_practice(prac_answers)
        if -1 in amt_tally:
            amt_tally[-1].add_turker(turker_data.user_id, turker_data.prac_data)
        else:
            amt_tally[-1] = AmtTest(-1, prac_json)
            amt_tally[-1].add_turker(turker_data.user_id, turker_data.prac_data)
        turker_data.eval_gold(gold_answers)
        start = turker_data.test_id * int(args["plotting"]["num_examples"])
        stop = start + int(args["plotting"]["num_examples"])
        test_answers = load_test_answers(gt_triples, gt_e2i, gt_r2i, test_json[start:stop])
        turker_data.eval_test_v2(test_answers)
        if turker_data.test_id in amt_tally:
            amt_tally[turker_data.test_id].add_turker(turker_data.user_id, turker_data.test_data)
        else:
            amt_tally[turker_data.test_id] = AmtTest(turker_data.test_id, test_json[start:stop])
            amt_tally[turker_data.test_id].add_turker(turker_data.user_id, turker_data.test_data)
        amt_data.append(turker_data)
    # make answer tally figures for practice and test
    figs = plot_answer_tally(amt_tally)
    # make individual, average, and majority vote performance figures
    correct_acc = []
    turker_ids = []
    for amt_result in amt_data:
        correct_acc.append([amt_result.prac_score, amt_result.test_score])
        turker_ids.append(amt_result.user_id)
    figs += check_variances(amt_data)
    correct_acc = np.asarray(correct_acc)
    correct_acc = np.append(correct_acc, [[np.mean(correct_acc[:,0]), np.mean(correct_acc[:,1])]], axis=0)
    correct_acc = np.append(correct_acc, [[np.std(correct_acc[:,0]), np.std(correct_acc[:,1])]], axis=0)
    m_prac, m_test_scores = get_majority_score(args, amt_tally, prac_json, test_json, gt_triples, gt_e2i, gt_r2i, 0)
    check, fig = test_normality(m_test_scores)
    figs.append(fig)
    if check:
        logout("Majority-vote turker feedback accuracy samples passed normality tests","s")
    else:
        logout("Majority-vote turker feedback accuracy samples did not normality tests","e")
    correct_acc = np.append(correct_acc, [[m_prac, np.mean(m_test_scores)]], axis=0)
    correct_acc = np.append(correct_acc, [[0.0, np.std(m_test_scores)]], axis=0)
    b_prac, b_test = get_best_turker_score(args, amt_tally, amt_data, test_json, gt_triples, gt_e2i, gt_r2i, 0)
    correct_acc = np.append(correct_acc, [[b_prac, b_test[0]]], axis=0)
    correct_acc = np.append(correct_acc, [[0.0, b_test[1]]], axis=0)
    ck_mean, ck_std = get_cohen_kappa(amt_tally, amt_data)
    correct_acc = np.append(correct_acc, [[0.0, ck_mean]], axis=0)
    correct_acc = np.append(correct_acc, [[0.0, ck_std]], axis=0)
    ca_mean, ca_std = get_cronbach_alpha(amt_tally, amt_data)
    correct_acc = np.append(correct_acc, [[0.0, ca_mean]], axis=0)
    correct_acc = np.append(correct_acc, [[0.0, ca_std]], axis=0)
    fig = plot_table(stats=np.asarray(correct_acc),
                     row_labels=turker_ids+["Individual Mean", "Individual STD", "Majority Mean", "Majority STD", "Best Mean", "Best STD", "CK Mean", "CK STD", "CA Mean", "CA STD"],
                     col_labels=["Practice", "Test"],
                     figure_size=(5,6))
    figs.append(fig)
    us, totals, accs = get_chance_stats(args, amt_tally, test_json, gt_triples, gt_e2i, gt_r2i, 0)
    chance_stats = []
    chance_prob = 0.0
    actul_acc = 0.0
    for i in range(len(us)):
        chance_stats.append([totals[i], accs[i]])
        chance_prob += (float(us[i]) / 5.0) * totals[i]
        actul_acc += accs[i] * totals[i]
    chance_stats.append([chance_prob / sum(totals), actul_acc / sum(totals)])
    fig = plot_table(stats=np.asarray(chance_stats),
                     row_labels=[str(u) for u in us]+["Chance Probability"],
                     col_labels=["Number Q in Test","Turker Acc"],
                     figure_size=(5,6))
    figs.append(fig)
    # make survey histogram figure
    figs += plot_exit_survey(amt_data)
    return figs


def amt_plot_rq1(args):
    # loads the test json
    results_fp = "explainable_kge/logger/logs"
    amt_folder = "rq1_amt_data"
    json_fp = os.path.join(results_fp, amt_folder, "rq1_amt_data.json")
    with open(json_fp, "r") as f:
        test_json = json.load(f)
    # load the amt results, tally answers, and score each turker individually
    amt_data = []
    amt_tally = {}
    amt_fp = os.path.join(results_fp, amt_folder)
    amt_results_fps = [os.path.join(amt_fp, file) for file in os.listdir(amt_fp) if file.endswith(".json")]
    for i, amt_results_fp in enumerate(amt_results_fps):
        if 'rq1_amt_data.json' in amt_results_fp: continue
        with open(amt_results_fp, "r") as f:
            turker_data = TurkerResultRQ1(json.load(f), amt_results_fp)
        turker_data.eval_test()
        turker_data.eval_gold()
        if turker_data.gold_score == 1.0:
            if turker_data.test_id in amt_tally:
                amt_tally[turker_data.test_id].add_turker(turker_data.user_id, turker_data.test_data)
            else:
                amt_tally[turker_data.test_id] = AmtTestRQ1(turker_data.test_id, test_json)
                amt_tally[turker_data.test_id].add_turker(turker_data.user_id, turker_data.test_data)
            amt_data.append(turker_data)
    # make answer tally figures for practice and test
    figs = plot_answer_tally_rq1(amt_tally)
    # make chi squared count figure
    tbls = get_cont_tables(amt_data)
    sigs = fishers_significance(tbls)
    figs += plot_cont_tables(tbls, sigs)

    # make individual performance figures
    correct_acc = []
    turker_ids = []
    for amt_result in amt_data:
        correct_acc.append([amt_result.q1_score, amt_result.q2_score])
        turker_ids.append(amt_result.user_id)
    correct_acc = np.asarray(correct_acc)
    correct_acc = np.append(correct_acc, [[np.mean(correct_acc[:,0]),np.mean(correct_acc[:,1])]], axis=0)
    correct_acc = np.append(correct_acc, [[np.std(correct_acc[:,0]),np.std(correct_acc[:,0])]], axis=0)
    fig = plot_table(stats=np.asarray(correct_acc),
                     row_labels=turker_ids+["Individual Mean", "Individual STD"],
                     col_labels=["XKGE Q Score","Causal Q Score"],
                     figure_size=(5,6))
    figs.append(fig)
    # make survey histogram figure
    figs += plot_exit_survey(amt_data)
    return figs

if __name__ == "__main__":
    plt.rcParams.update({'figure.max_open_warning': 0}) # hides the too many plots warning
    exp_config = load_config("Experiment Visualizations")
    plt.rcParams.update({'font.weight': 'bold'})
    figs = []
    if exp_config["plotting"]["mode"] == "amt1":
        figs += amt_plot_rq1(exp_config)
        figs_fp = os.path.join("explainable_kge/logger/logs", "rq1_amt_data", "amt_{}.pdf".format(exp_config["explain"]["xmodel"]))
    elif exp_config["plotting"]["mode"] == "amt2":
        figs += amt_plot_rq2(exp_config)
        main_fp = os.path.join("explainable_kge/logger/logs", exp_config["dataset"]["name"] + "_" + exp_config["model"]["name"] + "_" + str(exp_config["logging"]["log_num"]))
        figs_fp = os.path.join(main_fp, "results", "amt_{}.pdf".format(exp_config["explain"]["xmodel"]))
    else:
        figs += locality_plot(exp_config)
        figs += global_plot(exp_config)
        figs += best_locality(exp_config)
        main_fp = os.path.join("explainable_kge/logger/logs", exp_config["dataset"]["name"] + "_" + exp_config["model"]["name"] + "_" + str(exp_config["logging"]["log_num"]))
        figs_fp = os.path.join(main_fp, "results", "{}.pdf".format(exp_config["explain"]["xmodel"]))
    figs2pdf(figs, figs_fp)
