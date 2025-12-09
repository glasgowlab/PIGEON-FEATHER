import os
import re

import matplotlib.colors as col
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Rectangle
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd
import seaborn as sns

from pigeon_feather.data import *
from pigeon_feather.tools import get_num_d_from_iso


font = {"family": "Arial", "weight": "normal", "size": 36}
axes = {"titlesize": 36, "titleweight": "bold", "labelsize": 36}
plt.rc("font", **font)
plt.rc("axes", **axes)
plt.rc("lines", lw=3)
colors = ["k", "red", "blue", "purple", "gray", "orange", "yellow", "green", "brown"]


class UptakePlot:
    def __init__(
        self,
        hdxms_datas,
        identifier: str,
        states_subset=None,
        color_dict=None,
        if_plot_fit=True,
        figure=None,
        ax=None,
        if_d_percent=False,
        exp_only=False,
        calculate_num_d_from_envelope=True,
        auto_plot=True
    ):
        """
        hdxms_datas: list of class HDXMSData objects

        :param hdxms_datas: list of HDXMSData objects
        :param identifier: peptide identifier
        :param states_subset: list of states to plot
        :param color_dict: dictionary of colors for each state
        :param if_plot_fit: if True, plot the fit line
        :param figure: figure object
        :param ax: axis object

        :ivar hdxms_datas_df: pandas DataFrame of the HDX-MS data of the peptide
        """
        self.calculate_num_d_from_envelope=calculate_num_d_from_envelope
        self.hdxms_datas = hdxms_datas
        self.identifier = identifier
        self.sequence = identifier.split(" ")[1]
        self.states_subset = states_subset
        self.color_dict = self.make_color_dict(color_dict)
        self.if_plot_fit = if_plot_fit
        self.figure = figure
        self.ax = ax
        self.if_d_percent = if_d_percent
        self.exp_only = exp_only
        # self.title = self.make_title()
        # self.title = identifier
        if auto_plot:
            self.uptakeplot = self.make_uptakeplot()


    def make_uptakeplot(self):
        "make a uptakeplot for a peptide"
        plt.rcParams["legend.fontsize"] = 22

        if self.figure is None and self.ax is None:
            figure, ax = plt.subplots(1, 1, figsize=(9, 8))

        else:
            figure = self.figure
            ax = self.ax

        scatter_shapes = [
            "o",
            "v",
            "^",
            "<",
            ">",
            "s",
            "p",
            "*",
            "h",
            "H",
            "D",
            "d",
            "P",
            "X",
        ]

        sns.scatterplot(
            data=self.hdxms_datas_df[self.hdxms_datas_df["time"] != 0],
            x="time",
            y="deut",
            hue="state",
            style="data_set_index",
            alpha=1.0,
            ax=ax,
            palette=self.color_dict,
            s=600,
            markers={1: "o", 2: "s", 3: "^", 4: "P", 5: "X", 6: "H", 7: "v"},
        )


        # Plot the fit
        if self.if_plot_fit:
            for state_name in self.hdxms_datas_df.state.unique():
                avg_peptide = self.get_average_peptide(state_name)
                # trialT, y_pred, popt, perr =  avg_peptide.fit_results
                trialT, y_pred, popt, perr = avg_peptide.new_fit()
                # trialT, y_pred, popt, perr =  avg_peptide.fit_hdx_stats()
                if trialT is None:
                    # raw data, no fit
                    trialT = [
                        tp.deut_time
                        for tp in avg_peptide.timepoints
                        if tp.deut_time != np.inf or tp.deut_time != 0
                    ]
                    y_pred = [
                        #tp.num_d
                        get_num_d_from_iso(tp)/(avg_peptide.max_d / avg_peptide.theo_max_d)
                        for tp in avg_peptide.timepoints
                        if tp.deut_time != np.inf or tp.deut_time != 0
                    ]
                ax.plot(
                    trialT, y_pred, "-", color=self.color_dict[state_name], alpha=0.5
                )
        else:
            sns.lineplot(
            data=self.hdxms_datas_df[self.hdxms_datas_df["time"] != 0],
            x="time",
            y="deut",
            hue="state",
            alpha=0.5,
            marker=None,
            ax=ax,
            palette=self.color_dict,
            estimator="mean",
            legend=False 
            )            


        # set up the plot
        ax.set_ylabel("# Deuterons")
        ax.set_xlabel("Time (seconds)")
        ax.set_xscale("log")
        # avg_peptide = self.get_average_peptide(self.hdxms_datas_df.state.unique()[0])

        ax.set_ylim(
            min(self.hdxms_datas_df["deut"]) - 1, max(self.hdxms_datas_df["deut"]) + 1
        )
        # ax.set_xlim(ax.get_xlim()[0]/10, ax.get_xlim()[1] * 10)
        xlim = ax.get_xlim()
        left_limit = 10 ** np.floor(np.log10(xlim[0]))
        right_limit = 10 ** np.ceil(np.log10(xlim[1]))
        ax.set_xlim(left_limit, right_limit)
        # ax.set_ylim(-0.3, avg_peptide.max_d*1.1)
        # ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.legend()

        if len(self.sequence) < 20:
            ax.set_title(self.identifier)
        else:
            idf = self.identifier.split(" ")[0] + " " + self.identifier.split(" ")[1][:4] + "..." + self.identifier.split(" ")[1][-4:]
            ax.set_title(idf)
        plt.subplots_adjust(bottom=0.2)
        plt.close()

        return figure

    @property
    def hdxms_datas_df(self):
        hdxms_datas_df = pd.DataFrame()

        for hdxms_data_index, hdxms_data in enumerate(self.hdxms_datas):
            hdxms_data_df = pd.DataFrame()

            for state in hdxms_data.states:
                if (
                    self.states_subset is not None
                    and state.state_name not in self.states_subset
                ):
                    continue

                peptide = state.get_peptide(self.identifier)

                if peptide is not None:
                    if self.exp_only:
                        timepoint_objs = [
                            tp for tp in peptide.timepoints if tp.note is None
                        ]
                    else:
                        timepoint_objs = [
                            tp for tp in peptide.timepoints if tp.deut_time != np.inf
                        ]
                   
                    deut_list = [get_num_d_from_iso(tp)  if self.calculate_num_d_from_envelope else tp.num_d for tp in timepoint_objs]
                    peptide_df_i = pd.DataFrame(
                        {
                            "time": [tp.deut_time for tp in timepoint_objs],
                            #"deut": [tp.num_d for tp in timepoint_objs],
                            "deut": deut_list,
                            "state": state.state_name,
                            "charge_state": [tp.charge_state for tp in timepoint_objs],
                        }
                    )

                    if self.if_d_percent:
                        peptide_df_i["deut"] = [tp.d_percent for tp in timepoint_objs]

                    hdxms_data_df = pd.concat(
                        [hdxms_data_df, peptide_df_i], ignore_index=True
                    )

            hdxms_data_df["data_set_index"] = hdxms_data_index + 1
            hdxms_datas_df = pd.concat(
                [hdxms_datas_df, hdxms_data_df], ignore_index=True
            )
        return hdxms_datas_df

    @hdxms_datas_df.setter
    def hdxms_datas_df(self, value):
        self._hdxms_datas_df = value

    @property
    def uptake_std(self):
        df = self.hdxms_datas_df[~self.hdxms_datas_df["time"].isin([0, np.inf])]
        deut_std = df.groupby(["state", "time"]).std().reset_index()["deut"].mean()
        return deut_std

    def get_average_peptide(self, state_name):
        "return an averaged peptide for a state"
        grouped_df = self.hdxms_datas_df.groupby("state")
        group = grouped_df.get_group(state_name)

        final_group = group.groupby("time")
        group_mean = final_group.mean(numeric_only=True).reset_index()
        group_std = final_group.std(numeric_only=True).reset_index()

        idf_start, idf_end = re.match(r"(-?\d+)-(-?\d+)", self.identifier).group(1, 2)
        states = [
            state
            for data in self.hdxms_datas
            for state in data.states
            if state.state_name == state_name
        ]

        peptide = Peptide(
            self.sequence,
            int(idf_start),
            int(idf_end),
            # f"averaged peptide: {state_name}",
            states[0],
            n_fastamides=self.hdxms_datas[0].n_fastamides,
        )
        for i in range(len(group_mean)):
            timepoint = Timepoint(
                peptide,
                group_mean["time"][i],
                group_mean["deut"][i],
                group_std["deut"][i],
            )
            peptide.add_timepoint(timepoint)

        return peptide

    def make_title(self):
        "make a title for the plot"
        for hdxms_data in self.hdxms_datas:
            for state in hdxms_data.states:
                try:
                    start = state.get_peptide(self.sequence).start
                    end = state.get_peptide(self.sequence).end
                    return f"{start}-{end} {self.sequence}"
                except AttributeError:
                    pass
        return f"Missing data for {self.sequence}"

    def make_color_dict(self, color_dict=None):
        "make a color dictionary for the states in the plot"
        if color_dict is None:
            colors = [
                "k",
                "red",
                "blue",
                "purple",
                "gray",
                "orange",
                "yellow",
                "green",
                "brown",
            ]

            color_dict = {}
            if self.states_subset is None:
                state_names = list(
                    set(
                        [
                            state.state_name
                            for hdxms_data in self.hdxms_datas
                            for state in hdxms_data.states
                        ]
                    )
                )
                state_names.sort()
            else:
                state_names = self.states_subset

            for i, state_name in enumerate(state_names):
                color_dict[state_name] = colors[i]
        return color_dict



# new residue coverage plotting script heat map thing - sav
class ResidueCoverage:
    def __init__(self, hdxms_data):
        self.hdxms_data = hdxms_data

    def calculate_coverages(self, state_name):
        state = self.hdxms_data.get_state(state_name)
        coverage = np.zeros(len(self.hdxms_data.protein_sequence))
        for pep in state.peptides:
            coverage[pep.start - 1 : pep.end] += 1
        return coverage

    def plot(self):
        row_num = len(self.hdxms_data.states)
        fig, axes = plt.subplots(
            row_num, 1, figsize=(20, 3 * row_num), sharex=True, sharey=True
        )

        coverage_list = []
        for state in self.hdxms_data.states:
            coverage = self.calculate_coverages(state.state_name)
            coverage_list.append(coverage)

        max_coverage = max([max(coverage) for coverage in coverage_list])
        min_coverage = min([min(coverage) for coverage in coverage_list])

        for i, state in enumerate(self.hdxms_data.states):
            if row_num == 1:
                ax = axes
            else:
                ax = axes.flatten()[i]
            im = ax.imshow(
                coverage_list[i].reshape(1, -1),
                aspect="auto",
                cmap="Blues",
                extent=[0, len(self.hdxms_data.protein_sequence), 0, 1],
                vmin=min_coverage,
                vmax=max_coverage,
            )
            ax.set_yticks([])
            ax.set_ylabel(state.state_name)

        ax.set_xlabel("Resid")

        cbar_ax = fig.add_axes([0.92, 0.4, 0.02, 0.4])  # [left, bottom, width, height]
        fig.colorbar(im, cax=cbar_ax)

        plt.tight_layout(
            rect=[0, 0, 0.95, 1]
        )  # Adjust the layout to make room for the colorbar
        plt.show()


class UptakePlotsCollection:
    def __init__(self, color_dict=None, if_plot_fit=True, pdb_file=None):
        '''
        A class to store multiple UptakePlot objects

        :param color_dict: a dictionary of colors for each state, e.g. {'state1': 'red', 'state2': 'blue'}
        :param if_plot_fit: if True, plot the fit line
        '''
        # self.hdxms_datas = hdxms_datas
        self.plots = []
        self.color_dict = color_dict
        self.if_plot_fit = if_plot_fit
        self.pdb_file = pdb_file
        self.high_std_plots = []

    def add_plot(self, hdxms_datas, idf: str, uptake_std_threshold=0.5):
        'add a UptakePlot of a peptide to the collection'
        plot = UptakePlot(
            hdxms_datas, idf, if_plot_fit=self.if_plot_fit, color_dict=self.color_dict
        )
        if plot.uptake_std > uptake_std_threshold:
            print(f"{idf} not added due to high uptake std: {plot.uptake_std}")
            self.high_std_plots.append(plot)
        else:
            self.plots.append(plot)

    def add_plot_all(self, hdxms_datas):
        'add UptakePlot objects of all peptides to the collection'
        def get_unique_idfs(hdxms_datas):
            idfs = []
            for hdxms_data in hdxms_datas:
                for state in hdxms_data.states:
                    for peptide in state.peptides:
                        idfs.append(peptide.identifier)
            idfs = list(set(idfs))
            idfs.sort(key=lambda x: int(re.search(r"-?\d+", x).group(0)))
            return idfs

        unique_idfs = get_unique_idfs(hdxms_datas)
        for idf in unique_idfs:
            # (idf)
            self.add_plot(hdxms_datas, idf)

    def save_plots(self, path):
        'save all plots to a folder'
        folder_name = os.path.join(path, "uptake_plots")
        if not os.path.exists(folder_name):
            os.mkdir(folder_name)
        for plot in self.plots:
            fig = plot.uptakeplot
            fig.savefig(f"{folder_name}/{plot.identifier}.png", bbox_inches="tight")


def create_heatmap_compare(compare, colorbar_max, colormap="RdBu"):
    import matplotlib.colors as col
    from matplotlib.patches import Rectangle
    from matplotlib import cm

    font = {"family": "Arial", "weight": "normal", "size": 14}
    axes = {"titlesize": 18, "titleweight": "bold", "labelsize": 16}

    plt.rc("font", **font)
    plt.rc("axes", **axes)

    plt.rcParams["figure.figsize"] = (4, 25)
    plt.rcParams["font.size"] = 14
    plt.rcParams["font.family"] = "Arial"
    colormap = cm.get_cmap(colormap)

    fig, ax = plt.subplots(figsize=(20, 10))

    leftbound = compare.peptide_compares[0].peptide1_list[0].start - 10
    rightbound = compare.peptide_compares[-1].peptide1_list[0].end + 10
    ax.set_xlim(leftbound, rightbound)
    ax.xaxis.set_ticks(np.arange(round(leftbound, -1), round(rightbound, -1), 10))
    ax.set_ylim(-5, 110)
    ax.grid(axis="x")
    ax.yaxis.set_ticks([])

    # sns.heatmap(compare_df, cmap="RdBu", linewidths=1, vmin=-colorbar_max, vmax=colorbar_max, ax=ax)
    norm = col.Normalize(vmin=-colorbar_max, vmax=colorbar_max)

    for i, peptide_compare in enumerate(compare.peptide_compares):
        for peptide in peptide_compare.peptide1_list:
            rect = Rectangle(
                (peptide.start, (i % 20) * 5 + ((i // 20) % 2) * 2.5),
                peptide.end - peptide.start,
                4,
                fc=colormap(norm(peptide_compare.deut_diff_avg)),
            )
            ax.add_patch(rect)

    fig.colorbar(cm.ScalarMappable(cmap=colormap, norm=norm), ax=plt.gca())

    ax.set_title(
        compare.state1_list[0].state_name + "-" + compare.state2_list[0].state_name
    )
    fig.tight_layout()
    plt.close()

    return fig


def create_heatmap_single_state(hdxms_datas, colorbar_max, colormap="Greens"):
    import matplotlib.colors as col
    from matplotlib.patches import Rectangle
    from matplotlib import colormaps
    import matplotlib.patches as patches

    state_name = list(
        set([state.state_name for data in hdxms_datas for state in data.states])
    )
    if len(state_name) > 1:
        raise ValueError("More than one state name found")
    else:
        state_name = state_name[0]

    font = {"family": "Arial", "weight": "normal", "size": 14}
    axes = {"titlesize": 18, "titleweight": "bold", "labelsize": 16}

    plt.rc("font", **font)
    plt.rc("axes", **axes)

    plt.rcParams["figure.figsize"] = (4, 25)
    plt.rcParams["font.size"] = 14
    plt.rcParams["font.family"] = "Arial"

    # colormap = colormaps.get_cmap(colormap)
    colormap = sns.light_palette("seagreen", as_cmap=True)

    fig, ax = plt.subplots(1, 1, figsize=(20, 10))

    all_peptides = [
        pep for data in hdxms_datas for state in data.states for pep in state.peptides
    ]
    all_peptides.sort(key=lambda x: x.start)

    leftbound = all_peptides[0].start - 10
    rightbound = all_peptides[-1].end + 10
    ax.set_xlim(leftbound, rightbound)
    ax.xaxis.set_ticks(np.arange(round(leftbound, -1), round(rightbound, -1), 10))
    ax.set_ylim(-5, 110)
    ax.grid(axis="x")
    ax.yaxis.set_ticks([])

    # sns.heatmap(compare_df, cmap="RdBu", linewidths=1, vmin=-colorbar_max, vmax=colorbar_max, ax=ax)
    norm = col.Normalize(vmin=0, vmax=colorbar_max)

    for i, peptide in enumerate(all_peptides):
        avg_d_percent = np.average(
            [tp.d_percent for tp in peptide.timepoints if tp.deut_time != np.inf]
        )
        rect = Rectangle(
            (peptide.start, (i % 20) * 5 + ((i // 20) % 2) * 2.5),
            peptide.end - peptide.start,
            4,
            fc=colormap(norm(avg_d_percent)),
        )
        ax.add_patch(rect)

    #
    # coverage
    coverage = np.zeros(len(hdxms_datas[0].states[0].hdxms_data.protein_sequence))
    for pep in all_peptides:
        coverage[pep.start - 1 : pep.end] += 1
    height = 3
    for i in range(len(coverage)):
        color_intensity = (
            coverage[i] / 20
        )  # coverage.max()  # Normalizing the data for color intensity
        rect = patches.Rectangle(
            (i, 105), 1, height, color=plt.cm.Blues(color_intensity)
        )
        ax.add_patch(rect)

    fig.colorbar(cm.ScalarMappable(cmap=colormap, norm=norm), ax=plt.gca())

    ax.set_title(state_name)
    fig.tight_layout()
    plt.close()

    return fig


from matplotlib import cm


# Create a function to create a heatmap
def create_heatmap_compare_tp(compare, colorbar_max, colormap="RdBu"):
    font = {"family": "Arial", "weight": "normal", "size": 14}
    axes = {"titlesize": 18, "titleweight": "bold", "labelsize": 16}

    plt.rc("font", **font)
    plt.rc("axes", **axes)

    plt.rcParams["figure.figsize"] = (10, 25)
    plt.rcParams["font.size"] = 14
    plt.rcParams["font.family"] = "Arial"

    fig, ax = plt.subplots()

    df = pd.DataFrame()
    for pep_compare in compare.peptide_compares:
        pep_dict = {}
        pep_dict["title"] = pep_compare.compare_info.split(": ")[1]
        for tp, uptate_v in zip(pep_compare.common_timepoints, pep_compare.deut_diff):
            if tp != np.inf:
                pep_dict[int(tp)] = uptate_v
        df = pd.concat([df, pd.DataFrame(pep_dict, index=[0])], ignore_index=True)
    df = df.set_index("title")
    # sort the columns by timepoint
    df = df.reindex(sorted(df.columns), axis=1)
    # sort the rows by sequence
    # import re
    # df['Start'] = df.index.map(lambda x: int(re.search(r"(-?\d+)--?\d+ \w+", x).group(1)))
    # df = df.sort_values(by=['Start']).drop('Start', axis=1)

    ax = sns.heatmap(
        df, cmap=colormap, linewidths=0.75, vmin=-colorbar_max, vmax=colorbar_max
    )
    ax.set_title(
        compare.state1_list[0].state_name + "-" + compare.state2_list[0].state_name
    )
    ax.set_ylabel("")
    fig.tight_layout()
    plt.close()

    return fig


from matplotlib import cm


# Create a function to create a heatmap
def create_heatmap_tp_single_state(hdxms_datas, colorbar_max, colormap="BLues"):
    font = {"family": "Arial", "weight": "normal", "size": 14}
    axes = {"titlesize": 18, "titleweight": "bold", "labelsize": 16}

    plt.rc("font", **font)
    plt.rc("axes", **axes)

    plt.rcParams["figure.figsize"] = (10, 25)
    plt.rcParams["font.size"] = 14
    plt.rcParams["font.family"] = "Arial"

    state_name = list(
        set([state.state_name for data in hdxms_datas for state in data.states])
    )
    if len(state_name) > 1:
        raise ValueError("More than one state name found")
    else:
        state_name = state_name[0]

    fig, ax = plt.subplots()

    all_peptides = [
        pep for data in hdxms_datas for state in data.states for pep in state.peptides
    ]
    all_peptides.sort(key=lambda x: x.start)

    df = pd.DataFrame()
    for peptide in all_peptides:
        pep_dict = {}
        pep_dict["title"] = peptide.identifier
        # for tp, uptate_v in zip(pep_compare.common_timepoints, pep_compare.deut_diff):
        for tp in peptide.timepoints:
            if tp.deut_time != np.inf:
                pep_dict[int(tp.deut_time)] = tp.d_percent
        df = pd.concat([df, pd.DataFrame(pep_dict, index=[0])], ignore_index=True)
    df = df.set_index("title")
    # sort the columns by timepoint
    df = df.reindex(sorted(df.columns), axis=1)
    # sort the rows by sequence
    # import re
    # df['Start'] = df.index.map(lambda x: int(re.search(r"(-?\d+)--?\d+ \w+", x).group(1)))
    # df = df.sort_values(by=['Start']).drop('Start', axis=1)

    colormap = sns.light_palette("seagreen", as_cmap=True)
    ax = sns.heatmap(df, cmap=colormap, linewidths=0.75, vmin=0, vmax=100)
    ax.set_title(state_name)
    ax.set_ylabel("")
    fig.tight_layout()
    plt.close()

    return fig


# Function to make a heatmap with positive and negative deuterium uptake values separated by a dotted line
def create_heatmap_with_dotted_line(compare, colorbar_max, colormap="RdBu"):
    font = {"family": "Arial", "weight": "normal", "size": 14}
    axes = {"titlesize": 18, "titleweight": "bold", "labelsize": 16}

    plt.rc("font", **font)
    plt.rc("axes", **axes)

    plt.rcParams["figure.figsize"] = (4, 25)
    plt.rcParams["font.size"] = 14
    plt.rcParams["font.family"] = "Arial"
    colormap = cm.get_cmap(colormap)

    fig, ax = plt.subplots(figsize=(20, 10))

    leftbound = compare.peptide_compares[0].peptide1_list[0].start - 10
    rightbound = compare.peptide_compares[-1].peptide1_list[0].end + 10
    ax.set_xlim(leftbound, rightbound)
    ax.xaxis.set_ticks(np.arange(round(leftbound, -1), round(rightbound, -1), 10))
    y_min = -105
    y_max = 105
    ax.set_ylim(y_min, y_max)
    y_middle = (y_min + y_max) / 2
    ax.grid(axis="x")
    ax.yaxis.set_ticks([])
    # add a horizontal dotted line that matches with the 0.0 on the colorbar
    ax.axhline(y=y_middle, color="k", linestyle="--", linewidth=1)

    norm = col.Normalize(vmin=-colorbar_max, vmax=colorbar_max)

    fig.colorbar(cm.ScalarMappable(cmap=colormap, norm=norm))

    for i, peptide_compare in enumerate(compare.peptide_compares):
        for peptide in peptide_compare.peptide1_list:
            deut_diff_avg = peptide_compare.deut_diff_avg
            # print(deut_diff_avg)

            if deut_diff_avg > 0:
                y_position = (i % 20) * 5 + ((i // 20) % 2) * 2.5 + y_middle + 2

            else:
                y_position = (
                    y_middle - ((i % 20) * 5 + ((i // 20) % 2) * 2.5) - 5
                )  # Below the line

            rect = Rectangle(
                (peptide.start, y_position),
                peptide.end - peptide.start,
                3,
                fc=colormap(norm(deut_diff_avg)),
            )
            ax.add_patch(rect)

    ax.set_title(
        compare.state1_list[0].state_name + "-" + compare.state2_list[0].state_name
    )
    fig.tight_layout()

    return fig


def create_compare_pymol_plot(
    compares, colorbar_max, colormap="RdBu", pdb_file=None, path=None, save_pdb=False
):
    rgb_df = gen_rgb_df(compares, colorbar_max, colormap)

    from pymol import cmd

    cmd.delete("all")
    cmd.load(pdb_file)
    cmd.color("gray")

    if isinstance(compares, HDXStatePeptideCompares):
        # sort the peptide by length
        rgb_df.sort_values(
            by="title",
            key=lambda x: x.str.split().str.len(),
            ascending=False,
            inplace=True,
        )

        for row in rgb_df.iterrows():
            resi = f"{row[1]['title'].split(' ')[0]}"
            seq_n = f"res_{row[1]['title'].split(' ')[1]}"
            r_v, g_v, b_v = row[1]["r"], row[1]["g"], row[1]["b"]

            cmd.select(seq_n, "resi " + resi + " and polymer.protein")
            cmd.set_color(f"{seq_n}", [r_v, g_v, b_v])
            cmd.color(f"{seq_n}", seq_n)

    elif isinstance(compares, HDXStateResidueCompares):
        for row in rgb_df.iterrows():
            resi = f"{row[1]['title']}"
            cmd.select(f"resi_{resi}", "resi " + resi + " and polymer.protein")
            r_v, g_v, b_v = row[1]["r"], row[1]["g"], row[1]["b"]

            cmd.set_color(f"resi_{resi}", [r_v, g_v, b_v])
            cmd.color(f"resi_{resi}", f"resi_{resi}")
            cmd.delete(f"resi_{resi}")

    cmd.ray(1000, 1000)

    if path is not None:
        if isinstance(compares, HDXStatePeptideCompares):
            full_path = os.path.join(
                path,
                f"{compares.state1_list[0].state_name}-{compares.state2_list[0].state_name}_{colorbar_max}_pepcompare-pm.pse",
            )
        elif isinstance(compares, HDXStateResidueCompares):
            full_path = os.path.join(
                path,
                f"{compares.state1_list[0].state_name}-{compares.state2_list[0].state_name}_{colorbar_max}_rescompare-pm.pse",
            )
        cmd.save(full_path)
    else:
        raise ValueError("Please provide a path to save the pymol session")

    if isinstance(compares, HDXStateResidueCompares):
        if save_pdb:
            pdb_full_path = os.path.join(
                path,
                f"{compares.state1_list[0].state_name}-{compares.state2_list[0].state_name}_rescompare-pm.pdb",
            )
            save_pdb = plot_on_pdb(pdb_file, compares, pdb_full_path)


def plot_on_pdb(pdb, residue_compares, path=None):
    import MDAnalysis

    u = MDAnalysis.Universe(pdb)
    u.add_TopologyAttr("tempfactors")  # add empty attribute for all atoms
    protein = u.select_atoms("protein")  # select protein atoms

    for res_compare in residue_compares.residue_compares:
        res_id = res_compare.resid
        res = protein.select_atoms("resnum {}".format(res_id))
        res.tempfactors = res_compare.deut_diff_avg

    if path is None:
        raise ValueError("Please provide a path to save the pymol session")
    else:
        u.atoms.write(path)


def gen_rgb_df(compare, colorbar_max, colormap="RdBu"):
    colormap = plt.get_cmap(colormap)
    df = pd.DataFrame()

    compare_list = (
        compare.residue_compares
        if isinstance(compare, HDXStateResidueCompares)
        else compare.peptide_compares
    )

    for compare_i in compare_list:
        pep_dict = {}
        pep_dict["title"] = compare_i.compare_info.split(": ")[1]
        s_i = (compare_i.deut_diff_avg + colorbar_max) / (2 * colorbar_max)
        pep_dict["s"] = s_i
        df = pd.concat([df, pd.DataFrame(pep_dict, index=[0])], ignore_index=True)

    df["s"].clip(upper=1, lower=0, inplace=True)
    rgb = colormap(df["s"]) * 255
    df["r"] = rgb[:, 0]
    df["g"] = rgb[:, 1]
    df["b"] = rgb[:, 2]

    return df

import os
import re

import matplotlib.colors as col
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Rectangle
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd
import seaborn as sns

from pigeon_feather.data import *
from pigeon_feather.tools import get_num_d_from_iso


font = {"family": "Arial", "weight": "normal", "size": 36}
axes = {"titlesize": 36, "titleweight": "bold", "labelsize": 36}
plt.rc("font", **font)
plt.rc("axes", **axes)
plt.rc("lines", lw=3)
colors = ["k", "red", "blue", "purple", "gray", "orange", "yellow", "green", "brown"]


class UptakePlot:
    def __init__(
        self,
        hdxms_datas,
        identifier: str,
        states_subset=None,
        color_dict=None,
        if_plot_fit=True,
        figure=None,
        ax=None,
        if_d_percent=False,
        exp_only=False,
        calculate_num_d_from_envelope=True,
        auto_plot=True,
        markers={1: "o", 2: "s", 3: "^", 4: "P", 5: "X", 6: "H", 7: "v"}
    ):
        """
        hdxms_datas: list of class HDXMSData objects

        :param hdxms_datas: list of HDXMSData objects
        :param identifier: peptide identifier
        :param states_subset: list of states to plot
        :param color_dict: dictionary of colors for each state
        :param if_plot_fit: if True, plot the fit line
        :param figure: figure object
        :param ax: axis object

        :ivar hdxms_datas_df: pandas DataFrame of the HDX-MS data of the peptide
        """
        self.calculate_num_d_from_envelope=calculate_num_d_from_envelope
        self.hdxms_datas = hdxms_datas
        self.identifier = identifier
        self.sequence = identifier.split(" ")[1]
        self.states_subset = states_subset
        self.color_dict = self.make_color_dict(color_dict)
        self.if_plot_fit = if_plot_fit
        self.figure = figure
        self.ax = ax
        self.if_d_percent = if_d_percent
        self.exp_only = exp_only
        # self.title = self.make_title()
        # self.title = identifier
        if auto_plot:
            self.uptakeplot = self.make_uptakeplot()
            
        self.markers = markers


    def make_uptakeplot(self):
        "make a uptakeplot for a peptide"
        plt.rcParams["legend.fontsize"] = 22

        if self.figure is None and self.ax is None:
            figure, ax = plt.subplots(1, 1, figsize=(9, 8))

        else:
            figure = self.figure
            ax = self.ax

        sns.scatterplot(
            data=self.hdxms_datas_df[self.hdxms_datas_df["time"] != 0],
            x="time",
            y="deut",
            hue="state",
            style="data_set_index",
            alpha=1.0,
            ax=ax,
            palette=self.color_dict,
            s=600,
            markers=self.markers,
        )


        # Plot the fit
        if self.if_plot_fit:
            for state_name in self.hdxms_datas_df.state.unique():
                avg_peptide = self.get_average_peptide(state_name)
                # trialT, y_pred, popt, perr =  avg_peptide.fit_results
                trialT, y_pred, popt, perr = avg_peptide.new_fit()
                # trialT, y_pred, popt, perr =  avg_peptide.fit_hdx_stats()
                if trialT is None:
                    # raw data, no fit
                    trialT = [
                        tp.deut_time
                        for tp in avg_peptide.timepoints
                        if tp.deut_time != np.inf or tp.deut_time != 0
                    ]
                    y_pred = [
                        #tp.num_d
                        get_num_d_from_iso(tp)/(avg_peptide.max_d / avg_peptide.theo_max_d)
                        for tp in avg_peptide.timepoints
                        if tp.deut_time != np.inf or tp.deut_time != 0
                    ]
                ax.plot(
                    trialT, y_pred, "-", color=self.color_dict[state_name], alpha=0.5
                )
        else:
            sns.lineplot(
            data=self.hdxms_datas_df[self.hdxms_datas_df["time"] != 0],
            x="time",
            y="deut",
            hue="state",
            alpha=0.5,
            marker=None,
            ax=ax,
            palette=self.color_dict,
            estimator="mean",
            legend=False 
            )            


        # set up the plot
        ax.set_ylabel("# Deuterons")
        ax.set_xlabel("Time (seconds)")
        ax.set_xscale("log")
        # avg_peptide = self.get_average_peptide(self.hdxms_datas_df.state.unique()[0])

        ax.set_ylim(
            min(self.hdxms_datas_df["deut"]) - 1, max(self.hdxms_datas_df["deut"]) + 1
        )
        # ax.set_xlim(ax.get_xlim()[0]/10, ax.get_xlim()[1] * 10)
        xlim = ax.get_xlim()
        left_limit = 10 ** np.floor(np.log10(xlim[0]))
        right_limit = 10 ** np.ceil(np.log10(xlim[1]))
        ax.set_xlim(left_limit, right_limit)
        # ax.set_ylim(-0.3, avg_peptide.max_d*1.1)
        # ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.legend()

        ax.set_title(self.identifier)
        plt.subplots_adjust(bottom=0.2)
        plt.close()

        return figure

    @property
    def hdxms_datas_df(self):
        hdxms_datas_df = pd.DataFrame()

        for hdxms_data_index, hdxms_data in enumerate(self.hdxms_datas):
            hdxms_data_df = pd.DataFrame()

            for state in hdxms_data.states:
                if (
                    self.states_subset is not None
                    and state.state_name not in self.states_subset
                ):
                    continue

                peptide = state.get_peptide(self.identifier)

                if peptide is not None:
                    if self.exp_only:
                        timepoint_objs = [
                            tp for tp in peptide.timepoints if tp.note is None
                        ]
                    else:
                        timepoint_objs = [
                            tp for tp in peptide.timepoints if tp.deut_time != np.inf
                        ]

                    deut_list = [get_num_d_from_iso(tp)  if self.calculate_num_d_from_envelope else tp.num_d for tp in timepoint_objs]
                    peptide_df_i = pd.DataFrame(
                        {
                            "time": [tp.deut_time for tp in timepoint_objs],
                            #"deut": [tp.num_d for tp in timepoint_objs],
                            "deut": deut_list,
                            "state": state.state_name,
                            "charge_state": [tp.charge_state for tp in timepoint_objs],
                        }
                    )

                    if self.if_d_percent:
                        peptide_df_i["deut"] = [tp.d_percent for tp in timepoint_objs]

                    hdxms_data_df = pd.concat(
                        [hdxms_data_df, peptide_df_i], ignore_index=True
                    )

            hdxms_data_df["data_set_index"] = hdxms_data_index + 1
            hdxms_datas_df = pd.concat(
                [hdxms_datas_df, hdxms_data_df], ignore_index=True
            )
        return hdxms_datas_df

    @hdxms_datas_df.setter
    def hdxms_datas_df(self, value):
        self._hdxms_datas_df = value

    @property
    def uptake_std(self):
        df = self.hdxms_datas_df[~self.hdxms_datas_df["time"].isin([0, np.inf])]
        deut_std = df.groupby(["state", "time"]).std().reset_index()["deut"].mean()
        return deut_std

    def get_average_peptide(self, state_name):
        "return an averaged peptide for a state"
        grouped_df = self.hdxms_datas_df.groupby("state")
        group = grouped_df.get_group(state_name)

        final_group = group.groupby("time")
        group_mean = final_group.mean(numeric_only=True).reset_index()
        group_std = final_group.std(numeric_only=True).reset_index()

        idf_start, idf_end = re.match(r"(-?\d+)-(-?\d+)", self.identifier).group(1, 2)
        states = [
            state
            for data in self.hdxms_datas
            for state in data.states
            if state.state_name == state_name
        ]

        peptide = Peptide(
            self.sequence,
            int(idf_start),
            int(idf_end),
            # f"averaged peptide: {state_name}",
            states[0],
            n_fastamides=self.hdxms_datas[0].n_fastamides,
        )
        for i in range(len(group_mean)):
            timepoint = Timepoint(
                peptide,
                group_mean["time"][i],
                group_mean["deut"][i],
                group_std["deut"][i],
            )
            peptide.add_timepoint(timepoint)

        return peptide

    def make_title(self):
        "make a title for the plot"
        for hdxms_data in self.hdxms_datas:
            for state in hdxms_data.states:
                try:
                    start = state.get_peptide(self.sequence).start
                    end = state.get_peptide(self.sequence).end
                    return f"{start}-{end} {self.sequence}"
                except AttributeError:
                    pass
        return f"Missing data for {self.sequence}"

    def make_color_dict(self, color_dict=None):
        "make a color dictionary for the states in the plot"
        if color_dict is None:
            colors = [
                "k",
                "red",
                "blue",
                "purple",
                "gray",
                "orange",
                "yellow",
                "green",
                "brown",
            ]

            color_dict = {}
            if self.states_subset is None:
                state_names = list(
                    set(
                        [
                            state.state_name
                            for hdxms_data in self.hdxms_datas
                            for state in hdxms_data.states
                        ]
                    )
                )
                state_names.sort()
            else:
                state_names = self.states_subset

            for i, state_name in enumerate(state_names):
                color_dict[state_name] = colors[i]
        return color_dict



# new residue coverage plotting script heat map thing - sav
class ResidueCoverage:
    def __init__(self, hdxms_data):
        self.hdxms_data = hdxms_data

    def calculate_coverages(self, state_name):
        state = self.hdxms_data.get_state(state_name)
        coverage = np.zeros(len(self.hdxms_data.protein_sequence))
        for pep in state.peptides:
            coverage[pep.start - 1 : pep.end] += 1
        return coverage

    def plot(self):
        row_num = len(self.hdxms_data.states)
        fig, axes = plt.subplots(
            row_num, 1, figsize=(20, 3 * row_num), sharex=True, sharey=True
        )

        coverage_list = []
        for state in self.hdxms_data.states:
            coverage = self.calculate_coverages(state.state_name)
            coverage_list.append(coverage)

        max_coverage = max([max(coverage) for coverage in coverage_list])
        min_coverage = min([min(coverage) for coverage in coverage_list])

        for i, state in enumerate(self.hdxms_data.states):
            if row_num == 1:
                ax = axes
            else:
                ax = axes.flatten()[i]
            im = ax.imshow(
                coverage_list[i].reshape(1, -1),
                aspect="auto",
                cmap="Blues",
                extent=[0, len(self.hdxms_data.protein_sequence), 0, 1],
                vmin=min_coverage,
                vmax=max_coverage,
            )
            ax.set_yticks([])
            ax.set_ylabel(state.state_name)

        ax.set_xlabel("Resid")

        cbar_ax = fig.add_axes([0.92, 0.4, 0.02, 0.4])  # [left, bottom, width, height]
        fig.colorbar(im, cax=cbar_ax)

        plt.tight_layout(
            rect=[0, 0, 0.95, 1]
        )  # Adjust the layout to make room for the colorbar
        plt.show()


class UptakePlotsCollection:
    def __init__(self, color_dict=None, if_plot_fit=True, pdb_file=None):
        '''
        A class to store multiple UptakePlot objects

        :param color_dict: a dictionary of colors for each state, e.g. {'state1': 'red', 'state2': 'blue'}
        :param if_plot_fit: if True, plot the fit line
        '''
        # self.hdxms_datas = hdxms_datas
        self.plots = []
        self.color_dict = color_dict
        self.if_plot_fit = if_plot_fit
        self.pdb_file = pdb_file
        self.high_std_plots = []

    def add_plot(self, hdxms_datas, idf: str, uptake_std_threshold=0.5):
        'add a UptakePlot of a peptide to the collection'
        plot = UptakePlot(
            hdxms_datas, idf, if_plot_fit=self.if_plot_fit, color_dict=self.color_dict
        )
        if plot.uptake_std > uptake_std_threshold:
            print(f"{idf} not added due to high uptake std: {plot.uptake_std}")
            self.high_std_plots.append(plot)
        else:
            self.plots.append(plot)

    def add_plot_all(self, hdxms_datas):
        'add UptakePlot objects of all peptides to the collection'
        def get_unique_idfs(hdxms_datas):
            idfs = []
            for hdxms_data in hdxms_datas:
                for state in hdxms_data.states:
                    for peptide in state.peptides:
                        idfs.append(peptide.identifier)
            idfs = list(set(idfs))
            idfs.sort(key=lambda x: int(re.search(r"-?\d+", x).group(0)))
            return idfs

        unique_idfs = get_unique_idfs(hdxms_datas)
        for idf in unique_idfs:
            # (idf)
            self.add_plot(hdxms_datas, idf)

    def save_plots(self, path):
        'save all plots to a folder'
        folder_name = os.path.join(path, "uptake_plots")
        if not os.path.exists(folder_name):
            os.mkdir(folder_name)
        for plot in self.plots:
            fig = plot.uptakeplot
            fig.savefig(f"{folder_name}/{plot.identifier}.png", bbox_inches="tight")


def create_heatmap_compare(compare, colorbar_max, colormap="RdBu"):
    import matplotlib.colors as col
    from matplotlib.patches import Rectangle
    from matplotlib import cm

    font = {"family": "Arial", "weight": "normal", "size": 14}
    axes = {"titlesize": 18, "titleweight": "bold", "labelsize": 16}

    plt.rc("font", **font)
    plt.rc("axes", **axes)

    plt.rcParams["figure.figsize"] = (4, 25)
    plt.rcParams["font.size"] = 14
    plt.rcParams["font.family"] = "Arial"
    colormap = cm.get_cmap(colormap)

    fig, ax = plt.subplots(figsize=(20, 10))

    leftbound = compare.peptide_compares[0].peptide1_list[0].start - 10
    rightbound = compare.peptide_compares[-1].peptide1_list[0].end + 10
    ax.set_xlim(leftbound, rightbound)
    ax.xaxis.set_ticks(np.arange(round(leftbound, -1), round(rightbound, -1), 10))
    ax.set_ylim(-5, 110)
    ax.grid(axis="x")
    ax.yaxis.set_ticks([])

    # sns.heatmap(compare_df, cmap="RdBu", linewidths=1, vmin=-colorbar_max, vmax=colorbar_max, ax=ax)
    norm = col.Normalize(vmin=-colorbar_max, vmax=colorbar_max)

    for i, peptide_compare in enumerate(compare.peptide_compares):
        #for peptide in peptide_compare.peptide1_list:
        y_position =  i%100
        rect = Rectangle(
            (peptide_compare.start, (i % 20) * 5 + ((i // 20) % 2) * 2.5),
            #(peptide_compare.start, (i % 50) * 2 + ((i // 50) % 2) * 2.5),
            # (peptide_compare.start, y_position),
            peptide_compare.end - peptide_compare.start,
            3,
            fc=colormap(norm(peptide_compare.deut_diff_avg)),
            edgecolor='gray',
            linewidth=0.35
        )
        ax.add_patch(rect)

            
    # compares = sorted(compare.peptide_compares, key=lambda x: len(x.sequence), reverse=True)
    # for peptide_compare in compares:
    #     rect = Rectangle(
    #         # (peptide.start, (i % 20) * 5 + ((i // 20) % 2) * 2.5),
    #         (peptide_compare.start, -8),
    #         # (peptide.start, i%100),
    #         peptide_compare.end - peptide_compare.start,
    #         4,
    #         fc=colormap(norm(peptide_compare.deut_diff_avg)),
    #        alpha=0.7
    #     )
    #     ax.add_patch(rect)

    fig.colorbar(cm.ScalarMappable(cmap=colormap, norm=norm), ax=plt.gca())

    ax.set_title(
        compare.state1_list[0].state_name + "-" + compare.state2_list[0].state_name
    )
    fig.tight_layout()
    plt.close()

    return fig


def create_heatmap_single_state(hdxms_datas, colorbar_max, colormap="Greens"):
    import matplotlib.colors as col
    from matplotlib.patches import Rectangle
    from matplotlib import colormaps
    import matplotlib.patches as patches

    state_name = list(
        set([state.state_name for data in hdxms_datas for state in data.states])
    )
    if len(state_name) > 1:
        raise ValueError("More than one state name found")
    else:
        state_name = state_name[0]

    font = {"family": "Arial", "weight": "normal", "size": 14}
    axes = {"titlesize": 18, "titleweight": "bold", "labelsize": 16}

    plt.rc("font", **font)
    plt.rc("axes", **axes)

    plt.rcParams["figure.figsize"] = (4, 25)
    plt.rcParams["font.size"] = 14
    plt.rcParams["font.family"] = "Arial"

    # colormap = colormaps.get_cmap(colormap)
    colormap = sns.light_palette("seagreen", as_cmap=True)

    fig, ax = plt.subplots(1, 1, figsize=(20, 10))

    all_peptides = [
        pep for data in hdxms_datas for state in data.states for pep in state.peptides
    ]
    all_peptides.sort(key=lambda x: x.start)

    leftbound = all_peptides[0].start - 10
    rightbound = all_peptides[-1].end + 10
    ax.set_xlim(leftbound, rightbound)
    ax.xaxis.set_ticks(np.arange(round(leftbound, -1), round(rightbound, -1), 10))
    ax.set_ylim(-5, 110)
    ax.grid(axis="x")
    ax.yaxis.set_ticks([])

    # sns.heatmap(compare_df, cmap="RdBu", linewidths=1, vmin=-colorbar_max, vmax=colorbar_max, ax=ax)
    norm = col.Normalize(vmin=0, vmax=colorbar_max)

    for i, peptide in enumerate(all_peptides):
        avg_d_percent = np.average(
            [tp.d_percent for tp in peptide.timepoints if tp.deut_time != np.inf]
        )
        rect = Rectangle(
            (peptide.start, (i % 20) * 5 + ((i // 20) % 2) * 2.5),
            peptide.end - peptide.start,
            4,
            fc=colormap(norm(avg_d_percent)),
        )
        ax.add_patch(rect)

    #
    # coverage
    coverage = np.zeros(len(hdxms_datas[0].states[0].hdxms_data.protein_sequence))
    for pep in all_peptides:
        coverage[pep.start - 1 : pep.end] += 1
    height = 3
    for i in range(len(coverage)):
        color_intensity = (
            coverage[i] / 20
        )  # coverage.max()  # Normalizing the data for color intensity
        rect = patches.Rectangle(
            (i, 105), 1, height, color=plt.cm.Blues(color_intensity)
        )
        ax.add_patch(rect)

    fig.colorbar(cm.ScalarMappable(cmap=colormap, norm=norm), ax=plt.gca())

    ax.set_title(state_name)
    fig.tight_layout()
    plt.close()

    return fig


from matplotlib import cm


# Create a function to create a heatmap
def create_heatmap_compare_tp(compare, colorbar_max, colormap="RdBu"):
    font = {"family": "Arial", "weight": "normal", "size": 14}
    axes = {"titlesize": 18, "titleweight": "bold", "labelsize": 16}

    plt.rc("font", **font)
    plt.rc("axes", **axes)

    plt.rcParams["figure.figsize"] = (10, 25)
    plt.rcParams["font.size"] = 14
    plt.rcParams["font.family"] = "Arial"

    fig, ax = plt.subplots()

    df = pd.DataFrame()
    for pep_compare in compare.peptide_compares:
        pep_dict = {}
        pep_dict["title"] = pep_compare.compare_info.split(": ")[1]
        for tp, uptate_v in zip(pep_compare.common_timepoints, pep_compare.deut_diff):
            if tp != np.inf:
                pep_dict[int(tp)] = uptate_v
        df = pd.concat([df, pd.DataFrame(pep_dict, index=[0])], ignore_index=True)
    df = df.set_index("title")
    # sort the columns by timepoint
    df = df.reindex(sorted(df.columns), axis=1)
    # sort the rows by sequence
    # import re
    # df['Start'] = df.index.map(lambda x: int(re.search(r"(-?\d+)--?\d+ \w+", x).group(1)))
    # df = df.sort_values(by=['Start']).drop('Start', axis=1)

    ax = sns.heatmap(
        df, cmap=colormap, linewidths=0.75, vmin=-colorbar_max, vmax=colorbar_max
    )
    ax.set_title(
        compare.state1_list[0].state_name + "-" + compare.state2_list[0].state_name
    )
    ax.set_ylabel("")
    fig.tight_layout()
    plt.close()

    return fig


from matplotlib import cm


# Create a function to create a heatmap
def create_heatmap_tp_single_state(hdxms_datas, colorbar_max, colormap="BLues"):
    font = {"family": "Arial", "weight": "normal", "size": 14}
    axes = {"titlesize": 18, "titleweight": "bold", "labelsize": 16}

    plt.rc("font", **font)
    plt.rc("axes", **axes)

    plt.rcParams["figure.figsize"] = (10, 25)
    plt.rcParams["font.size"] = 14
    plt.rcParams["font.family"] = "Arial"

    state_name = list(
        set([state.state_name for data in hdxms_datas for state in data.states])
    )
    if len(state_name) > 1:
        raise ValueError("More than one state name found")
    else:
        state_name = state_name[0]

    fig, ax = plt.subplots()

    all_peptides = [
        pep for data in hdxms_datas for state in data.states for pep in state.peptides
    ]
    all_peptides.sort(key=lambda x: x.start)

    df = pd.DataFrame()
    for peptide in all_peptides:
        pep_dict = {}
        pep_dict["title"] = peptide.identifier
        # for tp, uptate_v in zip(pep_compare.common_timepoints, pep_compare.deut_diff):
        for tp in peptide.timepoints:
            if tp.deut_time != np.inf:
                pep_dict[int(tp.deut_time)] = tp.d_percent
        df = pd.concat([df, pd.DataFrame(pep_dict, index=[0])], ignore_index=True)
    df = df.set_index("title")
    # sort the columns by timepoint
    df = df.reindex(sorted(df.columns), axis=1)
    # sort the rows by sequence
    # import re
    # df['Start'] = df.index.map(lambda x: int(re.search(r"(-?\d+)--?\d+ \w+", x).group(1)))
    # df = df.sort_values(by=['Start']).drop('Start', axis=1)

    colormap = sns.light_palette("seagreen", as_cmap=True)
    ax = sns.heatmap(df, cmap=colormap, linewidths=0.75, vmin=0, vmax=100)
    ax.set_title(state_name)
    ax.set_ylabel("")
    fig.tight_layout()
    plt.close()

    return fig


# Function to make a heatmap with positive and negative deuterium uptake values separated by a dotted line
def create_heatmap_with_dotted_line(compare, colorbar_max, colormap="RdBu"):
    font = {"family": "Arial", "weight": "normal", "size": 14}
    axes = {"titlesize": 18, "titleweight": "bold", "labelsize": 16}

    plt.rc("font", **font)
    plt.rc("axes", **axes)

    plt.rcParams["figure.figsize"] = (4, 25)
    plt.rcParams["font.size"] = 14
    plt.rcParams["font.family"] = "Arial"
    colormap = cm.get_cmap(colormap)

    fig, ax = plt.subplots(figsize=(20, 10))

    leftbound = compare.peptide_compares[0].peptide1_list[0].start - 10
    rightbound = compare.peptide_compares[-1].peptide1_list[0].end + 10
    ax.set_xlim(leftbound, rightbound)
    ax.xaxis.set_ticks(np.arange(round(leftbound, -1), round(rightbound, -1), 10))
    y_min = -105
    y_max = 105
    ax.set_ylim(y_min, y_max)
    y_middle = (y_min + y_max) / 2
    ax.grid(axis="x")
    ax.yaxis.set_ticks([])
    # add a horizontal dotted line that matches with the 0.0 on the colorbar
    ax.axhline(y=y_middle, color="k", linestyle="--", linewidth=1)

    norm = col.Normalize(vmin=-colorbar_max, vmax=colorbar_max)

    fig.colorbar(cm.ScalarMappable(cmap=colormap, norm=norm))

    for i, peptide_compare in enumerate(compare.peptide_compares):
        for peptide in peptide_compare.peptide1_list:
            deut_diff_avg = peptide_compare.deut_diff_avg
            # print(deut_diff_avg)

            if deut_diff_avg > 0:
                y_position = (i % 20) * 5 + ((i // 20) % 2) * 2.5 + y_middle + 2

            else:
                y_position = (
                    y_middle - ((i % 20) * 5 + ((i // 20) % 2) * 2.5) - 5
                )  # Below the line

            rect = Rectangle(
                (peptide.start, y_position),
                peptide.end - peptide.start,
                3,
                fc=colormap(norm(deut_diff_avg)),
            )
            ax.add_patch(rect)

    ax.set_title(
        compare.state1_list[0].state_name + "-" + compare.state2_list[0].state_name
    )
    fig.tight_layout()

    return fig


def create_compare_pymol_plot(
    compares, colorbar_max, colormap="RdBu", pdb_file=None, path=None, save_pdb=False
):
    rgb_df = gen_rgb_df(compares, colorbar_max, colormap)

    from pymol import cmd

    cmd.delete("all")
    cmd.load(pdb_file)
    cmd.color("gray")

    if isinstance(compares, HDXStatePeptideCompares):
        # sort the peptide by length
        rgb_df.sort_values(
            by="title",
            key=lambda x: x.str.split().str.len(),
            ascending=False,
            inplace=True,
        )

        for row in rgb_df.iterrows():
            resi = f"{row[1]['title'].split(' ')[0]}"
            seq_n = f"res_{row[1]['title'].split(' ')[1]}"
            r_v, g_v, b_v = row[1]["r"], row[1]["g"], row[1]["b"]

            cmd.select(seq_n, "resi " + resi + " and polymer.protein")
            cmd.set_color(f"{seq_n}", [r_v, g_v, b_v])
            cmd.color(f"{seq_n}", seq_n)

    elif isinstance(compares, HDXStateResidueCompares):
        for row in rgb_df.iterrows():
            resi = f"{row[1]['title']}"
            cmd.select(f"resi_{resi}", "resi " + resi + " and polymer.protein")
            r_v, g_v, b_v = row[1]["r"], row[1]["g"], row[1]["b"]

            cmd.set_color(f"resi_{resi}", [r_v, g_v, b_v])
            cmd.color(f"resi_{resi}", f"resi_{resi}")
            cmd.delete(f"resi_{resi}")

    cmd.ray(1000, 1000)

    if path is not None:
        if isinstance(compares, HDXStatePeptideCompares):
            full_path = os.path.join(
                path,
                f"{compares.state1_list[0].state_name}-{compares.state2_list[0].state_name}_{colorbar_max}_pepcompare-pm.pse",
            )
        elif isinstance(compares, HDXStateResidueCompares):
            full_path = os.path.join(
                path,
                f"{compares.state1_list[0].state_name}-{compares.state2_list[0].state_name}_{colorbar_max}_rescompare-pm.pse",
            )
        cmd.save(full_path)
    else:
        raise ValueError("Please provide a path to save the pymol session")

    if isinstance(compares, HDXStateResidueCompares):
        if save_pdb:
            pdb_full_path = os.path.join(
                path,
                f"{compares.state1_list[0].state_name}-{compares.state2_list[0].state_name}_rescompare-pm.pdb",
            )
            save_pdb = plot_on_pdb(pdb_file, compares, pdb_full_path)


def plot_on_pdb(pdb, residue_compares, path=None):
    import MDAnalysis

    u = MDAnalysis.Universe(pdb)
    u.add_TopologyAttr("tempfactors")  # add empty attribute for all atoms
    protein = u.select_atoms("protein")  # select protein atoms

    for res_compare in residue_compares.residue_compares:
        res_id = res_compare.resid
        res = protein.select_atoms("resnum {}".format(res_id))
        res.tempfactors = res_compare.deut_diff_avg

    if path is None:
        raise ValueError("Please provide a path to save the pymol session")
    else:
        u.atoms.write(path)


def gen_rgb_df(compare, colorbar_max, colormap="RdBu"):
    colormap = plt.get_cmap(colormap)
    df = pd.DataFrame()

    compare_list = (
        compare.residue_compares
        if isinstance(compare, HDXStateResidueCompares)
        else compare.peptide_compares
    )

    for compare_i in compare_list:
        pep_dict = {}
        pep_dict["title"] = compare_i.compare_info.split(": ")[1]
        s_i = (compare_i.deut_diff_avg + colorbar_max) / (2 * colorbar_max)
        pep_dict["s"] = s_i
        df = pd.concat([df, pd.DataFrame(pep_dict, index=[0])], ignore_index=True)

    df["s"].clip(upper=1, lower=0, inplace=True)
    rgb = colormap(df["s"]) * 255
    df["r"] = rgb[:, 0]
    df["g"] = rgb[:, 1]
    df["b"] = rgb[:, 2]

    return df
