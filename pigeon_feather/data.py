import itertools
import math
import random
import re
import warnings
import time

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from pigeon_feather.tools import (
    average_timepoints,
    calculate_simple_deuterium_incorporation,
    exchange_fit,
    exchange_fit_low,
    event_probabilities,
    find_overlapped_peptides,
    find_peptide,
    fit_func,
    pdb2seq,
    subtract_peptides,
    _add_max_d_to_pep,
    normlize
)
import pigeon_feather.spectra as spectra
from pigeon_feather.hxio import (
    convert_dataframe_to_bayesianhdx_format,
    revert_hdxmsdata_to_dataframe,
)
from hdxrate import k_int_from_sequence
import uuid


class RangeList:
    def __init__(self, range_list_file=None, range_df=None):
        '''
        A class to handle a list of peptides.

        :param range_list_file: csv file with Start, End columns
        :param range_df: a dataframe
        
        :ivar range_list: a dataframe with Start, End columns
        :ivar range_set: a set of tuples (Start, End)
        '''
        
        if range_list_file is not None:
            df = pd.read_csv(range_list_file)
            if 'Start' in df.columns and 'End' in df.columns:
                self.range_list = df[["Start", "End"]]
            else:

                df = pd.read_csv(range_list_file, skiprows=1)
                self.range_list = (
                    df[["Start", "End"]]
                    .drop_duplicates()
                    .sort_values(by="Start")
                    .reset_index(drop=True)
                )
        else:
            self.range_list = range_df

    def to_set(self):
        'convert the range list to a set of tuples (Start, End)'
        return set(tuple(row) for _, row in self.range_list.iterrows())

    def to_dataframe(self):
        'return the range list as a dataframe'
        return self.range_list

    def to_csv(self, path):
        'save the range list to a csv file'
        self.range_list.to_csv(path, index=False)

    def union(self, other):
        'return the union of two range lists'
        if isinstance(other, RangeList):
            other_set = other.to_set()
        elif isinstance(other, set):
            other_set = other
        else:
            print("Invalid input. Please provide a RangeList object or a set.")
            return None

        result_set = self.to_set().union(other_set)
        result_df = pd.DataFrame(list(result_set), columns=["Start", "End"])
        result_df = result_df.astype(int)
        result_df = result_df.sort_values(by="Start").reset_index(drop=True)
        return RangeList(range_df=result_df)

    def intersection(self, other):
        'return the intersection of two range lists'
        if isinstance(other, RangeList):
            other_set = other.to_set()
        elif isinstance(other, set):
            other_set = other
        else:
            print("Invalid input. Please provide a RangeList object or a set.")
            return None

        result_set = self.to_set().intersection(other_set)
        result_df = pd.DataFrame(list(result_set), columns=["Start", "End"])
        result_df = result_df.astype(int)
        result_df = result_df.sort_values(by="Start").reset_index(drop=True)
        return RangeList(range_df=result_df)

    def difference(self, other):
        'return the difference of two range lists'
        if isinstance(other, RangeList):
            other_set = other.to_set()
        elif isinstance(other, set):
            other_set = other
        else:
            print("Invalid input. Please provide a RangeList object or a set.")
            return None

        result_set = self.to_set().difference(other_set)
        result_df = pd.DataFrame(list(result_set), columns=["Start", "End"])
        result_df = result_df.astype(int)
        result_df = result_df.sort_values(by="Start").reset_index(drop=True)
        return RangeList(range_df=result_df)


class HDXMSDataCollection:
    def __init__(self, hdxms_data_list):
        '''
        A cloolection of HDXMSData objects

        :param hdxms_data_list: a list of HDXMSData objects
        '''
        self.hdxms_data_list = hdxms_data_list


class HDXMSData:
    def __init__(
        self, protein_name, n_fastamides=2, protein_sequence=None, saturation=None, pH=None, temperature=None
    ):
        '''
        A class to store one HDX-MS replicate data. It can contain multiple states.

        :param protein_name: string, name of the protein
        :param n_fastamides: int, defaults to 2
        :param protein_sequence: string
        :param saturation: D saturation, defaults to 1
        
        :ivar states: a list of ProteinState objects
        '''
        self.protein_name = protein_name
        self.states = []
        self.n_fastamides = n_fastamides
        if protein_sequence is None or pH is None or temperature is None:
            raise ValueError("Protein sequence, pH, temperature and saturation are required!")
        self.protein_sequence = protein_sequence
        self.saturation = saturation
        self.pH = pH
        self.temperature = temperature

    def add_state(self, state):
        'add a state to the HDXMSData object'
        # Check if state already exists
        for existing_state in self.states:
            if existing_state.state_name == state.state_name:
                raise ValueError("State already exists")
        self.states.append(state)
        return state

    def load_protein_sequence(self, sequence):
        'load protein sequence to the HDXMSData object'
        self.protein_sequence = sequence

    @property
    def num_states(self):
        'return the number of states in the HDXMSData object'
        return len(self.states)

    def get_state(self, state_name):
        'return a state by name'
        for state in self.states:
            if state.state_name == state_name:
                return state
        return None

    def plot_res_coverage(self):
        'plot residue coverage of the protein'
        from pigeon_feather.plotting import ResidueCoverage

        res_cov = ResidueCoverage(self)
        return res_cov.plot()

    def reindex_peptide_from_pdb(self, pdb_file, first_residue_index=1):
        '''
        reindex peptides based on the pdb file, aften there is a index offset.

        :param pdb_file: path to the pdb file
        :param first_residue_index: the first residue index in the pdb file
        '''
        pdb_sequence = pdb2seq(pdb_file)
        a_middle_pep = self.states[0].peptides[int(len(self.states[0].peptides) / 2)]
        pdb_start, pdb_end = find_peptide(pdb_sequence, a_middle_pep.sequence)
        index_offset = a_middle_pep.start - pdb_start - first_residue_index

        for state in self.states:
            for peptide in state.peptides:
                peptide.start -= index_offset
                peptide.end -= index_offset

                # identifier
                old_idf = peptide.identifier
                idf_start, idf_end = re.match(r"(-?\d+)-(-?\d+)", old_idf).group(1, 2)
                idf_seq = old_idf.split(" ")[1]
                new_idf = f"{int(idf_start) - index_offset}-{int(idf_end) - index_offset} {idf_seq}"
                peptide.identifier = new_idf

        print(f"Peptide reindexed with offset {-1*index_offset}")

    def to_dataframe(self, if_percent=False):
        'convert the HDXMSData object to a dataframe'
        return revert_hdxmsdata_to_dataframe(self, if_percent=if_percent)

    def to_bayesianhdx_format(self, OUTPATH=None):
        'convert the HDXMSData object to BayesianHDX format and save to a file'
        convert_dataframe_to_bayesianhdx_format(
            self.to_dataframe(), self.protein_name, OUTPATH
        )

    def _drop_peptides(self, drop_list):
        '''
        drop peptides from the HDXMSData object

        :param drop_list: a lisf Peptide objects
        '''
        for state in self.states:
            for peptide in drop_list:
                state.peptides.remove(peptide)


class ProteinState:
    def __init__(self, state_name, hdxms_data=None):
        '''
        A class to store one state of a protein. It can contain multiple peptides.

        :param state_name: name of the state
        :param hdxms_data: HDXMSData object
        
        :ivar peptides: a list of Peptide objects
        :ivar if_subtracted: if subtracted peptides have been added
        :ivar num_subtracted_added: number of subtracted peptides added
        '''
        self.peptides = []
        self.state_name = state_name
        self.if_subtracted = False
        self.num_subtracted_added = 0
        self.hdxms_data = hdxms_data

    def add_peptide(self, peptide, allow_duplicate=False):
        'add a peptide to the ProteinState object'
        # Check if peptide already exists
        if not allow_duplicate:
            for existing_peptide in self.peptides:
                if existing_peptide.identifier == peptide.identifier:
                    raise ValueError(f"{ peptide.identifier} Peptide already exists")
        self.peptides.append(peptide)
        return peptide

    @property
    def num_peptides(self):
        'return the number of peptides in the ProteinState'
        return len(self.peptides)

    def get_peptide(self, identifier):
        'return a peptide by identifier'
        for peptide in self.peptides:
            if peptide.identifier == identifier:
                return peptide
        return None

    def add_new_peptides_by_subtract(self):
        """
        add new peptides to the protein state by subtracting the overlapped peptides
        """

        # check the input
        if not isinstance(self, ProteinState):
            raise TypeError("The input should be a protein state of hdxms_data object.")

        subgroups = find_overlapped_peptides(self)

        new_peptide_added = []
        for key in sorted(subgroups.keys()):
            subgroup = sorted(subgroups[key], key=lambda x: x.start)
            if len(subgroup) >= 2:
                # create all possible pairs of items
                pairs = list(itertools.combinations([i for i in subgroup], 2))

                for pair in pairs:

                    # skip if subtraction has been done
                    note1 = f"Subtraction: {pair[0].identifier} - {pair[1].identifier}"
                    note2 = f"Subtraction: {pair[1].identifier} - {pair[0].identifier}"
                    notes = [pep.note for pep in self.peptides if pep.note is not None]
                    
                    if note1 in notes or note2 in notes:
                        continue
                    
                    try:
                        new_peptide = subtract_peptides(pair[0], pair[1])
                    except:
                        continue

                    # skil if a None peptide is returned
                    if new_peptide is None:
                        continue

                    # skip if new_peptide has high recaculated num_d error
                    if hasattr(new_peptide.timepoints[0], "isotope_envelope"):
                        from pigeon_feather.tools import get_num_d_from_iso

                        avg_error = np.abs(
                            np.average(
                                [
                                    tp.num_d - get_num_d_from_iso(tp)
                                    for tp in new_peptide.timepoints
                                    if tp.deut_time != np.inf
                                ]
                            )
                        )
                        if avg_error > 0.15:
                            continue

                    # add the new peptide to the protein state object if not exists
                    existing_peptide = self.get_peptide(new_peptide.identifier)
                    
                    if existing_peptide is None:
                        self.add_peptide(new_peptide)
                        new_peptide_added.append(new_peptide)

                    # if exists, add the timepoints to the existing peptide
                    else:
                        for tp in new_peptide.timepoints:
                            tp.peptide = existing_peptide
                            existing_peptide.timepoints.append(tp)

        print(f"{len(new_peptide_added)} new peptides added to the {self.state_name} state.")

        self.subtracted_peptides = new_peptide_added
        self.num_subtracted_added += len(new_peptide_added)
        return new_peptide_added

    def add_all_subtract(self):
        'add all possible subtracted peptides to the protein state'
        if self.if_subtracted:
            print(
                f"{self.num_subtracted_added} subtracted peptides have already been subtracted."
            )

        while True:
            new_peptides = self.add_new_peptides_by_subtract()
            if len(new_peptides) == 0:
                break

        self.if_subtracted = True


class Peptide:
    def __init__(self, raw_sequence, raw_start, raw_end, protein_state=None, n_fastamides=0, RT=None):
        '''
        A class to store one peptide. It can contain multiple timepoints.

        :param raw_sequence: peptide sequence
        :param raw_start: peptide start residue number, including fastamides, 1-based
        :param raw_end: peptide end residue number, including fastamides, 1-based
        :param protein_state: ProteinState it belongs to
        :param n_fastamides: number of fastamides
        
        :ivar identifier: used for peptide identification, raw sequence including fastamides, e.g. "1-10 ABCDEFGHIJ"
        :ivar sequence: peptide sequence excluding fastamides, 1-based
        :ivar start: peptide start residue number excluding fastamides, 1-based
        :ivar end: peptide end residue number, 1-based
        :ivar timepoints: a list of Timepoint objects
        :ivar note: a note for the peptide
        '''
        
        self.raw_start = raw_start
        self.raw_end = raw_end
        self.raw_sequence = raw_sequence
        self.n_fastamides = n_fastamides
        
        self.identifier = (
            f"{raw_start}-{raw_end} {raw_sequence}"  # raw sequence without any modification
        )
        self.sequence = raw_sequence[n_fastamides:]
        self.start = raw_start + n_fastamides
        self.end = raw_end
        self.timepoints = []
        self.note = None
        self.n_fastamides = n_fastamides
        self.RT = RT
        self.unique_id = str(uuid.uuid4())

        if protein_state is not None:
            self.protein_state = protein_state

        # self.max_d = self.get_max_d()

    def add_timepoint(self, timepoint, allow_duplicate=False):
        'add a timepoint to the peptide'
        # Check if timepoint already exists
        if not allow_duplicate:
            for existing_timepoint in self.timepoints:
                if (
                    existing_timepoint.deut_time == timepoint.deut_time
                    and existing_timepoint.charge_state == timepoint.charge_state
                ):
                    raise ValueError(
                        f"{self.start}-{self.end} {self.sequence}: {timepoint.deut_time} (charge: {timepoint.charge_state})Timepoint already exists"
                    )
        if not allow_duplicate:
            for existing_timepoint in self.timepoints:
                if (
                    existing_timepoint.deut_time == timepoint.deut_time
                    and existing_timepoint.charge_state == timepoint.charge_state
                ):
                    raise ValueError(
                        f"{self.start}-{self.end} {self.sequence}: {timepoint.deut_time} (charge: {timepoint.charge_state})Timepoint already exists"
                    )

        self.timepoints.append(timepoint)
        return timepoint
    
    @property
    def id(self):
        return self.identifier

    @property
    def num_timepoints(self):
        'return the number of timepoints in the peptide'
        return len(self.timepoints)
    
    @property
    def unique_num_timepoints(self):
        'return the number of unique deuteration timepoints in the peptide'
        return len(set([tp.deut_time for tp in self.timepoints]))

    @property
    def max_d(self):
        'max deuterium incorporated in the peptide, determined by full-D experiment'
        # if no inf timepoint, return the theoretical max_d
        inf_tp = self.get_timepoint(np.inf)

        if inf_tp is None:
            max_d = self.theo_max_d
        else:
            max_d = inf_tp.num_d

        return max_d

    @property
    def theo_max_d(self):
        'theoretical max deuterium incorporated in the peptide'
        num_prolines = self.sequence.count("P")
        theo_max_d = len(self.sequence) - num_prolines
        return theo_max_d * self.protein_state.hdxms_data.saturation

    @property
    def fit_results(self):
        'fit the D uptakes to mutiple models and return the best fit'
        try:
            max_timepoint = max([tp.deut_time for tp in self.timepoints])
            trialT = np.logspace(1.5, np.log10(max_timepoint * 2), 100)

            if self.timepoints[-1].num_d > 0.5:
                x = [tp.deut_time for tp in self.timepoints]
                y = [tp.num_d for tp in self.timepoints]
                popt, pcov = curve_fit(
                    f=exchange_fit,
                    xdata=x,
                    ydata=y,
                    # bounds = (0, [self.max_d, self.max_d, self.max_d, 1, .1, .01, self.max_d, self.max_d]),
                    bounds=(
                        0,
                        [
                            self.max_d,
                            self.max_d,
                            self.max_d,
                            1,
                            0.1,
                            0.01,
                            self.max_d,
                            self.max_d,
                        ],
                    ),
                    # bounds = (0, [np.inf, np.inf, self.max_d]),
                    maxfev=100000,
                )
                y_pred = exchange_fit(trialT, *popt)
                perr = np.sqrt(np.diag(pcov))
            else:
                x = [tp.deut_time for tp in self.timepoints]
                y = [tp.num_d for tp in self.timepoints]
                popt, pcov = curve_fit(
                    f=exchange_fit_low,
                    xdata=x,
                    ydata=y,
                    bounds=(
                        0,
                        [self.max_d, self.max_d, 0.1, 0.01, self.max_d, self.max_d],
                    ),
                    maxfev=100000,
                )
                y_pred = exchange_fit_low(trialT, *popt)
                perr = np.sqrt(np.diag(pcov))
            return trialT, y_pred, popt, perr
        except:
            raise ValueError(
                f"Error in fitting peptide: {self.start}-{self.end} {self.sequence}"
            )

    def fit_hdx_stats(self, start={"a": None, "b": 0.001, "d": 0, "p": 1}):
        if start["a"] is None:
            start["a"] = self.max_d

        max_timepoint = max([tp.deut_time for tp in self.timepoints])
        trialT = np.logspace(1.5, np.log10(max_timepoint * 2), 100)

        x = [tp.deut_time for tp in self.timepoints]
        y = [tp.num_d for tp in self.timepoints]

        def model_func(timepoint, a, b, p, d):
            return a * (1 - np.exp(-b * (timepoint**p))) + d

        try:
            popt, pcov = curve_fit(
                model_func, x, y, p0=list(start.values()), maxfev=100000, method="lm"
            )
            y_pred = model_func(trialT, *popt)
            perr = np.sqrt(np.diag(pcov))
        except Exception as e:
            print(e)
            print(f"Error in fitting peptide: {self.start}-{self.end} {self.sequence}")
            y_pred = np.zeros(len(trialT))
            popt = np.zeros(4)
            perr = np.zeros(4)

        return trialT, y_pred, popt, perr

    def new_fit(self):
        from sklearn.metrics import mean_squared_error

        x = np.array([tp.deut_time for tp in self.timepoints if tp.deut_time != np.inf])
        y = np.array([tp.num_d for tp in self.timepoints if tp.deut_time != np.inf])
        max_timepoint = max(
            [tp.deut_time for tp in self.timepoints if tp.deut_time != np.inf]
        )
        min_timepoint = min(
            [tp.deut_time for tp in self.timepoints if tp.deut_time != 0]
        )
        trialT = np.logspace(np.log10(min_timepoint), np.log10(max_timepoint), 100)

        fit_resluts = {}
        for exp_num in range(2, 4):
            n = exp_num

            try:
                popt, pcov = curve_fit(
                    fit_func(n=n),
                    x,
                    y,
                    p0=[0.01] * n + [1] * n,
                    bounds=(0, [np.inf] * n + [self.max_d] * n),
                    maxfev=1000,
                )
                y_pred = fit_func(n=n)(trialT, *popt)
                perr = np.sqrt(np.diag(pcov))

                mse = mean_squared_error(y, fit_func(n=n)(x, *popt))
                loss = mse + np.sqrt(np.sum(perr)) * 0.1

                fit_resluts[exp_num] = {
                    "popt": popt,
                    "pcov": pcov,
                    "perr": perr,
                    "mse": mse,
                    "loss": loss,
                    "y_pred": y_pred,
                    "trialT": trialT,
                }
            except Exception as e:
                # print(f"Error in fitting peptide: exp_num={exp_num}")
                fit_resluts[exp_num] = {"loss": np.inf}
                warnings.warn(f"Error in fitting peptide: exp_num={exp_num}")

        best_fit = min(fit_resluts, key=lambda x: fit_resluts[x]["loss"])
        best_model = fit_resluts[best_fit]

        try:
            return (
                best_model["trialT"],
                best_model["y_pred"],
                best_model["popt"],
                best_model["perr"],
            )
        except:
            return None, None, None, None

    def get_deut(self, deut_time):
        'return deuterium num at a specific deuteration time'
        for timepoint in self.timepoints:
            if timepoint.deut_time == deut_time:
                return timepoint.num_d
        return None

    def get_deut_percent(self, deut_time):
        'return normalized deuterium incorporation at a specific deuteration time'
        for timepoint in self.timepoints:
            if timepoint.deut_time == deut_time:
                return timepoint.d_percent
        return None

    def get_timepoint(self, deut_time, charge_state=None):
        'return a timepoint by deuteration time and charge state'
        if charge_state is None:
            timepoints = [tp for tp in self.timepoints if tp.deut_time == deut_time]

            # if not empty return average timepoint
            if len(timepoints) == 1:
                return timepoints[0]

            elif len(timepoints) > 1:
                # avg_timepoint = Timepoint(self, deut_time, np.average([tp.num_d for tp in timepoints]), np.std([tp.num_d for tp in timepoints]))
                avg_timepoint = average_timepoints(timepoints)
                return avg_timepoint
            else:
                return None
        else:
            timepoints = [
                tp
                for tp in self.timepoints
                if tp.deut_time == deut_time and tp.charge_state == charge_state
            ]
            if len(timepoints) == 1:
                return timepoints[0]
            elif len(timepoints) > 1:
                avg_timepoint = average_timepoints(timepoints)
                return avg_timepoint
            else:
                return None


class Timepoint:
    def __init__(self, peptide, deut_time, num_d, stddev, charge_state=None):
        '''
        A class to store one timepoint of a peptide.

        :param peptide: Peptide object it belongs to
        :param deut_time: deuteration time
        :param num_d: number of deuterium incorporated
        :param stddev: standard deviation of the number of deuterium incorporated
        :param charge_state: charge state
        '''
        
        self.peptide = peptide
        self.deut_time = deut_time
        self.num_d = num_d
        self.stddev = stddev
        # self.d_percent = num_d / peptide.max_d
        self.charge_state = charge_state
        self.unique_id = str(uuid.uuid4())
        self.note = None

    def load_raw_ms_csv(self, csv_file):
        'load raw mass spec data from a HDExaminer csv file'
        df = pd.read_csv(csv_file, names=["m/z", "Intensity"])
        # normalize intensity to sum to 1
        # df['Intensity'] = df['Intensity'] / df['Intensity'].sum()
        self.raw_ms = df

        iso = spectra.get_isotope_envelope(self, add_sn_ratio_to_tp=True)
        if iso is not None:
            self.isotope_envelope = iso["Intensity"].values
        else:
            self.isotope_envelope = None

    @property
    def d_percent(self):
        'return normalized deuterium incorporation'
        return round(self.num_d / self.peptide.max_d * 100, 2)


    @property
    def num_d_backex_corrected(self):
        return self.num_d/(self.peptide.max_d/self.peptide.theo_max_d)


class HDXStatePeptideCompares:
    def __init__(self, state1_list, state2_list, threshold=0):
        '''
        A class to compare peptides between two states.

        :param state1_list: a list of ProteinState objects
        :param state2_list: a list of ProteinState objects
        '''
        
        self.state1_list = state1_list
        self.state2_list = state2_list
        self.peptide_compares = []
        self.threshold = threshold

    @property
    def common_idfs(self):
        'return a set of common peptide identifiers between two states'
        # indetifer f"{pep.start}-{pep.end} {pep.sequence}"
        peptides1 = set(
            [pep.identifier for state1 in self.state1_list for pep in state1.peptides]
        )
        peptides2 = set(
            [pep.identifier for state2 in self.state2_list for pep in state2.peptides]
        )
        common_sequences = peptides1.intersection(peptides2)
        return common_sequences

    def add_all_compare(self):
        'add all possible peptide compares between two states'
        import re

        peptide_compares = []
        for idf in self.common_idfs:
            peptide1_list = [state1.get_peptide(idf) for state1 in self.state1_list]
            peptide2_list = [state2.get_peptide(idf) for state2 in self.state2_list]
            peptide_compare = PeptideCompare(peptide1_list, peptide2_list)
            if abs(peptide_compare.deut_diff_avg) > self.threshold:
                peptide_compares.append(peptide_compare)
        self.peptide_compares = sorted(
            peptide_compares,
            key=lambda x: int(re.search(r"(-?\d+)--?\d+ \w+", x.compare_info).group(1)),
        )

    def to_dataframe(self):
        'convert the HDXStatePeptideCompares object to a dataframe'
        df = pd.DataFrame()
        for peptide_compare in self.peptide_compares:
            peptide_compare_df = pd.DataFrame(
                [
                    {
                        "Sequence": peptide_compare.compare_info.split(": ")[1],
                        "deut_diff_avg": peptide_compare.deut_diff_avg,
                    }
                ]
            )
            df = pd.concat([df, peptide_compare_df], ignore_index=True)
        df = df.pivot_table(index="Sequence", values="deut_diff_avg")
        df.index = pd.CategoricalIndex(
            df.index,
            categories=[i.compare_info.split(": ")[1] for i in self.peptide_compares],
        )
        df = df.sort_index()
        return df


class PeptideCompare:
    def __init__(self, peptide1_list, peptide2_list):
        '''
        A class to compare one peptide between two states.

        :param peptide1_list: a list of Peptide objects
        :param peptide2_list: a list of Peptide objects
        :raises ValueError: if peptides have different sequences
        '''
        
        self.peptide1_list = [
            peptide for peptide in peptide1_list if peptide is not None
        ]
        self.peptide2_list = [
            peptide for peptide in peptide2_list if peptide is not None
        ]

        # if not same sequence, raise error
        set_1 = set([peptide.sequence for peptide in self.peptide1_list])
        set_2 = set([peptide.sequence for peptide in self.peptide1_list])
        if set_1 != set_2:
            raise ValueError("Cannot compare peptides with different sequences")

    @property
    def compare_info(self):
        'compare information in the format of "State1-State2: Start-End Sequence"'
        peptide1 = self.peptide1_list[0]
        peptide2 = self.peptide2_list[0]

        return f"{peptide1.protein_state.state_name}-{peptide2.protein_state.state_name}: {peptide1.start}-{peptide1.end} {peptide1.sequence}"

    @property
    def identifier(self):
        return self.peptide1_list[0].identifier
    
    @property
    def sequence(self):
        return self.peptide1_list[0].sequence
    
    @property
    def start(self):
        return self.peptide1_list[0].start
    
    @property
    def end(self):
        return self.peptide1_list[0].end
    
    @property
    def common_timepoints(self):
        'common deuteration timepoints between two peptides'
        timepoints1 = set(
            [
                tp.deut_time
                for peptide1 in self.peptide1_list
                for tp in peptide1.timepoints
                if tp is not None
            ]
        )
        timepoints2 = set(
            [
                tp.deut_time
                for peptide2 in self.peptide2_list
                for tp in peptide2.timepoints
                if tp is not None
            ]
        )
        common_timepoints = list(timepoints1.intersection(timepoints2))
        common_timepoints.sort()

        return np.array(common_timepoints)

    @property
    def deut_diff(self):
        'deuterium difference between two peptides at each timepoint'
        deut_diff = []
        for timepoint in self.common_timepoints:
            deut_diff.append(self.get_deut_diff(timepoint))

        deut_diff = np.array(deut_diff)
        return deut_diff

    @property
    def deut_diff_sum(self):
        'sum of deuterium difference at all timepoint'
        return np.sum(self.deut_diff)

    @property
    def deut_diff_avg(self):
        'average of deuterium difference of all timepoints'
        return np.average(self.deut_diff)

    @deut_diff_avg.setter
    def deut_diff_avg(self, value):
        'set the average deuterium difference'
        self._deut_diff_avg = value

    def get_deut_diff(self, timepoint):
        'deuterium difference between two peptides at a specific timepoint'
        deut1_array = np.array(
            [
                tp.d_percent
                for pep1 in self.peptide1_list
                for tp in pep1.timepoints
                if tp.deut_time == timepoint
            ]
        )
        deut2_array = np.array(
            [
                tp.d_percent
                for pep2 in self.peptide2_list
                for tp in pep2.timepoints
                if tp.deut_time == timepoint
            ]
        )

        result = deut1_array.mean() - deut2_array.mean()

        #significance test: 
        # deut1_std = np.std(deut1_array)
        # deut2_std = np.std(deut2_array)

        # combined_std = np.sqrt(deut1_std**2 + deut2_std**2)

        # if result < combined_std: #if the result is less than combined std, it is not significant
        #     result = 0

        # if deut2_array.size == 1 and deut1_array.size == 1:
        #     result = 0

        return result


class HDXStateResidueCompares:
    def __init__(self, resids, state1_list, state2_list):
        '''
        A class to compare pseudo residues between two states.

        :param resids: a list of residue numbers
        :param state1_list: a list of ProteinState objects
        :param state2_list: a list of ProteinState objects
        '''
        self.resids = resids
        self.residue_compares = []
        self.state1_list = state1_list
        self.state2_list = state2_list

    def add_all_compare(self):
        'add all possible residue compares between two states'
        for resid in self.resids:
            res_compare = ResidueCompare(resid, self.state1_list, self.state2_list)
            if not res_compare.if_empty:
                self.residue_compares.append(res_compare)

    def get_residue_compare(self, resid):
        'return a residue compare by residue number'
        return self.residue_compares[self.resids.index(resid)]


class ResidueCompare:
    def __init__(self, resid, state1_list, state2_list):
        '''
        A class to compare one pseudo residue between two states.

        :param resid: residue number, 1-based
        :param state1_list: a list of ProteinState objects
        :param state2_list: a list of ProteinState objects
        '''
        
        self.resid = resid
        self.state1_list = state1_list
        self.state2_list = state2_list

    # @property
    # def resname(self):
    #   a_peptide = self.containing_peptides1[0]
    #    return a_peptide.sequence[a_peptide.start - self.resid]

    @property
    def compare_info(self):
        'compare information in the format of "State1-State2: Residue Number"'
        return f"{self.state1_list[0].state_name}-{self.state2_list[0].state_name}: {self.resid}"

    def find_peptides_containing_res(self, state_list):
        'find peptides containing the residue in a state list'
        res_containing_peptides = []
        for state in state_list:
            for pep in state.peptides:
                if self.resid >= pep.start and self.resid <= pep.end:
                    # if self.resid - pep.start < 5 and pep.end - self.resid < 5:
                    res_containing_peptides.append(pep)
        return res_containing_peptides

    @property
    def containing_peptides1(self):
        'return peptides containing the residue in state1'
        return self.find_peptides_containing_res(self.state1_list)

    @property
    def containing_peptides2(self):
        'return peptides containing the residue in state2'
        return self.find_peptides_containing_res(self.state2_list)

    @property
    def if_empty(self):
        'return True if no peptides contain the residue in either state'
        if len(self.containing_peptides1) == 0 or len(self.containing_peptides2) == 0:
            return True
        else:
            return False

    @property
    def common_timepoints(self):
        'return common deuteration timepoints between two residues'
        tp1 = set(
            [
                tp.deut_time
                for pep1 in self.containing_peptides1
                for tp in pep1.timepoints
                if tp is not None
            ]
        )
        tp2 = set(
            [
                tp.deut_time
                for pep2 in self.containing_peptides2
                for tp in pep2.timepoints
                if tp is not None
            ]
        )
        common_timepoints = list(tp1.intersection(tp2))
        common_timepoints.sort()

        return np.array(common_timepoints)

    @property
    def deut_diff(self):
        'deuterium difference between two residues at each timepoint'
        if len(self.common_timepoints) == 0:
            return np.nan
        else:
            deut_diff = []
            for timepoint in self.common_timepoints:
                deut_diff.append(self.get_deut_diff(timepoint))
            deut_diff = np.array(deut_diff)

            return deut_diff

    @property
    def deut_diff_sum(self):
        'sum of deuterium difference all timepoints'
        return np.sum(self.deut_diff)

    @property
    def deut_diff_avg(self):
        'average of deuterium difference of all timepoints'
        return np.average(self.deut_diff)

    def get_deut_diff(self, timepoint):
        'deuterium difference between two residues at a specific timepoint'
        deut1_array = np.array(
            [
                pep1.get_deut_percent(timepoint)
                for pep1 in self.containing_peptides1
                if pep1.get_deut_percent(timepoint) is not None
            ]
        )
        deut2_array = np.array(
            [
                pep2.get_deut_percent(timepoint)
                for pep2 in self.containing_peptides2
                if pep2.get_deut_percent(timepoint) is not None
            ]
        )

        result = deut1_array.mean() - deut2_array.mean()
        return result


class SimulatedData:
    def __init__(self, length=100, seed=42, noise_level=0, saturation=1.0, random_backexchange=False, drop_timepoints=True,
                 temperature=293.0, pH=7.0):
        '''
        A class to generate simulated HDX-MS data.

        :param length: protein length, defaults to 100
        :param seed: random seeds, defaults to 42
        :param noise_level: noise add to the isotopic envelope
        '''
        self.temperature = temperature
        self.pH = pH
        random.seed(seed)
        self.length = length
        self.gen_seq()
        self.gen_logP()
        self.cal_k_init()
        self.cal_k_ex()
        self.timepoints = np.insert(np.logspace(1, 12, 20), 0, 0)
        # self.timepoints = np.insert(self.timepoints, -1, 1e12)
        self.noise_level = noise_level
        self.saturation = saturation
        self.random_backexchange = random_backexchange
        self.drop_timepoints = drop_timepoints

        
    def gen_seq(
        self,
    ):
        'generate a random protein sequence of the given length'
        
        AA_list = ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]
        self.sequence = "".join(random.choices(AA_list, k=self.length))

    # def gen_logP(
    #     self,
    # ):
    #     'generate a random logP values for each residue in the sequence'
        
    #     logP = np.array([random.uniform(0.0, 14.0) for i in range(self.length)])
    #     # Proline residues are not exchangeable
    #     for i in range(self.length):
    #         if self.sequence[i] == "P":
    #             logP[i] = 0.0
    #     self.logP = logP
    
    def gen_logP(self, logP_range=[2, 12]):
        prob = [0.9, 0.05, 0.05]  # 90% probability in [2, 12], 5% probability in [12, 14] or [0, 2]
        logP = []
        
        while len(logP) < self.length:
            # Generate random number based on probability
            if random.choices([1, 2, 3], weights=prob)[0] == 1:
                new_value = random.uniform(logP_range[0], logP_range[1])
            elif random.choices([1, 2, 3], weights=prob)[0] == 2:
                new_value = random.uniform(logP_range[1], logP_range[1]+2)
            else:
                new_value = random.uniform(logP_range[0]-2, logP_range[0])
            
            # Check constraints
            if len(logP) >= 4:
                last_four = logP[-4:]
                if (
                    all(x > logP_range[1] for x in last_four) and new_value > logP_range[1]  # Avoid 5 consecutive values >12
                    or all(x < logP_range[0] for x in last_four) and new_value < logP_range[0]  # Avoid 5 consecutive values <2
                ):
                    continue  # Constraint not met, generate new value
            
            # Add value if constraints are met
            logP.append(new_value)
        
        logP = np.array(logP)
        logP[self.sequence == "P"] = 0.0
        self.logP = logP

    def cal_k_init(self):
        'calculate intrinsic exchange rate for each residue'
        self.log_k_init = np.log10(k_int_from_sequence(self.sequence, self.temperature, self.pH))

    def cal_k_ex(self):
        'calculate exchange rate for each residue'
        self.log_k_ex = self.log_k_init - self.logP

    def calculate_incorporation(self):
        'calculate deuterium incorporation for each residue'
        
        incorporations = {}
        for tp in self.timepoints:
            incorporations[tp] = calculate_simple_deuterium_incorporation(self.log_k_ex, tp)
        self.incorporations = incorporations

    def gen_peptides(self, min_len=5, max_len=12, max_overlap=5, num_peptides=30):
        'generate random peptides from the protein sequence'
        chunks = []
        avg_peptide_length = math.ceil(min_len + max_len) / 2
        sequence_length = len(self.sequence)

        avg_coverage = math.ceil(num_peptides / sequence_length) * 3

        #blank_peptide_obj_list = []  # just for calculating overlapping peptides
        for i in range(sequence_length):
            count = 0
            while count < avg_coverage:
                start = max(
                    0, random.randint(int(i - avg_peptide_length - 2), i)
                )  # n_fastamides = 2
                for _ in range(3):
                    end = min(
                        random.randint(i, int(i + avg_peptide_length)), sequence_length
                    )
                    pep_len = len([i for i in self.sequence[start:end] if i != "P"])
                    if pep_len > 3:
                        chunks.append(self.sequence[start:end])
                count += 1

        covered = np.zeros(sequence_length, dtype=bool)
        num_pairs = 0
        
        attempt = 0
        # while not (
        #     all(covered) and (num_peptides * 0.7 < num_pairs < num_peptides * 1.0)
        # ):
        while not (
            sum(covered) >= 0.9 * self.length and (num_peptides * 0.7 < num_pairs < num_peptides * 1.3)
        ):
        #top_range = 0.9
        #while not (
        #    (sum(covered) <= top_range * self.length) and (sum(covered) >= (top_range-0.05) * self.length) and (num_peptides * 0.7 < num_pairs < num_peptides * 1.3)
        #):
            random.seed(self.seed + attempt)
            attempt += 1
         
            reduced_chunks = random.sample(
                chunks,
                k=num_peptides,
            )

            # for similar overlapping peptides in the real data
            N_overlap_groups = {}
            C_overlap_groups = {}
            for pep in reduced_chunks:
                start = self.sequence.find(pep)
                end = start + len(pep)
                covered[start:end] = True

                if start in N_overlap_groups:
                    N_overlap_groups[start].append(pep)
                else:
                    N_overlap_groups[start] = [pep]

                if end in C_overlap_groups:
                    C_overlap_groups[end].append(pep)
                else:
                    C_overlap_groups[end] = [pep]

            combined_overlap_groups = {**N_overlap_groups, **C_overlap_groups}

            all_pairs = [
                i
                for k, v in combined_overlap_groups.items()
                for i in itertools.combinations(v, 2)
                if i[0] != i[1]
            ]
            num_pairs = len(set(all_pairs))

        # add more duplicates
        for i in range(3):
            duplicates = random.sample(reduced_chunks, k=int(0.1*len(reduced_chunks)))
            reduced_chunks = reduced_chunks + duplicates
            
        self.peptides = [
            i
            for i in sorted(reduced_chunks, key=lambda x: self.sequence.find(x))
            if len(i) > 3
        ]

    def convert_to_hdxms_data(self):
        'convert the simulated data to a HDXMSData object'
        hdxms_data = HDXMSData("simulated_data", protein_sequence=self.sequence, saturation=self.saturation, temperature=self.temperature, pH=self.pH)
        protein_state = ProteinState("SIM", hdxms_data=hdxms_data)
        hdxms_data.add_state(protein_state)

        # calculate incorporation
        self.calculate_incorporation()
        
        back_ex_dict = {}
        
        for peptide in self.peptides:
            start = self.sequence.find(peptide) + 1
            end = start + len(peptide) - 1

            peptide_obj = Peptide(peptide, start, end, protein_state, n_fastamides=2)

            if self.random_backexchange:
                if peptide_obj.identifier in back_ex_dict:
                    true_back_exchange = back_ex_dict[peptide_obj.identifier]
                    _add_max_d_to_pep(peptide_obj, max_d=(1-true_back_exchange)*peptide_obj.theo_max_d)
                else:
                    true_back_exchange = random.uniform(0.0, 0.4)
                    _add_max_d_to_pep(peptide_obj, max_d=(1-true_back_exchange)*peptide_obj.theo_max_d)
                    back_ex_dict[peptide_obj.identifier] = true_back_exchange
            else:
                true_back_exchange = 0.0
                
            try:
                protein_state.add_peptide(peptide_obj, allow_duplicate=True)
                
                # make sure 0 timepoint is included and is the first timepoint
                if self.drop_timepoints:
                    timepoints = self.timepoints.copy()
                    timepoints = random.sample(list(timepoints), k=int(0.8*len(timepoints)))
                else:
                    timepoints = self.timepoints.copy()
                
                if 0 not in timepoints:
                    timepoints = np.insert(timepoints, 0, 0)
                    
                # sort timepoints
                timepoints.sort()
                
                noise_multiplier = 1.0
                pep_length = peptide_obj.end - peptide_obj.start + 1
                if (pep_length <= 5 or pep_length >= 15) and random.uniform(0, 1) < 0.3:
                    noise_multiplier = 1.1
                            
                for tp_i, tp in enumerate(timepoints):
                    # tp_raw_deut = self.incorporations[
                    #     peptide_obj.start - 1 : peptide_obj.end
                    # ][:, tp_i]
                    tp_raw_deut = self.incorporations[tp][peptide_obj.start - 1 : peptide_obj.end]
                    
                    #add sidechain exchange noise, 5% of theo_max_d, only positive
                    if tp_i != 0 and random.uniform(0, 1) < 0.3:
                        tp_raw_deut = tp_raw_deut + random.uniform(0, 0.03) 
                    
                    tp_raw_deut = tp_raw_deut * (1 - true_back_exchange) * self.saturation
                    pep_incorp = sum(tp_raw_deut)
                    # random_stddev = peptide_obj.theo_max_d * self.noise_level * random.uniform(-1, 1)
                    random_stddev = 0
                    tp_obj = Timepoint(
                        peptide_obj,
                        tp,
                        pep_incorp + random_stddev,
                        random_stddev,
                    )
                    
                    if tp == 0.0:
                        t0_theo = spectra.get_theoretical_isotope_distribution(tp_obj)[1]
                        
                    p_D = event_probabilities(tp_raw_deut)

                    isotope_envelope = np.convolve(t0_theo, p_D)
                    # isotope_noise = np.array(
                    #     [
                    #         random.uniform(-1, 1) * self.noise_level * peak
                    #         for peak in isotope_envelope
                    #     ]
                    # )
                    isotope_noise = np.array(
                        [
                            random.uniform(-1, 1) * self.noise_level/10 +  random.uniform(-1, 1) * self.noise_level * peak 
                            for peak in isotope_envelope
                        ]
                    )
                    
                    if tp_i != 0:
                        isotope_noise += np.array([random.uniform(-1, 1) * self.noise_level/10 * (np.log10(tp)/18) for _ in isotope_envelope])
                    
                    isotope_envelope = isotope_envelope + isotope_noise * noise_multiplier
                    isotope_envelope[isotope_envelope < 0] = 0
                    isotope_envelope = normlize(isotope_envelope)
                    
                    tp_obj.isotope_envelope = isotope_envelope
                     
                    # update num_d
                    mass_num = np.arange(len(isotope_envelope))
                    if tp_i == 0:
                        t0_centroid = np.sum(isotope_envelope * mass_num)

                    tp_obj.num_d = np.sum(isotope_envelope * mass_num) - t0_centroid

                    peptide_obj.add_timepoint(tp_obj)

            except Exception as e:
                #print(e)
                continue
            
        #print(back_ex_dict)

        self.hdxms_data = hdxms_data

        # peptide_obj.add_timepoint(tp, self.incorporations[start:end])
