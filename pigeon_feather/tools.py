import numpy as np
import os
from collections import defaultdict
import re
import math

def refine_data(hdxms_data_list, std_threshold=1.0):
    """
    Remove peptides with large deviation across replicates
    all tps has std > 1.0 will be removed
    """
    all_tps = [
        tp
        for data in hdxms_data_list
        for state in data.states
        for peptide in state.peptides
        for tp in peptide.timepoints
        if tp.deut_time != np.inf and tp.deut_time != 0.0
    ]
    grouped_tps = group_by_attributes(
        all_tps, ["peptide.protein_state.state_name", "peptide.identifier", "deut_time"]
    )

    high_std_tps = {}
    all_std = []
    for k, v in grouped_tps.items():
        if len(v) > 1:
            std = np.std([i.num_d for i in v])
            if std > std_threshold:
                high_std_tps[k] = v
            all_std.append(std)
    for k, v in high_std_tps.items():
        std = np.std([i.num_d for i in v])
        # print(k, std)

    all_std = np.array(all_std)
    print(f"Average timepoint std: {np.mean(all_std)}")

    for k, v in high_std_tps.items():
        std = np.std([i.num_d for i in v])
        states = [data.get_state(k[0]) for data in hdxms_data_list if data.get_state(k[0]) is not None]
        for state in states:
            remove_tps_from_state(v, state)
        # print(k, std)


def find_non_compatible_peptides(hdxms_datas, protein_state, plot_results=False, threshold=0.3):
    import itertools
    from pigeon_feather.plot import UptakePlot
    from collections import Counter
    import matplotlib.pyplot as plt
    non_compatible_idfs = []
    fig_list = []

    subgroups = find_overlapped_peptides(protein_state)
    peptide_notes = [pep.note for pep in protein_state.peptides if pep.note]

    for peptides in subgroups.values():
        sorted_peptides = sorted(peptides, key=lambda x: x.start)
        for pair in itertools.combinations(sorted_peptides, 2):
            if any(f"Subtraction: {p[0].identifier} - {p[1].identifier}" in peptide_notes for p in [pair, pair[::-1]]):
                continue

            new_peptide = subtract_peptides(*pair, subtract_envelope=False, filter_result=False)
            if new_peptide is None:
                continue

            avg_num_d = np.mean([tp.num_d for tp in new_peptide.timepoints if tp.deut_time != np.inf])
            if avg_num_d <= -1 * threshold:
                idfs = new_peptide.note.split('Subtraction:')[1].strip().split(' - ')
                non_compatible_idfs.extend(idfs)

                if plot_results:
                    fig, ax = plt.subplots(1, 2, figsize=(18, 8))
                    for i, idf in enumerate(idfs):
                        UptakePlot(hdxms_datas, idf, figure=fig, ax=ax[i], states_subset=[protein_state.state_name])
                        if i > 0:
                            ax[i].set_ylim(ax[0].get_ylim())
                    fig_list.append(fig)

    counter = Counter(non_compatible_idfs)
    filtered_items = {item: count for item, count in counter.items() if count >= 2}

    if len(filtered_items) > 0:
        print('-'*50)
        print('STATE:', protein_state.state_name)
        print('Warning: the following peptides might be problematic, please check.')
    for idf, count in filtered_items.items():
        print(f'{idf}: conflict {count} times')

    return fig_list if plot_results else None



def find_overlapped_peptides(protein_state):
    """
    find overlapped peptides in the hdxms_data object that could be used for substation

    Parameters
    ----------
    protein_state : hdxms_data.states[0]
        the protein state of interest
    """

    from pigeon_feather.data import ProteinState

    # check the input
    if not isinstance(protein_state, ProteinState):
        raise TypeError("The input should be a protein state of hdxms_data object.")

    # create a dictionary to store the subgroups
    start_subgroups = {}  # subgroups with the same start position
    end_subgroups = {}  # subgroups with the same end position

    # iterate over all the peptides
    for peptide in protein_state.peptides:
        # check if the start position is already in the dictionary
        if peptide.start in start_subgroups:
            start_subgroups[peptide.start].append(f"{peptide.identifier}")
        else:
            start_subgroups[peptide.start] = [f"{peptide.identifier}"]

        # check if the end position is already in the dictionary
        if peptide.end in end_subgroups:
            end_subgroups[peptide.end].append(f"{peptide.identifier}")
        else:
            end_subgroups[peptide.end] = [f"{peptide.identifier}"]

    # combine two subgroups
    combined = {**start_subgroups, **end_subgroups}
    subgroups = {}
    for key in combined.keys():
        value = list(set(combined[key]))
        if len(value) > 1:
            subgroups[key] = [protein_state.get_peptide(idf) for idf in value]

    return subgroups


def subtract_peptides(peptide_1, peptide_2, subtract_envelope=True, filter_result=True):
    """
    Subtract two peptides and create a new peptide.
    """
    # Check if the two peptides have the same length
    if peptide_1.sequence == peptide_2.sequence:
        raise ValueError("Cannot subtract the same two peptides.")

    if peptide_1.start != peptide_2.start and peptide_1.end != peptide_2.end:
        raise ValueError("start or end need to be the different.")

    if peptide_1.protein_state.state_name != peptide_2.protein_state.state_name:
        raise ValueError("Cannot subtract peptides from different states.")

    # get longer peptide and shorter peptide
    if len(peptide_1.sequence) > len(peptide_2.sequence):
        longer_peptide = peptide_1
        shorter_peptide = peptide_2
    else:
        longer_peptide = peptide_2
        shorter_peptide = peptide_1

    timepoints1 = set([tp.deut_time for tp in longer_peptide.timepoints if tp.deut_time != np.inf])
    timepoints2 = set([tp.deut_time for tp in shorter_peptide.timepoints if tp.deut_time != np.inf])
    common_timepoints = list(timepoints1.intersection(timepoints2))
    common_timepoints.sort()

    if shorter_peptide.start == longer_peptide.start:
        start = shorter_peptide.end + 1
        end = longer_peptide.end
        # new_sequence = longer_peptide.sequence[
        #     shorter_peptide.end - shorter_peptide.start + 1 : longer_peptide.end
        #     - longer_peptide.start
        #     + 1
        # ]
    else:
        start = longer_peptide.start
        end = shorter_peptide.start - 1
        # new_sequence = longer_peptide.sequence[
        #     0 : shorter_peptide.start - longer_peptide.start
        # ]

    from pigeon_feather.data import Peptide, Timepoint

    # Create a new peptide (n_fastamides=0)
    new_sequence = peptide_1.protein_state.hdxms_data.protein_sequence[start - 1 -2: end]
    new_peptide = Peptide(
        raw_sequence=new_sequence,
        raw_start=start-2,
        raw_end=end,
        protein_state=peptide_1.protein_state,
        n_fastamides=2,
    )

    # return None if:
    # skip if new_peptide has less than 3 timepoints
    if len(common_timepoints) <= 3:
        return None
    # skip if new_peptide is single Proline
    if new_peptide.sequence == "P":
        return None
    if new_peptide.max_d == 0:
        return None

    # iterate over all the timepoints
    for tp in common_timepoints:
        longer_tp = longer_peptide.get_timepoint(tp)
        shorter_tp = shorter_peptide.get_timepoint(tp)

        std = np.sqrt(
            longer_peptide.get_timepoint(tp).stddev ** 2
            + shorter_peptide.get_timepoint(tp).stddev ** 2
        )
        timepoint = Timepoint(new_peptide, tp, longer_tp.num_d - shorter_tp.num_d, std)
        timepoint.note = (
        f"Subtraction: {longer_peptide.identifier} - {shorter_peptide.identifier}: {tp}")
        # timepoint = Timepoint(new_peptide, tp, longer_peptide.get_deut(tp) - shorter_peptide.get_deut(tp), std)

        if (
            hasattr(longer_tp, "isotope_envelope")
            and hasattr(shorter_tp, "isotope_envelope")
            and tp != np.inf
            and subtract_envelope
        ):
            # timepoint.isotope_envelope = deconvolute(longer_peptide.get_timepoint(tp).isotope_envelope, shorter_peptide.get_timepoint(tp).isotope_envelope)
            p1 = longer_tp.isotope_envelope.copy()
            p2 = shorter_tp.isotope_envelope.copy()
            # p3 = get_theo_ms(timepoint)
            p3 = force_to_envelope(deconvolute_fft(p1, p2))
            best_model = deconvolute_iso(p1, p2, p3, steps=2000)
            if best_model[1] is None:
                continue
            else:
                timepoint.isotope_envelope = best_model[1]
                if abs(get_num_d_from_iso(timepoint) - timepoint.num_d) > 0.5:
                    continue
                new_peptide.add_timepoint(timepoint)

        else:
            new_peptide.add_timepoint(timepoint)


    # back exchange correction
    new_peptide = _add_max_d_to_pep(new_peptide, max_d_theo_max_ratio=longer_peptide.max_d/longer_peptide.theo_max_d)

    

    if len(new_peptide.timepoints) < 3 and filter_result:
        return None

    # skip if new_peptide is negative
    if (
        (np.average([tp.num_d for tp in new_peptide.timepoints]) < -0.3
        or new_peptide.max_d < 0.0 
        or new_peptide.max_d/new_peptide.theo_max_d < 0.6
        or new_peptide.max_d/new_peptide.theo_max_d > 1.2)
        and filter_result
    ):
        return None

    new_peptide.note = (
        f"Subtraction: {longer_peptide.identifier} - {shorter_peptide.identifier}"
    )

    # if 0.0 not in [tp.deut_time for tp in new_peptide.timepoints]:
    #     print(longer_peptide.identifier, shorter_peptide.identifier)
    #     raise ValueError("Cannot subtract peptides without time 0.")

    return new_peptide



def average_hdxms_datas(hdxms_datas, states_subset=None):

    from pigeon_feather.data import HDXMSData, ProteinState, Peptide

    protein_sequence = hdxms_datas[0].protein_sequence
    state_names = [state.state_name for data in hdxms_datas for state in data.states]


    all_tps = [
        tp
        for data in hdxms_datas
        for state in data.states
        for peptide in state.peptides
        for tp in peptide.timepoints
        if states_subset is None or state.state_name in states_subset
        # if tp.deut_time != np.inf and tp.deut_time != 0.0
    ]

    grouped_tps = group_by_attributes(
        all_tps, ["peptide.protein_state.state_name", "peptide.identifier", "deut_time"]
    )

    hdxms_data_avg = HDXMSData(
        "Avg data",
        2,
        protein_sequence=protein_sequence,
        saturation=hdxms_datas[0].saturation,
    )

    for state in state_names:
        protein_state = hdxms_data_avg.get_state(state)
        if protein_state is None:
            protein_state = ProteinState(state, hdxms_data=hdxms_data_avg)
            hdxms_data_avg.add_state(protein_state)

    for k, v in grouped_tps.items():
        protein_state = hdxms_data_avg.get_state(k[0])

        if protein_state.get_peptide(k[1]) is None:
            raw_seq = v[0].peptide.identifier.split(" ")[-1]
            raw_start = int(v[0].peptide.identifier.split(" ")[0].split("-")[0])
            raw_end = int(v[0].peptide.identifier.split(" ")[0].split("-")[1])

            peptide = Peptide(
                raw_seq,
                raw_start,
                raw_end,
                hdxms_data_avg.get_state(k[0]),
                n_fastamides=2,
            )
            protein_state.add_peptide(peptide)
        else:
            peptide = protein_state.get_peptide(k[1])

        avg_tp = average_timepoints(v)
        peptide.add_timepoint(avg_tp)

    return hdxms_data_avg


def average_peptides(peptides):
    if len(set([pep.identifier for pep in peptides])) != 1:
        raise ValueError("Peptides must be the same sequence")
    
    from pigeon_feather.data import Peptide
    all_tps = [tp for pep in peptides for tp in pep.timepoints]
    all_tps_grouped = group_by_attributes(all_tps, ["deut_time"])


    avg_peptide = Peptide(
        raw_sequence = peptides[0].identifier.split(" ")[-1],
        raw_start = peptides[0].start-2,
        raw_end = peptides[0].end,
        protein_state = peptides[0].protein_state,
        n_fastamides=2,
    )

    if hasattr(peptides[0], "sidechain_exchange"):
        avg_peptide.sidechain_exchange = np.average([pep.sidechain_exchange for pep in peptides])

    all_tps_grouped_sorted = sorted(all_tps_grouped.items(), key=lambda x: x[0])

    for item in all_tps_grouped_sorted:
        avg_tp = average_timepoints(item[1])
        avg_peptide.add_timepoint(avg_tp)

    return avg_peptide
    


def average_timepoints(tps_list):
    # check if the same state, peptide, and timepoint
    if (
        len(
            set(
                [
                    (
                        tp.peptide.protein_state.state_name,
                        tp.peptide.identifier,
                        tp.deut_time,
                    )
                    for tp in tps_list
                ]
            )
        )
        != 1
    ):
        raise ValueError(
            "Timepoints must be from the same state, peptide, and timepoint"
        )

    deut_time = tps_list[0].deut_time
    num_d = np.average([tp.num_d for tp in tps_list])
    std_d = np.std([tp.num_d for tp in tps_list])

    from pigeon_feather.data import Timepoint

    avg_timepoint = Timepoint(tps_list[0].peptide, deut_time, num_d, std_d)
    try:
        size = max([tp.isotope_envelope.size for tp in tps_list])
        iso_list = [
            custom_pad(tp.isotope_envelope, size, pad_value=0.0) for tp in tps_list
        ]
        avg_iso = np.average(iso_list, axis=0)
        avg_timepoint.isotope_envelope = avg_iso / np.sum(avg_iso)

    except:
        # print('No isotope envelope found for timepoint')
        pass

    return avg_timepoint



def merge_hdxms_datas(hdxms_datas, states_subset=None):


    from pigeon_feather.data import HDXMSData, ProteinState, Peptide

    protein_sequence = hdxms_datas[0].protein_sequence
    
    if states_subset is None:
        state_names = list(set([state.state_name for data in hdxms_datas for state in data.states]))
    else:
        state_names = list(set([state.state_name for data in hdxms_datas for state in data.states if state.state_name in states_subset]))

    all_tps = [
        tp
        for data in hdxms_datas
        for state in data.states
        for peptide in state.peptides
        for tp in peptide.timepoints
        if states_subset is None or state.state_name in states_subset
        # if tp.deut_time != np.inf and tp.deut_time != 0.0
    ]


    hdxms_data_merged = HDXMSData(
        "merged data",
        2,
        protein_sequence=protein_sequence,
        saturation=hdxms_datas[0].saturation,
    )


    for state in state_names:
        protein_state = hdxms_data_merged.get_state(state)
        if protein_state is None:
            protein_state = ProteinState(state, hdxms_data=hdxms_data_merged)
            hdxms_data_merged.add_state(protein_state)

    for tp in all_tps:
        protein_state = hdxms_data_merged.get_state(tp.peptide.protein_state.state_name)

        if protein_state.get_peptide(tp.peptide.identifier) is None:
            raw_seq = tp.peptide.identifier.split(" ")[-1]
            raw_start = int(tp.peptide.identifier.split(" ")[0].split("-")[0])
            raw_end = int(tp.peptide.identifier.split(" ")[0].split("-")[1])

            peptide = Peptide(
                raw_seq,
                raw_start,
                raw_end,
                protein_state,
                n_fastamides=2,
            )
            protein_state.add_peptide(peptide)
        else:
            peptide = protein_state.get_peptide(tp.peptide.identifier)

        try:
            peptide.add_timepoint(tp)
        except:
            # skip if the timepoint already exists
            continue

    return hdxms_data_merged



def deconvolute_fft(p, p1):
    # Compute the Fourier Transforms
    P = np.fft.fft(p)
    P1 = np.fft.fft(p1, n=len(p))  # Make sure the arrays have the same length

    # Division in the Frequency Domain
    P2 = P / P1

    # Compute the Inverse Fourier Transform
    p2 = np.fft.ifft(P2).real  # Take the real part to ignore small numerical errors

    # wipe the negative values to 1e-10
    p2[p2 < 0] = 1e-10
    p2 = p2 / np.sum(p2)

    return p2


def metropolis_criterion(old_loss, new_loss, temperature):
    if new_loss < old_loss:
        return True
    else:
        delta_loss = new_loss - old_loss
        probability = np.exp(-delta_loss / temperature)
        return np.random.rand() < probability


def force_to_envelope(p3_array):
    max_index = np.argmax(p3_array)

    before_peak = np.sort(p3_array[0:max_index])
    after_peak = np.sort(p3_array[max_index + 1 :])[::-1]

    # make a new envelope
    new_envelope = np.concatenate(
        (before_peak, np.array(p3_array[max_index]), after_peak), axis=None
    )

    return new_envelope


def deconvolute_iso(p1, p2, p3, temperature=0.05, steps=2000, keep_best=True):
    """
    Deconvolution function that updates p3 all at once.
    p3: initial guess
    """

    max_length = max(p1.size, p2.size)
    p1 = custom_pad(p1, max_length)
    p2 = custom_pad(p2, max_length)
    p3 = p3[:max_length] / np.sum(p3[:max_length])

    p1_estimated = convolve(p3, p2)
    p1_estimated = normlize(p1_estimated)

    num_peaks = np.count_nonzero(p1 > 0.1)  # number of major peaks
    previous_loss = get_sum_ae(p1, p1_estimated) / num_peaks

    if previous_loss < 0.05:
        return (previous_loss, p3)

    best_models = []

    continus_rejects = 0

    for i in range(steps):
        for j in range(len(p3)):
            original_p3_j = p3[j]
            change = np.random.normal(0, 0.1)  # Generate change for the current peak
            p3[j] *= 1 + change  # Apply the change
            p3[j] = max(p3[j], 1e-10)  # Prevent negative values

            p3 = normlize(p3)  # Normalize p3 after all individual changes
            p3 = force_to_envelope(p3)  # Force to envelope

            # Calculate new loss
            p1_estimated = convolve(p3, p2)
            p1_estimated = normlize(p1_estimated)
            new_loss = get_sum_ae(p1, p1_estimated) / num_peaks

            # Decide whether to accept the change
            if metropolis_criterion(previous_loss, new_loss, temperature):
                previous_loss = new_loss
                continus_rejects = 0
            else:
                # Revert the change for the current peak
                p3[j] = original_p3_j
                continus_rejects += 1

        best_models.append((previous_loss, p3.copy()))
        temperature *= 0.99  # Decrease the temperature

        if new_loss < 0.1 or (0.1 < new_loss < 0.2 and continus_rejects > 20):
            best_models.sort(key=lambda x: x[0])
            return best_models[0]
        elif new_loss > 0.2 and continus_rejects > 500:
            return (previous_loss, None)

        # Optional: Uncomment to see progress every 100 steps
        # if i % 100 == 0:
        #     print(i, previous_loss)


def get_centroid_iso(tp):
    mz = np.arange(0, len(tp.isotope_envelope))
    intensity = tp.isotope_envelope
    return mz @ intensity


def get_t0_centroid_iso(tp):
    t0_centroid_iso = [
        get_centroid_iso(tp) for tp in tp.peptide.timepoints if tp.deut_time == 0
    ]
    return np.average(t0_centroid_iso)


def get_num_d_from_iso(tp):
    return (get_centroid_iso(tp) - get_t0_centroid_iso(tp)) / 1.00866491588


def exponential_sum_decorator(n):
    def decorator(func):
        def wrapper(*args):
            if len(args) != 2 * n + 1:
                raise ValueError(f"Expected {2*n + 1} arguments, got {len(args)}")

            t = args[0]  # assuming the first argument is the time variable 't'
            k_values = args[1 : n + 1]  # extracting the rate constants
            max_d_values = args[n + 1 :]  # extracting the max deuteration levels

            result = 0
            for k, max_d in zip(k_values, max_d_values):
                result += func(
                    t, k, max_d
                )  # call the original function with the ith rate constant, time, and max_d
            return result

        return wrapper

    return decorator


def fit_func(n=1):
    @exponential_sum_decorator(n=n)
    def single_amide_ex(t, k, max_d):
        return max_d * (1 - np.exp(-k * t))

    return single_amide_ex


def exchange_fit(x, a, b, c, d, e, f, g, max_d):
    return max_d - a * np.exp(-d * x) - b * np.exp(-e * x) - c * np.exp(-f * x) - g


def exchange_fit_low(x, b, c, e, f, g, max_d):
    return max_d - b * np.exp(-e * x) - c * np.exp(-f * x) - g


from numba import njit


@njit
def normlize(p):
    return p / np.sum(p)


@njit
def convolve(p1, p2):
    return np.convolve(p1, p2)


@njit
def custom_kl_divergence(p, q):
    """Calculate the Kullback-Leibler divergence between two probability distributions."""

    # Ensure the probabilities are normalized
    x1 = p / np.sum(p)
    x2 = q / np.sum(q)

    divergence = np.sum(x1 * np.log(x1 / x2))
    return divergence


@njit
def jensen_shannon_divergence(p, q):
    x1 = p / np.sum(p)
    x2 = q / np.sum(q)
    m = (x1 + x2) / 2
    return (custom_kl_divergence(x1, m) + custom_kl_divergence(x2, m)) / 2


@njit
def custom_pad(array, target_length, pad_value=1e-10):
    # replace 0s with a small value
    array[array == 0] = pad_value
    padded_array = np.full(target_length, pad_value)
    padded_array[: len(array)] = array
    return padded_array


@njit
def norm_pad_two_arrays(p, q):
    """
    normalize two arrays and pad them to the same size and
    """
    x1 = p / np.sum(p)
    x2 = q / np.sum(q)

    size = max(len(x1), len(x2))
    x1 = custom_pad(x1, size)
    x2 = custom_pad(x2, size)
    return x1, x2


@njit
def get_divergence(p, q, method="KL"):
    x1, x2 = norm_pad_two_arrays(p, q)

    if method == "KL":
        return custom_kl_divergence(x1, x2)
    elif method == "JS":
        return jensen_shannon_divergence(x1, x2)


@njit
def get_mse(p, q):
    x1, x2 = norm_pad_two_arrays(p, q)

    return np.mean(np.square(x1 - x2))


@njit
def get_mae(p, q):
    x1, x2 = norm_pad_two_arrays(p, q)

    return np.mean(np.abs(x1 - x2))


@njit
def get_sum_ae(p, q):
    x1, x2 = norm_pad_two_arrays(p, q)
    return np.abs(x1 - x2).sum()


@njit
def event_probabilities(p_array):
    n = p_array.shape[0]
    # dp[i][j] stores the probability of j successes in the first i events
    dp = np.zeros((n + 1, n + 1))
    dp[0][0] = 1  # Base case: probability of 0 successes in 0 events is 1

    for i in range(1, n + 1):
        p = p_array[i - 1]
        q = 1 - p
        dp[i][0] = dp[i - 1][0] * q  # Probability of 0 successes in i events

        for j in range(1, i + 1):
            # Probability of j successes in i events
            dp[i][j] = dp[i - 1][j - 1] * p + dp[i - 1][j] * q

    # The probabilities of 0, 1, 2, ..., n successes in n events
    probabilities = dp[n]

    # if negative values, set to 0
    if np.any(probabilities < 0):
        probabilities[probabilities < 0] = 0
        probabilities = probabilities / np.sum(probabilities)

    return probabilities


def remove_tps_from_state(removing_tps, state):
    # n_tp = len([tp for pep in state.peptides for tp in pep.timepoints])
    # print(f'Number of timepoints before removing: {n_tp}')

    num_pep_removed = 0
    peptides_removed = []

    new_peptides = []

    for pep in state.peptides:
        pep.timepoints = [tp for tp in pep.timepoints if tp not in removing_tps]

        if pep.timepoints and any(tp.deut_time == 0 for tp in pep.timepoints):
            new_peptides.append(pep)
            
        else:
            num_pep_removed += 1
            peptides_removed.append(pep)
            # print(f"{pep.identifier} removed")

    state.peptides = new_peptides


    final_peptides = []
    for pep in state.peptides:
        real_tps_num = len(
            set(
                tp.deut_time
                for tp in pep.timepoints
                if tp.deut_time != np.inf and tp.deut_time != 0.0
            )
        )
        if real_tps_num >= 2:
            final_peptides.append(pep)
        else:
            num_pep_removed += 1
            peptides_removed.append(pep)
            #print(f"{pep.identifier} removed")

    state.peptides = final_peptides
    removed_idfs = [pep.identifier for pep in peptides_removed]
    removed_idfs.sort(key=lambda x: int(x.split('-')[0]))
    # if print_details:
    #     print("-"*10)
    #     print(','.join(removed_idfs))
    #     print("-"*10)

    return num_pep_removed, removed_idfs


def group_by_attributes(objects, attributes):
    grouped_data = defaultdict(list)
    for obj in objects:
        key = []
        for attr in attributes:
            if "." in attr:
                nested_attrs = attr.split(".")
                value = obj
                for na in nested_attrs:
                    value = getattr(value, na)
                key.append(value)
            else:
                key.append(getattr(obj, attr))
        grouped_data[tuple(key)].append(obj)
    return grouped_data


def backexchange_correction(hdxms_data_list):
    """
    backexchange correction for all peptides in the hdxms_data_list
    """

    all_peps = [
        pep
        for data in hdxms_data_list
        for state in data.states
        for pep in state.peptides
    ]

    all_inf_tps = [
        tp
        for data in hdxms_data_list
        for state in data.states
        for pep in state.peptides
        for tp in pep.timepoints
        if tp.deut_time == np.inf
    ]


    all_inf_tps_grouped = group_by_attributes(
        all_inf_tps, ["peptide.identifier"]
    )


    pep_with_exp_max_d = [
        pep for pep in all_peps if pep.get_timepoint(np.inf) is not None
    ]

    pep_with_no_exp_mad_d = [
        pep for pep in all_peps if pep.get_timepoint(np.inf) is None
    ]

    print(f"Number of peptides with experimental max_d: {len(pep_with_exp_max_d)}")
    print(f"Number of peptides with no experimental max_d: {len(pep_with_no_exp_mad_d)}")

    import re

    # Function to extract the numbers from the key
    def _extract_numbers(key):
        return list(map(int, re.findall(r'\d+', key[0])))

    # Function to calculate the distance between two keys
    def _calculate_distance(key1, key2):
        num1 = _extract_numbers(key1)
        num2 = _extract_numbers(key2)
        return sum(abs(a - b) for a, b in zip(num1, num2))

    # Function to find the closest key
    def find_closest_key(keys, target_key):
        closest_key = None
        smallest_distance = float('inf')
        
        for key in keys:
            distance = _calculate_distance(key, target_key)
            if distance < smallest_distance:
                smallest_distance = distance
                closest_key = key
                
        return closest_key


    keys = [k for k,v  in all_inf_tps_grouped.items() if v != []]

    for pep in all_peps:
        if all_inf_tps_grouped[(pep.identifier,)] != []:
            target_key = (pep.identifier,)
        else:
            target_key = find_closest_key(keys, (pep.identifier,))
            #print(pep.identifier, target_key)

        max_d_ratios = [
            i.num_d/i.peptide.theo_max_d for i in all_inf_tps_grouped[target_key]
        ]
        
        max_d_ratios = list(set(max_d_ratios))
        avg_max_d_ratios = np.mean(max_d_ratios)        
        _add_max_d_to_pep(pep, max_d_theo_max_ratio=avg_max_d_ratios, force=True)



def _add_max_d_to_pep(pep, max_d_theo_max_ratio=None, max_d=None, force=False):
    from pigeon_feather.data import Timepoint
    if max_d_theo_max_ratio is None and max_d is None:
        raise ValueError("Either max_d_theo_max_ratio or max_d must be provided")
    
    if not force:

        if pep.get_timepoint(np.inf) is None:
            if max_d is None:
                max_d = max_d_theo_max_ratio * pep.theo_max_d
            inf_tp = Timepoint(pep, np.inf, max_d, np.nan)
            pep.add_timepoint(inf_tp)
        else:
            print("np.inf tp already exists")

    else:
        pep.timepoints = [tp for tp in pep.timepoints if tp.deut_time != np.inf]
        if max_d is None:
            max_d = max_d_theo_max_ratio * pep.theo_max_d
        inf_tp = Timepoint(pep, np.inf, max_d, np.nan)
        pep.add_timepoint(inf_tp)
    return pep



def calculate_coverages(hdxms_datas, state_name):
    states = [data.get_state(state_name) for data in hdxms_datas]
    coverage = np.zeros(len(states[0].hdxms_data.protein_sequence))
    for state in states:
        for pep in state.peptides:
            coverage[pep.start - 1 : pep.end] += 1
    return coverage


def generate_bayesian_hdx_script(
    exp_names,
    protein_sequence,
    state_name,
    base_directory,
    making_chunks=False,
    if_original_bayesian_hdx=False,
    install_dir=None,
    pH=None,
    temperature=None,
    saturation=None,
    rerun_num=3,
    extreme_value_prior=False,
    structural_prior=False,
):
    """
    Generates a script for the Bayesian HDX analysis based on the provided parameters and a template file.

    :param raw_spectra_paths: List of paths to raw spectra.
    :param protein_sequence: The protein sequence.
    :param protein_state: The state of the protein.
    :param base_directory: The base directory for the data.
    :param template_file: Path to the template file.
    :return: A string containing the generated script.
    """
    # Read the template from the file
    pigeon_feather_path = os.path.dirname(os.path.abspath(__file__))
    
    if making_chunks:
        with open(f"{pigeon_feather_path}/lib/run_bayesian_hdx_template_chunks2.txt", "r") as file:
            script_template = file.read()
            chunk_size = optimal_chunks(len(protein_sequence))[1]
    else:
        if if_original_bayesian_hdx:
            with open(f"{pigeon_feather_path}/lib/run_original_bayesian_hdx_template.txt", "r") as file:
                script_template = file.read()
        else:
            with open(f"{pigeon_feather_path}lib/run_bayesian_hdx_template.txt", "r") as file:
                script_template = file.read()

    if install_dir is None and if_original_bayesian_hdx:
        raise ValueError("Please provide the install directory for the original Bayesian HDX analysis.")
    
    if pH is None or temperature is None or saturation is None:
        raise ValueError("Please provide the pH, temperature, and saturation for the Bayesian HDX analysis.")

    if install_dir is None:
        # Generate the script using the template
        generated_script = script_template.format(
            protein_state=state_name,
            sequence=protein_sequence,
            exp_names=exp_names,
            base_directory=base_directory,
            pH=pH,
            temperature=temperature,
            saturation=saturation,
            rerun_num=rerun_num,
            chunk_size=chunk_size
        )
    else:
        # Generate the script using the template
        generated_script = script_template.format(
            protein_state=state_name,
            sequence=protein_sequence,
            exp_names=exp_names,
            base_directory=base_directory,
            install_dir=install_dir,
            pH=pH,
            temperature=temperature,
            saturation=saturation,
            rerun_num=rerun_num
        )

    def add_prior_lines(extreme_value_prior, structural_prior, state_name):
        prior_lines = ""
        if extreme_value_prior:
            prior_lines += f"""    extreme_prior = scoring.ExtremeValuePrior(np.load('extreme_prior_{state_name}.npy', allow_pickle=True), scale=0.5)
    state.scoring_function.add_prior(extreme_prior)
"""
        if structural_prior:
            prior_lines += f"""    structural_prior = scoring.ExtremeValuePrior(np.load('structural_prior_{state_name}.npy', allow_pickle=True), scale=0.75)
    state.scoring_function.add_prior(structural_prior)
"""
        return prior_lines

    prior_lines = add_prior_lines(extreme_value_prior, structural_prior, state_name)
    generated_script = generated_script.replace("    # priors", prior_lines)

    return generated_script


def calculate_simple_deuterium_incorporation(rate, time):
    # Calculates the deuterium incorporation for a log(kex)
    # at time = t (seconds) assuming full saturation
    return 1 - np.exp(-1 * (10**rate) * time)


def pdb2seq(pdb_file):
    import warnings
    from Bio import SeqIO
    from Bio import BiopythonWarning

    # Suppress all Biopython-specific warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", BiopythonWarning)
        records = list(SeqIO.parse(pdb_file, "pdb-atom"))
        return str(records[0].seq)


def find_peptide(seq, peptide):
    start_index = seq.find(peptide)
    if start_index == -1:
        return (-1, -1)
    end_index = start_index + len(peptide) - 1
    return (start_index, end_index)



def get_max_deuteration(pep):
    long_tps = [tp for tp in pep.timepoints if tp.deut_time != np.inf and tp.deut_time > 3600]
    if len(long_tps) == 0:
        return 100
    return max([tp.d_percent for tp in long_tps])


def get_min_deuteration(pep):
    shor_tps = [tp for tp in pep.timepoints if tp.deut_time != 0.0  and tp.deut_time < 100]
    if len(shor_tps) == 0:
        return 0
    else:
        return min([tp.d_percent for tp in shor_tps])


def get_pf_categories(states, protein_sequence):
    state_names = set([state.state_name for state in states])
    if len(state_names) > 1:
        raise ValueError("More than one state in the list")

    all_peptides_in_state = [pep for state in states for pep in state.peptides if pep.note is None]

    low_ex_peps = [(pep.start, pep.end) for pep in all_peptides_in_state if get_max_deuteration(pep) < 20 and pep.note is None]
    high_ex_peps = [(pep.start, pep.end) for pep in all_peptides_in_state if get_min_deuteration(pep) > 100 and pep.note is None]

    low_ex_res = list(set([j for i in low_ex_peps for j in range(i[0]-1, i[1])]))
    high_ex_res = list(set([j for i in high_ex_peps for j in range(i[0]-1, i[1])]))

    # print(set(low_ex_res).intersection(set(high_ex_res)))

    pf_categories = []
    for i in range(len(protein_sequence)):
        if i in low_ex_res and i not in high_ex_res:
            pf_categories.append(-1)
        elif i in high_ex_res and i not in low_ex_res:
            pf_categories.append(1)
        else:
            pf_categories.append(0)

    # print(f"Low exchange residues: {len(low_ex_res)}")
    # print(f"High exchange residues: {len(high_ex_res)}")

    pf_categories = np.array(pf_categories)
    return pf_categories


def generate_extreme_value_prior(hdx_datas, out_path):
    protein_sequence = hdx_datas[0].protein_sequence
    state_names = set([state.state_name for data in hdx_datas for state in data.states])

    for state_name in state_names:
        states = [state for data in hdx_datas for state in data.states if state.state_name == state_name]
        pf_categories = get_pf_categories(states, protein_sequence)
        pf_categories.dump(f"{out_path}/extreme_prior_{state_name}.npy")



def generate_structural_prior(protein_sequence, pdb_file, out_path=None, state_name=None):
    """
    Generate a structural prior for a state based on the pdb file
    pdb_file: need to be solvated first
    """

    from pigeon_feather.tools import pdb2seq, find_peptide

    pdb_sequence = pdb2seq(pdb_file)
    
    a_middle_pep = protein_sequence[80:90]
    pdb_start, pdb_end= find_peptide(pdb_sequence, a_middle_pep)
    index_offset = 80 - pdb_start

    if_exposed_array = get_if_exposed(pdb_file)
    
    pf_categories = []
    for i in range(len(protein_sequence)):
        res_i_in_pdb = i - index_offset
        if res_i_in_pdb >= 0 and res_i_in_pdb < len(if_exposed_array):
            if if_exposed_array[res_i_in_pdb] == 1:
                'exposed, high exchange'
                pf_categories.append(1)
            else:
                pf_categories.append(0)
        else:
            pf_categories.append(0)

    pf_categories = np.array(pf_categories)

    if out_path is not None:
        pf_categories.dump(f"{out_path}/structural_prior_{state_name}.npy")

    return pf_categories


def get_if_exposed(pdb_file,):
    
    import warnings
    import MDAnalysis
    from MDAnalysis.analysis.hydrogenbonds.hbond_analysis import HydrogenBondAnalysis as HBA
    
    warnings.filterwarnings("ignore")
    
    u = MDAnalysis.Universe(pdb_file,)
    
    protein = u.select_atoms('protein')

    if_exposed = []
    for res in protein.residues:

        hbonds = HBA(universe=u, d_a_cutoff=3.5, d_h_a_angle_cutoff=30)
        hbonds.donors_sel = f'protein and name N and resid {res.resid}'
        # caution, in pdb file generate by mdtraj, HNs are renamed to H
        
        selected_hydrogens = res.atoms.select_atoms(f'protein and name H* and resid {res.resid}')
        if selected_hydrogens.n_atoms == 0:
            raise ValueError(f'No amide hydrogen found for residue {res.resid}')

        hbonds.hydrogens_sel = 'protein and name H*'
        hbonds.acceptors_sel = 'name O* or name N*'
        hbonds.run()
        
        if hbonds.results.hbonds.size == 0:
            if_exposed.append(1)
        else:
            if_exposed.append(0)
        
    return np.array(if_exposed)


def optimal_chunks(total_size):
    '''
    calculate the optimal number of chunks and chunk size for a given total size
    '''
    residual = []
    num_chunks = []
    chunk_size = []

    for i in range(80, 121, 10): 
        num = math.ceil(total_size / i)
        res = num * i - total_size

        if res < 0:
            continue

        num_chunks.append(num)
        chunk_size.append(i)
        residual.append(res)
    
    # Find the best chunk size with the minimum number of chunks and least residual
    min_residual = min(residual)
    best_indices = [i for i, x in enumerate(residual) if x == min_residual]
    min_chunks = min([num_chunks[i] for i in best_indices])
    best_index = num_chunks.index(min_chunks)
    
    return num_chunks[best_index], chunk_size[best_index]
