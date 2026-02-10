"""
   funcs for refine spectra

"""
from pigeon_feather import tools
import itertools
import numpy as np
import pandas as pd
import pyopenms as oms
from itertools import groupby
from operator import attrgetter
from copy import deepcopy
import math
from scipy.signal import medfilt
import warnings
warnings.filterwarnings("ignore")



def get_theoretical_isotope_distribution(timepoint):
    """
    return the mono charge state iso
    """    
    # Create an AASequence object from the peptide sequence
    peptide_obj = oms.AASequence.fromString(timepoint.peptide.identifier.split(' ')[1])

    # Get the empirical formula of the peptide
    formula = peptide_obj.getFormula()
    isotope_generator = oms.CoarseIsotopePatternGenerator(len(timepoint.peptide.identifier.split(' ')[1]))
    isotope_distribution = isotope_generator.run(formula)

    # Get the monoisotopic mass
    mono_mz = peptide_obj.getMonoWeight() 

    # Adjust the m/z values relative to the monoisotopic mass and considering the charge state
    theo_mz = np.array([(iso.getMZ()) - mono_mz for iso in isotope_distribution.getContainer()])
    theo_intensity = np.array([iso.getIntensity() for iso in isotope_distribution.getContainer()])

    #plt.stem(theo_mz, theo_intensity, linefmt='-', markerfmt=' ', basefmt=" ",)
    #df_theo = pd.DataFrame({'m/z': theo_mz, 'Intensity': theo_intensity})
    return theo_mz, theo_intensity



def ms_smoothing(spectrum):
    exp = oms.MSExperiment()
    gf = oms.GaussFilter()
    param = gf.getParameters()
    param.setValue("gaussian_width", 0.001)  # Adjust the width as needed
    gf.setParameters(param)

    exp.addSpectrum(spectrum)
    gf.filterExperiment(exp)
    spectrum = exp[0]  # Get back the modified spectrum

    return spectrum


def calculate_SN(spectrum: oms.MSSpectrum, mz_value: float, window: float = 1.5):
    # Extract peaks
    mz, intensity = spectrum.get_peaks()
    
    # Find the peak intensity at the mz_value
    peak_intensity = intensity[np.abs(mz - mz_value) < window].max()

    # Estimate the noise level as the mean intensity in the window
    noise_level = np.mean(intensity[np.abs(mz - mz_value) < window])
    
    # Calculate the signal-to-noise ratio
    sn_ratio = peak_intensity / noise_level
    
    return sn_ratio


def MAD_filter(df, threshold_factor=20, fill_value=1e-10):

    # Calculate the median and MAD of the specified column
    median_value = np.median(df['Intensity'])
    mad = np.median(np.abs(df['Intensity'] - median_value))
    threshold = threshold_factor * mad
    df.loc[np.abs(df['Intensity'] - median_value) > threshold, 'Intensity'] = fill_value
    
    return df


def find_peak_data(data, peaks):
    peak_regions = []

    for peak in peaks:
        # Find start of the peak
        start = peak
        while start > 0 and data[start] >= data[start - 1]:
            start -= 1

        # Find end of the peak
        end = peak
        while end < len(data) - 1 and data[end] >= data[end + 1]:
            end += 1
            # Break if the signal starts increasing before reaching zero
            if end < len(data) - 1 and data[end] < data[end + 1]:
                break

        # Append the start and end index, including the peak data
        peak_region = data[start:end+1]
        peak_regions.append((start, end, peak_region))

    return peak_regions


    
    
def fill_missing_mz(df, default_intensity=1e-10, default_sn_ratio=0):
    
    df['m/z'] = df['m/z'].astype(int)
    min_mz = 0
    max_mz = df['m/z'].max()

    # Create a DataFrame with the complete range of m/z values
    full_range_df = pd.DataFrame({'m/z': range(min_mz, max_mz + 1)})

    merged_df = full_range_df.merge(df, on='m/z', how='left').fillna({
        'Intensity': default_intensity,
        'sn_ratio': default_sn_ratio
    })

    # Return the merged DataFrame sorted by m/z
    return merged_df.sort_values('m/z')


def get_theo_ms(timepoint):
    
    t0_theo = get_theoretical_isotope_distribution(timepoint)[1]
    tp_deut_iso_theo = convolute_deuterated_distribution(t0_theo, timepoint.num_d)
    
    return tp_deut_iso_theo



def filter_outliers(peaks, highest_peak_index):

    # Initialize a list to store the indices of outliers
    outliers = []
    
    # Check to the left of the highest peak
    for i in range(highest_peak_index - 1, -1, -1):
        if (peaks[i][1] - peaks[i + 1][1])/peaks[i + 1][1] > 0.1:
            outliers.append(i)
    
    # Check to the right of the highest peak
    for i in range(highest_peak_index + 1, len(peaks)):
        if (peaks[i][1] - peaks[i - 1][1])/peaks[i - 1][1] > 0.1:
            outliers.append(i)
    
    # # Filter out the outliers
    # filtered_peaks = [peak for i, peak in enumerate(peaks) if i not in outliers]

    # indexes
    filtered_peaks = np.array([i for i, peak in enumerate(peaks) if i not in outliers])
    
    return filtered_peaks


def get_isotope_envelope(timepoint, add_sn_ratio_to_tp=False, save_match=False):
    #print(f'{replicate.peptide.sequence} {replicate.timepoint.time} {replicate.charge_state}')
    
    # Step 1: create the spectrum object
    seq = oms.AASequence.fromString(timepoint.peptide.identifier.split(' ')[1]) 
    mfull = seq.getMonoWeight()
    mz_values = timepoint.raw_ms['m/z'].values*timepoint.charge_state - (mfull + 1.007276466812 * timepoint.charge_state)
    intensities = timepoint.raw_ms['Intensity'].values 
    spectrum = oms.MSSpectrum()
    spectrum.set_peaks((mz_values, intensities))

    # Step 2: smoothing 
    spectrum = ms_smoothing(spectrum)

    # Step 3: Centroiding
    picker = oms.PeakPickerHiRes()
    picked_spectrum = oms.MSSpectrum()
    picker.pick(spectrum, picked_spectrum)

    mz_picked, intensity_picked = picked_spectrum.get_peaks()
    df_picked = pd.DataFrame({'m/z': mz_picked, 'Intensity': intensity_picked})
    # filter out low intensity signals
    df_picked = df_picked[df_picked['Intensity'] > 1e3]

    tp_deut_iso_theo = get_theo_ms(timepoint)
    

    # # if bad drop replicate
    if df_picked.empty:
        #replicate.timepoint.replicates.remove(replicate)
        #print(f'bad replicate droped: {timepoint.peptide.sequence} {timepoint.deut_time} {timepoint.charge_state}')
        return None

    
    # Step 2: Select highest near integer
    max_mz = int(df_picked['m/z'].max())
    max_ins = df_picked['Intensity'].max()
    selected_mz = []
    selected_intensity = []
    sn_ratio = []
    
    
    #mz_start, mz_end = np.where(tp_deut_iso_theo> 0.01)[0][0], np.where(tp_deut_iso_theo> 0.01)[0][-1]
    mz_start = 0
    mz_end = np.where(get_theoretical_isotope_distribution(timepoint)[1] > 1e-3)[0][-1] + math.ceil(timepoint.num_d)
    #print(mz_start, mz_end)
    for mz_int in range(mz_start, mz_end+1):
        mask = (df_picked['m/z'] >= mz_int*1.00866491588-0.1) & (df_picked['m/z'] <= mz_int*1.00866491588+0.1)
        peaks_in_range = df_picked[mask]        
        if not peaks_in_range.empty:
            closest_peak = peaks_in_range.loc[peaks_in_range['Intensity'].idxmax()]
            # intensity_diff = abs(peaks_in_range['Intensity'].values - tp_deut_iso_theo[mz_int]*max_ins/tp_deut_iso_theo.max())
            # closest_peak_index = np.argmin(intensity_diff)
            # closest_peak = peaks_in_range.iloc[closest_peak_index]
            selected_mz.append(closest_peak['m/z'])
            selected_intensity.append(closest_peak['Intensity'])
            #print(max_peak)
            sn_ratio.append(calculate_SN(spectrum, closest_peak['m/z']))

    
    if selected_intensity == []:
        return None

    selected_mz = np.array(selected_mz)
    selected_intensity = np.array(selected_intensity)
    sn_ratio = np.array(sn_ratio)

    smoothed_selected_intensity = medfilt(selected_intensity, kernel_size=3)
    smoothed_highest_index = np.argmax(smoothed_selected_intensity)
    highest_index = np.where(selected_intensity == max(selected_intensity[max(smoothed_highest_index-3,0):min(smoothed_highest_index+3+1, len(selected_intensity))]))[0][0]


    selected_peak_indexes_1 = filter_outliers(np.stack([selected_mz, selected_intensity], axis=1),
                                            highest_index) 

    filtered_mz = selected_mz[selected_peak_indexes_1]
    filtered_intensity = selected_intensity[selected_peak_indexes_1]
    filtered_sn_ratio = sn_ratio[selected_peak_indexes_1]


    selected_peak_indexes_2 = np.arange(max(np.argmax(filtered_intensity)-5, 0), 
                                        min(np.argmax(filtered_intensity)+5+1, len(filtered_intensity))) 


    selected_df = pd.DataFrame({
        'm/z': filtered_mz[selected_peak_indexes_2],
        'Intensity': filtered_intensity[selected_peak_indexes_2],
        'sn_ratio': filtered_sn_ratio[selected_peak_indexes_2]
    })

    # round m/z to integer
    selected_df['m/z_raw'] = selected_df['m/z'].values
    selected_df['m/z'] = selected_df['m/z'].apply(round)

    # if bad drop replicate
    if selected_df.empty:
       #replicate.timepoint.replicates.remove(replicate)
       #print(f'bad replicate droped: {timepoint.peptide.sequence} {timepoint.deut_time} {timepoint.charge_state}')
       return None

    if add_sn_ratio_to_tp:
        timepoint.sn_ratio = selected_df['sn_ratio'].mean()
        
        
    # filter the peak pickings
    #selected_df = filter_peak_pickings(selected_df, timepoint)
    
    # fill the missing m/z
    selected_df = fill_missing_mz(selected_df)
    
    #selected_df = MAD_filter(selected_df)
    selected_df['Intensity'] /= selected_df['Intensity'].sum()
    
    
    #    mz_values = timepoint.raw_ms['m/z'].values*timepoint.charge_state - (mfull + 1.007276466812 * timepoint.charge_state)
    if save_match:
        timepoint.match = (selected_df['m/z_raw'].values + (mfull + 1.007276466812 * timepoint.charge_state)) / timepoint.charge_state
        timepoint.mono_mz = mfull
        
        # extract fine structure (uncentroided)
        timepoint.raw_ms_fine_structure = []
        raw_mz_all = timepoint.raw_ms['m/z'].values
        raw_int_all = timepoint.raw_ms['Intensity'].values

        for match_i in timepoint.match:
            mask = (raw_mz_all >= match_i - 0.1) & (raw_mz_all <= match_i + 0.1)
            indices = np.where(mask)[0]
            if len(indices) == 0:
                timepoint.raw_ms_fine_structure.append((np.array([np.nan]), np.array([np.nan])))
                continue

            local_int = raw_int_all[indices]
            center_local = np.argmax(local_int)

            # walk left: monotonically decreasing
            left = center_local
            while left > 0 and local_int[left-1] > 0 and local_int[left - 1] <= local_int[left] * 1.05:
                left -= 1

            # walk right: monotonically decreasing
            right = center_local
            while right < len(local_int) - 1 and local_int[right+1] > 0 and local_int[right + 1] <= local_int[right] * 1.05:
                right += 1

            peak_indices = indices[left:right + 1]
            timepoint.raw_ms_fine_structure.append(
                (raw_mz_all[peak_indices], raw_int_all[peak_indices])
            )
            # timepoint.raw_ms_fine_structure.append(
            #     list(zip(raw_mz_all[peak_indices], raw_int_all[peak_indices]))
            # )
        
    return selected_df



def flat_tp_list(tp_lists):
    return [tp for tp_list in tp_lists for tp in tp_list]



def get_large_error_tps(state, threshold=1.5):
    '''
    error across different charge states
    '''
    timepoints_list = []
    for pep in state.peptides:
        grouped_timepoints = [list(v) for k, v in groupby([tp for tp in pep.timepoints if tp.deut_time != np.inf], key=attrgetter('deut_time'))]
        for grouped_i in grouped_timepoints:
            timepoints_list.append(grouped_i)
    
    error_lsit = []
    large_error_list = []
    for tp_list in timepoints_list:
        if len(tp_list) == 1:
            continue
        if len(set([tp.deut_time for tp in tp_list])) != 1:
            continue    
        tp_combinations = list(itertools.combinations(tp_list, 2))
        abs_error =  np.average([tools.get_sum_ae(com[0].isotope_envelope, com[1].isotope_envelope) for com in tp_combinations])
        error_lsit.append(abs_error)
        if abs_error > threshold:
            #idf = get_identifer(rep_list[0].peptide)
            large_error_list.append(tp_list)
            #print(idf, abs_error)

    #flat_large_error_list = [rep for rep_list in large_error_list for rep in rep_list]

    return large_error_list



def find_low_intensity_tps(timepoints_list, threshold=10):

    low_intens_tps = []
    for tp_list in timepoints_list:
        max_intenstiy_tp = max(tp_list, key=lambda tp: tp.raw_ms['Intensity'].max())
        for tp in tp_list:
            intens_ratio = max_intenstiy_tp.raw_ms['Intensity'].max() / tp.raw_ms['Intensity'].max()
            if intens_ratio > threshold:
                #print(intens_ratio)
                low_intens_tps.append(tp)
    
    return low_intens_tps



def convolute_deuterated_distribution(non_deuterated, centroid_deuteration, sigma=0.5):

    # Creating a Gaussian distribution
    x = np.arange(len(non_deuterated) + centroid_deuteration)
    gaussian = np.exp(-(x - centroid_deuteration)**2 / (2 * sigma**2))
    gaussian /= np.sum(gaussian)
    
    # Convolving the non-deuterated distribution with the Gaussian
    convoluted = np.convolve(non_deuterated, gaussian,)

    return convoluted


def filter_by_thoe_ms(timepoint, add_avg_intensity=True):

    
    theoretical_peaks = get_theo_ms(timepoint)
    bool_array = theoretical_peaks < 0.05
    index_of_first_false = len(bool_array) - np.argmax(bool_array[::-1] == False)
    filtered = timepoint.isotope_envelope[:index_of_first_false+1]
    
    #filtered = deepcopy(timepoint.isotope_envelope)
    #filtered[bool_array] = 1e-10
    
    #timepoint.isotope_envelope = filtered/np.sum(filtered)
    return filtered/np.sum(filtered)


from scipy.signal import find_peaks

def filter_peak_pickings(data):

    data_x = np.arange(len(data))
    data_y = deepcopy(data)
    
    # def centroid_of_peak(peak_i, data_x, data_y):
    #     start, end, region = peak_i
    #     norm_ins = region/np.sum(region)
    #     mz = data_x[start:end+1]
    #     return norm_ins @ mz
    
    # Find peaks
    peaks, _ = find_peaks(data_y, height=0, prominence=0.01, distance=3)
    if peaks.size == 0:
        return peak_df
    
    # Get peak data
    peak_data = find_peak_data(data_y, peaks)
    #best_peak = min(peak_data, key=lambda x: centroid_of_peak(x, data_x, data_y)-tp.num_d)
    best_peak = max(peak_data, key=lambda x: len(x[2]))
    
    data_y[:best_peak[0]] = 1e-10
    data_y[best_peak[1]+1:] = 1e-10
    data_y = data_y/np.sum(data_y)
    
    
    return data_y


def refine_large_error_reps(state):
    
    # remove low intensity timepoints
    low_intens_tps = find_low_intensity_tps(get_large_error_tps(state, threshold=0.3), threshold=5)
    tools.remove_tps_from_state(low_intens_tps, state)
    
    # remove timepoints with large error across different charge states
    flat_large_error_list = flat_tp_list(get_large_error_tps(state, threshold=0.5)) 
    tools.remove_tps_from_state(flat_large_error_list, state)

    # remove timepoints with large deviation from HDExaminer centroided #D
    all_tps = [tp for pep in state.peptides for tp in pep.timepoints if tp.deut_time != np.inf]
    large_diviation_tps = [tp for tp in all_tps if abs(tp.num_d - tools.get_num_d_from_iso(tp)) > 0.5]
    tools.remove_tps_from_state(large_diviation_tps, state)


def calculate_isoenv_std(hdxms_data_list:list):
    '''
    calculate the standard deviation of isotope envelopes for each peptide
    input: a list of hdxms_data
    '''
    all_peptides = [peptide for data in hdxms_data_list for state in data.states for peptide in state.peptides]
    for k, v in tools.group_by_attributes(all_peptides, ['protein_state.state_name', 'identifier',]).items():
        tps = [tp for pep in v for tp in pep.timepoints if tp.deut_time != np.inf]
        for tp_k, tp_v in tools.group_by_attributes(tps, ['deut_time']).items():
            iso_list = [tp.isotope_envelope for tp in tp_v]
            max_len = max([len(l) for l in iso_list])
            padded_iso_list = [tools.custom_pad(iso, max_len) for iso in iso_list]
            iso_std = np.std(padded_iso_list, axis=0)
            
            num_d_list = [tp.num_d for tp in tp_v]
            num_d_std = np.std(num_d_list)

            percent_d_list = [tp.percent_d for tp in tp_v]
            percent_d_std = np.std(percent_d_list)
            for tp_i in tp_v:
                tp_i.isotope_envelope = tools.custom_pad(tp_i.isotope_envelope, max_len)
                tp_i.isotope_envelope_std = iso_std
                tp_i.stddev = num_d_std
                tp_i.percent_d_std = percent_d_std

    print('standard deviation of isotope envelopes calculated')
        




    


    
