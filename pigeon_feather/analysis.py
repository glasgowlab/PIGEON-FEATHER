
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.cluster import KMeans
from hdxrate import k_int_from_sequence
from pigeon_feather.tools import calculate_simple_deuterium_incorporation, event_probabilities, get_sum_ae
from pigeon_feather.plot import UptakePlot
from matplotlib.lines import Line2D
from sklearn.metrics import mean_squared_error
import MDAnalysis
import warnings


class Analysis:
    def __init__(self, protein_state, temperature=None, pH=None, global_dg_unfolding=None):
        '''
        :param protein_state: list of protein state objects
        :param temperature: temperature in K
        :param pH: pH
        :param global_dg_unfolding: global dg unfolding in kJ/mol
        '''

        if temperature is None or pH is None:
            raise ValueError('temperature and pH are required')
        
        if not isinstance(protein_state, list):
            self.protein_state = [protein_state]
        else:
            self.protein_state = protein_state

        initial_sliding_windows = self._get_initial_sliding_windows()
        self.__minimize_overlap_by_single_res(initial_sliding_windows)
        self.__minimize_overlap_by_adjacent_windows(initial_sliding_windows)
        self.maximum_resolution_limits = self.__find_continuous_blocks(initial_sliding_windows)
        self.protein_sequence = self.protein_state[0].hdxms_data.protein_sequence
        self.coverage = self.calculate_coverages()
        self.saturation = self.protein_state[0].hdxms_data.saturation
        self.temperature = temperature
        self.pH = pH
        self.log_k_init = np.log10(k_int_from_sequence(self.protein_sequence , temperature, pH, d_percentage=self.saturation*100))
        self.global_dg_unfolding = global_dg_unfolding
        self._calculate_valid_log_kex_range()
        self._calculate_log_kex_threshold_by_global_dg_unfolding()
        
    def find_mini_overlap(self, peps_covering,):

        '''
        find the mini overlap give peps covering a residue
        '''

        combinations_list = list(combinations(peps_covering, 2))

        overlaps = []

        for comb in combinations_list:
            #get the overlap
            overlap_start = max(comb[0].start, comb[1].start)
            overlap_end = min(comb[0].end, comb[1].end)
            overlap = (overlap_start, overlap_end)
            #if res_i_start in range(overlap_start, overlap_end):
            overlaps.append(overlap)

        min_overlap = min(overlaps, key=lambda x: x[1]-x[0])

        return min_overlap
    

    def _get_initial_sliding_windows(self):
        
        '''
        get the initial sliding windows for each residue
        input: protein_state
        '''

        all_peptides = [pep for state in self.protein_state for pep in state.peptides ]

        min_covering_pep_dict = {}
        for i in range(len(self.protein_state[0].hdxms_data.protein_sequence)):

            #! all_peptides must contain substracted peptides
            peps_covering = [pep for pep in all_peptides if pep.start <= i+1 <= pep.end]
            
            #if len(peps_covering) == 0 or self.protein_state[0].hdxms_data.protein_sequence[i] == 'P':
            if len(peps_covering) == 0:
                continue
            elif len(peps_covering) == 1:
                min_covering_pep = (peps_covering[0].start, peps_covering[0].end)
            else:
                min_covering_pep = self.find_mini_overlap(peps_covering)
            
            min_covering_pep_dict[i+1] = min_covering_pep
            #print(i+1, min_covering_pep)

        return min_covering_pep_dict
    


    def __minimize_overlap_by_single_res(self, my_dict):
        changes_made = True

        while changes_made:
            changes_made = False
            single_res = [k for k, v in my_dict.items() if v[0] == v[1]]

            for key, value in list(my_dict.items()):
                v0, v1 = value
                original_value = (v0, v1)

                if (v0 in single_res or v1 in single_res) and v0 != v1:
                    if v0 in single_res:
                        v0 += 1
                    if v1 in single_res:
                        v1 -= 1

                    if v0 > v1:
                        v0, v1 = v1, v0

                    my_dict[key] = (v0, v1)

                if my_dict[key] != original_value:
                    changes_made = True


    def __minimize_overlap_by_adjacent_windows(self, my_dict):
        changes_made = True

        while changes_made:
            changes_made = False
            sorted_keys = sorted(my_dict.keys())

            # Adjust overlapping ranges
            for i in range(len(sorted_keys)):
                key = sorted_keys[i]
                v0, v1 = my_dict[key]

                # Check and adjust overlaps with the next tuple
                if i < len(sorted_keys) - 1:
                    next_key = sorted_keys[i + 1]
                    next_v0, next_v1 = my_dict[next_key]
                    if v1 >= next_v0:
                        # Adjust the range of the next tuple to remove overlap
                        new_v0 = v1 + 1
                        if new_v0 <= next_v1:
                            my_dict[next_key] = (new_v0, next_v1)
                            changes_made = True

                # Check and adjust overlaps with the previous tuple
                if i > 0:
                    prev_key = sorted_keys[i - 1]
                    prev_v0, prev_v1 = my_dict[prev_key]
                    if prev_v1 >= v0:
                        # Adjust the range of the current tuple to remove overlap
                        new_v1 = v0 - 1
                        if new_v1 >= prev_v0:
                            my_dict[prev_key] = (prev_v0, new_v1)
                            changes_made = True

        return my_dict




    def __find_continuous_blocks(self, input_dict):
        # Sort the keys
        sorted_keys = sorted(input_dict.keys())

        # Initialize the first block
        blocks = []
        current_block = [sorted_keys[0]]
        current_range = input_dict[sorted_keys[0]]

        # Function to check if two ranges overlap
        def ranges_overlap(range1, range2):
            return range1[1] >= range2[0] and range2[1] >= range1[0]

        # Iterate through the sorted keys
        for key in sorted_keys[1:]:
            key_range = input_dict[key]

            # Check if the current key overlaps with the current block's range
            if ranges_overlap(current_range, key_range):
                current_block.append(key)
                # Update the block's range to the overlapping portion
                current_range = (max(current_range[0], key_range[0]), min(current_range[1], key_range[1]))
            else:
                # Save the current block and start a new one
                blocks.append((current_block, current_range))
                current_block = [key]
                current_range = key_range

        # Add the last block
        blocks.append((current_block, current_range))

        # make it a dictionary
        blocks = {res: block[1] for block in blocks for res in block[0]}


        return blocks


    def _convert_logP_to_log_kex(self, df):
        bayesian_hdx_df_log_kex = pd.DataFrame()
        for col in df.columns:
            bayesian_hdx_df_log_kex[col] = self.log_k_init[col] - df[col]

        bayesian_hdx_df_log_kex *= -1 # convert to -log_kex
        self.bayesian_hdx_df_log_kex = bayesian_hdx_df_log_kex
        
    
    def _calculate_valid_log_kex_range(self, threshold=0.001):
        '''
        get the valid log_kex range by timewindow
        '''
        all_tps = [tp for state in self.protein_state for pep in state.peptides for tp in pep.timepoints if tp.deut_time != np.inf and tp.deut_time != 0]
        max_time = max([tp.deut_time for tp in all_tps])
        min_time = min([tp.deut_time for tp in all_tps])
        
        def single_exp_reverse(y, t):
            return -np.log(1-y) / t
        
        log_kex_min = np.log10(single_exp_reverse(threshold, max_time))
        log_kex_max = np.log10(single_exp_reverse(1-threshold, min_time))

        self.log_kex_min = log_kex_min
        self.log_kex_max = log_kex_max
        
    def _calculate_log_kex_threshold_by_global_dg_unfolding(self):
        '''
        calculate the log_kex threshold by global dg unfolding
        '''
        if self.global_dg_unfolding is not None:
            log_P = np.log10(np.exp(self.global_dg_unfolding*1000 / (8.314*self.temperature)))
            log_kex = self.log_k_init - log_P
            self._log_kex_min_by_global_dg_unfolding = log_kex
            self._log_P_max_by_global_dg_unfolding = log_P
        else:
            self._log_kex_min_by_global_dg_unfolding = None
            self._log_P_max_by_global_dg_unfolding = None


    def load_bayesian_hdx_oupt(self, bayesian_hdx_data_file, N=50):
        pof = Bayesian_hdx_ParseOutputFile(bayesian_hdx_data_file)
        self.bayesian_hdx_df = pof.get_best_scoring_models_pf_df(N=N)
        
        self._convert_logP_to_log_kex(self.bayesian_hdx_df)

        # run clustering
        self.clustering_results()
        

    def load_bayesian_hdx_oupt_chunks(self, state_name, chunk_size, chunk_num, bayesian_hdx_data_folder, run_num=1, N=50):

        df_list_all = []      
        df_be_list_all = []  
        df_sidechain_ex_list_all = []
        range_chunks = [(i*chunk_size+1, (i+1)*chunk_size) for i in range(chunk_num)]
        print(range_chunks)

        for run_index in range(1, run_num+1):
            df_list_i = []
            df_be_list_i = []
            df_sidechain_ex_list_i = []
            bayesian_hdx_data_files = sorted([os.path.join(bayesian_hdx_data_folder, f) for f in os.listdir(bayesian_hdx_data_folder) if f.endswith(f'{run_index}.dat') and state_name in f])
            print(f"Found {len(bayesian_hdx_data_files)} files for state {state_name}, run {run_index}")

            for i, chunk in enumerate(range_chunks):
                # if i >= len(bayesian_hdx_data_files):
                #     print(f"Error: Not enough files for chunk {i}. Only {len(bayesian_hdx_data_files)} files found.")
                #     break
                # print(f"Processing chunk {i+1}/{chunk_num}, file: {bayesian_hdx_data_files[i]}")

                pof = Bayesian_hdx_ParseOutputFile(bayesian_hdx_data_files[i])
                df = pof.get_best_scoring_models_pf_df(N=N)
                if isinstance(df, pd.DataFrame):
                    df_list_i.append(df.iloc[:, chunk[0]-1:chunk[1]]) 
                else:
                    df_list_i.append(df[0].iloc[:, chunk[0]-1:chunk[1]])
                    df_be_list_i.append(df[1])
                    if len(df) == 3:
                        df_sidechain_ex_list_i.append(df[2])
            if df_be_list_i != []:
                df_be_run_i = pd.concat(df_be_list_i, axis=1)
                df_be_run_i = df_be_run_i.groupby(by=lambda x: x, axis=1).mean()
                df_be_list_all.append(df_be_run_i)

            if df_sidechain_ex_list_i != []:
                df_sidechain_ex_run_i = pd.concat(df_sidechain_ex_list_i, axis=1)
                df_sidechain_ex_run_i = df_sidechain_ex_run_i.groupby(by=lambda x: x, axis=1).mean()
                df_sidechain_ex_list_all.append(df_sidechain_ex_run_i)

            df_run_i = pd.concat(df_list_i, axis=1)
            #df_run_i['run_index'] = run_index
            df_list_all.append(df_run_i)

        
        self.bayesian_hdx_df = pd.concat(df_list_all, axis=0)
        
        if df_be_list_all != []:
            self.bayesian_hdx_df_be = pd.concat(df_be_list_all, axis=0)

        if df_sidechain_ex_list_all != []:
            self.bayesian_hdx_df_sidechain_ex = pd.concat(df_sidechain_ex_list_all, axis=0)
        
        self._convert_logP_to_log_kex(self.bayesian_hdx_df)

        # run clustering
        self.clustering_results()

        # load the sampled sidechain exchange data
        self.load_sampled_sidechain_exchange()


    def load_sampled_sidechain_exchange(self):
        '''
        Load the sampled sidechain exchange data from the bayesian hdx output files
        '''

        # if not hasattr(self, 'bayesian_hdx_df_sidechain_ex'):
        #     warnings.warn('sidechain exchange data not found in the bayesian hdx output files')

        try:
            all_peps = [pep for state in self.protein_state for pep in state.peptides]
            bayesian_hdx_df_sidechain_ex = self.bayesian_hdx_df_sidechain_ex

            avg_sidechain_exchange = np.nanmean(bayesian_hdx_df_sidechain_ex.values.flatten())
        except:
            avg_sidechain_exchange = 0.0

        for pep in all_peps:
            try:
                #avg_sidechain_exchange = bayesian_hdx_df_sidechain_ex[pep.identifier].mean() # this be consider the saturation level
                pep.sidechain_exchange = avg_sidechain_exchange
            except Exception as e:
                pep.sidechain_exchange = 0.0
                print(e)

    def load_sampled_back_exchange(self):
        '''
        overwrite the back exchange with the sampled back exchange from the bayesian analysis

        :param protein_state: protein state object
        :param ana_obj: analysis object
        '''

        if not hasattr(self, 'bayesian_hdx_df_be'):
            raise Exception('back exchange data not found in the bayesian hdx output files')
        
        all_peps = [pep for state in self.protein_state for pep in state.peptides]
        bayesian_hdx_df_be = self.bayesian_hdx_df_be

        from pigeon_feather.tools import _add_max_d_to_pep

        for pep in all_peps:
            try:
                avg_be = bayesian_hdx_df_be[pep.identifier].mean() # this be consider the saturation level
                max_d_ratio = 1- avg_be*self.saturation

                _add_max_d_to_pep(pep, max_d_theo_max_ratio=max_d_ratio, force=True)
            except Exception as e:
                print(e)

    

    def clustering_results(self):
        
        results = Results(self)
        
        for k,v in self.maximum_resolution_limits.items():
            try:
                mini_pep = results.get_mini_pep(v[0], v[1])
            except:
                mini_pep = MiniPep(v[0], v[1])
                results.add_mini_pep(mini_pep)
                
            #print(k,v, clustering_a_mini_pep(k, v[0], v[1]))
            res = results.get_residue_by_resid(k)
            mini_pep.add_residue(res)
            mini_pep.set_clustering_results(self._clustering_a_mini_pep(k, v[0], v[1]))
            
        self.results_obj = results


        for res in results.residues:
            if hasattr(res, 'mini_pep'):
                res.clustering_results_logP = np.array([res.log_k_init + i for i in res.mini_pep.clustering_results_log_kex]) # log_kex is negative
                res.std_within_clusters_logP = res.mini_pep.std_within_clusters_log_kex


    def _apply_log_kex_range(self, data, resindex):
        '''
        data: 1D array of -log_kex
        apply the log_kex range to the data
        '''
        if self.global_dg_unfolding is not None:
            log_kch = self.log_k_init[resindex] 
            log_P = np.log10(np.exp(self.global_dg_unfolding*1000 / (8.314*self.temperature)))
            log_kex = log_kch - log_P
            # data = data[data <= -1*log_kex]
            data = np.clip(data, -10, -1*log_kex)
        
        # apply the log_kex range to the data
        #data = data[(data <= -1*self.log_kex_min) & (data >= -1*self.log_kex_max)]
        data = np.clip(data, -1*self.log_kex_max, -1*self.log_kex_min)
        return data
            

    def _clustering_a_mini_pep(self, k, v0, v1):

        
        # v0 and v1 are 1-based
        # Apply remove_outliers to each column and collect results
        #cleaned_data = [remove_outliers(self.bayesian_hdx_df[col]) for col in self.bayesian_hdx_df.iloc[:,v0-1:v1].columns] 
        cleaned_data = [remove_outliers(self.bayesian_hdx_df_log_kex[col]) for col in self.bayesian_hdx_df_log_kex.iloc[:,v0-1:v1].columns] 
        
        # apply the log_kex range to the data
        # cleaned_data = [self._apply_log_kex_range(col, range(v0-1, v1)[i]) for i, col in enumerate(cleaned_data)]
        
        initial_centers = np.array([np.mean(col[:20]) for col in cleaned_data if col.size !=0]).reshape(-1, 1)  
        num_clusters = initial_centers.shape[0]
        
        # add 0 if Proline in the mini_pep
        num_Ps = self.protein_sequence[v0-1:v1].count('P')
        
        if num_clusters == 0:
            if num_Ps > 0:
                return np.array([np.inf]*num_Ps), np.array([np.inf]*num_Ps), np.array([np.inf]*num_Ps)
            num_nan = v1 - v0 + 1
            return np.array([np.nan]*num_nan), np.array([np.nan]*num_nan), np.array([np.nan]*num_nan)
        
        pool_values = np.concatenate(cleaned_data).flatten()
        pool_values = pool_values[~np.isnan(pool_values)]
        
        if len(pool_values) != 0:
            k_cluster = KMeans(n_clusters=num_clusters, random_state=0, init=initial_centers, n_init='auto').fit(pool_values.reshape(-1, 1))
            sorted_indices = np.argsort(k_cluster.cluster_centers_.flatten())

            std_within_cluster = np.zeros(num_clusters)
            data_within_cluster = np.zeros(num_clusters, dtype=object)
            for i in range(num_clusters):
                cluster_mask = k_cluster.labels_ == i
                cluster_points = pool_values[cluster_mask]
                center = k_cluster.cluster_centers_[i]
                std_within_cluster[i] = np.std(cluster_points)
                data_within_cluster[i] = cluster_points
                




            cluster_centers = k_cluster.cluster_centers_.flatten()[sorted_indices]
            std_within_cluster = std_within_cluster[sorted_indices]
            data_within_cluster = data_within_cluster[sorted_indices]
            
            if num_Ps > 0:

                for seq_i, seq in enumerate(self.protein_sequence[v0-1:v1]):
                    if seq == 'P':
                        cluster_centers = np.insert(cluster_centers, seq_i, np.inf)
                        std_within_cluster = np.insert(std_within_cluster, seq_i, 0)
                        data_within_cluster = np.insert(data_within_cluster, seq_i, np.inf)

                return cluster_centers, std_within_cluster, data_within_cluster
            else:
                return cluster_centers, std_within_cluster, data_within_cluster

    
        

    def plot_kex_bar(self, ax=None, label=None, resolution_indicator_pos=15, seq_pos=17, color=None, 
                     show_coverage=True, show_seq=True, show_residue_range=None, show_out_of_range_indicator=True):
        #mini_peps_index = sorted(list(set([(v[0]-1, v[1]-1) for k, v in self.maximum_resolution_limits.items()])))
        # xx = self.bayesian_hdx_df.mean().values

        # for mini_pep in mini_peps_index:
        #     indexes = [i for i in range(mini_pep[0], mini_pep[1]+1) if np.isnan(self.bayesian_hdx_df.mean()).iloc[i] == False] 
        #     xx[indexes] = sorted(xx[indexes])

        #self.clustering_results()
        
        ax.axhline(-self.log_kex_min, color='gray', linestyle='--', lw=1, alpha=0.5)
        ax.axhline(-self.log_kex_max, color='gray', linestyle='--', lw=1, alpha=0.5)
            
        xx = np.array([res.resid for res in self.results_obj.residues if res.is_nan() == False])
        yy = np.concatenate([mini_pep.clustering_results_log_kex for mini_pep in self.results_obj.mini_peps])
        yy_std = np.concatenate([mini_pep.std_within_clusters_log_kex for mini_pep in self.results_obj.mini_peps])

        # padding all the residues
        padded_xx = np.arange(1, self.results_obj.n_residues+1)
        padded_yy = np.zeros(self.results_obj.n_residues)
        padded_yy_std = np.zeros(self.results_obj.n_residues)

        for i in range(len(xx)):
            padded_yy[xx[i]-1] = yy[i]
            padded_yy_std[xx[i]-1] = yy_std[i]
        

        if ax is None:
            fig, ax = plt.subplots(figsize=(20, 5))
        
        if color is None:
            sns.barplot(x=padded_xx, y=padded_yy, ax=ax, label=label, alpha=0.5)
        else:
            sns.barplot(x=padded_xx, y=padded_yy, ax=ax, label=label, alpha=0.5, color=color)
        #ax.errorbar(np.arange(len(padded_xx)), padded_yy, yerr=padded_yy_std, fmt='none', color='black')
        
        #manutally add the error bar
        for i in range(len(padded_xx)):
            ax.plot([i, i], [padded_yy[i]-padded_yy_std[i], padded_yy[i]+padded_yy_std[i]], color='gray', linewidth=3, alpha=0.7)
            
        # indicate the residues out of the log_kex range
        if show_out_of_range_indicator:
            mask = ((padded_yy >= -self.log_kex_min) | (padded_yy <= -self.log_kex_max)) & (padded_yy != 0)
            sns.scatterplot(x=padded_xx[mask]-1, y=padded_yy[mask]+0.5, ax=ax, color='red', s=100, alpha=0.7)
            
        # indicator of the log_kex range from global dg unfolding
        if self.global_dg_unfolding is not None:
            #mask = (padded_yy >= -self._log_kex_min_by_global_dg_unfolding) & (padded_yy != 0)
            mask = np.array([res.if_off_global_dg_unfolding for res in self.results_obj.residues])
            sns.scatterplot(x=padded_xx[mask]-1, y=padded_yy[mask]-0.5, ax=ax, color='blue', s=100, alpha=0.7)
        
        #ax.bar(padded_xx, padded_yy, yerr=padded_yy_std, alpha=0.5, label=label)
        #sns.barplot(x=range(1, len(xx)+1), y=xx, alpha=0.5, label='bayesian_hdx', ax=ax)
        
        for mini_pep in self.results_obj.mini_peps:
            width = mini_pep.end - mini_pep.start
            height = 0.5  # Height can be adjusted based on visual preference
            #rect = patches.Rectangle((mini_pep.start-1, ax.get_ylim()[1]-0.7), width, height, linewidth=1, edgecolor='k', facecolor='lightgrey')
            rect = patches.Rectangle((mini_pep.start-1, resolution_indicator_pos), width, height, linewidth=1, edgecolor='k', facecolor='lightgrey')
            ax.add_patch(rect)

        #plt.legend(loc='upper left')
        plt.tight_layout()
        plt.xlabel('Resid')
        plt.ylabel('-log_kex')
        plt.xticks(rotation=90);
        ax.set_ylim(0, 18)
        ax.set_xlim(-3, self.results_obj.n_residues+3)
        
        
        #add the coverage heatmap
        #coverage = self.calculate_coverages()
        if show_coverage:
            for i in range(len(self.coverage)):
                color_intensity = self.coverage[i] / self.coverage.max() # coverage.max()  # Normalizing the data for color intensity
                rect = patches.Rectangle((i-0.5, 16), 1, height, color=plt.cm.Blues(color_intensity))
                ax.add_patch(rect)
        #seq 
        if show_seq:
            for i in range(len(self.protein_sequence)):
                ax.text(i, seq_pos, self.protein_sequence[i], ha='center', va='center', fontsize=22)

        return ax
    
    #print(mini_pep[0]+1, mini_pep[1]+1)

    def calculate_coverages(self, n_fastamides=2):
        all_peptides = [pep for state in self.protein_state for pep in state.peptides if pep.note is None] # exp peptides only
        coverage = np.zeros(len(self.protein_state[0].hdxms_data.protein_sequence))
        for pep in all_peptides:
            coverage[pep.start-1:pep.end] += 1
            # num_tps = len([tp for tp in pep.timepoints if tp.deut_time != 0 and tp.deut_time != np.inf])
            # coverage[pep.start-1:pep.end] += num_tps
        return coverage


class Results(object):
    
    def __init__(self, analysis_object: Analysis):
        
        self.analysis_object = analysis_object
        
        self.protein_name = analysis_object.protein_state[0].hdxms_data.protein_name
        self.protein_sequence = analysis_object.protein_state[0].hdxms_data.protein_sequence
        self.protein_state = analysis_object.protein_state
        self.add_residues()
        self.mini_peps = []
    
    def add_residues(self):
        self.residues = []
        for i in range(len(self.protein_sequence)):
            res = Residue(i+1, i, self.protein_sequence[i])
            res.set_resluts_obj(self)
            self.residues.append(res)
            
    @property
    def n_residues(self):
        return len(self.residues)
            
    def get_residue_by_resid(self, resid):
        for res in self.residues:
            if res.resid == resid:
                return res
        raise Exception('residue not found')
    
    def get_residue_by_resindex(self, resindex):
        return self.get_residue_by_resid(resindex+1)
    
    
    def add_mini_pep(self, mini_pep):
        if mini_pep in self.mini_peps:
            print('mini_pep already exists')
        self.mini_peps.append(mini_pep)
    
     
    def get_mini_pep(self, start, end):
        for mini_pep in self.mini_peps:
            if mini_pep.start == start and mini_pep.end == end:
                return mini_pep
        raise Exception('mini_pep not found')
    
    @property
    def log_k_init(self):
        return self.analysis_object.log_k_init

       
class Residue(object):
    def __init__(self, resid, resindex, resname):
        self.resid = resid # 1-based
        self.resindex = resindex # 0-based
        self.resname = resname # one letter code
        
        
    def set_resluts_obj(self, resluts_obj):
        self.resluts_obj = resluts_obj

    def set_mini_pep(self, mini_pep):
        self.mini_pep = mini_pep

    def is_nan(self):
        #return self.resluts_obj.analysis_object.bayesian_hdx_df.mean().isna()[self.resindex]
        return self.resluts_obj.analysis_object.coverage[self.resindex] == 0
    
    @property
    def log_k_init(self):
        return self.resluts_obj.log_k_init[self.resindex]
    
    @property
    def _log_kex_min_by_global_dg_unfolding(self):
        if self.resluts_obj.analysis_object._log_kex_min_by_global_dg_unfolding is None:
            return None
        else:
            return self.resluts_obj.analysis_object._log_kex_min_by_global_dg_unfolding[self.resindex]
    
    @property
    def _log_P_max_by_global_dg_unfolding(self):
        if self.resluts_obj.analysis_object._log_P_max_by_global_dg_unfolding is None:
            return None
        else:
            return self.resluts_obj.analysis_object._log_P_max_by_global_dg_unfolding
        
    @property
    def _capped_log_kex(self):
        if self.if_slow_exchanger:
            return -self.resluts_obj.analysis_object.log_kex_min
        elif self.if_fast_exchanger:
            return -self.resluts_obj.analysis_object.log_kex_max
        else:
            raise Exception('residue is not a slow or fast exchanger')
        
    @property
    def _capped_log_P(self):
        if self.if_slow_exchanger or self.if_fast_exchanger:
            return self.log_k_init + self._capped_log_kex
        else:
            raise Exception('residue is not a slow or fast exchanger')
            

    @property
    def if_off_global_dg_unfolding(self):
        if self._log_kex_min_by_global_dg_unfolding is None:
            return False

        if self.resname == 'P' or self.is_nan():
            return False

        return np.all(
            self.mini_pep.clustering_results_log_kex
            > -self._log_kex_min_by_global_dg_unfolding
        )
        
    @property
    def if_off_log_kex_range_by_time_window(self):
        if self.resname == 'P' or self.is_nan():
            return False

        return self.if_slow_exchanger or self.if_fast_exchanger
        
    @property
    def if_slow_exchanger(self):
        if self.resname == 'P' or self.is_nan():
            return False

        return np.all(
            self.mini_pep.clustering_results_log_kex > -self.resluts_obj.analysis_object.log_kex_min
        )
        
    @property
    def if_fast_exchanger(self):
        if self.resname == 'P' or self.is_nan():
            return False

        return np.all(
            self.mini_pep.clustering_results_log_kex < -self.resluts_obj.analysis_object.log_kex_max
        )
        
        
class MiniPep(object):
    '''
    maximum_resolution_limits: {resid: [v0, v1]}
    can be a single residue or a range of residues
    '''
    def __init__(self, start, end):
        self.start = start
        self.end = end
        self.residues = []

    def set_clustering_results(self, clustering_results):
        self.clustering_results_log_kex = clustering_results[0]
        self.std_within_clusters_log_kex = clustering_results[1]
        self.data_within_clusters = clustering_results[2]
        
    def if_single_residue(self):
        return self.start == self.end
    
    def add_residue(self, residue):
        if residue in self.residues:
            raise Exception('residue already exists')
        self.residues.append(residue)
        residue.set_mini_pep(self)
        

class BFactorPlot(object):
    def __init__(
        self,
        analysis_object_1,
        analysis_object_2=None,
        pdb_file=None,
        plot_deltaG=False,
        temperature=None,
        logP_threshold=3.0,
    ):
        """
        _summary_

        :param analysis_object_1: an instance of Analysis and data should be loaded
        :param analysis_object_2: an instance of Analysis and data should be loaded
        """
        self.analysis_object_1 = analysis_object_1
        self.analysis_object_2 = analysis_object_2

        if self.analysis_object_2 is not None:
            if analysis_object_1.protein_sequence != analysis_object_2.protein_sequence:
                raise ValueError(
                    "The protein sequences of the two analysis objects are different"
                )

        if pdb_file is None:
            raise ValueError("pdb_file is required")

        self.pdb_file = pdb_file
        self.index_offset = get_index_offset(analysis_object_1, pdb_file)

        self.plot_deltaG = plot_deltaG

        if plot_deltaG:
            if temperature is None:
                raise ValueError("temperature is required to convert logP to deltaG")
            self.temperature = temperature
        
        self.logP_threshold = logP_threshold

    def plot(self, out_file):
        """
        if analysis_object_2 is None, plot the logP values of the first analysis object
        if analysis_object_2 is not None, plot the difference of logP values between the two analysis objects

        :param out_file: output pdb file with B-factors set to logP values
        """

        if self.analysis_object_2 is None:
            self._plot_single_state(out_file)
        else:
            self._plot_two_states_diff(out_file)

    def _plot_single_state(self, out_file):
        u = MDAnalysis.Universe(self.pdb_file)
        u.atoms.tempfactors = 0.0

        for res_i, _ in enumerate(self.analysis_object_1.results_obj.protein_sequence):
            res = self.analysis_object_1.results_obj.get_residue_by_resindex(res_i)
            avg_logP, std_logP = get_res_avg_logP(res)
            
            if res.if_off_log_kex_range_by_time_window:
                avg_logP = res._capped_log_P

            if std_logP > self.logP_threshold:
                avg_logP = np.nan
                continue
                # print(res.resid, res.resname ,avg_logP, std_logP, res.clustering_results_logP, res.std_within_clusters_logP)

            protein_res = u.select_atoms(
                f"protein and resid {res.resid - self.index_offset}"
            )
            if self.plot_deltaG:
                avg_deltaG = self._logPF_to_deltaG(avg_logP)
                protein_res.atoms.tempfactors = avg_deltaG

            else:
                protein_res.atoms.tempfactors = avg_logP
        
        PROs = u.select_atoms(f"resname PRO")
        PROs.atoms.tempfactors = np.nan

        u.atoms.write(out_file)

    def _plot_two_states_diff(self, out_file):
        u = MDAnalysis.Universe(self.pdb_file)
        u.atoms.tempfactors = 0.0

        for res_i, _ in enumerate(self.analysis_object_1.results_obj.protein_sequence):
            res_obj_1 = self.analysis_object_1.results_obj.get_residue_by_resindex(
                res_i
            )
            res_obj_2 = self.analysis_object_2.results_obj.get_residue_by_resindex(
                res_i
            )

            avg_logP_1, std_logP_1 = get_res_avg_logP(res_obj_1)
            avg_logP_2, std_logP_2 = get_res_avg_logP(res_obj_2)
            
            if res_obj_1.if_off_log_kex_range_by_time_window:
                avg_logP_1 = res_obj_1._capped_log_P

            if res_obj_2.if_off_log_kex_range_by_time_window:
                avg_logP_2 = res_obj_2._capped_log_P

            combined_std = np.sqrt(std_logP_1 ** 2 + std_logP_2 ** 2)

            # if not siginificant difference, 0.35 is the grid size
            if (
                abs(avg_logP_1 - avg_logP_2) < max(combined_std, 0.35)
                and combined_std <= self.logP_threshold
            ):
                continue
            elif combined_std > self.logP_threshold:
                diff_logP = np.nan
            else:
                diff_logP = avg_logP_1 - avg_logP_2

            protein_res = u.select_atoms(
                f"protein and resid {res_obj_1.resid - self.index_offset}"
            )

            if self.plot_deltaG:
                diff_deltaG = self._logPF_to_deltaG(diff_logP)
                protein_res.atoms.tempfactors = diff_deltaG

            else:
                protein_res.atoms.tempfactors = diff_logP

        PROs = u.select_atoms(f"resname PRO")
        PROs.atoms.tempfactors = np.nan

        u.atoms.write(out_file)

    def _logPF_to_deltaG(self, logPF):
        """
        :param logPF: logP value
        :return: deltaG in kJ/mol, local unfolding energy
        """

        return 8.3145 * self.temperature * np.log(10) * logPF / 1000


class Bayesian_hdx_ParseOutputFile(object):
    '''
    !!! Forked from salilab/bayesian_hdx
    
    An object that stores all of the information from an hxio.Output file.
    Also can be used as a general tool for analyzing sets of data.

    To create a copy of this object, do >new_pof = deepcopy(pof) to retain all
    of the header information and then clear the models

    '''
    def __init__(self, output_file):
        self.output_file = output_file
        self.datafiles = []
        self.datasets=[]
        self.sectors = []
        self.models=[]
        self.pf_grids = {}
        self.observed_residues = []
        self.parse_header()
        self.path = os.path.dirname(os.path.realpath(output_file))

    def get_sequence(self):
        '''
        Return the macromolecule sequence from the output file
        '''
        return self.get_datasets()[0].sequence


    def parse_header(self):
        '''
        Function that parses the header of output files.
        Stores the logk grid, along with other experimental information
        '''
        f = open(self.output_file, "r")
        for line in f.readlines():
            
            # > means model data (so header is over.)
            if line[0]==">":
                break
            
            # #-symbol means datasets
            elif line[0:2]=="# ":
                self.datafiles.append( (line[2:].split("|")[0].strip(), float(line[2:].split("|")[2].strip())) )
            
            # @-symbol means sectors
            elif line[0:2]=="@ ":
                for s_string in line[2:].strip().split("|"):
                    sector = []
                    for r in s_string.strip().split(" "):
                        if r != "":
                            sector.append(int(r))
                            self.observed_residues.append(int(r))
                    self.sectors.append(sector)

            elif line[0:2]=="$ ":
                if line[2:].split("|")[0].strip() != "Residue_number":
                    res = int(line[2:].split("|")[0].strip())
                    grid = []
                    for pf in line[2:].split("|")[1].strip().split(" "):

                        grid.append(float(pf))
                    self.pf_grids[res] = grid

            elif line.split(":")[0].strip() == "grid_size":
                self.grid_size = int(line.split(":")[1].strip())

            elif line.split(":")[0].strip() == "State":
                self.state_name = line.split(":")[1].strip()

            elif line.split(":")[0].strip() == "Molecule_Name":
                self.molecule_name = line.split(":")[1].strip()

            elif line[0:3] == "!! ":
                self.pep_idfs = [str(i) for i in line[3:].strip().strip('|').split("|")]
        f.close()

    def clear_models(self):
        self.models = []

    def cluster_models_kmeans(self, nmodels, nclust):
        # Uses kmeans to cluster models
        from sklearn.cluster import KMeans
        mods = [m[1] for m in self.get_best_scoring_models(N=nmodels)]
        models = np.array(mods)
        kmeans = KMeans(n_clusters=nclust).fit(models)

        for i in range(nclust):
            unique, counts = np.unique(kmeans.labels_, return_counts=True)
            print(nclust, " | ", i, dict(zip(unique, counts))[i] *1.0/nmodels)
        '''
        for c in range(len(kmeans.cluster_centers_)):
            for i in range(c,len(kmeans.cluster_centers_)):
                print(i, c, np.linalg.norm(kmeans.cluster_centers_[c]-kmeans.cluster_centers_[i]))
        '''

    def get_datasets(self):
        if len(self.datasets) == 0:
            self.generate_datasets()
        return self.datasets


    def get_models(self, return_pf=False):
        # returns all models stored in the POF (not the data file)
        if return_pf:
            return self.models_to_protection_factors(self.models)
        else:
            return self.models

    def get_all_models(self, return_pf=False):
        f = open(self.output_file, "r")
        models = []
        # Cycle over all lines
        for line in f.readlines():       
            if line[0]==">":
                score = float(line.split("|")[1].strip())

                model_string = line[1:].split("|")[0].strip()
                model_list = []
                for m in model_string.split(" "):
                    model_list.append(int(m))
                if return_pf:
                    ml1 = model_list
                    model_list = self.models_to_protection_factors(model_list)
                models.append((score, model_list)) 

        self.models = models

        return models

    def get_scores(self):
        try:
            models = self.models
        except:
            models = self.get_all_models()

        scores = [mod[0] for mod in models]

        return np.sort(scores)


    def get_best_scoring_models(self, N="all", sigmas=False, return_pf=False, sort_sectors=False):
        ''' Get the N best scoring models from the output file
        Returns a list of tuples of best_scoring_models 
            [(score, [model])]
        and (if sigmas=True)
        a grid of the timepoint sigma values.
        '''
        # Model entries are marked with a > as the first character

        if N=="all":
            N=100000000000

        f = open(self.output_file, "r")
        best_scoring_models = []
        # Cycle over all lines
        for line in f.readlines():
            if line[0]==">":
                score = float(line.split("|")[1].strip())

                # if the score is better than the last best score
                if len(best_scoring_models) < N or score < best_scoring_models[-1][0]:
                    if len(best_scoring_models) >= N:
                        del best_scoring_models[-1]
                        #print(score, best_scoring_models[-1])
                    model_string = line[1:].split("|")[0].strip()
                    model_list = []
                    for m in model_string.split(" "):
                        model_list.append(int(m))
                    
                    try:
                        back_exchange_list = [float(be) for be in line.split("||")[1].strip().split(" ")]
                    except:
                        back_exchange_list = []

                    try:
                        sidechain_exchange_list = [float(be) for be in line.split("||")[2].strip().split(" ")]
                    except:
                        sidechain_exchange_list = []

                    
                    if sort_sectors:
                        model_list = self.sort_model_by_sector(model_list)
                    if return_pf:
                        ml1 = model_list
                        model_list = self.models_to_protection_factors(model_list)
                        #print(score, model_list, ml1)
                        best_scoring_models.append((score, model_list, back_exchange_list, sidechain_exchange_list))

                    best_scoring_models = sorted(best_scoring_models, key=lambda x: x[0])

        self.best_scoring_models = best_scoring_models

        return best_scoring_models

    def models_to_protection_factors(self, models):
        # Input a list of list of integers.  
        # CHECK THAT THE MODEL SIZE IS CORRECT!

        if type(models[0]) != list:
            models = [models]

        protection_factor_models = []

        for m in models:
            pf_model = []
            for res in range(len(m)):
                if res + 1 in self.observed_residues:
                    #print("RES", res+1, m[res], self.pf_grids[res+1][m[res]-1])
                    pf_model.append(float(self.pf_grids[res+1][m[res]-1]))
                else:
                    pf_model.append(np.nan)
            protection_factor_models.append(pf_model)
            #print("MOD:", m[38:45])
            #print("PF:",pf_model[38:45])
        return protection_factor_models

    def get_sectors(self):
        # returns the list of sectors
        return self.sectors

    def sort_model_by_sector(self, model):
        # Given an input of a single model (as a list of integers), and a list
        # of sectors (as a list of list of residue numbers), return the model
        # with the integers in each sector sorted in increasing order. 

        # check that the model and total length of the sectors is the same:
        s_len = 0
        s_len = sum([len(s) for s in self.sectors])

        out_model = np.zeros(len(model))
        # Loop over all sectors
        for s in self.sectors:
            if len(s) == 0:
                continue

            sec_model = model[s[0]-1:s[-1]]
            sort_model = np.sort(sec_model) #sort indexes in increasing order

            idx_zero = [index for index, v in enumerate(sec_model) if v == 0]
            idx_nonzero = [index for index, v in enumerate(sec_model) if v != 0]
            offset=0
            for i in range(len(s)):
                if i in idx_zero:
                    out_model[i] = 0
                    offset += 1
                else:
                    out_model[i+offset] = sort_model[i + len(idx_zero)-offset]
                out_model[idx_nonzero[i] - 1 + s[0]] = sort_model[i + len(idx_zero)]
        return out_model.astype(int)

    def calculate_random_sample_convergence(self, replicates=100, pct_values=10):
        """ 
            For each sample of good scoring models (self.get_best_scoring_models)

            Get a list of top scores from each.

            Store the results as a list of two lists (one for each sample)
            with each sample list containing a tuple :: (pct, avg, stdev)
        """
        delpct = 1.0 / pct_values

        pct_grid = np.arange(delpct, 1.0, delpct)

        convergence_tuples = []

        scores = [g[0] for g in self.get_best_scoring_models()]

        pct_tuples = []

        for p in pct_grid:
            min_scores = []
            # For each percent value, take N samples and report avg/std of minimum values
            for r in range(replicates):
                subset = self._random_subset(scores, p)
                min_scores.append(min(subset))
            array = np.array(min_scores)

            pct_tuples.append((p, np.average(array), np.std(array)))

            print((p, np.average(array), np.std(array)), len(subset))

        convergence_tuples.append(pct_tuples)

        #print(convergence_tuples)
        del scores


    def _random_subset(self, models, pct):
        """
            Generate a random subset of pct percent of the given list of things
            @param models - python list of model dictionaries (really, could be anything)
            @param pct - the fraction of samples to return
        """
        import random
        if pct > 1.0:
            print("WARNING: percent value should be between 0 and 1.0. Diving by 100")
            pct = pct/100.0
        if pct > 1.0 or pct <= 0:
            print("WARNING: percent value is outside of bounds")

        num_samples = int(len(models) * pct)
        rand_smpl = [models[i] for i in random.sample(range(len(models)), num_samples)]
        #print(num_samples, len(models), len(rand_smpl), models)
        return rand_smpl
    
    
    def get_best_scoring_models_pf_df(self,  N=200):
        
        self.get_best_scoring_models(N=N, return_pf=True, sort_sectors=False)
        
        pfs = []     
        for i in self.best_scoring_models:
            # Append just the model itself. 
            pfs.append(np.array(i[1][0]))
            
        df_pfs = pd.DataFrame(pfs)

        # if hasattr(self, 'pep_idfs'):

        #     bes = [np.array(i[2]) for i in self.best_scoring_models]
        #     sidechain_ex = [np.array(i[3]) for i in self.best_scoring_models]

        #     df_bes = pd.DataFrame(bes, columns=self.pep_idfs) if bes[0].size !=0 else pd.DataFrame()
        #     df_sidechain_ex = pd.DataFrame(sidechain_ex, columns=self.pep_idfs) if sidechain_ex[0].size !=0 else pd.DataFrame()

        #     return df_pfs, df_bes, df_sidechain_ex

        
        return df_pfs
    
    

# some helper functions

def remove_outliers(data):

    # Convert input to numpy array in case it's a list or other array-like object
    if len(set(~np.isnan(data))) == 2: #if a mix of nan and non-nan due the pep drops
        data = np.array(data[~np.isnan(data)])
    else:
        data = np.array(data)

    # Remove all zeros, bc PFs can't be zero
    #data = data[data != 1e-3]
    
    # Calculate Q1, Q3, and IQR
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1

    # Identify outliers
    lower_bound = Q1 - (1.5 * IQR)
    upper_bound = Q3 + (1.5 * IQR)

    # Remove outliers
    return data[(data >= lower_bound) & (data <= upper_bound)]


def check_fitted_peptide_uptake(
    ana_obj,
    hdxms_data_list,
    peptide_obj,
    if_plot=False,
    state_name="SIM",
    figure=None,
    ax=None,
):
    pep_start = peptide_obj.start
    pep_end = peptide_obj.end

    time_points = sorted(
        [
            tp.deut_time
            for tp in peptide_obj.timepoints
            if tp.deut_time > 0 and tp.deut_time != np.inf
        ]
    )
    # time_points = np.logspace(1, 5, 1000)
    # deut_times = np.logspace(np.log10(time_points[0]), np.log10(time_points[-1]), 1000)

    fitted_uptakes = []
    fitted_uptakes_lower = []
    fitted_uptakes_upper = []

    for deut_time in time_points:
        peptide_incorporation = 0
        mini_peps_inrange = [
            mini_pep
            for mini_pep in ana_obj.results_obj.mini_peps
            if mini_pep.start >= pep_start and mini_pep.end <= pep_end
        ]

        mini_cover_resids = set(
            [res.resid for mini_pep in mini_peps_inrange for res in mini_pep.residues]
        )

        if set(range(pep_start, pep_end + 1)) != mini_cover_resids:
            raise ValueError("mini_peps not cover all residues")

        else:
            log_kex_inrange = [
                ki
                for mini_pep in mini_peps_inrange
                for ki in mini_pep.clustering_results_log_kex
                if not np.isinf(ki) and not np.isnan(ki)
            ]

            log_kex_std_inrange = [
                mini_pep.std_within_clusters_log_kex[i]
                for mini_pep in mini_peps_inrange
                for i, ki in enumerate(mini_pep.clustering_results_log_kex)
                if not np.isinf(ki) and not np.isnan(ki)
            ]

            peptide_incorporation = sum(
                calculate_simple_deuterium_incorporation(-1 * log_kex, deut_time)
                + peptide_obj.sidechain_exchange
                for log_kex in log_kex_inrange
            )
            # fitted_uptakes.append(peptide_incorporation*ana_obj.saturation)
            fitted_uptakes.append(peptide_incorporation)

            peptide_incorporation_lower = sum(
                calculate_simple_deuterium_incorporation(
                    -1 * log_kex - log_kex_std_inrange[i], deut_time
                )
                + peptide_obj.sidechain_exchange
                for i, log_kex in enumerate(log_kex_inrange)
            )
            # fitted_uptakes_lower.append(peptide_incorporation_lower*ana_obj.saturation)
            fitted_uptakes_lower.append(peptide_incorporation_lower)

            peptide_incorporation_upper = sum(
                calculate_simple_deuterium_incorporation(
                    -1 * log_kex + log_kex_std_inrange[i], deut_time
                )
                + peptide_obj.sidechain_exchange
                for i, log_kex in enumerate(log_kex_inrange)
            )
            # fitted_uptakes_upper.append(peptide_incorporation_upper*ana_obj.saturation)
            fitted_uptakes_upper.append(peptide_incorporation_upper)

    try:
        full_d_scaler = peptide_obj.max_d / peptide_obj.theo_max_d
        # print(full_d_scaler)
    except:
        full_d_scaler = 1  # for simulated data

    fitted_uptakes = np.array(fitted_uptakes) * full_d_scaler * ana_obj.saturation
    fitted_uptakes_lower = (
        np.array(fitted_uptakes_lower) * full_d_scaler * ana_obj.saturation
    )
    fitted_uptakes_upper = (
        np.array(fitted_uptakes_upper) * full_d_scaler * ana_obj.saturation
    )

    exp_uptakes = np.array([peptide_obj.get_timepoint(tp).num_d for tp in time_points])
    rmse = np.sqrt(mean_squared_error(exp_uptakes, fitted_uptakes)) / (
        pep_end - pep_start + 1
    )

    if if_plot:
        uptake = UptakePlot(
            hdxms_data_list,
            peptide_obj.identifier,
            states_subset=[state_name],
            if_plot_fit=False,
            figure=figure,
            ax=ax,
        )
        # uptake.uptakeplot.axes[0].plot(time_points, fitted_uptakes, label='model', color='red', linestyle='-')
        ax.plot(
            time_points,
            fitted_uptakes,
            label="model",
            color="g",
            linestyle="-",
            marker="s",
            markersize=15,
        )

        ax.fill_between(
            time_points,
            fitted_uptakes_lower,
            fitted_uptakes_upper,
            color="g",
            alpha=0.1,
        )
        # legend_elements = [Line2D([0], [0], color='black', lw=4, linestyle='-', label='exp'),
        #                    Line2D([0], [0], color='red', lw=4, linestyle='-', label='model')]

        # uptake.uptakeplot.axes[0].legend(handles=legend_elements, loc='lower right')

        return rmse, uptake.uptakeplot

    else:
        return rmse



def check_fitted_isotope_envelope(ana_obj, timepont_obj, if_plot=False):


    pep_start = timepont_obj.peptide.start
    pep_end = timepont_obj.peptide.end

    deut_time = timepont_obj.deut_time
    
    mini_peps_inrange = [mini_pep for mini_pep in ana_obj.results_obj.mini_peps if mini_pep.start >= pep_start and mini_pep.end <= pep_end]

    mini_cover_resids = set([res.resid for mini_pep in mini_peps_inrange for res in mini_pep.residues])

    if set(range(pep_start, pep_end+1)) != mini_cover_resids:
        raise ValueError('mini_peps not cover all residues')
    
    else:
        log_kex_inrange = [
            ki
            for mini_pep in mini_peps_inrange
            for ki in mini_pep.clustering_results_log_kex
            if not np.isinf(ki) and not np.isnan(ki)
        ]

        tp_raw_deut = []
        for log_kex in log_kex_inrange:
            tp_raw_deut.append(calculate_simple_deuterium_incorporation(-1*log_kex, deut_time)*ana_obj.saturation)

        try:
            full_d_scaler = timepont_obj.peptide.get_timepoint(np.inf).num_d/timepont_obj.peptide.theo_max_d
        except:
            full_d_scaler = 1 # for simulated data
        tp_raw_deut = np.array(tp_raw_deut)*full_d_scaler
        p_D = event_probabilities(tp_raw_deut)

        #t0_theo = spectra.get_theoretical_isotope_distribution(timepont_obj)['Intensity'].values
        #fitted_isotope_envelope = np.convolve(t0_theo, p_D)

        fitted_isotope_envelope = np.convolve(timepont_obj.peptide.get_timepoint(0).isotope_envelope, p_D)



    if if_plot:

        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        peak_num = np.where(timepont_obj.isotope_envelope>1e-6)[0][-1]
        
        stem1 = plt.stem(fitted_isotope_envelope[:peak_num], linefmt='r-', markerfmt='ro', basefmt='r-', label='model')
        stem2 = plt.stem(timepont_obj.isotope_envelope[:peak_num], linefmt='k-', markerfmt='ko', basefmt='k-', label='exp')
        plt.legend()
        

        title  =  f'{timepont_obj.peptide.protein_state.state_name}_{timepont_obj.peptide.identifier}_tp{int(deut_time)}'
        plt.title(title, fontdict={'fontsize': 24})
          
        plt.setp(stem1[1], linewidth = 10,)
        plt.setp(stem1[0], markersize = 20)
        plt.setp(stem1[2], linewidth = 1.5)
        
        
        plt.setp(stem2[1], linewidth = 5)
        plt.setp(stem2[0], markersize = 10)
        plt.setp(stem2[2], linewidth = 1)
        
        plt.xlabel('#heavy isotopes')
        plt.ylabel('probability')
        

    sum_ae = get_sum_ae(fitted_isotope_envelope, timepont_obj.isotope_envelope)
    return sum_ae




def get_res_avg_logP(res_obj):
    '''
    calculate the average logP value of a residue

    :param res_obj: residue object
    :return: average logP value, std from the average
    '''

    if res_obj.is_nan():
        return np.nan, 0.0
    if res_obj.resname == "P":
        return 0.0, 0.0
    else:
        logPs = [i for i in res_obj.clustering_results_logP if i != np.inf and not np.isnan(i)]
        stds = [i for i in res_obj.std_within_clusters_logP if i != np.inf and not np.isnan(i)]

        avg_logP = np.mean(logPs)
        avg_std = np.sqrt(np.sum(np.square(stds)) / len(stds) ** 2)

        return avg_logP, avg_std


def get_res_avg_logP_std(res_obj):
    '''
    calculate the average logP standard deviation of a residue

    :param res_obj: residue object
    :return: average logP standard deviation
    '''
    if res_obj.is_nan():
        return np.nan
    if res_obj.resname == "P":
        return 0.0
    else:

        stds = [i for i in res_obj.std_within_clusters_logP if i != np.inf and not np.isnan(i)]
        avg_std = np.sqrt(np.sum(np.square(stds)) / len(stds) ** 2)

        return avg_std


def get_res_avg_log_kex(res_obj):
    '''
    calculate the average log_kex value of a residue

    :param res_obj: residue object
    :return: average log_kex value
    '''

    if res_obj.is_nan():
        return np.nan
    if res_obj.resname == "P":
        return np.inf
    else:
        return np.average(
            [
                i
                for i in res_obj.mini_pep.clustering_results_log_kex
                if np.abs(i) != np.inf and not np.isnan(i)
            ]
        )  # skip Pro



def get_avg_envelope_uptake_errors(protein_states, ana_obj, res):
    all_peps = [pep  for state in protein_states for pep in state.peptides]

    peps_covering = [
        pep
        for pep in all_peps
        if pep.start <= res.resid <= pep.end and pep.note is None
    ]

    peps_covering_tps = [
        tp
        for pep in peps_covering
        for tp in pep.timepoints
        if tp.deut_time != 0 and tp.deut_time != np.inf
    ]

    envelope_errors = [
        check_fitted_isotope_envelope(ana_obj, tp) for tp in peps_covering_tps
    ]

    peptide_uptake_errors = [
        check_fitted_peptide_uptake(ana_obj=ana_obj,hdxms_data_list=None, peptide_obj=pep) for pep in peps_covering
    ]

    return np.average(envelope_errors), np.average(peptide_uptake_errors)



def if_off_time_window(log_kex, time_window, threshold=0.01):
    if (
        calculate_simple_deuterium_incorporation(log_kex, time_window[0])
        > 1 - threshold
    ):
        return True
    elif calculate_simple_deuterium_incorporation(log_kex, time_window[1]) < threshold:
        return True
    else:
        return False


def get_index_offset(
    ana_obj,
    pdb_file,
):
    from pigeon_feather.tools import pdb2seq, find_peptide

    u = MDAnalysis.Universe(pdb_file)
    protein = u.select_atoms("protein")
    first_protein_res_index = protein.resids[0] - 1

    pdb_sequence = pdb2seq(pdb_file)
    half_length = int(len(ana_obj.results_obj.protein_sequence) / 2)
    a_middle_pep = ana_obj.results_obj.protein_sequence[half_length-5:half_length+5]
    pdb_start, pdb_end = find_peptide(pdb_sequence, a_middle_pep)
    index_offset = half_length-5 - (pdb_start+first_protein_res_index)

    # check if the offset is correct
    for residue in protein.residues[20:30]:
        pdb_resname = MDAnalysis.lib.util.convert_aa_code(residue.resname)
        res = ana_obj.results_obj.get_residue_by_resid(residue.resid + index_offset).resname

        if pdb_resname != res:
            raise ValueError("The offset is not correct")

    
    return index_offset