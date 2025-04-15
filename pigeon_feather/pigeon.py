import numpy as np
import pandas as pd
import scipy.sparse as sparse
import scipy.optimize as opt
import scipy.stats as stats
from pyteomics import mass
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import rcParams
from matplotlib import rc
import time
import argparse
import os

# PARAMS

# maximum charge of peptide ions in search
default_maxz = 3
# length range for peptides in search (inclusive)
default_minl = 3
default_maxl = 15
# m/z error cutoff for initial search and for cut about trendline (in ppm)
default_ppmc_wide = 30
default_ppmc_narrow = 7
# score cutoff for truth peptides (for systematic error fit)
default_scorec = 0.05
# fit to use for systematic error ('quad' or 'inv')
default_fit = 'quad'
# maximum iterations for systematic error fit
default_maxfev = 1000
# types of fragment ions to consider in search
default_ions = ['b', 'b-H2O', 'b-NH3', 'y', 'y-H2O', 'y-NH3']
# threshold in Da/e for matching fragments
default_threshold = 0.02
# cutoff for matches to be considered degenerate, in m/z (Da/e) and RT (min)
default_mzc = 0.1
default_rtc = 0.5
# significance threshold for p-value for disambiguation
default_pvc = 0.05
# method to use for disambiguation: keep or drop coeluting peptides? ('keep' or 'drop')
default_cut_method = 'drop'
# score floor (drop all below given match score)
default_sf = 0


# FOR VECTORIZED FRAGMENT MATCHING
def match_fragment(mz1, mz2, threshold=default_threshold):
    return abs(mz1 - mz2) <= threshold and (mz1 or mz2)

vecmf = np.vectorize(match_fragment, otypes=[bool])


class Pool:
    def __init__(self, proteins, msms, fit=default_fit):
        self.proteins = proteins
        self.msms = msms
        self.fit = fit

    def fitfn(self, x, a, b, c):
        if self.fit == 'inv':
            return c + a / (x - b)
        return c + a * x + b * (x ** 2)

    # PLOTTING FNS

    def scorehist(self, matches, path=None, title=None, xmax=None, ymin=None, ymax=None, figsize=109.7):
        rcParams['font.sans-serif'][7] = 'DejaVu Sans'
        rcParams['font.sans-serif'][0] = 'Arial'
        scores = [match.score for match in matches]
        afont = {'size': 6}
        rc('font', **afont)

        if not ymin:
            ymin = 9e-1
        if not ymax:
            ymax = len(matches)
        if not xmax:
            xmax = max(scores)
        pt = 72
        figure, (ax1) = plt.subplots(1, 1, figsize=(figsize / pt, figsize / pt))
        plt.xlabel('Score')
        plt.ylabel('n (matches)')
        plt.ylim(ymin, ymax)
        if title:
            plt.title(title)
        plt.hist(scores, log=True, bins=100, range=[0, xmax])
        plt.tight_layout()
        if path:
            plt.savefig(path)
            plt.close(figure)

    def mzscatter(self, matches, path=None, popt=None, xmax=None, mzmin=None, mzmax=None, figsize=109.7):
        rcParams['font.sans-serif'][7] = 'DejaVu Sans'
        rcParams['font.sans-serif'][0] = 'Arial'

        afont = {'size': 6}
        rc('font', **afont)

        pt = 72
        figure, (ax1) = plt.subplots(1, 1, figsize=(figsize / pt, figsize / pt))

        mzs = np.asarray([match.pep.mz[match.z] for match in matches])
        ppms = np.asarray([match.ppmerr for match in matches])
        scores = np.asarray([match.score for match in matches])
        if not xmax:
            xmax = max(scores)
        if popt is not None:
            plt.plot(np.sort(mzs), self.fitfn(np.sort(mzs), *popt), linewidth=0.5)
        plt.scatter(mzs, ppms, c=scores + 1e-7,
                    norm=colors.LogNorm(vmin=1e-7, vmax=xmax), s=50 / pt, linewidth=0.05)
        plt.colorbar(label='Score')
        plt.xlabel('$\it{{m}}$/$\it{{z}}$ (D/e)')
        plt.ylabel('mass error (ppm)')
        plt.ylim(-30, 30)
        if mzmin and mzmax:
            plt.xlim(mzmin, mzmax)
        plt.tight_layout()
        if path:
            plt.savefig(path)
            plt.close(figure)

    # MAKE TABLE OF FEATURES FOR MATCHED PEPTIDES
    def pool_output(self, matches, path=None):
        cmpdnum = [match.peak.cmpdnum for match in matches]
        runid = [match.peak.runid for match in matches]
        sequence = [match.pep.sequence for match in matches]
        start = [match.pep.start + 1 for match in matches]
        end = [match.pep.end for match in matches]

        int = [match.peak.intensity for match in matches]
        z = [match.z for match in matches]
        m_calc = [match.pep.m for match in matches]
        mz_calc = [match.pep.mz[match.z] for match in matches]
        m_meas = [match.peak.mz * match.z for match in matches]
        mz_meas = [match.peak.mz for match in matches]
        ppm_err = [match.ppmerr for match in matches]
        da_err = [match.peak.mz * match.z - match.pep.mz[match.z] for match in matches]

        rt = [match.peak.rt for match in matches]
        n_hits = [match.fragmatch.size for match in matches]
        n_tries = [np.prod(match.fragmatch.shape) for match in matches]
        score = [match.score for match in matches]

        df = pd.DataFrame(data={'compound #': cmpdnum, 'run ID': runid, 'sequence': sequence,
                                'Start': start, 'End': end, 'intensity': int,
                                'z': z, 'mass (calc)': m_calc, 'm/z (calc)': mz_calc,
                                'mass (meas)': m_meas, 'm/z (meas)': mz_meas,
                                'error (ppm)': ppm_err, 'error (Da/e)': da_err,
                                'RT (min)': rt, '# fragment matches': n_hits,
                                '# peaks * fragments': n_tries, 'score': score
                                })
        df.sort_values(by=['Start', 'End', 'z', 'score'], inplace=True)
        if path:
            df.to_csv(path, index=False)
        return df

    # MAKE TABLE OF SEQUENCE RANGES FOR MATCHED PEPTIDES
    def rangeslist(self, matches, path=None):
        start = [match.pep.start + 1 for match in matches]
        end = [match.pep.end for match in matches]
        df = pd.DataFrame(data={'Start': start, 'End': end})
        df.sort_values(by=['Start', 'End'], inplace=True)
        if path:
            df.to_csv(path, index=False)
        return df

    # LOG STATS ABOUT EACH STEP IN ANALYSIS
    def nums_out(self, match_dict, popt, path=None):
        out = ''
        for key in match_dict:
            n = len(match_dict[key])
            p_0 = len([match for match in match_dict[key] if not match.score]) / n * 100
            mean_score = np.mean([match.score for match in match_dict[key]])
            rmsdev = np.sqrt(
                np.mean([(match.ppmerr - self.fitfn(match.pep.mz[match.z], *popt)) ** 2 for match in match_dict[key]]))
            out = out + f'{key}: n={n}, %0={p_0}, mean_score={mean_score}, RMSD={rmsdev}\n'
        if path:
            with open(path, 'w') as file:
                file.write(out)
        return out

    # INITIAL MATCH COMPARING ALL PEAKS TO ALL THEORETICAL CHARGE STATES
    def match(self, mzc=default_ppmc_wide, maxz=default_maxz):
        self.matches = []
        for protein in self.proteins:
            for pep in protein.peps:
                for peak in self.msms.peaks:
                    if peak.z:
                        if peak.z in pep.mz:
                            if abs(pep.mz[peak.z] - peak.mz) <= mzc * pep.mz[peak.z] * 1e-6:
                                ppmerr = (peak.mz - pep.mz[peak.z]) / pep.mz[peak.z] * 1e6
                                self.matches.append(Match(pep, peak, peak.z, ppmerr, self))
                    else:
                        for z in range(1, maxz + 1):
                            if abs(pep.mz[z] - peak.mz) <= mzc * pep.mz[z] * 1e-6:
                                ppmerr = (peak.mz - pep.mz[z]) / pep.mz[z] * 1e6
                                self.matches.append(Match(pep, peak, z, ppmerr, self))
                                break
        self.score_max = max([match.score for match in self.matches])

    # CUT ALL BUT THOSE PASSING HIGH SCORE CUTOFF, FIT TRENDLINE FOR SYSTEMATIC ERROR
    def truth_cut(self, matches, score_c=default_scorec, maxfev=default_maxfev, fit=default_fit):
        self.fit = fit
        best_peaks = []
        mzs = []
        ppms = []
        scores = []
        for match in matches:
            if match.score >= score_c:
                best_peaks.append(match)
                mzs.append(match.pep.mz[match.z])
                ppms.append(match.ppmerr)
                scores.append(match.score)
        popt, pcov = opt.curve_fit(self.fitfn, mzs, ppms, maxfev=maxfev,
                                   bounds=([-np.inf, -np.inf, -np.inf],
                                           [np.inf, min(mzs) - 100 if fit == 'inv' else np.inf, np.inf]))
        return best_peaks, popt, pcov

    # CUT ALL FARTHER THAN ppm_c FROM TRENDLINE
    def ppm_cut(self, popt, matches, ppm_c=default_ppmc_narrow):
        return [match for match in matches if abs(match.ppmerr - self.fitfn(match.pep.mz[match.z], *popt)) < ppm_c]

    # CUT ALL BUT BEST MATCH FOR EACH PEPTIDE
    def best_peak_cut(self, matches):
        best_peaks = []
        peps = list(set([match.pep for match in matches]))
        for pep in peps:
            maxscore = max([match.score for match in pep.matches if match in matches])
            best = [match for match in pep.matches if match.score >= maxscore and match in matches]
            if len(best) > 1:
                maxint = max([match.peak.intensity for match in best])
                best = [match for match in best if match.peak.intensity >= maxint]
            best_peaks = best_peaks + best
        # self.matches = best_peaks
        return best_peaks

    # CUT ALL BELOW GIVEN SCORE (OPTIONAL)
    def scorefloor_cut(self, matches, score_f=default_sf):
        return [match for match in matches if match.score >= score_f]

    # CUT USING FRAGMENT-BASED DISAMBIGUATION
    def fragment_cut(self, matches, allmatches, threshold=default_threshold, pvc=0, mzc=default_mzc, rtc=default_rtc,
                     cut_method=default_cut_method):
        degen_dict = {}
        best_peaks = []
        exclude = []
        for match1 in matches:
            print()
            print()
            print('match1:')
            print(f'{match1.pep.sequence}-{match1.pep.start+1}-{match1.pep.end}+{match1.z}-{match1.pep.mz[match1.z]}-{match1.peak.runid}-cmpd{match1.peak.cmpdnum}')
            print()
            degen = [match2 for match2 in allmatches
                     if (np.abs(match1.pep.mz[match1.z] - match2.pep.mz[match2.z]) < mzc
                         and np.abs(match1.peak.rt - match2.peak.rt) < rtc
                         and (match1.pep.sequence != match2.pep.sequence
                              or match1.pep.start != match2.pep.start
                              or match1.pep.end != match2.pep.end))]
            degen_dict[match1] = degen
            if not degen:
                print('not degen. keeping')
                best_peaks.append(match1)
            elif match1.fragmatch.size == 0:
                exclude.append(match1)
                print('no support, degen. dropping.')
            else:
                same_pep = [match2 for match2 in allmatches
                            if (np.abs(match1.pep.mz[match1.z] - match2.pep.mz[match2.z]) < mzc
                                and np.abs(match1.peak.rt - match2.peak.rt) < rtc
                                and match1.pep.sequence == match2.pep.sequence)]
                flag = False

                pvs = []
                match1_mzs = match1.fragmatch.sum(0) * match1.pep.fragpeaks
                mask = np.ones(match1_mzs.shape)
                print('degen:')
                for match2 in degen:
                    match2_mzs = match2.fragmatch.sum(0) * match2.pep.fragpeaks
                    same_frags_1 = sparse.coo_array(
                        vecmf(match1_mzs, match2.pep.fragpeaks.reshape(1, match2.pep.fragpeaks.shape[0]).T,
                              threshold=threshold))
                    same_frags_2 = sparse.coo_array(
                        vecmf(match2_mzs, match1.pep.fragpeaks.reshape(1, match1.pep.fragpeaks.shape[0]).T,
                              threshold=threshold))
                    mask1 = np.logical_not(same_frags_1.sum(0))
                    mask = np.logical_and(mask, mask1)
                    mask2 = np.logical_not(same_frags_2.sum(0))
                    print(f'{match2.pep.sequence}-{match2.pep.start+1}-{match2.pep.end}+{match2.z}-{match2.pep.mz[match2.z]}-{match2.peak.runid}-cmpd{match2.peak.cmpdnum}')
                    print(match2_mzs)
                    print(mask2 * match2_mzs)
                    if not (mask1 * match1_mzs).any():
                        flag = True
                        print('no unique support, degen. dropping.')
                        break
                    elif cut_method == 'drop':
                        n_unique = np.count_nonzero(mask2 * match2_mzs)
                        if not pvc and n_unique:
                            flag = True
                            print('(WRONG) 1 hit, dropping')
                            break
                        elif pvc:
                            maxmz = max([max(match2.pep.fragpeaks), max(match2.peak.ms2s)])
                            minmz = min([min(match2.pep.fragpeaks), min(match2.peak.ms2s)])
                            prob = match2.peak.ms2s.size * 2 * threshold / (maxmz - minmz)
                            pvs.append(
                                stats.binomtest(n_unique, match2.pep.fragpeaks.size, prob, alternative='greater').pvalue)
                            print(pvs[-1])

                print()
                print('same_pep:')
                if flag:
                    exclude.append(match1)
                else:
                    if not pvc:
                        print('(WRONG) no hits, keeping')
                        best_peaks.append(match1)
                    else:
                        samepvs = []
                        for match2 in same_pep:
                            match2_mzs = match2.fragmatch.sum(0) * match2.pep.fragpeaks
                            print(f'{match2.pep.sequence}-{match2.pep.start+1}-{match2.pep.end}+{match2.z}-{match2.pep.mz[match2.z]}-{match2.peak.runid}-cmpd{match2.peak.cmpdnum}')
                            print(match2_mzs)
                            print(mask * match2_mzs)
                            n_unique = np.count_nonzero(mask * match2_mzs)
                            maxmz = max([max(match2.pep.fragpeaks), max(match2.peak.ms2s)])
                            minmz = min([min(match2.pep.fragpeaks), min(match2.peak.ms2s)])
                            prob = match2.peak.ms2s.size * 2 * threshold / (maxmz - minmz)
                            samepvs.append(
                                stats.binomtest(n_unique, match2.pep.fragpeaks.size, prob, alternative='greater').pvalue)
                            print(samepvs[-1])
                        if stats.combine_pvalues(samepvs).pvalue >= pvc or (
                                cut_method == 'drop' and stats.combine_pvalues(pvs).pvalue <= pvc):
                            exclude.append(match1)
                            print('pvalue, dropping')
                        else:
                            print('pvalue, keeping')
                            best_peaks.append(match1)
                        print(samepvs)
                        print(stats.combine_pvalues(samepvs).pvalue)
                        print()
                        print(pvs)
                        print(stats.combine_pvalues(pvs).pvalue)
                        print()
        return best_peaks, exclude, degen_dict


class Match:
    def __init__(self, pep, peak, z, ppmerr, pool):
        self.pep = pep
        self.peak = peak
        self.z = z
        self.ppmerr = ppmerr
        self.pool = pool
        self.fragmatch = sparse.coo_array(vecmf(pep.fragpeaks, peak.ms2s.reshape(1, peak.ms2s.shape[0]).T))
        self.score = self.fragmatch.size ** 2 / np.prod(self.fragmatch.shape) if np.prod(self.fragmatch.shape) > 0 else 0
        pep.matches.append(self)
        peak.matches.append(self)

    def remove(self):
        pep.matches.remove(self)
        peak.matches.remove(self)
        pool.matches.remove(self)


class Protein:
    def __init__(self, sequence, minl=default_minl, maxl=default_maxl, maxz=default_maxz, ions=default_ions):
        self.sequence = sequence
        self.peps = []
        self.make_peps(minl=minl, maxl=maxl)
        for pep in self.peps:
            pep.get_mass(maxz=maxz)
            pep.get_fragments(maxz=maxz, ions=ions)

    def make_peps(self, minl=default_minl, maxl=default_maxl):
        for start in range(len(self.sequence) - minl):
            for end in range(start + minl, min(len(self.sequence) + 1, start + maxl + 1)):
                self.peps.append(Peptide(self.sequence[start:end], start, end))


class Peptide:
    def __init__(self, sequence, start, end):
        self.sequence = sequence
        self.start = start
        self.end = end
        self.matches = []

    def get_mass(self, maxz=default_maxz):
        self.m = mass.calculate_mass(sequence=self.sequence)
        self.mz = {}
        for z in range(1, maxz + 1):
            self.mz[z] = mass.calculate_mass(sequence=self.sequence, charge=z)

    def get_fragments(self, maxz=default_maxz, ions=default_ions):
        self.fragments = []
        for ion in ions:
            for i in range(len(self.sequence)):
                for z in range(1, maxz + 1):
                    if ion[0] == 'b':
                        seq = self.sequence[0:i + 1]
                        self.fragments.append(Fragment(len(self.fragments), seq, 0, i, ion,
                                                       mass.calculate_mass(sequence=seq, ion_type=ion, charge=z), z))
                    else:
                        seq = self.sequence[i:]
                        self.fragments.append(Fragment(len(self.fragments), seq, i, len(self.sequence) - 1, ion,
                                                       mass.calculate_mass(sequence=seq, ion_type=ion, charge=z), z))
        self.fragpeaks = np.empty(len(self.fragments), np.float64)
        for frag in range(len(self.fragments)):
            self.fragpeaks[frag] = self.fragments[frag].mz


class Fragment:
    def __init__(self, i, seq, start, end, ion, mz, z):
        self.index = i
        self.seq = seq
        self.start = start
        self.end = end
        self.ion = ion
        self.mz = mz
        self.z = z


class MSMS:
    def __init__(self):
        self.peaks = []
        self.runids = []

    def first(self, s, l):
        for i in range(len(l)):
            if s == l[i]:
                return i

    # MGF FORMAT-SPECIFIC I/O
    def read_block(self, block, runid):
        z = 0
        mz = np.nan
        cmpdnum = np.nan
        rt = np.nan
        ms1int = np.nan

        ms2s = []
        for line in block:
            if 'CHARGE=' in line:
                z = int(line.split('CHARGE=')[1].split('+')[0])
            if 'RTINSECONDS=' in line:
                rt = np.float64(line.split('=')[1]) / 60
            if 'TITLE=' in line:
                if 'TITLE=Cmpd' in line:
                    cmpdnum = int(line.split('TITLE=Cmpd ')[1].split(',')[0])
                if 'scan=' in line:
                    cmpdnum = int(line.split('scan=')[1].split('"')[0])
            if 'PEPMASS=' in line:
                if len(line.split()) > 1:
                    ms1int = np.float64(line.split()[1])
                mz = np.float64(line.split()[0].split('=')[1])
            if line[0].isdigit():
                ms2s.append(line)
        ms2ints = np.asarray([np.float64(line.split()[1]) for line in ms2s])
        ms2mzs = np.asarray([np.float64(line.split()[0]) for line in ms2s])
        return Peak(runid, z, mz, cmpdnum, rt, ms1int, ms2ints, ms2mzs)

    def read_mgf(self, paths):
        for path in paths:
            f = open(path, 'r')
            lines = f.readlines()
            f.close()
            lines = [line for line in lines if 'MaxRes' not in line and '###' not in line]
            header = lines[0:self.first('BEGIN IONS\n', lines)]
            data = lines[self.first('BEGIN IONS\n', lines):]
            startpoints = [i for i in range(len(data)) if 'BEGIN IONS' in data[i]]
            ms1blocks = [data[startpoints[i]:startpoints[i + 1]] for i in range(len(startpoints) - 1)] + [
                data[startpoints[-1]:]]
            for block in ms1blocks:
                self.peaks.append(self.read_block(block, path))
            self.runids.append(path)


class Peak:
    def __init__(self, runid, z, mz, cmpdnum, rt, ms1int, ms2ints, ms2mzs):
        self.runid = runid
        self.z = z
        self.mz = mz
        self.cmpdnum = cmpdnum
        self.rt = rt
        self.intensity = ms1int
        self.ms2ints = ms2ints
        self.ms2s = ms2mzs
        self.matches = []


# END-TO-END MATCHING + DISAMBIGUATION

def analyze(seqs, files, maxz=default_maxz, minl=default_minl, maxl=default_maxl, ppmc_wide=default_ppmc_wide,
            ppmc_narrow=default_ppmc_narrow, scorec=default_scorec, fit=default_fit, maxfev=default_maxfev,
            ions=default_ions, threshold=default_threshold, mzc=default_mzc, rtc=default_rtc, pvc=default_pvc,
            cut_method=default_cut_method, sf=default_sf,
            path='./', plots_dir=None, pool_out=None, rangeslist=None, nums_out=None):

    t0 = time.time()
    pool = Pool([Protein(seq, minl=minl, maxl=maxl, maxz=maxz, ions=ions) for seq in seqs], MSMS())
    print(f'Generating sequences {time.time() - t0}')

    t0 = time.time()
    pool.msms.read_mgf(files)
    print(f'Reading files: {time.time() - t0}')

    t0 = time.time()
    pool.match(mzc=ppmc_wide, maxz=maxz)
    print(f'Matching: {time.time() - t0}')

    t0 = time.time()
    truth, popt, pcov = pool.truth_cut(pool.matches, score_c=scorec, maxfev=maxfev, fit=fit)
    print(f'Truth cut: {time.time() - t0}')

    t0 = time.time()
    ppmc = pool.ppm_cut(popt, pool.matches, ppm_c=ppmc_narrow)
    print(f'PPM cut: {time.time() - t0}')

    t0 = time.time()
    best = pool.best_peak_cut(ppmc)
    print(f'single-peak cut: {time.time() - t0}')

    t0 = time.time()
    scorefloor = pool.scorefloor_cut(best, score_f=sf)
    print(f'scorefloor cut: {time.time() - t0}')

    t0 = time.time()
    disambig, ambig, degen_dict = pool.fragment_cut(scorefloor, ppmc, threshold=threshold, pvc=pvc,
                                                    mzc=mzc, rtc=rtc, cut_method=cut_method)
    print(f'disambiguation cut: {time.time() - t0}')

    t0 = time.time()
    match_dict = {'all': pool.matches, 'truth': truth, 'ppm_cut': ppmc, 'single': best, 'clean': disambig,
                  'ambig': ambig}
    if sf:
        match_dict['scorefloor'] = scorefloor
    if plots_dir:
        if not os.path.isdir(path + plots_dir):
            os.mkdir(path + plots_dir)
        xmax = max([match.score for match in pool.matches])
        mzmin = min([match.pep.mz[match.z] for match in pool.matches])
        mzmax = max([match.pep.mz[match.z] for match in pool.matches])
        for step in match_dict:
            pool.mzscatter(match_dict[step], popt=popt, xmax=xmax, mzmin=mzmin, mzmax=mzmax,
                           path=f'{path}{plots_dir}/{step}-scatter.pdf')
            pool.scorehist(match_dict[step], title=step, ymax=len(pool.matches), xmax=xmax,
                           path=f'{path}{plots_dir}/{step}-hist.pdf')
    if rangeslist:
        for i, rl in enumerate(rangeslist):
            ml = [match for match in disambig if seqs[i][match.pep.start:match.pep.end] == match.pep.sequence]
            pool.rangeslist(ml, path=f'{path}{rl}')
    if pool_out:
        for i, ol in enumerate(pool_out):
            ml = [match for match in disambig if seqs[i][match.pep.start:match.pep.end] == match.pep.sequence]
            pool.pool_output(ml, path=f'{path}{ol}')
    if nums_out:
        nums = pool.nums_out(match_dict, popt, path=f'{path}{nums_out}')

    print(f'plots + output: {time.time() - t0}')
    if nums_out:
        print(nums)
    return pool, match_dict


def parse_args():
    # CMD LINE ARGS CODE

    parser = argparse.ArgumentParser(description='PIGEON: match peptides and disambiguate duplicates')
    parser.add_argument('--s', '--seq', '--sequence', dest='sequence', help="input sequence", nargs='+', required=True)
    parser.add_argument('--f', '--files', dest='files', help="paths to input files", nargs='+', required=True)

    parser.add_argument('--path', dest='path', help="path for outputs")
    parser.add_argument('--p', '--plots', dest='plots', help="directory for output plots")
    parser.add_argument('--o', '--output', dest='output', nargs='+', help="pool output")
    parser.add_argument('--r', '--rangeslist', dest='rangeslist', nargs='+', help="rangeslist output")
    parser.add_argument('--n', '--nums', dest='nums', help="diagnostic stats output")

    parser.add_argument('--maxz', dest='maxz', type=int, help='max charge for peptides')
    parser.add_argument('--minl', dest='minl', type=float, help='min length for peptides')
    parser.add_argument('--maxl', dest='maxl', type=float, help='max length for peptides')
    parser.add_argument('--ppmc_w', dest='ppmc_wide', type=float, help='ppm error cutoff for initial match')
    parser.add_argument('--ppmc_n', dest='ppmc_narrow', type=float, help='ppm cutoff after curve fit')
    parser.add_argument('--scorec', dest='scorec', type=float, help='score threshold for initial curve fit')
    parser.add_argument('--fit', dest='fit', help='how to fit systematic error: quad or inv?')
    parser.add_argument('--maxfev', dest='maxfev', type=int, help='maximum iterations for systematic error fit')

    parser.add_argument('--ions', dest='ions', help="which ions to consider in match", nargs='+')
    parser.add_argument('--ft', '--threshold', dest='threshold', type=float,
                        help='threshold (Da/e) for fragment matches')
    parser.add_argument('--mzc', dest='mzc', type=float, help='m/z cutoff for duplicates')
    parser.add_argument('--rtc', dest='rtc', type=float, help='RT cutoff for duplicates')
    parser.add_argument('--pvc', dest='pvc', type=float, help='p value cutoff for duplicates')
    parser.add_argument('--c', '--method', '--cm', dest='cut_method', help="how to treat coeluting peptides: keep or drop?")
    parser.add_argument('--scorefloor', '--sf', dest='scorefloor', type=float, help='minimum score for score cut')

    args = parser.parse_args()

    return args


# FOR CMD LINE

def main():
    args = parse_args()
    print(args)

    seqs = args.sequence
    files = args.files

    maxz = args.maxz if args.maxz else default_maxz
    minl = args.minl if args.minl else default_minl
    maxl = args.maxl if args.maxl else default_maxl
    ppmc_wide = args.ppmc_wide if args.ppmc_wide else default_ppmc_wide
    ppmc_narrow = args.ppmc_narrow if args.ppmc_narrow else default_ppmc_narrow
    scorec = args.scorec if args.scorec else default_scorec
    fit = args.fit if args.fit else default_fit
    maxfev = args.maxfev if args.maxfev else default_maxfev

    ions = args.ions if args.ions else default_ions
    threshold = args.threshold if args.threshold else default_threshold
    mzc = args.mzc if args.mzc else default_mzc
    rtc = args.rtc if args.rtc else default_rtc
    pvc = args.pvc if args.pvc else default_pvc
    cut_method = args.cut_method if args.cut_method else default_cut_method
    sf = args.scorefloor if args.scorefloor else default_sf

    path = args.path if args.path else ''
    plots_dir = args.plots if args.plots else None
    pool_out = args.output if args.output else None
    rangeslist = args.rangeslist if args.rangeslist else None
    nums_out = args.nums if args.nums else None

    analyze(seqs, files, maxz=maxz, minl=minl, maxl=maxl, ppmc_wide=ppmc_wide,
            ppmc_narrow=ppmc_narrow, scorec=scorec, fit=fit, maxfev=maxfev,
            ions=ions, threshold=threshold, mzc=mzc, rtc=rtc, pvc=pvc,
            cut_method=cut_method, sf=sf,
            path=path, plots_dir=plots_dir, pool_out=pool_out,
            rangeslist=rangeslist, nums_out=nums_out)
    return


if __name__ == '__main__':
    main()