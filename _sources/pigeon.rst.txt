
PIGEON
======

**pigeon** is a stand-only program that takes in a list of peptide pool csv 
and outputs a cleaned csv of pooled data. It also outputs a rangeslist table for merge in FEATHER.

CMD Options
-----------

Type ``pigeon --help`` to see the following options:

| ``--s``, ``--seq``, ``--sequence``: sequence(s) of target proteins. Required
| ``--f``, ``--files``: path(s) to input mgf(s). Required
| ``--path``: path for output directory.
| ``--o``, ``--output``: path(s) for peptide pool output, if any.
| ``--r``, ``--rangeslist``: path(s) for output rangeslist(s), if any.
| ``--n``, ``--nums``: path for cut statistics ouput, if any.
| ``--p``, ``--plots``: directory for output plots.

| ``--maxz``: maximum charge for theoretical peptides and fragments. Default: 3
| ``--minl``: minimum length for theoretical peptides. Default: 3
| ``--maxl``: maximum length for theoretical peptides. Default: 15
| ``--ions``: which fragment ions to consider in match. Default: 'b', 'b-H2O', 'b-NH3', 'y', 'y-H2O', 'y-NH3'
| ``--ft``, ``--threshold``: m/z threshold for fragment matches. Default: 0.02

| ``--ppmc_w``: ppm error cutoff for initial match. Default: 30
| ``--ppmc_n``: ppm error cutoff after fit. Default: 7
| ``--scorec``: score threshold for provisional cut for trendline fit. Default: 0.05
| ``--fit``: how to fit systematic error. Options: 'quad', 'inv'. Default: 'quad'
| ``--maxfev``: maximum iterations for systematic error fit.


| ``--c``, ``--cm``, ``--method``: flag specifying how to treat co-eluting peptides. Options: 'keep', 'drop'. Default: 'drop'
| ``--rtc``: RT cutoff for duplicate peptides. Default: 0.5
| ``--mzc``: m/z cutoff for duplicate peptides. Default: 0.1
| ``--pvc``: p-value cutoff for duplicate peptides. Default: 0.05
| ``--scorefloor``, ``--sf``: minimum score for score cut, if any. Default: 0

Default inputs

| ``--f`` (list of MS2 files in .mgf format)
| ``--s`` (list of protein sequences)

Default outputs
  
| ``--o`` (peptide pool csv(s) batched and cleaned)
| ``--r`` (rangeslist table(s) for merge in post-PIGEON)
| ``--n`` (peptide pool features at each step)

In the plot directory, the following files are generated:

score histograms:

  | `all-hist.pdf`: Histogram of score for all pooled data.
  | `truth-hist.pdf`: Histogram of score for high scoring data used for curve fit.
  | `ppm_cut-hist.pdf`: Histogram of score after cut at ppmC from trendline.
  | `single-hist.pdf`: Histogram of score after dropping all but best match for each peptide.
  | `ambig-hist.pdf`: Histogram of score for discarded duplicates.
  | `clean-hist.pdf`: Histogram of score for final cleaned data.

scatterplots:

  | `all-scatter.pdf`: PPM error vs. m/z (colorbar score) for all pooled data.
  | `truth-scatter.pdf`: PPM error vs. m/z (colorbar score) for high scoring data used for curve fit.
  | `ppm_cut-scatter.pdf`: PPM error vs. m/z (colorbar score) after cut at ppmC from trendline.
  | `single-scatter.pdf`: PPM error vs. m/z (colorbar score) after dropping all but best match for each peptide.
  | `ambig-scatter.pdf`: PPM error vs. m/z (colorbar score) for discarded duplicates.
  | `clean-scatter.pdf`: PPM error vs. m/z (colorbar score) for final cleaned data.

Example Usage
-------------

Below are examples of how to use the ``pigeon`` program with various arguments:

.. code-block:: bash

   pigeon --s [protein sequence] [pepsin sequence] --c keep --p keep-plots --o keep-pool.csv keep-pepsin-pool.csv --r keep-ranges.csv keep-pepsin-ranges.csv --n keep-nums.out --f apo.mgf

Alternative usage example:

.. code-block:: bash

   pigeon --f apo1.mgf apo2.mgf apo3.mgf --s [protein 1 sequence] [protein 2 sequence] --p drop-plots --ov drop --o drop-pooled-1.csv drop-pooled-2 --mzc 0.05 --rtc 0.25 --pvc 1 --scorec 0.01 --ppmc_n 5 --maxfev 2000
