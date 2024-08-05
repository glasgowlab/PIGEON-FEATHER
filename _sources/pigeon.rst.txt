
PIGEON
======

**pigeon** is a stand-only program that takes in a list of peptide pool csv 
and outputs a cleaned csv of pooled data. It also outputs a rangeslist table for merge in FEATHER.

CMD Options
-----------

Type ``pigeon --help`` to see the following options:

| ``--t``, ``--table``: Path(s) to input csv. Required
| ``--o``, ``--out``: path to output csv, if any.
| ``--p``, ``--plots``: directory for output plots. Default: './csvplots'
| ``--ov``, ``--overlap``: flag specifying how to treat overlapping duplicates. Options: 'keep', 'drop', 'select' (or any other). Default: 'select'
| ``--r``, ``--rangeslist``: path to output rangeslist, if any.
| ``--RTC``: RT cutoff for duplicate peptides. Default: 0.5
| ``--MZC``: m/z cutoff for duplicate peptides. Default: 0.1
| ``--scoreC``: Score threshold for provisional cut for trendline fit. Default: 150
| ``--ppmC``: m/z deviation cutoff (in ppm) for cut about trendline. Default: 7
| ``--maxfev``: maxfev for curve_fit for trendline.

Default inputs

| ``--t`` (list of peptide pool csvs)

Default outputs
  
| ``--o`` (batched and cleaned peptide pool csv)
| ``--r`` (rangeslist table for merge in post-PIGEON)

In the plot directory, the following files are generated:

score histograms:

  | `hist-scores-all.png`: Histogram of score for all pooled data.
  | `hist-scores-cut.png`: Histogram of score for high scoring data used for curve fit.
  | `hist-scores-ppm.png`: Histogram of score after cut at ppmC from trendline.
  | `hist-scores-topscores.png`: Histogram of score after dropping all but best match for each charge state.
  | `hist-scores-flagged.png`: Histogram of score for non-overlapping duplicates.
  | `hist-scores-yellow.png`: Histogram of score for overlapping duplicates.
  | `hist-scores-clean.png`: Histogram of score for final cleaned data.

scatterplots:

  | `scatter-30ppm.png`: All data before cuts, ppm m/z deviation vs m/z measured, colorbar Score.
  | `scatter-hscoring.png`: High scoring data used for curve fit.
  | `scatter-7ppm.png`: All data after cut at ppmC from trendline.
  | `scatter-topscore.png`: Data after dropping all but best match for each charge state.
  | `scatter-flagged.png`: Data dropped for being duplicates, non-overlapping.
  | `scatter-yellow.png`: Data dropped for being duplicates, overlapping.
  | `scatter-clean.png`: Final cleaned data.

ppm histogram:

  `hist-ppms.png`: Histogram of abs(distance from trendline) for all data, showing location of cut.


Example Usage
-------------

Below are examples of how to use the ``pigeon`` program with various arguments:

.. code-block:: bash

   pigeon --t pool-1.csv pool-2.csv pool-3.csv --p csvplots --ov keep --o keep-pooled.csv --r rangeslist.csv --MZC 0.05 --RTC 0.25 --scoreC 200 --ppmC 5 --maxfev 2000

Alternative usage example:

.. code-block:: bash

   pigeon --t pool-1.csv --o select-pooled.csv --r rangeslist.csv
