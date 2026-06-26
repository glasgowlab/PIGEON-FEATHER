pigeon_feather.data.RangeList
-----------------------------

.. currentmodule:: pigeon_feather.data
.. autoclass:: RangeList

    .. automethod:: __init__

    .. rubric:: Methods

    .. autosummary::
    
     ~RangeList.to_set
     ~RangeList.to_dataframe
     ~RangeList.to_csv
     ~RangeList.union
     ~RangeList.intersection
     ~RangeList.difference


pigeon_feather.data.HDXMSDataCollection
---------------------------------------

.. currentmodule:: pigeon_feather.data
.. autoclass:: HDXMSDataCollection

    .. automethod:: __init__



pigeon_feather.data.HDXMSData
-----------------------------

.. currentmodule:: pigeon_feather.data
.. autoclass:: HDXMSData

    .. automethod:: __init__

    .. rubric:: Methods

    .. autosummary::
    
     ~HDXMSData.add_state
     ~HDXMSData.load_protein_sequence
     ~HDXMSData.get_state
     ~HDXMSData.plot_res_coverage
     ~HDXMSData.reindex_peptide_from_pdb
     ~HDXMSData.to_dataframe
     ~HDXMSData.to_bayesianhdx_format

    .. rubric:: Attributes

    .. autosummary::
     
     ~HDXMSData.num_states


pigeon_feather.data.ProteinState
--------------------------------

.. currentmodule:: pigeon_feather.data

.. autoclass:: ProteinState

   
   .. automethod:: __init__

   
   .. rubric:: Methods

   .. autosummary::
   
  
    ~ProteinState.add_peptide
    ~ProteinState.num_peptides
    ~ProteinState.get_peptide
    ~ProteinState.add_new_peptides_by_subtract
    ~ProteinState.add_all_subtract



pigeon_feather.data.Peptide
----------------------------

.. currentmodule:: pigeon_feather.data

.. autoclass:: Peptide

   
   .. automethod:: __init__

   .. rubric:: Methods

   .. autosummary::
   
    ~Peptide.add_timepoint
    ~Peptide.get_deut
    ~Peptide.get_deut_percent
    ~Peptide.get_timepoint



pigeon_feather.data.Timepoint
-----------------------------

.. currentmodule:: pigeon_feather.data

.. autoclass:: Timepoint

   
   .. automethod:: __init__

   .. rubric:: Methods

   .. autosummary::
    
    ~Timepoint.load_raw_ms_csv



pigeon_feather.HDXStatePeptideCompares
---------------------------------------

.. currentmodule:: pigeon_feather.data

.. autoclass:: HDXStatePeptideCompares

   
   .. automethod:: __init__

   .. rubric:: Methods

   .. autosummary::
   
    ~HDXStatePeptideCompares.add_all_compare
    ~HDXStatePeptideCompares.to_dataframe




pigeon_feather.PeptideCompare
-----------------------------

.. currentmodule:: pigeon_feather.data

.. autoclass:: PeptideCompare

    .. automethod:: __init__
       
    .. rubric:: Methods

    .. autosummary::
    
     ~PeptideCompare.get_deut_diff




pigeon_feather.HDXStateResidueCompares
--------------------------------------

.. currentmodule:: pigeon_feather.data
.. autoclass:: HDXStateResidueCompares

    .. automethod:: __init__

    .. rubric:: Methods

    .. autosummary::
    
     ~HDXStateResidueCompares.add_all_compare
     ~HDXStateResidueCompares.get_residue_compare



pigeon_feather.ResidueCompare
-----------------------------

.. currentmodule:: pigeon_feather.data
.. autoclass:: ResidueCompare

    .. automethod:: __init__

    .. rubric:: Methods

    .. autosummary::
    
     ~ResidueCompare.find_peptides_containing_res
     ~ResidueCompare.get_deut_diff

    

pigeon_feather.SimulatedData
----------------------------

.. currentmodule:: pigeon_feather.data
.. autoclass:: SimulatedData

    .. automethod:: __init__

    .. rubric:: Methods

    .. autosummary::
    
     ~SimulatedData.gen_seq
     ~SimulatedData.gen_logP
     ~SimulatedData.cal_k_init
     ~SimulatedData.cal_k_ex
     ~SimulatedData.calculate_incorporation
     ~SimulatedData.gen_peptides
     ~SimulatedData.convert_to_hdxms_data




pigeon_feather.plot.UptakePlotsCollection
-----------------------------------------

.. currentmodule:: pigeon_feather.plot
.. autoclass:: UptakePlotsCollection

    .. automethod:: __init__

    .. rubric:: Methods

    .. autosummary::
    
     ~UptakePlotsCollection.add_plot
     ~UptakePlotsCollection.add_plot_all
     ~UptakePlotsCollection.save_plots


pigeon_feather.plot.UptakePlot
------------------------------

.. currentmodule:: pigeon_feather.plot
.. autoclass:: UptakePlot

    .. automethod:: __init__

    .. rubric:: Methods

    .. autosummary::
    
     ~UptakePlot.make_uptakeplot
     ~UptakePlot.get_average_peptide
     ~UptakePlot.make_title
     ~UptakePlot.make_color_dict




