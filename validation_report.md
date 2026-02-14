============================================================
 DATA VALIDATION REPORT
 Li Keqiang Index for Indian States - Phase 0
 Generated: 2026-02-13 23:28:26
============================================================

------------------------------------------------------------
SOURCE: GST State Collections
  Status:      GO
  Files:       statewise_GST_collection_2017-18.xlsx, statewise_GST_collection_2018-19.xlsx, statewise_GST_collection_2019-20.xlsx, statewise_GST_collection_2020-21.xlsx, statewise_GST_collection_2021-22.xlsx, statewise_GST_collection_2022-23.xlsx, statewise_GST_collection_2023-24.xlsx, statewise_GST_collection_2024-25.xlsx, statewise_GST_collection_2025-26.xlsx
  Rows:        439
  States:      36 matched to canonical list
  Time Range:  FY 2017-18 to FY 2025-26
  Parsed:      9/9 files
  Unmatched:   ['Daman and Diu', 'Note :', 'CBIC', 'OIDAR', 'The above numbers are provisional and the actual numbers may slightly vary on finalisation', 'Other Territory']

------------------------------------------------------------
SOURCE: Electricity Consumption (POSOCO)
  Status:      GO
  Files:       POSOCO_data.csv
  Rows:        4,788
  Columns:     139
  States:      33 matched to canonical list
  Time Range:  2013-01-02 to 2026-02-10
  File Size:   3.4 MB
  Missing:     3.2%
  Not Found:   ['Andaman & Nicobar Islands', 'Ladakh', 'Lakshadweep']
  Sample Cols: ['Punjab: EnergyMet', 'Haryana: EnergyMet', 'Rajasthan: EnergyMet', 'Delhi: EnergyMet', 'UP: EnergyMet']

------------------------------------------------------------
SOURCE: RBI Bank Credit
  Status:      CONDITIONAL GO
  Files:       rbi_scb_credit_by_state.xlsx, rbi_scb_deposits_by_state.xlsx
  Shape:       [49, 13]
  States:      36 matched to canonical list
  Unmatched:   ['(As at end-March)', '(₹ crore)', 'Region/State/Union Territory', 'NORTHERN REGION', 'NORTH-EASTERN REGION', 'EASTERN REGION', 'CENTRAL REGION', 'WESTERN REGION', 'Dadra & Nagar Haveli*', 'SOUTHERN REGION']

------------------------------------------------------------
SOURCE: EPFO Net Payroll (Formal Employment)
  Status:      GO
  Files:       epfo_sample_sep2025.xlsx
  Shape:       [213, 13]
  States:      32 matched to canonical list
  Time Range:  2017-18 From Sep-17 to 2024-25
  Time Series: Yes
  Not Found:   ['Dadra & Nagar Haveli and Daman & Diu', 'Lakshadweep', 'Puducherry', 'Sikkim']

============================================================
 SUMMARY: 3 GO | 1 CONDITIONAL | 0 NO-GO

 GATE: PASSED (review any CONDITIONAL items)
 Ready to proceed to Phase 1: Data Pipeline
============================================================