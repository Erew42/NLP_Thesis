# LM2011 Additional Data and Item Sample Paths

Created: 2026-04-10

## LM2011 additional data directory

`C:\Users\erik9\Documents\SEC_Data\code\NLP_Thesis\full_data_run\LM2011_additional_data`

- `Data_material_weakness.xls` — material weakness reference sheet used in LM2011 controls/quality workflows.
- `F-F_Research_Data_Factors_daily.csv` — Fama-French daily factor time series.
- `FF_Siccodes_48_Industries.txt` — Fama-French 48-industry SIC classification mapping.
- `FF_Siccodes_5_Industries.txt` — Fama-French 5-industry SIC classification mapping.
- `Fin-Lit.txt` — litigation/litigious lexicon from Loughran-McDonald.
- `Fin-Neg.txt` — negative sentiment word list.
- `Fin-Pos.txt` — positive sentiment word list.
- `Fin-Unc.txt` — uncertainty word list.
- `Harvard_IV_NEG_Inf.txt` — Harvard IV negative lexical list.
- `LM2011_MasterDictionary.txt` — legacy/custom dictionary reference for LM2011 runs.
- `Loughran-McDonald_10X_Summaries_1993-2024.csv` — Loughran-McDonald summaries with year range context.
- `Loughran-McDonald_MasterDictionary_1993-2024.csv` — canonical Loughran-McDonald dictionary (1993–2024).
- `MW-Strong.txt` — strong modal words list.
- `MW-Weak.txt` — weak modal words list.
- `Stop_Words_Currencies.txt` — currency-specific stop words.
- `Stop_Words_Dates_Numbers.txt` — date/number stop words.
- `Stop_Words_Generic.txt` — general stop words.
- `Stop_Words_Geographic.txt` — geographic stop words.
- `Stop_Words_Names.txt` — names/entities stop-word list.
- `Word_lists_for_22When_Is_a.xlsx` — additional word-list workbook used in preprocessing.

## Sample directory for LM2011 item data

`C:\Users\erik9\Documents\SEC_Data\code\NLP_Thesis\full_data_run\sample_5pct_seed42\results\sec_ccm_unified_runner\local_sample\items_analysis`

- Contains year-partitioned parquet item shards used as sample item inputs:
  - `1995.parquet` through `2024.parquet`.

## Related FinBERT sample item path (smoke input)

`C:\Users\erik9\Documents\SEC_Data\code\NLP_Thesis\full_data_run\sample_5pct_seed42\results\finbert_item_analysis_runner\real_subset_smoke_input\items_analysis`

- Contains `2006.parquet`.

## Default-script paths referenced

- `src\thesis_pkg\notebooks_and_scripts\lm2011_sample_post_refinitiv_runner.py`:
  - Additional data: `full_data_run\LM2011_additional_data`
  - Upstream items dir default: `results\sec_ccm_unified_runner\local_sample\items_analysis` under the sample root.
- `src\thesis_pkg\notebooks_and_scripts\finbert_item_analysis_runner.py`:
  - Default sample items dir: `full_data_run\sample_5pct_seed42\results\sec_ccm_unified_runner\local_sample\items_analysis`.
