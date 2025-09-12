# EDA module (app/eda/eda.py)

Functions
- infer_format(path) → 'csv' | 'tsv' | 'excel'
  - Sniffs extension and simple content
- detect_delimiter(sample) → delimiter
  - Chooses among tab/comma/semicolon/pipe by median split count
- is_large_file(path, large_file_mb)
  - Returns True if file size > threshold
- load_sampled_chunked_csv(path, delimiter, sample_target)
  - Returns {'df': sample_df, 'missing': {col: {count,pct}}, 'total_rows': int}
- load_dataframe(path, file_format, sheet_name, delimiter, sample_rows)
  - Loads CSV/TSV/Excel with delimiter sniffing when needed
- compute_eda(df, max_topk=5)
  - Core EDA: shape, dtypes, missing, nunique, id/constant cols, numeric stats,
    top correlations (on small frames), categorical top-k, rare levels, time-like hints, recommendations
- compute_target_relations(df, target, max_features=10)
  - Task type and simple numeric/categorical relationships
- compute_timeseries_hints(df, time_col, metric_col)
  - Daily aggregation and coarse autocorr hints (acf7/acf30)
- generate_basic_plots(job_id, df, plots_dir)
  - Saves hist/bar/missingness PNGs to plots_dir; returns /static URLs
- generate_basic_plots_storage(storage, job_id, df)
  - Stores PNGs via storage adapter; returns signed/served URLs

Notes
- All functions are defensive; EDA is designed to be fast and resilient.
- Pandas >=2: avoid deprecated infer_datetime_format; rely on errors='coerce'.

