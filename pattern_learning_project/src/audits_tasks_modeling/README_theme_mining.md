# Audits + Tasks + Incident Theme Mining

This pipeline clusters the unified safety event table into source-aware themes:

1. `incident_hazard` records: incidents, injuries, near misses, and hazard identifications
2. `audit_observation` records: audits, inspections, observations, unsafe acts/conditions
3. `task_action` records: tasks, corrective actions, open actions, overdue actions

The pipeline now defaults to a small 20,000-row concept-proof sample. You can increase it to 60,000 or 100,000 in `config.py` after the first test run.

The pipeline keeps `00_build_unified_text_events.py` as the first step. That script builds the common table `safety_text_event.csv.gz` with shared fields such as `event_id`, `source_type`, `event_date`, `location_id`, `location_path`, `clean_text`, injury flags, and task status fields.

## Why POC sampling is now the default

Embedding every event can take a long time. For fast concept proof, `01_prepare_theme_text.py` now:

1. Reads the full unified event table.
2. Scores locations by safety-signal volume: injuries, near misses, hazard identifications, audits/observations, tasks/actions, open tasks, and overdue tasks.
3. Selects the top case-heavy locations.
4. Samples no more than `POC_MAX_TOTAL_RECORDS` rows from those selected locations.
5. Saves the selected sample to the theme input files used by embeddings and clustering.

This lets you test the concept quickly before running the full dataset.

## Main config settings

Edit `src/audits_tasks_modeling/config.py`.

```python
ENABLE_POC_MAJOR_LOCATION_SAMPLE = True
POC_TOP_LOCATIONS = 12
POC_MAX_TOTAL_RECORDS = 20000
POC_FAMILY_QUOTAS = {
    FAMILY_INCIDENT_HAZARD: 8000,
    FAMILY_AUDIT_OBSERVATION: 7000,
    FAMILY_TASK_ACTION: 5000,
}
```

For a larger POC:

```python
POC_MAX_TOTAL_RECORDS = 100000
POC_FAMILY_QUOTAS = {
    FAMILY_INCIDENT_HAZARD: 40000,
    FAMILY_AUDIT_OBSERVATION: 30000,
    FAMILY_TASK_ACTION: 30000,
}
```

For the full dataset later:

```python
ENABLE_POC_MAJOR_LOCATION_SAMPLE = False
MAX_RECORDS_PER_FAMILY = 0
```

## Run order

Use the existing unified dataset:

```bash
python pattern_learning_project/src/audits_tasks_modeling/run_theme_mining_end_to_end.py
```

Or run step by step:

```bash
python pattern_learning_project/src/audits_tasks_modeling/01_prepare_theme_text.py
python pattern_learning_project/src/audits_tasks_modeling/02_generate_theme_embeddings.py
python pattern_learning_project/src/audits_tasks_modeling/03_cluster_by_family.py
python pattern_learning_project/src/audits_tasks_modeling/04_label_theme_clusters.py
python pattern_learning_project/src/audits_tasks_modeling/05_build_location_theme_period_profiles.py
python pattern_learning_project/src/audits_tasks_modeling/06_build_cross_family_theme_links.py
```

To rebuild the unified dataset first, set:

```python
RUN_STEP_00_IN_END_TO_END = True
```

then run `run_theme_mining_end_to_end.py`.

## Key outputs

All outputs are saved under:

```text
pattern_learning_project/outputs/audits_tasks_modeling/
```

### POC sampling outputs

```text
02_theme_input/poc_major_location_profile.csv
02_theme_input/poc_selected_locations.csv
02_theme_input/poc_sampling_summary.json
02_theme_input/theme_input_all.csv
02_theme_input/theme_input_incident_hazard.csv
02_theme_input/theme_input_audit_observation.csv
02_theme_input/theme_input_task_action.csv
```

Use `poc_selected_locations.csv` to confirm which major locations were selected before embedding.

### Clustering outputs

```text
04_theme_clusters/event_theme_assignments.csv
05_theme_catalog/theme_catalog_review.csv
05_theme_catalog/theme_representative_examples.csv
06_location_theme_profiles/location_theme_period_profile_Y.csv
06_location_theme_profiles/location_period_top_themes_Y.csv
06_location_theme_profiles/theme_period_trends_Y.csv
07_theme_links/cross_family_theme_links_Y.csv
```

## Notes

- No OpenAI or paid API is used.
- Embeddings use `sentence-transformers/all-MiniLM-L6-v2` by default.
- If sentence-transformers cannot load, the embedding script falls back to TF-IDF + SVD.
- Cross-family links are review candidates, not causal proof.

## Audit clustering improvement

The audit/observation family is now preprocessed differently from incidents and tasks.
The previous audit clusters were often dominated by routine form language such as
`Scheduled`, `Inspection`, `Safety Observation`, `Inspecion montacargas`, and
`Inspecion vehicular`. Those records are useful for accounting, but they should
not drive safety theme clustering.

Step 01 now splits audit records into two paths:

```text
meaningful audit findings / unsafe observations -> embedding + clustering
routine scheduled inspections / generic safe observations -> accounting only
```

Important outputs:

```text
02_theme_input/audit_activity_accounting.csv
02_theme_input/audit_cluster_eligibility_summary.csv
02_theme_input/audit_excluded_from_clustering_sample.csv
02_theme_input/theme_input_audit_observation.csv
```

Review `audit_cluster_eligibility_summary.csv` first. It shows how many audit
records were kept for clustering versus excluded as routine/scheduled/generic.
The file `theme_input_audit_observation.csv` should now contain mostly specific
unsafe acts, unsafe conditions, and meaningful risk observations rather than
routine inspection titles.

Useful config switches:

```python
AUDIT_CLUSTER_ONLY_MEANINGFUL_FINDINGS = True
AUDIT_KEEP_ROUTINE_INSPECTIONS_FOR_ACCOUNTING = True
AUDIT_INCLUDE_GENERAL_OBSERVATIONS_WITH_RISK_KEYWORDS = True
AUDIT_EXCLUDE_SAFE_POSITIVE_OBSERVATIONS_FROM_CLUSTERING = True
AUDIT_MIN_MEANINGFUL_OBSERVATION_CHARS = 60
```

After changing audit filtering rules, rerun from Step 01:

```bash
python pattern_learning_project/src/audits_tasks_modeling/01_prepare_theme_text.py
python pattern_learning_project/src/audits_tasks_modeling/02_generate_theme_embeddings.py
python pattern_learning_project/src/audits_tasks_modeling/03_cluster_by_family.py
python pattern_learning_project/src/audits_tasks_modeling/04_label_theme_clusters.py
```

## Audit clustering update: risk vs positive control vs activity

The audit path is now split before embedding/clustering:

```text
audit_risk
  Unsafe Act, Unsafe Condition, and meaningful risk observations.
  These records are embedded and clustered.

audit_positive
  Safe Act, Safe Condition, and meaningful positive-control observations.
  These records are embedded and clustered separately from unsafe findings.

audit_activity
  Scheduled inspections, checklists, risk-assessment/admin records, generic/low-information observations.
  These records are not clustered. They are retained for accounting outputs.
```

This is intentional. Safe and unsafe observations often share the same nouns such as PPE, guard, forklift, cable, or housekeeping. If they are clustered together, the algorithm tends to create broad "PPE observation" or "housekeeping observation" clusters that mix control failures and working controls. Splitting the audit path keeps risk themes and positive-control themes interpretable.

### New audit clustering outputs

Step 01 now writes:

```text
outputs/audits_tasks_modeling/02_theme_input/theme_input_audit_risk.csv
outputs/audits_tasks_modeling/02_theme_input/theme_input_audit_positive.csv
outputs/audits_tasks_modeling/02_theme_input/audit_activity_accounting.csv
outputs/audits_tasks_modeling/02_theme_input/audit_cluster_eligibility_summary.csv
outputs/audits_tasks_modeling/02_theme_input/audit_risk_for_clustering_sample.csv
outputs/audits_tasks_modeling/02_theme_input/audit_positive_for_clustering_sample.csv
outputs/audits_tasks_modeling/02_theme_input/audit_excluded_from_clustering_sample.csv
```

The rest of the pipeline automatically treats `audit_risk` and `audit_positive` as separate source families. Final cluster files therefore include:

```text
outputs/audits_tasks_modeling/04_theme_clusters/event_theme_assignments_audit_risk.csv
outputs/audits_tasks_modeling/04_theme_clusters/event_theme_assignments_audit_positive.csv
```

The main review catalog still remains:

```text
outputs/audits_tasks_modeling/05_theme_catalog/theme_catalog_review.csv
```

### POC sample quotas

The default POC sample is still capped at 20,000 records, now split as:

```python
POC_FAMILY_QUOTAS = {
    FAMILY_INCIDENT_HAZARD: 8000,
    FAMILY_AUDIT_RISK: 3500,
    FAMILY_AUDIT_POSITIVE: 3500,
    FAMILY_TASK_ACTION: 5000,
}
```

If there are fewer eligible audit-risk or audit-positive rows at the selected locations, the script keeps all eligible rows and does not fill the quota with routine inspection records.

### How to run

Run from Step 01 after replacing these files:

```bash
python pattern_learning_project/src/audits_tasks_modeling/01_prepare_theme_text.py
python pattern_learning_project/src/audits_tasks_modeling/02_generate_theme_embeddings.py
python pattern_learning_project/src/audits_tasks_modeling/03_cluster_by_family.py
python pattern_learning_project/src/audits_tasks_modeling/04_label_theme_clusters.py
python pattern_learning_project/src/audits_tasks_modeling/05_build_location_theme_period_profiles.py
python pattern_learning_project/src/audits_tasks_modeling/06_build_cross_family_theme_links.py
```

Or run end-to-end:

```bash
python pattern_learning_project/src/audits_tasks_modeling/run_theme_mining_end_to_end.py
```
