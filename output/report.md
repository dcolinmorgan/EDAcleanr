# Data Cleaning & EDA Report

## Dataset Overview

- **Original shape**: 5 rows × 7 columns
- **Cleaned shape**: 5 rows × 7 columns
- **Rows removed**: 0
- **Columns removed**: 0

## Issues Found

### Missing Values

| Column | Missing % |
|--------|-----------|
| name | 0.0% |
| node_type | 0.0% |
| score | 0.0% |
| id | 100.0% |
| result_category | 0.0% |
| source | 100.0% |
| relationship_count | 0.0% |

### Duplicate Rows

- **0** duplicate rows detected

### High Cardinality Columns

- name

### Zero Variance Columns

- node_type
- id
- result_category
- source
- relationship_count

## Cleaning Actions

No cleaning actions were performed.


## Key Statistics

### Numeric Statistics

```
          score   id  source  relationship_count
count  5.000000  0.0     0.0                 5.0
mean   0.818983  NaN     NaN                 0.0
std    0.151234  NaN     NaN                 0.0
min    0.661465  NaN     NaN                 0.0
25%    0.713583  NaN     NaN                 0.0
50%    0.761331  NaN     NaN                 0.0
75%    0.958535  NaN     NaN                 0.0
max    1.000000  NaN     NaN                 0.0
```

### Categorical Statistics

```
Column: name
name
heart failure             3
Valvular heart disease    1
heart                     1

Column: node_type
node_type
Entity    5

Column: result_category
result_category
general    5
```

### Correlation Matrix

```
                    score  id  source  relationship_count
score                 1.0 NaN     NaN                 NaN
id                    NaN NaN     NaN                 NaN
source                NaN NaN     NaN                 NaN
relationship_count    NaN NaN     NaN                 NaN
```

## Insights

1. [{'type': 'text', 'text': '<thinking> To analyze the cleaned dataset and provide insights, I will first need to gather some basic statistics and information about the dataset. I will start by using the `eda_numeric_stats` and `eda_categorical_stats` tools to get descriptive statistics for numeric and categorical columns, respectively. Additionally, I will use the `eda_correlation` tool to understand the relationships between numeric columns. Finally, I will use the `eda_figures` tool to see if there are any generated figures that can provide additional insights. </thinking>\n'}, {'type': 'tool_use', 'name': 'eda_numeric_stats', 'input': {}, 'id': 'tooluse_vZpeMQ7sSjc9Xz62qFydeP'}, {'type': 'tool_use', 'name': 'eda_categorical_stats', 'input': {}, 'id': 'tooluse_Y61ChzHzrCOokKBwdI534S'}, {'type': 'tool_use', 'name': 'eda_correlation', 'input': {}, 'id': 'tooluse_3KY88niv7xGVvp75HH5lO4'}, {'type': 'tool_use', 'name': 'eda_figures', 'input': {}, 'id': 'tooluse_HbvaHAou1G1si6K3qJOaPp'}]

## Figures

![Hist Score](figures/hist_score.png)

![Hist Id](figures/hist_id.png)

![Hist Source](figures/hist_source.png)

![Hist Relationship Count](figures/hist_relationship_count.png)

![Box Score](figures/box_score.png)

![Box Id](figures/box_id.png)

![Box Source](figures/box_source.png)

![Box Relationship Count](figures/box_relationship_count.png)

![Correlation Heatmap](figures/correlation_heatmap.png)
