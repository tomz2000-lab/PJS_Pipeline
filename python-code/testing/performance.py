"""
Performance evaluation module for incentive extraction from job advertisements.

This module calculates accuracy and recall metrics by comparing extracted job data 
against a validation dataset. It processes incentive and non-incentive columns 
separately and generates both overall and per-column performance metrics.

The results are printed to console and saved to a CSV file for historical tracking.
"""

import pandas as pd
import numpy as np
import csv
import os
from collections import defaultdict
from datetime import datetime


"""
Load datasets for validation and testing.

The job_analysis_export can be generated running read_db.py.
See this Documentation for more information.
"""
validation_df = pd.read_csv('validation_file_new.csv')
export_df = pd.read_csv('job_analysis_export_05.05.csv')

"""
Clean up column names in dataframes to ensure consistency.

:param dataframes: List of dataframes to process
:type dataframes: list
:return: List of dataframes with cleaned column names
:rtype: list
"""
validation_df.columns = [col.replace(' ', '_') for col in validation_df.columns]
export_df.columns = [col.replace(' ', '_') for col in export_df.columns]

incentive_columns = [
    'Gehalt_anhand_von_Tarifklassen', 'Überstundenvergütung', 'Gehaltserhöhungen',
    'Aktienoptionen/Gewinnbeteiligung', 'Boni', 'Sonderzahlungen', '13._Gehalt', 'Betriebliche_Altersvorsorge',
    'Flexible_Arbeitsmodelle', 'Homeoffice', 'Weiterbildung_und_Entwicklungsmöglichkeiten',
    'Gesundheit_und_Wohlbefinden', 'Finanzielle_Vergünstigungen', 'Mobilitätsangebote',
    'Verpflegung', 'Arbeitsumfeld_Ausstattung', 'Zusätzliche_Urlaubstage',
    'Familien_Unterstützung', 'Onboarding_und_Mentoring_Programme', 'Teamevents_Firmenfeiern'
]

non_incentive_columns = [
    'Job_Titel', 'Portal_Name', 'Datum', 'Stadt', 'Bundesland', 'Land',
    'Zeitmodell', 'Position', 'Beschäftigungsart', 'Berufserfahrung_vorausgesetzt'
]

# Function to calculate accuracy and recall
def calculate_accuracy_recall(true_positives, false_positives, false_negatives, true_negatives):
    """Calculate accuracy and recall metrics for classification results.
    
    :param true_positives: Number of correctly identified positive cases
    :type true_positives: int
    :param false_positives: Number of incorrectly identified positive cases
    :type false_positives: int
    :param false_negatives: Number of incorrectly identified negative cases
    :type false_negatives: int
    :param true_negatives: Number of correctly identified negative cases
    :type true_negatives: int
    :return: Tuple containing accuracy and recall values
    :rtype: tuple(float, float)
    """
    accuracy = (true_positives + true_negatives) / (true_positives + false_positives + false_negatives + true_negatives) if (true_positives + false_positives + false_negatives + true_negatives) > 0 else 1.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 1.0
    return accuracy, recall

# Create a dictionary to store metrics for each column
metrics = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0})

# Create a dictionary to store metrics by category
category_metrics = {
    'overall': {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0},
    'incentive': {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0},
    'non_incentive': {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0}
}

# Create URL-based keys for validation and export dataframes
validation_df['url_key'] = validation_df['Job_URL'].apply(lambda x: x.split('?')[0] if isinstance(x, str) else x)
export_df['url_key'] = export_df['Job_URL'].apply(lambda x: x.split('?')[0] if isinstance(x, str) else x)

# Group by company and URL
validation_grouped = validation_df.groupby(['Unternehmen', 'url_key'])
export_grouped = export_df.groupby(['Unternehmen', 'url_key'])

# Get unique keys from both datasets
validation_keys = set(validation_grouped.groups.keys())
export_keys = set(export_grouped.groups.keys())

# Find common keys and keys unique to each dataset
common_keys = validation_keys.intersection(export_keys)
validation_only_keys = validation_keys - export_keys
export_only_keys = export_keys - validation_keys

"""
Process the jobs in both files based on thier uniquness. 
There are 59 unique jobs in the validation set.
The jobs are assigned as unique, if the have a unique locatiion, position, time model and URL.
The comparison is further divided by inccnetives and non-incenitves.
"""
for key in common_keys:
    company, url = key
    val_group = validation_grouped.get_group(key)
    exp_group = export_grouped.get_group(key)
    
    # Use the first entry from each group for comparison
    val_job = val_group.iloc[0]
    exp_job = exp_group.iloc[0]
    
    # Compare each incentive column
    for col in incentive_columns:
        if col in validation_df.columns and col in export_df.columns:
            val_value = val_job[col] if col in val_job else np.nan
            exp_value = exp_job[col] if col in exp_job else np.nan
            
            # Convert to same data type for comparison
            if not pd.isna(val_value) and not pd.isna(exp_value):
                val_value = int(val_value) if isinstance(val_value, (int, float, bool, np.number)) else val_value
                exp_value = int(exp_value) if isinstance(exp_value, (int, float, bool, np.number)) else exp_value
            
            # Compare values
            if pd.isna(val_value) and pd.isna(exp_value):
                # Both NaN - true negative
                metrics[col]['tn'] += 1
                category_metrics['incentive']['tn'] += 1
                category_metrics['overall']['tn'] += 1
            elif pd.isna(val_value) and not pd.isna(exp_value):
                # NaN in validation but value in export - false positive
                metrics[col]['fp'] += 1
                category_metrics['incentive']['fp'] += 1
                category_metrics['overall']['fp'] += 1
            elif not pd.isna(val_value) and pd.isna(exp_value):
                # Value in validation but NaN in export - false negative
                metrics[col]['fn'] += 1
                category_metrics['incentive']['fn'] += 1
                category_metrics['overall']['fn'] += 1
            elif val_value == exp_value:
                # Values match - true positive
                metrics[col]['tp'] += 1
                category_metrics['incentive']['tp'] += 1
                category_metrics['overall']['tp'] += 1
            else:
                # Values don't match - false positive and false negative
                metrics[col]['fp'] += 1
                metrics[col]['fn'] += 1
                category_metrics['incentive']['fp'] += 1
                category_metrics['incentive']['fn'] += 1
                category_metrics['overall']['fp'] += 1
                category_metrics['overall']['fn'] += 1
    
    # Compare each non-incentive column
    for col in non_incentive_columns:
        if col in validation_df.columns and col in export_df.columns:
            val_value = val_job[col] if col in val_job else np.nan
            exp_value = exp_job[col] if col in exp_job else np.nan
            
            # Compare values
            if pd.isna(val_value) and pd.isna(exp_value):
                # Both NaN - true negative
                metrics[col]['tn'] += 1
                category_metrics['non_incentive']['tn'] += 1
                category_metrics['overall']['tn'] += 1
            elif pd.isna(val_value) and not pd.isna(exp_value):
                # NaN in validation but value in export - false positive
                metrics[col]['fp'] += 1
                category_metrics['non_incentive']['fp'] += 1
                category_metrics['overall']['fp'] += 1
            elif not pd.isna(val_value) and pd.isna(exp_value):
                # Value in validation but NaN in export - false negative
                metrics[col]['fn'] += 1
                category_metrics['non_incentive']['fn'] += 1
                category_metrics['overall']['fn'] += 1
            elif str(val_value).strip() == str(exp_value).strip():
                # Values match - true positive
                metrics[col]['tp'] += 1
                category_metrics['non_incentive']['tp'] += 1
                category_metrics['overall']['tp'] += 1
            else:
                # Values don't match - false positive and false negative
                metrics[col]['fp'] += 1
                metrics[col]['fn'] += 1
                category_metrics['non_incentive']['fp'] += 1
                category_metrics['non_incentive']['fn'] += 1
                category_metrics['overall']['fp'] += 1
                category_metrics['overall']['fn'] += 1

# Process validation-only keys (jobs only in validation)
for key in validation_only_keys:
    val_group = validation_grouped.get_group(key)
    val_job = val_group.iloc[0]
    
    # Count all columns as false negatives
    for col in incentive_columns:
        if col in validation_df.columns:
            val_value = val_job[col] if col in val_job else np.nan
            if not pd.isna(val_value):
                metrics[col]['fn'] += 1
                category_metrics['incentive']['fn'] += 1
                category_metrics['overall']['fn'] += 1
    
    for col in non_incentive_columns:
        if col in validation_df.columns:
            val_value = val_job[col] if col in val_job else np.nan
            if not pd.isna(val_value):
                metrics[col]['fn'] += 1
                category_metrics['non_incentive']['fn'] += 1
                category_metrics['overall']['fn'] += 1

# Process export-only keys (jobs only in export)
for key in export_only_keys:
    exp_group = export_grouped.get_group(key)
    exp_job = exp_group.iloc[0]
    
    # Count all columns as false positives
    for col in incentive_columns:
        if col in export_df.columns:
            exp_value = exp_job[col] if col in exp_job else np.nan
            if not pd.isna(exp_value):
                metrics[col]['fp'] += 1
                category_metrics['incentive']['fp'] += 1
                category_metrics['overall']['fp'] += 1
    
    for col in non_incentive_columns:
        if col in export_df.columns:
            exp_value = exp_job[col] if col in exp_job else np.nan
            if not pd.isna(exp_value):
                metrics[col]['fp'] += 1
                category_metrics['non_incentive']['fp'] += 1
                category_metrics['overall']['fp'] += 1

# Calculate accuracy and recall for each category
results = {}
for category, values in category_metrics.items():
    accuracy, recall = calculate_accuracy_recall(values['tp'], values['fp'], values['fn'], values['tn'])
    results[category] = {'accuracy': accuracy, 'recall': recall}

# Calculate accuracy and recall for each column
column_results = {}
for col, values in metrics.items():
    accuracy, recall = calculate_accuracy_recall(values['tp'], values['fp'], values['fn'], values['tn'])
    column_results[col] = {'accuracy': accuracy, 'recall': recall}

# Print results
print(f"Total jobs in validation: {len(validation_df)}")
print(f"Total jobs in export: {len(export_df)}")
print(f"Unique job combinations in validation: {len(validation_keys)}")
print(f"Unique job combinations in export: {len(export_keys)}")
print(f"Jobs found in both datasets: {len(common_keys)}")
print(f"Jobs only in validation: {len(validation_only_keys)}")
print(f"Jobs only in export: {len(export_only_keys)}")

print("\nOverall Metrics:")
print(f"Accuracy: {results['overall']['accuracy']:.4f}")
print(f"Recall: {results['overall']['recall']:.4f}")

print("\nIncentive Metrics:")
print(f"Accuracy: {results['incentive']['accuracy']:.4f}")
print(f"Recall: {results['incentive']['recall']:.4f}")

print("\nNon-Incentive Metrics:")
print(f"Accuracy: {results['non_incentive']['accuracy']:.4f}")
print(f"Recall: {results['non_incentive']['recall']:.4f}")

print("\nMetrics by Incentive Column:")
for col in incentive_columns:
    if col in column_results:
        print(f"{col}:")
        print(f" Accuracy: {column_results[col]['accuracy']:.4f}")
        print(f" Recall: {column_results[col]['recall']:.4f}")

print("\nMetrics by Non-Incentive Column:")
for col in non_incentive_columns:
    if col in column_results:
        print(f"{col}:")
        print(f" Accuracy: {column_results[col]['accuracy']:.4f}")
        print(f" Recall: {column_results[col]['recall']:.4f}")

"""
The found values for accuracy and recall are now written in a csv with the current date.
This data can be used to track process in performance over al the columns.
"""
current_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

csv_row = {
    'date': current_date,
    'overall_accuracy': results['overall']['accuracy'],
    'overall_recall': results['overall']['recall'],
    'incentive_accuracy': results['incentive']['accuracy'],
    'incentive_recall': results['incentive']['recall'],
    'non_incentive_accuracy': results['non_incentive']['accuracy'],
    'non_incentive_recall': results['non_incentive']['recall'],
}
# Add all per-column results
for col in incentive_columns:
    if col in column_results:
        csv_row[f'{col}_accuracy'] = column_results[col]['accuracy']
        csv_row[f'{col}_recall'] = column_results[col]['recall']
for col in non_incentive_columns:
    if col in column_results:
        csv_row[f'{col}_accuracy'] = column_results[col]['accuracy']
        csv_row[f'{col}_recall'] = column_results[col]['recall']

csv_file = 'performance_history.csv'
fieldnames = list(csv_row.keys())

write_header = not os.path.isfile(csv_file)

with open(csv_file, 'a', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()
    writer.writerow(csv_row)

print(f"\nPerformance row added to {csv_file} for {current_date}.")
