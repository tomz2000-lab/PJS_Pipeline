"""
This file can be used to extract the current content from the SQlite-Database into an csv-file.
This file can then be used to run the ``performance.py`` code and see the actual accuracy and recall of the model.
"""

import sqlite3
import pandas as pd
import os

def export_db_to_csv(db_path, csv_path):
    """
    Export all the data from db_path
    
    :param db_path: path to SQLite-DB
    :param csv_path: path to the created csv-file
    """
    # See, if DB excists
    if not os.path.exists(db_path):
        print(f"Database {db_path} not found.")
        return
        
    try:
        # Connect to the DB
        conn = sqlite3.connect(db_path)
        
        # See, if table already excists
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='job_analysis'")
        if not cursor.fetchone():
            print("Tabelle 'job_analysis' existiert nicht in der Datenbank.")
            conn.close()
            return
            
        # Load data into data frame
        query = "SELECT * FROM job_analysis"
        df = pd.read_sql_query(query, conn)
        
        # Daten in CSV exportieren
        df.to_csv(csv_path, index=False, encoding='utf-8')
        
        print(f"Daten erfolgreich in {csv_path} exportiert.")
        
        # close connection
        conn.close()
        
    except Exception as e:
        print(f"Fehler beim Exportieren der Daten: {e}")

"""
Adapt the paths for the db-file and csv-file for your usage.
"""
db_file = "job_analysis.db"  # adapt path to database
csv_file = "job_analysis_export.csv"  # adapt path to export-file

export_db_to_csv(db_file, csv_file)
