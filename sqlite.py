"""
SQLite Database Manager for Job Analysis

This module provides functionality to create and interact with a SQLite database
for storing job analysis data. It includes functions for database initialization,
checking if jobs exist, and inserting or updating job records with conflict resolution.
"""

import sqlite3

def create_database(job_analysis):
    """
    Create and initialize SQLite database for job analysis data.
    
    This function creates a SQLite database file and sets up the job_analysis
    table with all necessary columns for storing job listing data. The table includes
    fields for job metadata, location information, company details, and binary flags
    for various incentive categories.
    
    :param job_analysis: Path to the SQLite database file
    :type job_analysis: str
    :return: None
    :rtype: NoneType
    
    The table includes a composite UNIQUE constraint on Job_URL, Stadt, Zeitmodell,
    and Position to prevent duplicate entries while allowing the same job to appear
    with different location/time model combinations.
    
    The schema includes:

    - Basic job metadata (ID, MongoDB_ID, URL, title, portal, date)
    - Location data (city, state, country)
    - Company information (name, size)
    - Job attributes (time model, position level, employment type)
    - Binary flags for 20+ incentive categories
    - Text field for unclassified incentives
    """
    conn = sqlite3.connect(job_analysis)
    cursor = conn.cursor()

    # Create table with UNIQUE constraint on Job_URL
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS job_analysis (
            ID INTEGER PRIMARY KEY AUTOINCREMENT,
            MongoDB_ID TEXT,
            Job_URL TEXT TEXT,
            Job_Titel TEXT,
            Portal_Name TEXT,
            Datum TEXT,
            Stadt TEXT,
            Bundesland TEXT,
            Land TEXT,
            Unternehmen TEXT,
            Unternehmensgröße TEXT,
            Zeitmodell TEXT,
            Position TEXT,
            Beschäftigungsart TEXT,
            Berufserfahrung_vorausgesetzt INTEGER,
            Kategorie TEXT,
            Gehalt_anhand_von_Tarifklassen INTEGER,
            Überstundenvergütung INTEGER,
            Gehaltserhöhungen INTEGER,
            Aktienoptionen_Gewinnbeteiligung INTEGER,
            Boni INTEGER,
            Sonderzahlungen INTEGER,
            "13. Gehalt" INTEGER,
            Betriebliche_Altersvorsorge INTEGER,
            Flexible_Arbeitsmodelle INTEGER,
            Homeoffice INTEGER,
            Weiterbildung_und_Entwicklungsmöglichkeiten INTEGER,
            Gesundheit_und_Wohlbefinden INTEGER,
            Finanzielle_Vergünstigungen INTEGER,
            Mobilitätsangebote INTEGER,
            Verpflegung INTEGER,
            Arbeitsumfeld_Ausstattung INTEGER,
            Zusätzliche_Urlaubstage INTEGER,
            Familien_Unterstützung INTEGER,
            Onboarding_und_Mentoring_Programme INTEGER,
            Teamevents_Firmenfeiern INTEGER,
            others TEXT,
            UNIQUE(Job_URL, Stadt, Zeitmodell, Position)
        );
    ''')

    conn.commit()
    conn.close()

#impoertant check to see, if the current job already got processed
def job_exists(db_name, mongodb_id):
    """
    Check if a job with the given Mongo-DB ID already exists in SQLite database.
    
    This function queries the job_analysis table to determine if a record
    with the specified Mongo-DB ID already exists. It's used to prevent duplicate
    processing of jobs that have already been analyzed and stored.
    
    :param db_name: Path to the SQLite database file
    :type db_name: str
    :param mongodb_id: Mongo-DB document ID to check for existence
    :type mongodb_id: str
    :return: True if the job exists, False otherwise
    :rtype: bool
    
    The function uses a parameterized query with placeholders to safely handle
    the Mongo-DB ID value and prevent SQL injection vulnerabilities.
    """
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    
    cursor.execute("SELECT 1 FROM job_analysis WHERE MongoDB_ID = ?", (mongodb_id,))
    exists = cursor.fetchone() is not None
    
    conn.close()
    return exists

#insert or replace function to see if thhe URL, Stadt; Zeitmodell and Position are similar
def insert_or_replace_job(job_analysis, job_data):
    """
    Insert a new job record or update an existing one in the SQLite database.
    
    This function adds job data to the database with conflict resolution based on a 
    composite unique constraint. If a record with the same Job_URL, Stadt, Zeitmodell, 
    and Position already exists, it will be skipped.
    
    :param job_analysis: Path to the SQLite database file
    :type job_analysis: str
    :param job_data: Dictionary containing job data to insert or update
    :type job_data: dict
    :return: None
    :rtype: NoneType
    
    :raises sqlite3.Error: If there's an issue with the database connection or query execution
    
    The composite key (Job_URL, Stadt, Zeitmodell, Position) allows the same job posting
    to appear multiple times with different location/work model/position combinations,
    while preventing true duplicates.
    """
    conn = sqlite3.connect(job_analysis)
    cursor = conn.cursor()

    # Build conflict resolution clause
    columns = ', '.join(f'"{col}"' for col in job_data.keys())
    placeholders = ', '.join(['?'] * len(job_data))
    update_columns = ', '.join([f'"{col}"=excluded."{col}"' 
                              for col in job_data.keys()])

    sql = f"""
        INSERT INTO job_analysis ({columns}) 
        VALUES ({placeholders})
        ON CONFLICT(Job_URL, Stadt, Zeitmodell, Position) 
        DO UPDATE SET {update_columns}
    """

    # Execute with parameterized values
    cursor.execute(sql, list(job_data.values()))
    conn.commit()
    conn.close()




