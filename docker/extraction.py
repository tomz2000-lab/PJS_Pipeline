"""
Job Analysis Pipeline for Extracting and Categorizing Incentives

This module implements a comprehensive data processing pipeline that extracts, analyzes,
and categorizes job incentives and their descriptions from online job listings. It connects to Mongo-DB to retrieve
job data, processes text using Llama 3.2 for information extraction, classifies incentives
using Sentence-Transformers, and stores results in a SQLite-DB.

Key components:

    * Mongo-DB data extraction with bson handling
    * Location normalization and geocoding with offline database
    * Synonym handling and regex-extraction for non-benefits related data
    * Incentive extraction, industry category detection, and experience knowledge extraction using LLMs
    * Incentive Classification using Sentence-Transformer with few-shot learning and context information
    * Memory-efficient batch processing and model offloading/cleaning

The pipeline is designed to handle different job portal formats (stepstone, indeed) and
implements robust error handling and fallback mechanisms throughout the extraction process.
"""

import os
import re
import json
import torch
import pandas as pd
import csv
from huggingface_hub import login
from dotenv import load_dotenv
from json_repair import repair_json
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F
from sqlite import create_database, insert_or_replace_job, job_exists
from deep_translator import GoogleTranslator
from itertools import product
from mongo_db import get_mongo_jobs
from datetime import datetime


#log into Hugging Face for Model Download
load_dotenv(override=True)  # Force reload with override
hf_api_key = os.environ.get("HF_API_KEY")

if not hf_api_key:
    print("Warning: HF_API_KEY not found in environment variables")

login(hf_api_key)



# Predefined incentive categories
INCENTIVE_CATEGORIES = [
    "Gehalt_anhand_von_Tarifklassen",	"Überstundenvergütung",	"Gehaltserhöhungen",
	"Aktienoptionen_Gewinnbeteiligung",	"Boni", "Sonderzahlungen",	"13. Gehalt",	"Betriebliche_Altersvorsorge",	"Flexible_Arbeitsmodelle",	"Homeoffice",
	"Weiterbildung_und_Entwicklungsmöglichkeiten","Gesundheit_und_Wohlbefinden", "Finanzielle_Vergünstigungen", "Mobilitätsangebote",
    "Verpflegung", "Arbeitsumfeld_Ausstattung", "Zusätzliche_Urlaubstage",
    "Familien_Unterstützung", "Onboarding_und_Mentoring_Programme", "Teamevents_Firmenfeiern", "others", "non_incentive" 
]


#get the benefits text from stepstone and indeed structure
def get_benefits_text(job):
    """
    Extract benefits text from job data for LLM processing.
    
    This function extracts benefits-related text from job listings in different formats depending on the job protal, the data is from.

    The function handles two main data structures:

    1. Stepstone format with 'lists' containing 'content/benefits'
    2. Indeed format with 'paragraphs' as either list or dictionary
    
    If no benefits text is found in either structure, an empty string is returned.

    This text passages are later used for the :func:`incentive extraction<extraction.process_jobs>` as input for the corresponding prompt.
    
    :param job: Job listing data containing benefits information
    :type job: dict
    :return: Concatenated string of benefits text, separated by newlines
    :rtype: str
    """
    benefits_texts = []

    if 'lists' in job and 'content/benefits' in job['lists']:
        benefits = job['lists']['content/benefits']
        if isinstance(benefits, list) and len(benefits) > 0:
            return '\n'.join(str(b) for b in benefits)
    #Check for paragraphs
    if 'paragraphs' in job:
        job_desc = job['paragraphs']
        if isinstance(job_desc, list):
            benefits_texts.extend(str(b) for b in job_desc)
        elif isinstance(job_desc, dict):
            for section in job_desc.values():
                if isinstance(section, list):
                    benefits_texts.extend(str(b) for b in section)
                elif isinstance(section, str):
                    benefits_texts.append(section)
    # If neither is present, return empty string
    return '\n'.join(benefits_texts)

#get the benefits directly from the data not through the llm
def get_direct_benefits(job):
    """
    Extract benefits directly from the 'benefits' array for indeed jobs.
    Adds them to the :func:`classification<extraction.process_jobs>`-function together with the incentives found by the LLM.
    """
    if 'benefits' in job and isinstance(job['benefits'], list):
        # Filter out placeholder values like "Nicht gefunden"
        return [benefit for benefit in job['benefits'] if benefit and benefit.lower() != "nicht gefunden"]
    return []



#bert classification sertup
def classify_incentives_with_few_shot(incentives, device, threshold=0.45):
    """Classify incentives using a few-shot learning approach with context implementation.
    
    This function uses a Sentence-Transformer model to compare job incentives against
    predefined examples for each category. It combines direct similarity matching with
    contextual understanding to achieve more accurate classification. (80/20 ratio)

    It further creates a logging-file that shows the found incentives with their 
    regarding values and assignments. This can be used to adapt the global threshold.
    
    :param incentives: List of incentive strings to classify
    :type incentives: list
    :param device: Computation device ('cuda' or 'cpu')
    :type device: str
    :param threshold: Minimum similarity score to assign a category, defaults to 0.45
    :type threshold: float, optional
    :return: Tuple containing classification results and unmatched incentives
    :rtype: tuple(dict, list)
    """
    # Load model
    model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v1').to(device)
    
    # Define few-shot examples for each category
    few_shot_examples = {
    "Gehalt_anhand_von_Tarifklassen": [
        "Gehalt nach Tarifvertrag",
        "Attraktive Bezahlung nach jeweiligem Tarifvertrag",
        "Tarifliche Vergütung",
        "Vergütung entsprechend Tarifvertrag"
    ],
    "Überstundenvergütung": [
        "Bezahlte Überstunden",
        "Überstundenausgleich",
        "Vergütung von Mehrarbeit",
        "Nacht- und Feiertagszuschüsse"
    ],
    "Gehaltserhöhungen": [
        "Regelmäßige Gehaltsanpassungen",
        "Jährliche Gehaltssteigerungen",
        "Leistungsbezogene Gehaltserhöhungen",
        "Gehaltsentwicklung nach Karrierestufen"
    ],
    "Aktienoptionen_Gewinnbeteiligung": [
        "Mitarbeiteraktien zu vergünstigten Konditionen",
        "Aktienbeteiligungsprogramm",
        "Gewinnbeteiligung am Unternehemenserfolg",
        "Optionen zum Erwerb von Unternehmensanteilen"
    ],
    "Boni": [
        "Leistungsboni für persönliche Zielerreichung",
        "Verkaufsprovision bei Zielerreichung",
        "Erfolgsprämien für übertroffene Umsatzziele",
        "Projektabschlussprämien"
    ],
    "Sonderzahlungen": [
        "Leistungsprämien und Bonuszahlungen",
        "Weihnachtsgeld und Urlaubsgeld",
        "Erfolgsbeteiligung am Unternehmensergebnis",
        "Sonderzuwendungen und Gratifikationen"
    ],
    "13. Gehalt": [
        "13. Monatsgehalt",
        "Zusätzliches volles Monatsgehalt",
        "13. Gehalt",
        "Jahressonderzahlung in Höhe eines vollen Monatsgehalts"
    ],
    "Betriebliche_Altersvorsorge": [
        "Betriebliche Altersvorsorge",
        "Beiträge zur Altersvorsorge",
        "Arbeitgeberzuschuss zur Altersvorsorge",
        "Betriebsrente und Pensionsplan"
    ],
    "Flexible_Arbeitsmodelle": [
        "Flexible Arbeitszeiten",
        "Gleitzeit mit Kernarbeitszeiten",
        "Individuelle Arbeitszeitmodelle",
        "Vertrauensarbeitszeit",
        "individuell anpassbare Arbeitszeiten"
    ],
    "Homeoffice": [
        "Homeoffice möglich",
        "Mobiles Arbeiten möglich",
        "Remote Work Option",
        "Hybrides Arbeitsmodell",
        "Homeoffice-Möglichkeit"
    ],
    "Weiterbildung_und_Entwicklungsmöglichkeiten": [
        "Berufliche Weiterbildungsmöglichkeiten",
        "Fort- und Weiterbildungsangebote",
        "Persönliche und fachliche Entwicklung",
        "Coaching und Trainings",
        "Karriereprogramme"
    ],
    "Gesundheit_und_Wohlbefinden": [
        "Betriebliche Gesundheitsförderung",
        "Zuschuss zum Fitnessstudio",
        "Gesundheitsprogramme und Vorsorgeuntersuchungen",
        "Betriebssport",
        "Lauftreff"
    ],
    "Finanzielle_Vergünstigungen": [
        "Mitarbeiterrabatte auf Produkte",
        "Versicherungsleistungen",
        "Corporate Benefits und Vorteilsprogramme",
        "Finanzielle Zuschüsse und Sonderkonditionen",
        "Rabattangebote"
    ],
    "Mobilitätsangebote": [
        "Firmenwagen",
        "Jobticket",
        "Fahrtkostenzuschuss",
        "Leasing",
        "Zuschuss zur Bahn- oder Autofahrt",
        "Job Rad"
    ],
    "Verpflegung": [
        "Bezuschusstes Betriebsrestaurant",
        "Kostenlose Getränke",
        "Essenszuschuss",
        "Obstkorb und Kaffeeflatrate",
        "Subventioniertes Essen"
    ],
    "Arbeitsumfeld_Ausstattung": [
        "Moderne Büroausstattung",
        "Firmenlaptop und Smartphone",
        "Ergonomischer Arbeitsplatz",
        "Höhenverstellbare Schreibtische",
        "Firmenlaptop"
    ],
    "Zusätzliche_Urlaubstage": [
        "30 Tage Urlaubsanspruch",
        "Zusätzliche Urlaubstage über dem gesetzlichen Minimum",
        "Sonderurlaubstage für besondere Anlässe",
        "Sabbatical-Möglichkeit"
    ],
    "Familien_Unterstützung": [
        "Betriebsnahe Kita und Eltern-Kind Büros",
        "Zuschuss zur Kinderbetreuung",
        "Familienfreundliche Arbeitszeiten",
        "Wiedereinstiegsprogramme nach Elternzeit",
        "Kitaplätze"
        
    ],
    "Onboarding_und_Mentoring_Programme": [
        "Individuelle Einarbeitung und Onboarding",
        "Mentorenprogramm für neue Mitarbeiter",
        "Strukturierte Einarbeitung",
        "Patenschaftsprogramm",
        "Mentoring",
        "Onboardingtag und individuelles Einarbeitungsprogramm"
    ],
    "Teamevents_Firmenfeiern": [
        "Teamevents",
        "Firmenfeiern und Betriebsausflüge",
        "Teambuilding-Aktivitäten",
        "Gemeinsame Mittagessen und Afterwork-Events",
        "Weihnachstfeiern und Sommerfeste",
        "Kulturevents"
    ],
    "non_incentive": [
        "Gehalt anzeigen",
        "Teilzeit möglich",
        "Feste Anstellung",
        "Tolles Teamwork und ﬂache Hierarchien sichern kurze Kommunikationswege",
        "freundschaftliche Atmosphäre",
        "gemeinsame Zeit"
    ]
}
    
    # Define category contexts (separate from examples)
    category_contexts = {
    "Gehalt_anhand_von_Tarifklassen": "Compensation based on collective bargaining agreements or tariff classifications, providing standardized salary structures according to industry standards",
    "Überstundenvergütung": "Compensation for work performed beyond regular working hours, including overtime pay, time off in lieu, or special allowances for night or holiday work",
    "Gehaltserhöhungen": "Regular or performance-based salary increases, including annual raises, systematic salary reviews, salary development frameworks, or regular feedback processes that impact compensation",
    "Aktienoptionen_Gewinnbeteiligung": "Opportunities for employees to own company shares or stock options, including employee stock purchase plans, equity compensation, or discounted company shares",
    "Boni": "Performance-based additional payments that reward individual or company achievements, typically variable in amount and directly tied to reaching specific targets or goals. Examples include sales commissions, performance bonuses, and profit-sharing payments.",
    "Sonderzahlungen": "One-time or regular additional payments beyond the normal salary that are typically tied to specific occasions, celebrations, or as general appreciation. These include holiday bonuses, vacation allowances, anniversary payments, and special situation payments like inflation compensation",
    "13. Gehalt": "An additional full month's salary paid once per year, separate from regular monthly payments and distinct from other bonuses or special payments",
    "Betriebliche_Altersvorsorge": "Employer-sponsored retirement plans or pension schemes, including employer contributions to retirement funds, pension plans, or retirement insurance",
    "Flexible_Arbeitsmodelle": "Work arrangements that allow employees to vary when and how they work, including flexible start/end times, compressed work weeks, or trust-based working time",
    "Homeoffice": "Ability to work from home or remotely instead of commuting to an office location, including full remote work, hybrid models, or occasional work-from-home options",
    "Weiterbildung_und_Entwicklungsmöglichkeiten": "Opportunities for professional growth and skill development, including training programs, educational courses or career advancement paths",
    "Gesundheit_und_Wohlbefinden": "Programs and benefits focused on employee health and wellness, including fitness subsidies, wellness programs, company sports or joined sport groups",
    "Finanzielle_Vergünstigungen": "Financial benefits beyond direct compensation, including employee discounts, insurance benefits, corporate perks, or financial subsidies",
    "Mobilitätsangebote": "Benefits related to transportation and commuting, including company cars, public transport subsidies, bicycle leasing, or parking facilities",
    "Verpflegung": "Food and beverage benefits provided in the workplace, including subsidized meals, free drinks, snacks, or meal allowances",
    "Arbeitsumfeld_Ausstattung": "Physical workplace environment and equipment provided to employees, including modern office furniture, technology, ergonomic equipment, or workspace design",
    "Zusätzliche_Urlaubstage": "Vacation days beyond the legal minimum requirement of 20 days, including extra holiday allowances, special occasion leave, sabbaticals, or extended time off",
    "Familien_Unterstützung": "Benefits that support employees with family responsibilities, including childcare assistance, parental leave, family-friendly work hours, parent-child offices, family services, emergency support, or specific allowances for family events",
    "Onboarding_und_Mentoring_Programme": "Structured programs to help new employees integrate into the company, including orientation processes, mentoring relationships, buddy systems, or training periods",    
    "Teamevents_Firmenfeiern": "Social activities and events organized by the company to foster team spirit and company culture, including team outings, company celebrations, or social gatherings",
    "non_incentive": "Phrases indicating employment terms and tasks rather than benefits - salary disclosures, employment types (full/part-time), or work tasks that don't constitute additional incentives"

}
   
    # Prepare results dictionary
    results = {cat: 0 for cat in INCENTIVE_CATEGORIES if cat != "non_incentive"}  #Exclude from output
    others = []

    #csv file for logging
    csv_file = 'bert_classification_log.csv'
    write_header = not os.path.exists(csv_file) or os.path.getsize(csv_file) == 0
    csvfile = open(csv_file, mode='a', newline='', encoding='utf-8')
    writer = csv.writer(csvfile)
    if write_header:
        writer.writerow([
            'incentive',
            'top_1_category', 'top_1_score',
            'top_2_category', 'top_2_score',
            'top_3_category', 'top_3_score',
            'top_4_category', 'top_4_score',
            'top_5_category', 'top_5_score',
            'best_category', 'best_score',
            'threshold', 'assigned'
        ])
    
    # Get embeddings for all incentives
    if incentives:
        print("\n" + "="*50 + "\nDETAILED CONTEXTUAL FEW-SHOT CLASSIFICATION:\n" + "="*50)
        
        incentive_embeddings = model.encode(incentives, batch_size=4, convert_to_tensor=True)
        
        # Process each category with few-shot examples
        direct_similarities = {}
        context_similarities = {}
        
        for category, examples in few_shot_examples.items():
            # 1. Direct example matching
            example_embeddings = model.encode(examples, convert_to_tensor=True)
            direct_matrix = F.cosine_similarity(
                incentive_embeddings.unsqueeze(1),
                example_embeddings.unsqueeze(0),
                dim=2
            )
            direct_max, _ = torch.max(direct_matrix, dim=1)
            direct_similarities[category] = direct_max
            
            # 2. Context similarity (with lower weight)
            context_embedding = model.encode([category_contexts[category]], convert_to_tensor=True)
            context_matrix = F.cosine_similarity(
                incentive_embeddings.unsqueeze(1),
                context_embedding.unsqueeze(0),
                dim=2
            )
            context_similarities[category] = context_matrix.squeeze(1)
        
        # Combine direct and contextual similarities (with weights)
        combined_similarities = {}
        for category in few_shot_examples.keys():
            # 80% direct similarity with examples, 20% contextual influence
            combined_similarities[category] = 0.8 * direct_similarities[category] + 0.2 * context_similarities[category]
        
        # Process each incentive
        for i, incentive in enumerate(incentives):
            # Get the best matching category for this incentive
            best_category = None
            best_similarity = 0
            
            # Print all similarity scores for this incentive as an overview
            print(f"\nIncentive: '{incentive}'")
            print("-" * 40)
            print("Top 5 category matches:")
            
            # Collect all category similarities for this incentive
            incentive_cats = []
            for category in few_shot_examples.keys():
                similarity = float(combined_similarities[category][i])
                incentive_cats.append((category, similarity))
            
            # Sort by similarity score (highest first)
            incentive_cats.sort(key=lambda x: x[1], reverse=True)
            
            # Print top 5 most similar categories
            for category, similarity in incentive_cats[:5]:
                print(f"  - {category}: {similarity:.4f}" + 
                      (" (MATCHED)" if similarity > threshold else ""))
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_category = category
            
            # Assign to the best category if above threshold
            if best_similarity > threshold:
                results[best_category] = 1
                print(f"  → Incentive assigned to: {best_category} (score: {best_similarity:.4f})")
            else:
                others.append(incentive)
                print(f"  → No category matched above threshold ({threshold}). Added to 'others'.")

            # Write log row here
            assigned = int(best_similarity > threshold)
            top5 = incentive_cats[:5]
            while len(top5) < 5:
                top5.append(('', ''))
            writer.writerow([
                incentive,
                top5[0][0], top5[0][1],
                top5[1][0], top5[1][1],
                top5[2][0], top5[2][1],
                top5[3][0], top5[3][1],
                top5[4][0], top5[4][1],
                best_category, best_similarity,
                threshold, assigned
            ])

    #clean up model
    model = model.to('cpu')  # Move model back to CPU
    del model
    torch.cuda.synchronize()
    torch.cuda.empty_cache() # Clear CUDA cache

    csvfile.close()
    return results, others


#setup google translator for location extraction in exctract_entities_from_json
def translate_text(text, source='de', target='en'):
    """
    Translate text between languages using Google Translator.
    
    This function provides text translation capabilities used for location data
    processing. It handles translation errors gracefully by returning the original text
    when translation fails.
    
    :param text: The text string to be translated
    :type text: str
    :param source: Source language code (ISO 639-1), defaults to 'de'
    :type source: str, optional
    :param target: Target language code (ISO 639-1), defaults to 'en'
    :type target: str, optional
    :return: Translated text or original text if translation fails
    :rtype: str
    
    :raises: No exceptions are raised as errors are caught internally
    """
    try:
        return GoogleTranslator(source=source, target=target).translate(text)
    except Exception as e:
        print(f"Translation error: {e}")
        return text  # Fallback to original text


#clean location string to see multiple locations     
def normalize_location_string(location_str):
    """
    Clean and standardize location strings for consistent processing.
    
    This function normalizes location strings by converting various separators to commas,
    removing annotations, parenthetical content, and standardizing spacing. It helps
    prepare location data for further processing and geocoding.
    
    :param location_str: Raw location string that may contain multiple locations
    :type location_str: str
    :return: Cleaned and standardized location string with consistent separators
    :rtype: str
    
    The function performs the following transformations:

    1. Converts separators (semicolons, slashes, bullets) and conjunctions to commas
    2. Removes qualifying phrases like "bei Stuttgart"
    3. Removes parentheses and their content
    4. Standardizes spacing around commas
    """
    import re
    # Replace common separators with commas
    location_str = re.sub(r'[;/\u2022]| und | oder ', ', ', location_str)
    # Remove non-city annotations like "bei Stuttgart"
    location_str = re.sub(r'\bbei\b.*', '', location_str, flags=re.IGNORECASE)
    # Remove parentheses and their content
    location_str = re.sub(r'\(.*?\)', '', location_str)
    # Fix spacing around commas and remove extra spaces
    return re.sub(r'\s*,\s*', ', ', location_str).strip()


#split the now cleaned locations
def split_locations(normalized_str):
    """
    Split a normalized location string into individual location components.
    
    This function intelligently splits location strings on commas while handling
    complex cases such as parenthetical content. It preserves commas that appear
    within parentheses to maintain the integrity of location names that naturally
    contain commas.
    
    :param normalized_str: Normalized location string with comma separators
    :type normalized_str: str
    :return: List of individual location strings
    :rtype: list
    
    :example:
        >>> split_locations("Berlin, Frankfurt (Main), München")
        ['Berlin', 'Frankfurt (Main)', 'München']
    """
    import re
    # Split on commas, but ignore commas in city names (e.g. "Mönchengladbach")
    return re.split(r',\s*(?![^(]*\))', normalized_str)


#validate cities against the json-database before translation, to avoid mistranslations (e.g. Hannover gets then Hanover, whic is in the USA)
def validate_city(city_name, world_data):
    """
    Check if city exists in offline database before translating it.
    
    This function normalizes and compares city names against an offline database
    to verify their existence before translation. It helps prevent translation errors
    that could lead to geographic misidentification (e.g., translating "Hannover" 
    to "Hanover" which would incorrectly place it in the US instead of Germany).
    
    :param city_name: Name of the city to validate
    :type city_name: str
    :param world_data: Dictionary containing hierarchical country/state/city data
    :type world_data: list
    :return: True if the city exists in the database, False otherwise
    :rtype: bool
    
    The function performs Unicode normalization to handle special characters like
    umlauts, making the comparison case-insensitive and accent-insensitive.
    """
    import unicodedata
    # Normalize city name (e.g., handle umlauts)
    city_lower = unicodedata.normalize('NFKD', city_name).encode('ascii', 'ignore').decode('utf-8').lower()
    for country in world_data:
        for state in country["states"]:
            for city in state["cities"]:
                city_normalized = unicodedata.normalize('NFKD', city["name"]).encode('ascii', 'ignore').decode('utf-8').lower()
                if city_normalized == city_lower:
                    return True
    return False

#open world json-database with locations
"""
Access the offline states and countries database.

This function loads a comprehensive database of countries, states, and cities from a json-file.
The database is used for location validation, geocoding, and standardization throughout the
application, particularly for job location processing.

The file structure contains a hierarchical representation of:

- Countries
- States/provinces within each country
- Cities within each state

This database helps avoid API calls to external geocoding services and ensures consistent
location data handling even when network access is limited.
"""
# Get the directory where the current script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

json_path = os.path.join(script_dir, "countries+states+cities.json")

with open(json_path, "r", encoding="utf-8") as f:
    world_data = json.load(f)
    
# Mapping of Bundesländer, to avoid English Bundesländer in the databse
"""
Map German federal states (Bundesländer) to standardized German names.

This mapping dictionary provides translations from English names and common variations
to standardized German names for all 16 federal states of Germany. It's used to ensure
consistent naming in the database when processing location data that might come from
different sources or through translation.

:return: Dictionary mapping English state names and variations to standardized German names
:rtype: dict
"""
BUNDESLAENDER_MAPPING = {
    # English to German
    "Bavaria": "Bayern",
    "Lower Saxony": "Niedersachsen",
    "Baden-Württemberg": "Baden-Württemberg",
    "Rhineland-Palatinate": "Rheinland-Pfalz",
    "Saxony": "Sachsen",
    "Thuringia": "Thüringen",
    "Hesse": "Hessen",
    "North Rhine-Westphalia": "Nordrhein-Westfalen",
    "Saxony-Anhalt": "Sachsen-Anhalt",
    "Brandenburg": "Brandenburg",
    "Mecklenburg-Western Pomerania": "Mecklenburg-Vorpommern",
    "Hamburg": "Hamburg",
    "Schleswig-Holstein": "Schleswig-Holstein",
    "Saarland": "Saarland",
    "Bremen": "Bremen",
    "Berlin": "Berlin",
    
    # Add common variations
    "Mecklenburg-Vorpommern": "Mecklenburg-Vorpommern",
    "North Rhine Westphalia": "Nordrhein-Westfalen",
    "Northrhine-Westphalia": "Nordrhein-Westfalen"
}


# Example function to fetch state and country for a city
def get_state_and_country(city_name):
    """
    Fetch state and country for a given city using offline json database.
    
    This function searches through a hierarchical database of countries, states, and cities
    to find location information for a given city name. It performs a case-insensitive match
    and handles translation of state and country names to German.
    It first tries to find the name in the German part of the json, then switches over to a full-json search,
    if the name was not found in the German part.
    
    :param city_name: Name of the city to search for in the database
    :type city_name: str
    :return: Dictionary containing state and country names in German
    :rtype: dict
    
    The function follows these steps:

    1. Searches for an exact match of the city name in the database, prioritizing German cities
    2. If found, checks if the state name exists in BUNDESLAENDER_MAPPING
    3. If not in mapping, translates state name from English to German
    4. Translates country name from English to German if no direct German match
    5. Returns both in a dictionary with "state" and "country" keys
    6. Returns {"state": "Unknown", "country": "Unknown"} if city not found
    """
    #first search for geman cities
    for country in world_data:
        if country["name"] == "Germany":  # Zuerst in Deutschland suchen
            for state in country["states"]:
                for city in state["cities"]:
                    if city["name"].lower() == city_name.lower():
                        #check mapping first
                        state_de = BUNDESLAENDER_MAPPING.get(state["name"], None)
                        # Translate state and country names to German
                        if state_de is None:
                            state_de = GoogleTranslator(source='en', target='de').translate(state["name"])
                        
                        country_de = GoogleTranslator(source='en', target='de').translate(country["name"])
                        return {"state": state_de, "country": "Deutschland"}
                    
    #when there is no german city with this name go over all cities
    for country in world_data:  # Iterate over list of countries
        if country["name"] != "Germany": #if no german city
            for state in country["states"]:  # Iterate over states in each country
                for city in state["cities"]:  # Iterate over cities in each state
                    if city["name"].lower() == city_name.lower():
                        #check mapping first
                        state_de = BUNDESLAENDER_MAPPING.get(state["name"], None)
                        # Translate state and country names to German
                        if state_de is None:
                            state_de = GoogleTranslator(source='en', target='de').translate(state["name"])
                        
                        country_de = GoogleTranslator(source='en', target='de').translate(country["name"])
                        return {"state": state_de, "country": country_de}
    return {"state": "Unknown", "country": "Unknown"}


#remove postal code from city if existnet
def remove_leading_postal_code(location):
    """
    Remove leading postal codes from location strings.
    
    This function strips any numeric postal codes that appear at the beginning of
    location strings, allowing for cleaner location data processing. It matches
    one or more digits at the start of the string followed by optional whitespace.
    
    :param location: Location string that may contain a leading postal code
    :type location: str
    :return: Location string with any leading postal code removed
    :rtype: str
    
    :example:
        >>> remove_leading_postal_code("12345 Berlin")
        'Berlin'
        >>> remove_leading_postal_code("Berlin")
        'Berlin'
    """
    return re.sub(r'^\d+\s*', '', location)


#extract company size
def extract_company_size(job):
    """
    Extract company size information from stepstone job listings.
    
    This function searches through the 'CompanyInfo' field in stepstone job listings
    to find company size information. 
    The information is stored in the last element of the list-elements.
    
    Note that this function only works with stepstone's data structure, as indeed job
    listings don't provide company size information.
    
    :param job: Job listing data dictionary from Stepstone
    :type job: dict
    :return: Company size string or default value if not found
    :rtype: str
    
    The function follows these steps:

    1. Check if 'CompanyInfo' exists in the job data
    2. Iterate through each list in 'CompanyInfo'
    3. Look for "Unternehmensgröße" label and return the next item
    4. If not found, try to use the last element as fallback
    5. Return default value if no company size information is found
    """
    default_size = "Keine Angaben"
    
    # look, if CompanyInfo exists
    if 'lists' in job and 'CompanyInfo' in job['lists']:
        for info_list in job['lists']['CompanyInfo']:
            if isinstance(info_list, list) and info_list:
                # Return the last element of the list (if it's a string)
                if isinstance(info_list[-1], str):
                    return info_list[-1]
    return default_size


#calculate company size
def categorize_company_size(size_str):
    """
    Categorize company size into eight predefined categories based on employee count.
    
    This function extracts numeric values from a company size string and maps them
    to standardized size categories. It handles various input formats by extracting
    all numbers and using the largest one as the size indicator.
    
    :param size_str: String containing company size information
    :type size_str: str
    :return: Categorized company size as a string label
    :rtype: str
    
    The function uses the following size categories:

    - '0-10': Very small companies/startups
    - '11-50': Small companies
    - '51-250': Medium-sized companies
    - '251-500': Mid-large companies
    - '501-1000': Large companies
    - '1001-2500': Very large companies
    - '2501-10000': Enterprise-level companies
    - '10000+': Major corporations
    
    If no numbers are found in the input string, 'Keine Angaben' (No information) is returned.
    """
    import re
    
    # Define threshold values
    thresholds = [0, 10, 50, 250, 500, 1000, 2500, 10000, float('inf')]
    labels = ['0-10', '11-50', '51-250', '251-500', '501-1000', '1001-2500', '2501-10000', '10000+']
    
    # Remove commas from the size string
    size_str = size_str.replace(',', '')

    # extract numbers from string
    numbers = re.findall(r'\d+', size_str)
    if not numbers:
        return 'Keine Angaben'
    
    # convert numbers to strings
    nums = list(map(int, numbers))
    
    # Verwende die größte Zahl als Größenindikator
    size = max(nums)
    
    # find suitable lable
    for i in range(len(thresholds)-1):
        if thresholds[i] < size <= thresholds[i+1]:
            return labels[i]
    
    return 'Keine Angaben'


#get date in dd.mm.yyyy
def get_formatted_date(job):
        """
        Extract and format the job posting date in dd.mm.yyyy format.
        
        This function retrieves the posting date from datePosted in the stepstone-data and converts it to a
        standardized German date format. It handles ISO 8601 formatted dates and
        provides a fallback to the current date when no valid date is found.
        
        :param job: Dictionary containing job listing data
        :type job: dict
        :return: Formatted date string in DD.MM.YYYY format
        :rtype: str
        
        The function follows these steps:

        1. Attempt to extract the 'datePosted' field from the stepstone-jobs
        2. If present, try to parse it as an ISO 8601 date string
        3. Handle timezone information by normalizing 'Z' notation
        4. If parsing fails or no date is provided, use the current date as fallback
        
        This approach ensures consistent date formatting across different job board
        sources.
        """
        date_str = job.get("datePosted", "")
        # Try to parse the datePosted field
        if date_str:
            try:
                # Handles ISO 8601 format, e.g. "2025-04-15T09:21:38+02:00"
                dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                return dt.strftime("%d.%m.%Y")
            except Exception:
                pass  # If parsing fails, fallback to current date
        # Fallback: use current date
        return datetime.now().strftime("%d.%m.%Y")


#look for homeoffice directly under lists/company, look at fallback and in the end try to use the llm (s. later)
def detect_homeoffice(job_data):
    """
    Detect mentions of remote work options in job listings.
    
    This function uses a multi-stage approach to identify homeoffice/remote work
    options in job listings. 

    1. Check company info section (most reliable location for this information)
    2. Fall back to full-text search using synonyms across all job description sections
    3. Return 1 if any homeoffice pattern is found by synonym-matching, 0 otherwise
    4. If no homeoffice is found the code overwrites the 0 with a 1 if the LLM later finds homeoffice in the text
    
    :param job_data: Job listing data dictionary
    :type job_data: dict
    :return: Binary indicator (1 if homeoffice is mentioned, 0 if not)
    :rtype: int
    
    The detection follows these steps in order:

    This direct pattern matching approach is more efficient than directly/always using LLM-based
    extraction for this specific attribute.
    """
    homeoffice_patterns = [
        "homeoffice möglich", "Homeoffice möglich", "Homeoffice Möglich"
    ]
    
    # Check company info section first (most reliable)
    if 'lists' in job_data and 'company' in job_data['lists']:
        company_info = ' '.join(map(str, job_data['lists']['company'])).lower()
        if any(pattern in company_info for pattern in homeoffice_patterns):
            return 1
    
    # Check full text as fallback
    full_text = ''
    if 'paragraphs' in job_data:
        full_text += ' '.join(map(str, job_data['paragraphs'])).lower()
    if 'lists' in job_data:
        for lst in job_data['lists'].values():
            full_text += ' '.join(map(str, lst)).lower()
    
    if any(pattern in full_text for pattern in homeoffice_patterns):
        return 1
    
    return 0

#synonyms for time_model
ZEITMODELL_SYNONYME = {
    "Vollzeit": [
        #german
        "vollzeit", "vollzeitstelle", 
        "ganztags", "40 stunden", "vollbeschäftigung",
        "vollzeitjob", "vollzeitarbeit", "voll- und teilzeit", 
        "voll und teilzeit", "voll- oder teilzeit", 
        "voll oder teilzeit", "voll- & teilzeit", 
        "voll & teilzeit",
        #english
        "full- and part time", "full and part time"
        "full- & part time", "full & part time"
        "full time", "full-time"
        "full or part time", "full- or part time"
         "40 h",
    ],
    "Teilzeit": [
        #german
        "teilzeit", "teilzeitstelle", "halbtags", 
        "20 stunden", "teilbeschäftigung",
        "teilzeitjob", "teilzeitarbeit", "reduzierte stunden",
        "voll- und teilzeit", "voll und teilzeit", 
        "voll- oder teilzeit", "voll oder teilzeit", 
        "voll- & teilzeit", "voll & teilzeit",
        #english
        "full- and part time", "full and part time"
        "full- & part time", "full & part time"
        "part time", "part-time"
        "full or part time", "full- or part time"
         "20 h",
    ]
}

#synonyms for Beschäftigungsart
BESCHAEFTIGUNGSART_SYNONYME = {
    "Feste Anstellung": [
        # Deutsch
        "unbefristet", "dauerhaft", "festanstellung", "festangestellt",
        "unbefristeter vertrag", "dauerstelle", "festvertrag", "Feste Anstellung",
        # Englisch
        "permanent", "full-time", "regular employment", "indefinite contract"
    ],
    "befristet": [
        # Deutsch
        "befristeter vertrag", "zeitlich begrenzt", "projektbezogen", 
        "temporär", "vertrag auf zeit", "zeitvertrag", "befristet",
        # Englisch
        "fixed-term", "temporary", "contract-based", "limited duration"
    ]
}

#synonyms for position
POSITION_SYNONYME = {
    "Praktikant": [
        #german
        "praktikum", "praktikant", "trainee", "praktikantin", "volontär",
        #english
        "internship"
    ],
    "Werkstudent": [
        #german
        "werkstudent", "werkstudentin", "studentische hilfskraft",
        #english
        "student assistant", "student worker", "working student"
    ]
}


def extract_entities_from_json(job):
    """
    Extract location, company, and employment details from job listing json.
    
    This function processes job data to extract and normalize key entities including location,
    company information, job type, working model, and position level. It handles different
    data formats from various job portals (stepstone, indeed) and normalizes the extracted
    information into a consistent structure.
    
    The function performs several key operations:

    1. Extracts basic job metadata (title, URL, portal)
    2. Identifies company information from different data structures
    3. Processes location data with geocoding and translation
    4. Analyzes job descriptions to determine employment type and work model
    5. Categorizes company size and position level
    
    Time model, position level, and employment type are extracted using word-matching
    against predefined synonym dictionaries.
    
    :param job: Job listing data in JSON format
    :type job: dict
    :return: Dictionary containing extracted and normalized job entities
    :rtype: dict
    
    :raises IndexError: When accessing company_info elements that don't exist
    :raises KeyError: When accessing non-existent dictionary keys
    :raises Exception: For general errors during location processing
    """
    # Initialize default values
    company = ""
    location = ""
    job_type = "befristet"  
    time_model = []
    entry_level = "Normaler Angestellter"  
    url = ""
    portal_name = ""
    company_info = []


    #search for Job Title
    job_title = job.get("Job Title", "").strip()
    if not job_title:
        job_title = "Kein Titel"

    # Extract URL - handle both structures
    if 'url' in job:
        url = job['url']
    elif 'URL' in job:
        url = job['URL']
    
    # Extract portal name from URL
    portal_name = extract_website_name(url)
    
    # Handle IT-Consultant JSON structure
    if 'Company Name' in job and job['Company Name']:
        company = job['Company Name']
    elif 'lists' in job and 'company' in job['lists']:
        company_info = []
        for item in job['lists']['company']:
            if isinstance(item, list):
                company_info.extend(item)
            else:
                company_info.append(item)
        
        # Extract company and location from company info
        company = company_info[0] if len(company_info) > 0 else ""
        location = company_info[1] if len(company_info) > 1 else ""

        # Extract job type from company info
        if len(company_info) >= 3 and "Feste Anstellung" in company_info[2]:
            job_type = "Feste Anstellung"  
    
    # Handle Data-Projekt-Manager JSON structure
    if 'jobLocationText' in job:
        location = job.get('jobLocationText', "")
        # Clean location by removing leading postal code
        location = remove_leading_postal_code(location)
        print(f"Location after postal code removal: {location}")
    
    # Extract company size and categorize it
    company_size_raw = extract_company_size(job)
    company_size = categorize_company_size(company_size_raw)

    # Collect all text for further analysis
    full_text = []
    desc_parts = []
    
    # For stepstone structure
    if 'lists' in job:
        for section in job['lists'].values():
            full_text.extend(str(item) for item in section)
    
    # For indeed structure
    if 'paragraphs' in job:
        job_desc = job.get('paragraphs', {})
        if isinstance(job_desc, list):
            desc_parts.extend(map(str, job_desc))
        elif isinstance(job_desc, dict):
            for section in job_desc.values():
                if isinstance(section, list):
                    desc_parts.extend(map(str, section))

    full_text.extend(desc_parts)
        
    # Common text analysis for both structures
    search_text = ' '.join(full_text).lower()
    
    # Analyze time model
    for modell, synonyme in ZEITMODELL_SYNONYME.items():
        #direct word matching
        # Remove word boundary markers for partial matching
        if any(re.search(r'(?:\w*?)' + re.escape(syn) + r'(?:\w*?)', search_text) for syn in synonyme):
            time_model.append(modell)
        #Defasult zu Vollzeit
    if not time_model:
        time_model = ["Vollzeit"]  # Standardwert, wenn nichts gefunden wurde
    # Deduplicate and normalize time model
    time_model = list(set(time_model))
    time_model = [m.capitalize() for m in time_model]
    
    # Analyze entry level
    for pos, synonyme in POSITION_SYNONYME.items():
        #direct word matching
        if any(re.search(r'\b' + re.escape(syn) + r'\b', search_text) for syn in synonyme):
            entry_level = pos
            break
    
    # Analyze job type for both structures
    if any(re.search(r'\b' + re.escape(syn) + r'\b', search_text) for syn in BESCHAEFTIGUNGSART_SYNONYME["Feste Anstellung"]):
        job_type = "Feste Anstellung"
    # Process location data
    normalized = normalize_location_string(location)
    raw_locations = split_locations(normalized)

    translated_states_countries = []
    for loc in raw_locations:
        clean_loc = re.sub(r'\s+', ' ', loc).strip()
        clean_loc = remove_leading_postal_code(clean_loc)
        # Special handling for "bundesweit"
        if clean_loc.lower() == 'bundesweit':
            translated_states_countries.append({
                "city": clean_loc,
                "state": "bundesweit",
                "country": "Germany"
            })
            continue  # Skip the regular location processing
        try:
            # 1. First try with original German name
            result_de = get_state_and_country(clean_loc)
            
            if result_de["state"] != "Unknown":
                # Use German results directly
                translated_states_countries.append({
                    "city": clean_loc,
                    "state": result_de["state"],
                    "country": result_de["country"]
                })
            else:
                # 2. Fallback to English translation
                city_en = translate_text(clean_loc, source='de', target='en')
                result_en = get_state_and_country(city_en)

                if result_en["state"] != "Unknown":
                    # 3. Translate state/country back to German
                    state_de = translate_text(result_en["state"], 'en', 'de')
                    country_de = translate_text(result_en["country"], 'en', 'de')
                    translated_states_countries.append({
                        "city": clean_loc,  # Keep original German name
                        "state": state_de,
                        "country": country_de
                    })
                else:
                    # Complete fallback
                    translated_states_countries.append({
                        "city": clean_loc,
                        "state": "Unbekannt",
                        "country": "Unbekannt"
                    })
                    
        except Exception as e:
            print(f"Error processing location '{clean_loc}': {e}")
            translated_states_countries.append({
                "city": clean_loc,
                "state": "Unbekannt",
                "country": "Unbekannt"
            })



    # Prepare final state and country values
    unique_states = set([entry["state"] for entry in translated_states_countries])
    unique_countries = set([entry["country"] for entry in translated_states_countries])
    state = ', '.join(unique_states) if unique_states else "Unbekannt"
    country = ', '.join(unique_countries) if unique_countries else "Unbekannt"

    #print statements for correct extraction and connection to the mongo-db
    print("\nFINAL EXTRACTED ENTITIES:")
    print(f"Job Title: {job_title}")
    print(f"Company: {company}")
    print(f"Locations: {translated_states_countries}")
    print(f"State: {state}")
    print(f"Country: {country}")
    print(f"Job type: {job_type}")
    print(f"Time model: {time_model}")
    print(f"Entry level: {entry_level}")
    print(f"URL: {url}")
    print(f"Portal: {portal_name}")
    print(f"Unternehmensgröße: {company_size}")
    print("="*50)

    return {
         "job_title": job_title,
        "location": translated_states_countries,
        "state": state,
        "country": country,
        "company": company,
        "company_size": company_size,
        "job_type": job_type,
        "time_model": time_model,
        "entry_level": entry_level
    }


#Get portal name
def extract_website_name(url):
    """
    Extract the website name from the provided URL using regex pattern matching.
    
    This function attempts to extract the domain name from a URL by applying two different
    regex patterns. It first tries to match URLs with 'www' prefix, then falls back to
    matching standard HTTP/HTTPS URLs without 'www'.
    
    :param url: The URL from which to extract the website name
    :type url: str
    :return: The extracted website name or 'Unknown' if no pattern matches
    :rtype: str
    
    :example:
        >>> extract_website_name("https://www.example.com/jobs")
        'example'
        >>> extract_website_name("https://subdomain.indeed.com/job/123")
        'indeed'
    """
    import re
    
    # Try to match www.domain.com pattern
    match = re.search(r'www\.(.*?)\.', url)
    if match:
        return match.group(1)
    
    # Try to match domain.com pattern without www
    match = re.search(r'https?://(?:[\w-]+\.)*([\w-]+)\.com', url)
    if match:
        return match.group(1)
    
    return 'Unknown'


def sanitize_json_response(response):
    """
    Sanitize invalid escape sequences in json responses from language models.
    
    This function removes invalid escape sequences that may appear in json responses
    generated by large language models. It specifically targets hexadecimal escape
    sequences that are not valid in standard json and could cause
    parsing errors.
    
    :param response: Text response from a language model that may contain invalid json
    :type response: str
    :return: Sanitized response with invalid escape sequences removed
    :rtype: str
    
    The function is particularly useful when processing outputs from LLMs that may
    hallucinate or generate malformed json with escape sequences that aren't properly
    encoded according to the json specification.
    """
    import re
    # Replace invalid \x escape sequences
    sanitized_response = re.sub(r'\\x[0-9A-Fa-f]{2}', '', response)
    return sanitized_response


def parse_json_response(response):
    """
    Extract and parse the first json object from the LLMs respond wheather expericne is required
    for the job or not.
    
    This function searches for the first json object in a text string and attempts
    to parse it. It's particularly useful for extracting structured data from
    LLM responses that might contain additional text before or after the json.
    
    :param response: Text string that may contain a json object
    :type response: str
    :return: Parsed json object as a dictionary, or empty dictionary if no valid json found
    :rtype: dict
    """
    import json, re
    # Find the first json object in the text
    match = re.search(r'\{.*?\}', response, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except:
            pass
    return {}


def parse_json_incentives(response):
    """
    Parse model response to extract incentives as a json array.
    
    This function extracts benefits/incentives from language model responses using
    multiple fallback strategies. It handles various response formats and potential
    parsing errors with a robust multi-step approach.
    
    :param response: Text response from the language model containing benefits data
    :type response: str
    :return: List of unique incentives/benefits extracted from the response
    :rtype: list
    
    The function implements the following extraction strategies in order:

    1. Sanitizes and repairs the entire response as json
    2. Attempts to find and repair json objects within the response
    3. Extracts the benefits array directly using regex patterns
    4. Falls back to extracting bulleted items as a last resort
    
    All exceptions are handled internally, ensuring the function always returns
    a list without raising exceptions to the caller.
    
    :example:
        >>> parse_json_incentives('{"benefits": ["Flexible hours", "Health insurance"]}')
        ['Flexible hours', 'Health insurance']
    """
    try:
        # Sanitize the response to remove invalid escape sequences
        response = sanitize_json_response(response)
        
        # First try to repair and parse the entire response as JSON
        try:
            repaired_json = repair_json(response)
            result = json.loads(repaired_json)
            
            # Handle the case where result is a list of objects with "benefit" key
            if isinstance(result, list) and all(isinstance(item, dict) and "benefit" in item for item in result):
                benefits = [item["benefit"] for item in result]
                filtered_benefits = benefits
                return list(dict.fromkeys(filtered_benefits))
            
            # Handle the standard case with "benefits" key
            if isinstance(result, dict) and "benefits" in result and isinstance(result["benefits"], list):
                benefits = result["benefits"]
                filtered_benefits = benefits
                return list(dict.fromkeys(filtered_benefits))
                
        except Exception as e:
            print(f"Full json repair attempt failed: {e}")
        
        # Try to find a JSON object pattern and repair it
        match = re.search(r'(\{.*\})', response, re.DOTALL)
        if match:
            try:
                json_str = match.group(1)
                repaired_json = repair_json(json_str)
                result = json.loads(repaired_json)
                
                if "benefits" in result and isinstance(result["benefits"], list):
                    benefits = result["benefits"]
                    filtered_benefits = benefits
                    return list(dict.fromkeys(filtered_benefits))
            except Exception as e:
                print(f"json object repair attempt failed: {e}")
        
        # If that fails, try to extract the benefits array directly
        benefits_match = re.search(r'"benefits":\s*\[(.*?)(?:\]|}|$)', response, re.DOTALL)
        if benefits_match:
            benefits_str = benefits_match.group(1)
            # Try to repair and parse just the array part
            try:
                array_json = "[" + benefits_str + "]"
                repaired_array = repair_json(array_json)
                benefits = json.loads(repaired_array)
                if isinstance(benefits, list):
                    filtered_benefits = benefits
                    return list(dict.fromkeys(filtered_benefits))
            except Exception as e:
                print(f"Benefits array repair attempt failed: {e}")
            
            # Fallback to regex extraction if repair fails
            benefits = re.findall(r'"([^"]*)"', benefits_str)
            filtered_benefits = benefits
            return list(dict.fromkeys(filtered_benefits))
        
        # If all else fails, try to extract bulleted items
        incentives = clean_incentives(response)
        return list(dict.fromkeys(incentives))
        
    except Exception as e:
        print(f"Error parsing JSON incentives: {e}")
        # Fall back to extracting bullet points
        incentives = clean_incentives(response)
        return list(dict.fromkeys(incentives))


def clean_incentives(response):
    """
    Extract bulleted list items from the model response when json parsing fails.
    
    This function serves as a fallback extraction method for when structured json
    parsing fails. It identifies and extracts items from bulleted or numbered lists
    in the model's text response by looking for common list markers.
    
    :param response: Text response from the language model that may contain bulleted items
    :type response: str
    :return: List of extracted incentives/benefits from bulleted items
    :rtype: list
    
    The function identifies list items by matching lines that start with:

    - Hyphens (-)
    - Asterisks (*)
    - Bullet points (•)
    - Numbered items (1., 2., etc.)
    
    It then cleans these items by removing the leading markers and whitespace.
    """
    incentives = []
    for line in response.split('\n'):
        # Match lines starting with - * or numbers
        if re.match(r'^(\s*[-*•]|\d+\.)\s*', line):
            cleaned = re.sub(r'^[\s\-*•]+\s*', '', line).strip()
            if cleaned:
                incentives.append(cleaned)
    return incentives



def process_jobs(insert_or_replace_job, batch_size=1):
    """
    Process job listings from Mongo-DB and store in SQLite-Database.
    
    This function implements a complete pipeline for processing job listings data:

    1. Extracts required experience, branche based on job title and raw-incentives/benefits using LLama
    2. Classifies raw-incentives/benefits using Sentence-Transformers
    3. Stores the processed data in a SQLite-Database
    
    For key design choices see :ref:`here<Model Selection and Implementation>`)
    
    :param insert_or_replace_job: Function to insert or update job records in the database
    :type insert_or_replace_job: callable
    :param batch_size: Number of jobs to process in each batch, defaults to 1
    :type batch_size: int, optional
    :return: None
    :rtype: NoneType
    
    :raises ConnectionError: If connection to Mongo-DB fails
    :raises RuntimeError: If model loading fails
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cursor = get_mongo_jobs()
    
    #initialize the database, if not already excisting
    db_name = "job_analysis.db"
    create_database(db_name)
    print("created db or found excisting")

    # Get all jobs from Mongo-DB and convert to a list
    mongo_results = list(get_mongo_jobs())
    all_jobs = [job for job, coll_name in mongo_results]
    total_jobs = len(all_jobs)
    print(f"Processing {total_jobs} jobs from Mongo-DB")
    
    # Count for new jobs
    new_jobs_count = 0
    skipped_jobs_count = 0

    # Process jobs in batches to manage memory
    for batch_idx in range(0, len(all_jobs), batch_size):
        batch = all_jobs[batch_idx:batch_idx+batch_size]
        #load model outside loop
        model = AutoModelForCausalLM.from_pretrained(
                "meta-llama/Llama-3.2-3B-Instruct",
                torch_dtype=torch.float16 #reduced due to memory constraints
            ).to(device).eval()
            
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
        tokenizer.pad_token = tokenizer.eos_token
            
        generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)

        # Process each job in the batch
        with torch.inference_mode():  # Disables gradients and reduces memory
            for job in batch:
                # Check if the job already exists in SQLite
                mongodb_id = str(job.get('_id', ''))
                if mongodb_id and job_exists(db_name, mongodb_id):
                    print(f"Job with Mongo-DB ID {mongodb_id} already exists in SQLite. Skipping.")
                    skipped_jobs_count += 1
                    continue
                new_jobs_count += 1
                print(f"Processing new job with Mongo-DB ID: {mongodb_id}")

                # Extract job description for experience-required prompt
                desc_parts = []
                #indeed strucuture
                if 'paragraphs' in job:
                    job_desc = job.get('paragraphs', [])
                    if isinstance(job_desc, list):
                        desc_parts.extend(map(str, job_desc))
                    elif isinstance(job_desc, dict):
                        for section in job_desc.values():
                            if isinstance(section, list):
                                desc_parts.extend(map(str, section))
                #stepstone structure
                if 'lists' in job:
                    for lst in job['lists'].values():
                        desc_parts.extend(map(str, lst))

                full_desc = ' '.join(desc_parts)

                #print("\n" + "="*50 + "\nFULL DESCRIPTION FOR LLM (Berufserfahrung_vorausgesetzt):\n" + "="*50)
                #print(full_desc)
                
                entities = extract_entities_from_json(job)
                
                # First phase: Extract job details
                details_prompt = f"""You are a helpful assistant that extracts structured information from job descriptions.
                Extract the following information from this text {full_desc}:

                For Experience Required:
                - Return 1 if the job description explicitly mentions requiring professional experience
                - Else return 0 
                - Only return 0 or 1

                EXAMPLE OUTPUT:
                {{
                "Experience_Required": 1/0
                }}
                Respond only with valid JSON. Do not write an introduction or summary.
                """
                
                response = generator(details_prompt, return_full_text=False, max_new_tokens=15, do_sample=True, temperature=0.1, repetition_penalty=1.2)
                details = parse_json_response(response[0]['generated_text'])
                
                # Print raw outputs before parsing
                print("\n" + "="*50 + "\nRAW LLM OUTPUT (DETAILS):\n" + "="*50)
                print(response[0]['generated_text'])
                
                # Print details response
                print("\n" + "="*50 + "\nJOB DETAILS ANALYSIS:\n" + "="*50)
                print(json.dumps(details, indent=2, ensure_ascii=False))
                
                # Free memory
                del response
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                
                # Second phase: Extract incentives
                benefits_text = get_benefits_text(job)
                #direct benefits
                direct_benefits = get_direct_benefits(job)
                
                # Only benefits section
                extraction_prompt = f"""Antworte ausschließlich auf Deutsch.

                Extrahiere alle Incentives/Benefitss aus folgender Stellenanzeige, soweit vorhanden:

                {benefits_text}

                REGELN:
                1. Extrahiere nur Incentives/Benefits, die sich direkt auf die Incentives/Benefits, Vergütungen oder Zusatzleistungen für Mitarbeitende beziehen (z.B. finanzielle Anreize, flexible Arbeitsmodelle, Zusatzleistungen).
                2. Schließe Aufgaben, Tätigkeiten, Anforderungen oder allgemeine Rollenbeschreibungen aus.
                3. WICHTIG: Erfinde oder ergänze KEINE Incentives/Benefits – nur exakt das übernehmen, was im Text steht.
                4. Vermeide Wiederholungen mit leicht abweichenden Formulierungen.
                5. Übernimm die Incentives/Benefits wortwörtlich aus dem Text, NICHT umformulieren.
                6. Gib keine allgemeinen Begriffe wie „finanzielle Anreize“, „flexible Arbeitszeitmodelle“ oder „zusätzliche Ausbildungsmöglichkeiten“ aus, wenn diese nicht wortwörtlich im Text stehen.


                BEISPIELE:
                Keine Incentives/Benefits (NICHT übernehmen!):
                - "Herausfordernde Aufgaben" → Beschreibung der Tätigkeit
                - "Wir entwickeln innovative Lösungen" → Allgemeine Aussage
                - "Erfahrung mit Cloud-Technologien" → Anforderung
                - "Verantwortung für IT-Projekte" → Aufgabenbeschreibung
                - "Engagiertes Team" -> kein direkter Mitarbeitervorteil
                - "Vollzeit/Teilzeit" -> Arbeitsmodelle, keine Incentives/Benefits
                - "Gehalt" -> nur ein generischer Begriff ohne Aussage

                Gib die Antwort ausschließlich als gültiges JSON-Objekt im folgenden Format zurück und nichts anderes:
                {{
                    "benefits": [
                        "benefit 1",
                        "benefit 2",
                        "benefit 3"
                    ]
                }}

                Keine Erklärungen, keine Beispiele, keine Einleitung, keine Zusammenfassung. Nur das JSON-Objekt im vorgegebenen Format!
                """
                
                response = generator(extraction_prompt, return_full_text=False, do_sample=False, max_new_tokens=350, repetition_penalty=1.1)
                raw_response = response[0]['generated_text']
                
                print("\n" + "="*50 + "\nRAW LLM OUTPUT (INCENTIVES):\n" + "="*50)
                print(response[0]['generated_text'])
                
                incentives = parse_json_incentives(raw_response)
                
                # Print incentives as a list
                print("\n" + "="*50 + "\nINCENTIVES ANALYSIS:\n" + "="*50)
                if incentives:
                    print("Found Incentives:", json.dumps(incentives, indent=2, ensure_ascii=False))
                else:
                    print("No incentives were found.")
                
                # Free memory
                del response
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

                #Third phase: find jobs branche
                job_title = entities.get("job_title", "Kein Titel")
                #categorize the branche
                category_prompt = f"""Klassifiziere diesen Job-Titel in genau eine der folgenden Kategorien:
                - IT: Softwareentwicklung, Programmierung, Cloud, Backend, DevOps, KI, Datenanalyse
                - Gesundheit: Medizin, Pflege, Pharma, Krankenhauswesen, Gesundheitswesen
                - Technik: Maschinenbau, Elektrotechnik, Produktion, Fertigung
                - Bildung & Forschung: Lehre, Wissenschaft, Forschungsinstitute, Universitäten
                - Finanzen: Bankwesen, Versicherungen, Buchhaltung, Steuerberatung
                - Recht: Jura, Compliance, Notardienste, Rechtsberatung
                - Marketing & Medien: Werbung, Journalismus, Social Media, Content Creation
                - Handel & E-Commerce: Einzelhandel, Online-Handel, Vertrieb, E-Commerce
                - Bau & Handwerk: Baugewerbe, Handwerksbetriebe, Sanierung
                - Logistik: Transport, Lagerhaltung, Supply Chain Management
                - Öffentlicher Dienst: Behörden, öffentliche Einrichtungen, Verwaltung
                - Andere: Keine Übereinstimmung mit oben genannten

                Job-Titel: {job_title} 

                Regeln:
                1. Wähle die spezifischste passende Kategorie
                2. Bei Überschneidungen (z.B. IT in Gesundheitsbereich) primäre Branche wählen
                3. Im Zweifelsfall "Andere"

                Antworte NUR im JSON-Format: {{"Kategorie": "<Kategorie>"}} Schreibe keine Beispiele, Einleitungen oder Erklärungen."""

                category_response = generator(
                    category_prompt,
                    return_full_text=False,
                    max_new_tokens=20,
                    do_sample=True,
                    temperature=0.1,
                    repetition_penalty=1.1,
                    top_p=0.95
                )

                raw_output = category_response[0]['generated_text']
                matches = list(re.finditer(r'\{.*?\}', raw_output, re.DOTALL))
                job_category = "Andere"
                if matches:
                    try:
                        last_json = matches[-1].group(0)
                        category_data = json.loads(last_json)
                        job_category = category_data.get("Kategorie", "Andere")
                    except:
                        pass

                # Normalisierung & Validierung
                normalized = job_category.lower().replace(" ", "")
                valid_categories_norm = [c.lower().replace(" ", "") for c in [
                    "IT", "Gesundheit", "Technik", "Bildung & Forschung", 
                    "Finanzen", "Recht", "Marketing & Medien", "Handel & E-Commerce",
                    "Bau & Handwerk", "Logistik", "Öffentlicher Dienst", "Andere"
                ]]

                if normalized not in valid_categories_norm:
                    job_category = "Andere"
                
                #free memory
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

                #Fourth phase: Categorize incentives with DistilBERT
                categorized = {cat: 0 for cat in INCENTIVE_CATEGORIES}
                others = []
                
                #incentives:
                incentives = list(dict.fromkeys(incentives))
                    
                print("\n" + "="*50 + "\nINCENTIVES PASSED TO CATEGORIZATION:\n" + "="*50)
                print(json.dumps(incentives, indent=2, ensure_ascii=False))
                    
                # Use sentence-transformer for classification
                #put direct incentives and llm incentives together
                all_incentives = direct_benefits + incentives
                print("\n" + "="*50 + "\nCOMBINED INCENTIVES (DIRECT + LLM):\n" + "="*50)
                print(json.dumps(all_incentives, indent=2, ensure_ascii=False))

                categorized, others = classify_incentives_with_few_shot(all_incentives, device, threshold=0.45)

                # Combined Homeoffice detection (rule-based + Sentence Transformers)
                homeoffice_direct = detect_homeoffice(job)
                homeoffice_llm = categorized.get("Homeoffice", 0)  # From Sentence Transformers
                categorized["Homeoffice"] = 1 if (homeoffice_direct or homeoffice_llm) else 0
                    
                # Print classification results
                print("\n" + "="*50 + "\nDISTILBERT CLASSIFICATION RESULTS:\n" + "="*50)
                for cat, val in categorized.items():
                    if val == 1:
                        print(f"{cat}: {val}")
                
                if others:
                    print("\nUnmatched incentives:", json.dumps(others, indent=2, ensure_ascii=False))
            
                # Process locations and save to database
                locations = entities.get("location", [{"city": "Unspecified", "state": "Unbekannt", "country": "Unbekannt"}])
                time_models = entities.get("time_model", ["nan"])
                entry_levels = [entities.get("entry_level", "nan")]
                
                # Process each location
                seen_locations = set()

                #initialize combo count
                combo_count = 0

                # Create row with single location
                for loc, tm, el in product(locations, time_models, entry_levels):
                    row_data = {
                        'MongoDB_ID': mongodb_id,
                        'Job_URL': job.get('url', ''),
                        'Portal_Name': extract_website_name(job.get('url', '')),
                        'Job_Titel': entities.get('job_title', 'Kein Titel'),
                        'Datum': get_formatted_date(job),
                        'Stadt': loc["city"],
                        'Bundesland': loc["state"],
                        'Land': loc["country"],
                        'Unternehmen': entities.get("company", 'nan'),
                        'Unternehmensgröße': entities.get("company_size", 'Keine Angaben'),
                        'Zeitmodell': tm,
                        'Position': el,
                        'Beschäftigungsart': entities.get("job_type", 'nan'),
                        'Berufserfahrung_vorausgesetzt': details.get('Experience_Required', '0'),
                        'Kategorie': job_category,
                        # Incentives
                        'Gehalt_anhand_von_Tarifklassen': categorized.get("Gehalt_anhand_von_Tarifklassen", 0),
                        'Überstundenvergütung': categorized.get("Überstundenvergütung", 0),
                        'Gehaltserhöhungen': categorized.get("Gehaltserhöhungen", 0),
                        'Aktienoptionen_Gewinnbeteiligung': categorized.get("Aktienoptionen_Gewinnbeteiligung", 0),
                        'Boni': categorized.get("Boni", 0),
                        'Sonderzahlungen': categorized.get("Sonderzahlungen", 0),
                        '13. Gehalt': categorized.get("13. Gehalt", 0),
                        'Betriebliche_Altersvorsorge': categorized.get("Betriebliche_Altersvorsorge", 0),
                        'Flexible_Arbeitsmodelle': categorized.get("Flexible_Arbeitsmodelle", 0),
                        'Homeoffice': categorized.get("Homeoffice", 0),
                        'Weiterbildung_und_Entwicklungsmöglichkeiten': categorized.get("Weiterbildung_und_Entwicklungsmöglichkeiten", 0),
                        'Gesundheit_und_Wohlbefinden': categorized.get("Gesundheit_und_Wohlbefinden", 0),
                        'Finanzielle_Vergünstigungen': categorized.get("Finanzielle_Vergünstigungen", 0),
                        'Mobilitätsangebote': categorized.get("Mobilitätsangebote", 0),
                        'Verpflegung': categorized.get("Verpflegung", 0),
                        'Arbeitsumfeld_Ausstattung': categorized.get("Arbeitsumfeld_Ausstattung", 0),
                        'Zusätzliche_Urlaubstage': categorized.get("Zusätzliche_Urlaubstage", 0),
                        'Familien_Unterstützung': categorized.get("Familien_Unterstützung", 0),
                        'Onboarding_und_Mentoring_Programme': categorized.get("Onboarding_und_Mentoring_Programme", 0),
                        'Teamevents_Firmenfeiern': categorized.get("Teamevents_Firmenfeiern", 0),
                        # Add unmatched benefits to "others"
                        'others': '; '.join(others)
                    }
                    
                    #print examples for verification
                    print("\n" + "="*50 + "\nWRITING TO DATABASE:\n" + "="*50)
                    for k, v in row_data.items():
                        print(f"{k}: {v}")

                    #test für Job-Kategorie
                    print("\n" + "="*50 + "\nKLASSIFIZIERUNGSAUSGABE:\n" + "="*50)
                    print(f"Kategorie: {job_category}")
                    print(f"LLM-Antwort: {category_response[0]['generated_text']}")

                    # Insert or replace the row in the database
                    insert_or_replace_job(db_name, row_data)
                    del row_data

                    # Memory management within the location loop
                    combo_count += 1
                    if combo_count % 1 == 0:  # Clear cache every 2 combinations
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()

        # This section should be outside all loops (location, time_model, entry_level)
        # Clean up models after processing the current job
        import gc
        del generator
        del model
        del tokenizer
        torch.cuda.synchronize()  # Wait for all operations to complete
        torch.cuda.empty_cache()
        gc.collect()

        #control print statements to see, if the job got processed properly
        print("\n" + "="*50 + "\nLLAMA MODEL UNLOADED - BATCH COMPLETE\n" + "="*50)
        print(f"Processed {new_jobs_count} new jobs, skipped {skipped_jobs_count} existing jobs out of {total_jobs} total jobs.")


process_jobs(insert_or_replace_job)
