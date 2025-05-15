from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import json
import re

def classify_job_category_interactive():
    # Modell- und Tokenizer-Konfiguration
    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    
    # Modell mit accelerate laden (device_map="auto")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"  # Wichtig: accelerate verwaltet das Gerät
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Pipeline OHNE device-Parameter erstellen
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.float16  # dtype beibehalten
    )

    # Kategoriedefinition wie im Hauptcode
    kategorien = [
        "Gehalt_anhand_von_Tarifklassen",	"Überstundenvergütung",	"Gehaltserhöhungen",
        "Aktienoptionen",	"Boni_und_Sonderzahlungen",	"13. Gehalt",	"Betriebliche_Altersvorsorge",	"Flexible_Arbeitsmodelle",	"Homeoffice",
        "Weiterbildung_und_Entwicklungsmöglichkeiten","Gesundheit_und_Wohlbefinden", "Finanzielle_Vergünstigungen", "Mobilitätsangebote",
        "Verpflegung", "Arbeitsumfeld_Ausstattung", "Zusätzliche_Urlaubstage",
        "Familien_Unterstützung", "Onboarding_und_Mentoring_Programme", "Teamevents_Firmenfeiern", "others", "non_incentive" 
        ]

    while True:
        print("\n" + "="*50)
        job_desc = input("Geben Sie eine Stellenbeschreibung ein (oder 'exit'):\n")
        
        if job_desc.lower() == 'exit':
            break

        # Original-Prompt aus dem Hauptcode
        category_prompt = f"""Aufgabe: Ordne diese Benefits {job_desc[:1500]} diesen vordefinierten Kategorien zu: {kategorien}.

        **Regeln:**
        1. Keine persönlichen Interpretationen, keine Kombinationen, keine Erklärungen, keine Kategorien weglassen, keine neuen Kategorien erfinden.
        2. Wenn du dir bei der Zuordnung unsicher bist, entscheide dich für die wahrscheinlichste Kategorie.
        3. Nutze die Beispiele unten, um die Klassifizierungslogik und das korrekte Format zu verstehen.

        1. **Gehalt_anhand_von_Tarifklassen**  
           Beschreibung: Explizite Bezugnahme auf Tarifverträge  
           Beispiel: „Bezahlung nach TVöD“, „Eingruppierung nach IG-Metall-Tarif“

        2. **Überstundenvergütung**  
           Beschreibung: Kompensation für Mehrarbeit  
           Beispiel: „Überstunden mit 25% Zuschlag“, „Gleitzeitausgleich“

        3. **Gehaltserhöhungen**  
           Beschreibung: Regelmäßige Gehaltsanpassungen  
           Beispiel: „Jährliche Gehaltssteigerungen“, „Leistungsabhängige Erhöhungen“

        4. **Aktienoptionen**  
           Beschreibung: Beteiligungen am Unternehmen  
           Beispiel: „Mitarbeiteraktienprogramm“, „Vergünstigte Aktienpakete“

        5. **Boni_und_Sonderzahlungen**  
           Beschreibung: Zusätzliche Zahlungen außerhalb des Grundgehalts  
           Beispiel: „Weihnachtsgeld“, „Leistungsboni“

        6. **13._Gehalt**  
           Beschreibung: Explizit genanntes 13. Monatsgehalt  
           Beispiel: „13. Monatsgehalt“, „Jahressonderzahlung im Dezember“

        7. **Betriebliche_Altersvorsorge**  
           Beschreibung: Altersvorsorgeleistungen des Arbeitgebers  
           Beispiel: „Betriebliche Altersvorsorge (bAV)“, „Pensionskasse“

        8. **Flexible_Arbeitsmodelle**  
           Beschreibung: Flexible Zeitgestaltung  
           Beispiel: „Vertrauensarbeitszeit“, „Kernzeitmodell“

        9. **Homeoffice**  
           Beschreibung: Mobiles Arbeiten außerhalb des Büros  
           Beispiel: „3 Tage Homeoffice pro Woche“, „Remote Work Optionen“

        10. **Weiterbildung_und_Entwicklungsmöglichkeiten**  
            Beschreibung: Qualifizierungsangebote  
            Beispiel: „Zertifizierungskurse“, „Fachliche Weiterbildungen“

        11. **Gesundheit_und_Wohlbefinden**  
            Beschreibung: Betriebliche Gesundheitsförderung  
            Beispiel: „Firmenfitnessstudio“, „Kostenlose Gesundheitschecks“

        12. **Finanzielle_Vergünstigungen**  
            Beschreibung: Monetäre Vorteile außerhalb des Gehalts  
            Beispiel: „Corporate Benefits“, „Mitarbeiterrabatte“

        13. **Mobilitätsangebote**  
            Beschreibung: Unterstützung bei Transportkosten  
            Beispiel: „Jobticket“, „Dienstwagen“, „Fahrradleasing“

        14. **Verpflegung**  
            Beschreibung: Subventionierte Essensangebote  
            Beispiel: „Kantine mit Zuschuss“, „Kostenlose Getränke“

        15. **Arbeitsumfeld_Ausstattung**  
            Beschreibung: Arbeitsmittel und Ergonomie  
            Beispiel: „Ergonomischer Arbeitsplatz“, „Firmenlaptop“

        16. **Zusätzliche_Urlaubstage**  
            Beschreibung: Urlaub über gesetzliches Minimum  
            Beispiel: „35 Tage Jahresurlaub“, „Sonderurlaub bei Hochzeit“

        17. **Familien_Unterstützung**  
            Beschreibung: Vereinbarkeit von Familie und Beruf  
            Beispiel: „Betriebskindergarten“, „Eltern-Kind-Büro“

        18. **Onboarding_und_Mentoring_Programme**  
            Beschreibung: Einarbeitung und Karrierebegleitung  
            Beispiel: „Patenprogramm für Neueinsteiger“, „Karrierecoaching“

        19. **Teamevents_Firmenfeiern**  
            Beschreibung: Gemeinschaftsaktivitäten  
            Beispiel: „Jährliche Skiwoche“, „Sommerfest“

        20. **non_incentives**  
            Beschreibung: Keine echten Mitarbeitervorteile  
            Beispiel: „Vollzeitstelle“, „Homeoffice-Pflicht“, „Gehaltsangabe im Stelleninserat“

        ### Few-Shot Beispiele:
        **Beispiel 1:**
        Eingabe: 
        - Bezahlung nach IG-Metall-Tarif
        - Jährliche Gehaltsanpassungen
        - Betriebliches Fitnessstudio

        Ausgabe:
        Gehalt_anhand_von_Tarifklassen:1
        Gehaltserhöhungen:1
        Gesundheit_und_Wohlbefinden:1
        [Alle anderen Kategorien]:0

        **Beispiel 2:**
        Eingabe: 
        - 13. Monatsgehalt
        - Jobrad-Leasing
        - Vollzeitstelle

        Ausgabe:
        13._Gehalt:1
        Mobilitätsangebote:1
        non_incentives:1  Vollzeitstelle
        [Alle anderen Kategorien]:0

        AUSGABEFORMAT:
        Gehalt_anhand_von_Tarifklassen: 0
        Überstundenvergütung: 0
        ... [Alle Kategorien auflisten]



        Antworte NUR im vorgegebenen Format. Maximal 250 Tokens. Keine Einleitung."""


        # Generierung mit Originalparametern
        response = generator(
            category_prompt,
            max_new_tokens=250,
            do_sample=True,
            temperature=0.1,
            repetition_penalty=1.1,
            return_full_text=False
        )

        # Response-Verarbeitung wie im Hauptcode
        raw_output = response[0]['generated_text']
        print("\n" + "="*50)
        print("ROHE LLM-ANTWORT:")
        print(raw_output)

        # JSON-Extraktion
        try:
            json_match = re.search(r'\{.*?\}', raw_output, re.DOTALL)
            if json_match:
                category_data = json.loads(json_match.group(0))
                job_category = category_data.get("Kategorie", "Andere")
                
                # Validierung
                if job_category not in kategorien:
                    job_category = "Andere"
            else:
                job_category = "Andere"
        except:
            job_category = "Andere"

        print("\nERGEBNIS:")
        print(f"Erkannte Kategorie: {job_category}")

        # Speicherbereinigung
        torch.cuda.empty_cache()

if __name__ == "__main__":
    classify_job_category_interactive()
