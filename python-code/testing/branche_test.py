from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import json
import re

def classify_job_category_interactive():
    # model- and tokenizer configuration
    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto" 
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # pipeline without device-parameter 
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.float16 
    )

    # define categories
    kategorien = [
        "IT", "Gesundheit", "Technik", "Bildung & Forschung",
        "Finanzen", "Recht", "Marketing & Medien", "Handel & E-Commerce",
        "Bau & Handwerk", "Logistik", "Öffentlicher Dienst", "Andere"
    ]

    while True:
        print("\n" + "="*50)
        job_desc = input("Geben Sie eine Stellenbeschreibung ein (oder 'exit'):\n")
        
        if job_desc.lower() == 'exit':
            break

        # origional-prompt from extraction.py
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

        Job-Titel: {job_desc[:1500]}

        Regeln:
        1. Wähle die spezifischste passende Kategorie
        2. Bei Überschneidungen (z.B. IT in Gesundheitsbereich) primäre Branche wählen
        3. Im Zweifelsfall "Andere"

        Antworte NUR im JSON-Format: {{"Kategorie": "<Kategorie>"}} Schreibe keine Beispiele, Einleitungen oder Erklärungen."""

        # origional parameters from extraction.py
        response = generator(
            category_prompt,
            max_new_tokens=250,
            do_sample=True,
            temperature=0.1,
            repetition_penalty=1.1,
            return_full_text=False
        )

        # response-parsing as in extraction.py
        raw_output = response[0]['generated_text']
        print("\n" + "="*50)
        print("ROHE LLM-ANTWORT:")
        print(raw_output)

        # json-extraction
        try:
            json_match = re.search(r'\{.*?\}', raw_output, re.DOTALL)
            if json_match:
                category_data = json.loads(json_match.group(0))
                job_category = category_data.get("Kategorie", "Andere")
                
                # validation
                if job_category not in kategorien:
                    job_category = "Andere"
            else:
                job_category = "Andere"
        except:
            job_category = "Andere"

        print("\nERGEBNIS:")
        print(f"Erkannte Kategorie: {job_category}")

        # empty cache
        torch.cuda.empty_cache()

if __name__ == "__main__":
    classify_job_category_interactive()
