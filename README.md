## 📚 Table of Contents

- [🏃 General Idea and Motivation](#-general-idea-and-motivation)
- [🏛️ Overview on my Part of the Project](#-overview-on-my-part-of-the-project)
- [✨ Features](#-features)
- [💻 Technologies Used](#-technologies-used)
- [📁 Repository Structure](#-repository-structure)
- [⚙️ Setup and Installation](#-setup-and-installation)
  - [Prerequisites](#prerequisites)
  - [Clone the Repository](#clone-the-repository)
  - [Install Dependencies](#install-dependencies)
  - [Configure Environment](#configure-environment)
  - [Build and Run with Docker](#build-and-run-with-docker)
- [▶️ Usage](#-usage)
- [📄 Documentation](#-documentation)
- [🧪 Testing](#-testing)
- [👥 Contributions](#-contributions)



# 🏃 General Idea and Motivation

Mergers and acquisitions present substantial challenges for companies, especially in aligning compensation and benefits systems between merged organizations. PwC, as a leading German auditing and consulting firm, regularly addresses these issues during post-merger integration. Job advertisements offer valuable but unstructured data on current market compensation and benefits trends. Our project, in collaboration with PwC, developed an AI-powered tool that uses web scraping and NLP to collect and analyze this data efficiently. This tool provides PwC consultants with actionable, evidence-based insights to support fair and effective post-merger integration.



## 🏛️ Overview on my Part of the Project

**PJS_Pipeline** is designed to process job listing data for analytical and dashboard purposes. The pipeline:

- **Extracts raw data** from a MongoDB database.
- **Processes and structures** the information using Natural Language Processing (NLP) techniques.
- **Loads the structured data** into an SQLite database, making it accessible for visualization and analysis.

This repository includes Python code, Docker support, and comprehensive documentation.



## ✨ Features

- **Extracts job listing data** from MongoDB.
- **Applies NLP for data structuring**.
- **Stores processed data** in SQLite for dashboard use.
- **Containerized with Docker** for easy deployment.
- **Includes testing suite** for accuracy and recall evaluation.
- **Detailed documentation** available under [Documentation](#documentation).



## 💻 Technologies Used

- **Python 3.x** – Core language for pipeline scripting and data processing.
- **Natural Language Processing (NLP)** – For structuring and enriching extracted data.
- **MongoDB** – NoSQL database for raw job listing storage.
- **SQLite** – Lightweight database for structured data output.
- **Docker** – Containerization for consistent deployment.
- **GitHub Pages/Sphinx** – Hosting for documentation.



## 📁 Repository Structure

- **PJS_Pipeline/**
  - README.md
  - **python-code/**
    - countries+states+cities.zip
    - extraction.py
    - main.py
    - mongo_db.py
    - requirements.txt
    - sqlite.py
    - **testing/**
         - branche_test.py
         - performance.py
         - read_db.py
         - test.json
         - validation_file_new.csv
  - **docker/**
   - Dockerfile
   - Python-Code
   - ...



## ⚙️ Setup and Installation

### 1. Prerequisites

- **Docker** – Required for containerized deployment.
- **MongoDB** – Local or cloud-hosted instance.
- **SQLite** – Included with Python.
- **Hadware** - Have at least 8 GB of VRAM available and a GPU supporting CUDA.

### 2. Clone the Repository

```bash
git clone https://github.com/tomz2000-lab/PJS_Pipeline.git
cd PJS_Pipeline
```

### 3. Install Dependencies

For local use (without Docker):

```bash
pip install -r requirements.txt
```

### 4. Configure Environment

Ensure your MongoDB connection and Hugging-Face API-Key details are set in a `.env` file.


### 5. Build and Run with Docker

5.1 Build the Docker image:

```bash
sudo docker build -t my-pipeline .
```

5.2 Run the container (with GPU support and persistent cache):

```bash
sudo docker run --gpus all -v ~/.cache/huggingface:/root/.cache/huggingface -v ~/docker-data/sqlite:/app/data my-pipeline
```


5.3 To run every hour:

```bash
0 * * * * sudo docker run --rm --gpus all -v ~/.cache/huggingface:/root/.cache/huggingface -v ~/docker-data/sqlite:/app/data my-pipeline
```



## ▶️ Usage

- **With Docker:** Follow the [Build and Run with Docker](#build-and-run-with-docker) steps.
- **Locally:** After setting up dependencies, MongoDB and Hugging-Face access, run `main.py`.



## 📄 Documentation

Detailed documentation is available at:  
[https://tomz2000-lab.github.io/PJS_Pipeline/](https://tomz2000-lab.github.io/PJS_Pipeline/)



## 🧪 Testing

The `testing/` folder contains all necessary code and files to evaluate the pipeline's accuracy and recall. See the documentation under Performance Moduel and Metrics for instructions and explanations.



## 👥 Contributions

My main tasks in this project were:

- **Extraction of raw data from MongoDB**  
  Retrieve the necessary data directly from the MongoDB database.
- **Structuring the information using natural language processing**  
  Process and organize the extracted data into a coherent format suitable for further analysis and visualization.
- **Loading the structured data into SQLite database**  
  Import the structured data into an SQLite database, ensuring it is ready for use by the project’s dashboard.



*© Tom Ziegler, 2025*
