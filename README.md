# PJS_Pipeline

This repo includes:
1. Python Code
   - main pipeline files (main, extraction, mongodb and sqlite)
   - a zip containing the json country database (needs to be unpacked in the same directory)
   - a testing folder with all the necessary code and files to test the pipeline on accuracy and recall (see documentation under instructions for more information)
     
2. Documentation (branch gh-pages)
   - https://tomz2000-lab.github.io/PJS_Pipeline/
   - html files with a detailed documentation on all the code snippets and extraction details
     
3. Docker
   - all the python code for the docker plus the docker-file for creation
   - run sudo docker build -t my-pipeline . for creation
   - run sudo docker run \--gpus all \-v ~/.cache/huggingface:/root/.cache/huggingface \-v ~/docker-data/sqlite:/app/data \my-pipeline to run the docker (stores the llms in cache for reuse and uses all available gpus)
   - run sudo 0 * * * * docker run --rm --gpus all -v ~/.cache/huggingface:/root/.cache/huggingface -v ~/docker-data/sqlite:/app/data my-pipeline if you want to run the docker every hour in the same manner as described above
  

As part of this project, my main tasks were:

- Extraction of Raw Data from MongoDB
-> Retrieve the necessary data directly from the MongoDB database.

- Structuring the Information using Natural Language Processing
-> Process and organize the extracted data into a coherent format suitable for further analysis and visualization.

- Loading the Structured Data into SQLite Database
-> Import the structured data into an SQLite database, ensuring it is ready for use by the project’s dashboard.
