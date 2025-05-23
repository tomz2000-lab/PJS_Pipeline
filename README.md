# PJS_Pipeline

This repo includes:
1. python code
   - main pipeline files (main, extraction, mongodb and sqlite)
   - a zip containing the country database (needs to be unpacked in the same directory)
   - a testing folder with all the necessary code and files to test the pipeline on accuracy and recall (see readme and docu for more information)
     
2. documentation (branch gh-pages)
   - https://tomz2000-lab.github.io/PJS_Pipeline/
   - html files with a detailed documentation on all the code snippets and extraction details
   - also contains the code with the docstrings again to generate the documentation via sphinx
     
3. docker
   - all the python code for the docker plus the docker-file for the image creation
   - the .bat file of the docker for sirect usage
   - run sudo docker build -t my-pipeline . for creation
   - run sudo docker run \--gpus all \-v ~/.cache/huggingface:/root/.cache/huggingface \-v ~/docker-data/sqlite:/app/data \my-pipeline to run the docker (stores the llms in cache for reuse and uses all available gpus)
   - run sudo 0 * * * * docker run --rm --gpus all -v ~/.cache/huggingface:/root/.cache/huggingface -v ~/docker-data/sqlite:/app/data my-pipeline if you want to run the docker every hour in the same manner as described above
