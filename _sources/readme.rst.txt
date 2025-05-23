Instructions
=============

This section documents how the code is constituted and how it runs.

Running the Code
----------------

The code consists of three main modules:

* ``mongo_db.py``: controls for the extraction of the jobs from the Mongo-DB (See :ref:`here<MongoDB Module>` for more information) 
* ``extraction.py``: handles the structuring of the raw data (See :ref:`here<Extraction Module>` for more information)
* ``sqlite.py``: stores the structured data in the SQLite-DB (See :ref:`here<SQlite Module>` for more information)

-> this three parts are started by running ``main.py`` (See :ref:`here<Main Module>` for more information)

For Testing
-----------

Use the ``test.json`` file and put it into your Mongo-DB database, where you can access it by changing lines 35&51 in ``mongo_db.py`` 
to your desired database. 

Then run the code using ``main.py``, which created the local :ref:`job_analysis.db<Job Analysis SQlite Database>`. 

After running successfully start ``read_db.py`` to extract :ref:`job_analysis_export.csv<Job Analysis Export CSV>` from the database.

Then use :ref:`performance.py<Performance Module>`, which will compare the ``validation_file.csv`` with the database-export. (change the file names in line 25&26 if necessary)
The values will appear in the terminal and in a csv called ``performance-history.csv``.

For my performance results look under :ref:`Performance<Performance Metrics>`.

Access the model's env-file
--------------------------

The code expects a .env you have to create, which contains the following keys:

* MONGODB_URI=your_mongodb_connection_string

* HF_API_KEY=your_huggingface_api_key


