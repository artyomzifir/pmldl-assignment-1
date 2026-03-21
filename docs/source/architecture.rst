Architecture
============

Overview
--------

The project is split into three services:

- ``prepare`` downloads MNIST, preprocesses the tensors, and trains the model
- ``api`` loads the trained weights and exposes prediction endpoints
- ``app`` provides a browser UI for drawing a digit and querying the API

Runtime flow
------------

#. The ``prepare`` service checks whether ``train.pt`` and ``test.pt`` already exist
#. If the dataset artifacts are missing, ``data_processing.py`` generates them
#. The same service checks whether ``simple_nn.pth`` already exists
#. If the model weights are missing, ``model.py`` trains the network and saves them
#. The ``api`` service starts after ``prepare`` and loads the saved checkpoint on demand
#. The ``app`` service starts after ``api`` and sends preprocessed ``28 x 28`` images to ``POST /predict``

Repository layout
-----------------

.. code-block:: text

   .
   ├── docker-compose.yml
   ├── README.md
   └── src
       ├── api
       │   ├── api.py
       │   ├── Dockerfile
       │   └── requirements.txt
       ├── app
       │   ├── app.py
       │   ├── Dockerfile
       │   └── requirements.txt
       └── prepare
           ├── dataset
           │   └── data_processing.py
           ├── Dockerfile
           ├── model
           │   └── model.py
           └── requirements.txt

Service graph
-------------

.. code-block:: text

   +-----------+      writes dataset and weights      +-------------------+
   |  prepare  | -----------------------------------> | train.pt          |
   |           | -----------------------------------> | test.pt           |
   |           | -----------------------------------> | simple_nn.pth     |
   +-----------+                                      +-------------------+
         |
         | artifacts available in the shared project tree
         v
   +-----------+        HTTP JSON        +-----------+
   |    api    | <---------------------> |    app    |
   | FastAPI   |                         | Streamlit |
   +-----------+                         +-----------+

Public endpoints
----------------

.. code-block:: text

   GET  /health
   POST /predict

Prediction contract
-------------------

Request body:

.. code-block:: json

   {
     "data": [
       [0, 0, 0, 1, 0],
       [0, 1, 1, 1, 0],
       [0, 0, 0, 1, 0]
     ]
   }

Response body:

.. code-block:: json

   {
     "prediction": 7,
     "probs": [0.0, 0.0, 0.0, 0.01, 0.03, 0.0, 0.0, 0.92, 0.03, 0.01]
   }
