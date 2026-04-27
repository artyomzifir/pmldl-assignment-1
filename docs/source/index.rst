PMLDL Assignment 1 -- Getting Started
======================================

This guide walks you through running the MNIST CNN Predictor project from scratch.
By the end, you will have three services running locally: a data preparation pipeline,
a FastAPI inference backend, and a Streamlit UI where you can draw a digit and get
a real-time prediction from the trained model.

**Who this is for.** You are comfortable writing Python and have trained a neural
network before -- in a notebook or a plain script. You have not worked with Docker,
FastAPI, or multi-service deployments yet. This guide assumes no prior knowledge
of those tools.

.. toctree::
   :maxdepth: 2
   :hidden:

   architecture
   code_reference

----

Prerequisites
-------------

You need two tools installed on your machine before you start.

**Docker Desktop** handles everything else -- Python, PyTorch, TensorFlow, and all
other dependencies are installed automatically inside containers. Download it from
`https://www.docker.com/products/docker-desktop <https://www.docker.com/products/docker-desktop>`_
and follow the installer for your operating system. After installation, open Docker
Desktop and make sure the engine is running (you will see a green indicator in the
system tray).

**Git** is needed to clone the repository. Check whether it is already installed:

.. code-block:: bash

   git --version

If the command is not found, download Git from `https://git-scm.com <https://git-scm.com>`_.

No Python environment setup is required on your host machine. All three services run
in isolated containers, so there are no version conflicts with your existing setup.

----

Clone the Repository
--------------------

Open a terminal and run:

.. code-block:: bash

   git clone https://github.com/artyomzifir/pmldl-assignment-1.git
   cd pmldl-assignment-1

The repository has the following structure:

.. code-block:: text

   .
   +-- docker-compose.yml       # Defines and wires all three services
   +-- src
       +-- prepare              # Downloads MNIST and trains the model
       |   +-- dataset
       |   |   +-- data_processing.py
       |   +-- model
       |   |   +-- model.py
       |   +-- Dockerfile
       |   +-- requirements.txt
       +-- api                  # FastAPI inference service
       |   +-- api.py
       |   +-- Dockerfile
       |   +-- requirements.txt
       +-- app                  # Streamlit frontend
           +-- app.py
           +-- Dockerfile
           +-- requirements.txt

----

Start the Services
------------------

Run the following command from the repository root:

.. code-block:: bash

   docker compose up --build

The ``--build`` flag tells Docker to build images from the Dockerfiles before starting.
You only need it on the first run or after changing source files. On subsequent runs,
``docker compose up`` is enough.

.. note::

   The first run can take up to 10 minutes depending on your internet connection
   and hardware. Docker pulls base images, installs all dependencies, downloads
   the MNIST dataset, and trains the CNN for 5 epochs. Subsequent runs skip all
   of that and start in seconds.

----

What Happens During Startup
----------------------------

Understanding the startup sequence helps you read the logs and diagnose issues.

Step 1 -- prepare
~~~~~~~~~~~~~~~~~~

The ``prepare`` service runs first. It checks two conditions before doing any work:

- If ``src/prepare/dataset/train.pt`` and ``test.pt`` do not exist, it runs
  ``data_processing.py``, which downloads MNIST via TensorFlow's Keras API,
  normalizes pixel values from ``[0, 255]`` to ``[0.0, 1.0]``, converts the images
  from NumPy arrays to PyTorch tensors in ``[N, C, H, W]`` format, and saves them
  to disk as ``.pt`` files.

- If ``src/prepare/model/simple_nn.pth`` does not exist, it runs ``model.py``, which
  loads the saved tensors, trains the ``SimpleNN`` convolutional classifier for
  5 epochs using the Adam optimizer and cross-entropy loss, and saves the model weights.

On the first run you will see output similar to this:

.. code-block:: text

   prepare-1  | ===> Generating dataset...
   prepare-1  | Saved train.pt and test.pt in src/prepare/dataset/
   prepare-1  | ===> Training model...
   prepare-1  | Using device: cpu
   prepare-1  | Epoch 1, Loss: 0.2099
   prepare-1  | Epoch 2, Loss: 0.0676
   prepare-1  | Epoch 3, Loss: 0.0503
   prepare-1  | Epoch 4, Loss: 0.0410
   prepare-1  | Epoch 5, Loss: 0.0364
   prepare-1  | Test Accuracy: 99.10%
   prepare-1  | Model saved to src/prepare/model/simple_nn.pth

On every subsequent run, both files already exist on the shared volume, so
``prepare`` skips training entirely and exits in under a second:

.. code-block:: text

   prepare-1  | ===> Dataset already exists
   prepare-1  | ===> Model already exists

Step 2 -- api
~~~~~~~~~~~~~~

Once ``prepare`` exits, Docker starts the ``api`` service. It loads
``simple_nn.pth`` into memory, selects the best available compute backend
(CUDA -> MPS -> CPU), and starts a Uvicorn server on port ``8085``.

The API exposes two endpoints:

.. list-table::
   :header-rows: 1
   :widths: 10 20 40

   * - Method
     - Path
     - Description
   * - GET
     - ``/health``
     - Returns ``{"status": "ok"}`` -- used by container health checks
   * - POST
     - ``/predict``
     - Accepts a 28x28 digit array, returns prediction and probabilities

You will see this line when the API is ready:

.. code-block:: text

   api-1  | INFO:     Application startup complete.

Step 3 -- app
~~~~~~~~~~~~~~

After the API is up, Docker starts the ``app`` service -- a Streamlit application
on port ``8501``. It is ready when you see:

.. code-block:: text

   app-1  |   You can now view your Streamlit app in your browser.
   app-1  |   URL: http://0.0.0.0:8501

----

Known Issue: api Fails on the First Run
-----------------------------------------

On the first run, ``api`` may start before ``prepare`` has finished writing
``simple_nn.pth`` and crash with ``FileNotFoundError``:

.. code-block:: text

   api-1  | FileNotFoundError: [Errno 2] No such file or directory:
   api-1  |     'src/prepare/model/simple_nn.pth'
   api-1 exited with code 1

This is expected. The ``depends_on`` directive in ``docker-compose.yml`` waits for
``prepare`` to start, but not for it to finish training. When this happens, wait
for ``prepare`` to complete and then run:

.. code-block:: bash

   docker compose up

On the second run, the weights already exist and ``api`` starts successfully.

----

Use the Application
-------------------

Open `http://localhost:8501 <http://localhost:8501>`_ in your browser.

The interface has a black canvas on the left and a sidebar with a brush thickness
slider. Draw any digit from 0 to 9 using your mouse or trackpad. A 28x28 preview
of how the model will see your drawing appears below the canvas in real time.

When you are ready, click the **Predict** button. The app sends your drawing to
``POST /predict`` on the API service, and the result appears below: the predicted
digit and a bar chart showing the probability distribution across all 10 classes.

If the model predicts the wrong digit, try drawing larger and centered in the canvas.
The model was trained on MNIST digits that fill most of the 28x28 frame.

----

Explore the API Directly
------------------------

FastAPI generates interactive documentation automatically. While the services are
running, open `http://localhost:8085/docs <http://localhost:8085/docs>`_ in your browser.

You can call ``/predict`` directly from this page without writing any code. Click
**POST /predict -> Try it out**, paste a 28x28 nested list of ``0`` and ``1`` values
into the request body, and click **Execute**. The response will contain the predicted
digit and a ``probs`` array with 10 probabilities that sum to ``1.0``:

.. code-block:: json

   {
     "prediction": 7,
     "probs": [0.0, 0.0, 0.0, 0.01, 0.03, 0.0, 0.0, 0.92, 0.03, 0.01]
   }

----

Stop and Restart
----------------

To stop all services, press ``Ctrl+C`` in the terminal where ``docker compose up`` is
running, or run this in a separate terminal:

.. code-block:: bash

   docker compose down

The trained weights and dataset files persist on the Docker volume between restarts,
so the next ``docker compose up`` will skip training and start in seconds.

----

Troubleshooting
---------------

**Port already in use.**
If port ``8501`` or ``8085`` is occupied, Docker will print
``Bind for 0.0.0.0:8501 failed: port is already allocated``.
Open ``docker-compose.yml`` and change the left side of the port mapping:

.. code-block:: yaml

   ports:
     - "8502:8501"   # host port 8502 -> container port 8501

Then access the app at ``http://localhost:8502`` instead.

**Streamlit canvas does not respond.**
Hard-refresh the page with ``Ctrl+Shift+R`` (Windows/Linux) or ``Cmd+Shift+R`` (macOS).

**Training is slow.**
The model trains on CPU by default when no GPU is available. Five epochs on MNIST
takes approximately 5-10 minutes on a modern laptop CPU. This only happens once --
subsequent starts use the cached weights.