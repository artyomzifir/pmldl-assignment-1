Architecture
============

This page explains why the project is structured the way it is. If you are looking
for setup instructions, see :doc:`getting-started`.

----

Three Services, Not One Script
--------------------------------

The most obvious way to build this project would be a single Python script that
downloads data, trains the model, and then starts a web server. That works, but
it creates a problem: every time the server restarts, it re-trains the model from
scratch.

Splitting the work into three services solves this:

- ``prepare`` runs once, produces artifacts, and exits.
- ``api`` loads those artifacts and stays running to serve requests.
- ``app`` talks to the API and handles the user interface.

Each service has a single responsibility. Changing the UI does not require
restarting the inference server. Re-training the model does not require rebuilding
the frontend image.

----

How the Services Connect
-------------------------

``prepare`` writes three files to a directory that is mounted as a shared volume
into the ``api`` container:

.. code-block:: text

   src/prepare/dataset/train.pt      — training split as PyTorch tensors
   src/prepare/dataset/test.pt       — test split as PyTorch tensors
   src/prepare/model/simple_nn.pth   — trained model weights (state dict)

``docker-compose.yml`` uses ``depends_on`` to enforce startup order:
``api`` starts only after ``prepare`` exits, and ``app`` starts only after ``api``
is up. This prevents the inference server from crashing on a missing weights file.

.. code-block:: text

   ┌───────────┐   writes artifacts to shared volume   ┌──────────────────┐
   │  prepare  │ ─────────────────────────────────────▶│  simple_nn.pth   │
   │  (exits)  │                                        │  train.pt        │
   └───────────┘                                        │  test.pt         │
                                                        └──────────────────┘
                                                                │
                                              volume mounted into api container
                                                                │
                                                                ▼
   ┌───────────┐          HTTP JSON            ┌───────────────────────────┐
   │    app    │ ◀───────────────────────────▶ │           api             │
   │ Streamlit │    POST /predict              │  FastAPI + SimpleNN       │
   │ :8501     │    GET  /health               │  :8085                    │
   └───────────┘                               └───────────────────────────┘

----

Why TensorFlow for Data, PyTorch for the Model
-----------------------------------------------

``data_processing.py`` downloads MNIST using ``tf.keras.datasets.mnist.load_data()``.
This is not a deliberate architectural choice — TensorFlow's Keras API provides
the most convenient one-line access to MNIST, and the ``prepare`` container is
the only place TensorFlow appears. Everything downstream uses PyTorch.

After downloading, the script converts the NumPy arrays to PyTorch tensors and saves
them as ``.pt`` files. From that point on, TensorFlow is no longer involved.

The conversion also handles the channel dimension. Keras returns images in
``[N, H, W]`` format (no channel axis). PyTorch's ``Conv2d`` expects
``[N, C, H, W]``, so ``np.expand_dims(..., 1)`` inserts the missing axis before
the tensors are saved.

----

Model Architecture
------------------

``SimpleNN`` is a compact convolutional classifier defined in ``model.py``:

.. code-block:: text

   Input [N, 1, 28, 28]
       │
       ▼
   Conv2d(1 → 32, kernel 3×3, padding 1)  →  ReLU  →  MaxPool2d(2)
       │                                                output: [N, 32, 14, 14]
       ▼
   Conv2d(32 → 64, kernel 3×3, padding 1)  →  ReLU  →  MaxPool2d(2)
       │                                                output: [N, 64, 7, 7]
       ▼
   Flatten  →  Dropout(0.2)  →  Linear(3136 → 128)  →  ReLU  →  Dropout(0.2)
       │
       ▼
   Linear(128 → 10)
       │
       ▼
   Logits [N, 10]

Two dropout layers with rate 0.2 are added after the flatten and after the first
linear layer. On a dataset as clean as MNIST this has minimal effect, but it
demonstrates a standard practice for regularization.

Training runs for 5 epochs with the Adam optimizer (learning rate 1e-3) and
cross-entropy loss. This consistently reaches around 98–99% accuracy on the
test set.

----

Inference Pipeline
------------------

When ``app`` sends a drawing to ``POST /predict``, the following steps happen:

1. The Streamlit canvas produces a 280×280 RGBA image.
2. ``preprocess_to_28x28()`` converts it to grayscale, resizes to 28×28 using
   nearest-neighbor interpolation, and binarizes: pixels above zero become 1,
   everything else stays 0. This matches the format the model was trained on.
3. The resulting array is serialized as a JSON nested list and sent to the API.
4. The API deserializes the list, calls ``torch.tensor(...).unsqueeze(0).unsqueeze(0)``
   to produce a ``[1, 1, 28, 28]`` tensor, and passes it through the model.
5. ``argmax`` on the output logits gives the predicted class. ``softmax`` converts
   the logits to a probability distribution over all 10 digits.
6. Both values are returned as JSON and rendered in the Streamlit UI.

----

What Is Not Here
----------------

This project deliberately omits several things that a production deployment would
need: authentication, input validation beyond Pydantic's type checking, model
versioning, logging, and monitoring. The goal is to show the minimum viable
structure for serving a model — not to build production infrastructure.