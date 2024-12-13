Tutorial
========

Introduction
------------

NanoFed is a Python library designed to simplify the implementation of federated learning systems, offering out-of-the-box support for coordination, client-server communication, and model aggregation.

In this tutorial, we guide you step-by-step through setting up a federated learning experiment using NanoFed. You will learn how to configure a federated server, manage clients, and utilize the built-in aggregation strategies to perform FL on an example dataset. This tutorial uses the MNIST dataset and a simple classification model, but NanoFed can work with any PyTorch-based classification model and dataset.

First, make sure you have PyTorch and NanoFed installed:

.. code-block:: bash

    pip install nanofed

Step 1: Import Required Modules
-------------------------------

Start by importing the necessary modules.

.. note::
    ``load_mnist_data`` and ``MNISTModel`` are provided as examples in the NanoFed library. You can replace these with any PyTorch ``nn.Module`` that performs classification and PyTorch ``DataLoader``.

.. code-block:: python

    import asyncio
    from pathlib import Path

    import torch

    from nanofed import (
        Coordinator,
        CoordinatorConfig,
        FedAvgAggregator,
        HTTPClient,
        HTTPServer,
        ModelManager,
        coordinate,
        TrainingConfig,
        TorchTrainer
    )
    from nanofed.data import load_mnist_data
    from nanofed.models import MNISTModel

Step 2: Preparing the Global Model
----------------------------------

Set up the global model and initialize the model manager.

The **global model** is a shared model that all clients collaboratively train. At the beginning of the trianing process, the global model is initialized and distributed to all participating clients. Each client then trains this model locally on its private dataset and submits updates back to the server. The server aggregates these updates to refine the global model.

In this step, we define the global model and set up a ``ModelManager`` to handle its versions and storage.

.. code-block:: python

    # Define the base directory for outputs and checkpoints
    base_dir = Path("runs/")

    # Initialize the global model
    model = MNISTModel()  # Any PyTorch classification model
    model_manager = ModelManager(model=model)

Step 3: Setting up the Federated Server
---------------------------------------

The server is the central communication hub in a FL setup. It facilitates interactions between the global model and participating clients. It is responsible for:
1. **Distributing the Global Model**: Clients fetch the current state of the global model from the server.
2. **Collecting Updates**: Cleints send their locally computed updates to the server after training on their private dataset.
3. **Orchestrating Rounds**: The server manages the flow of training rounds.

In NanoFed, the ``HTTPServer`` class implements these functionalities using an HTTP-based protocol.

.. code-block:: python

    server = HTTPServer(
        host="0.0.0.0",
        port=8080,
        # Limit the size of incoming requests. Useful for controlling the size of model updates sent by clients.
        max_request_size=100 * 1024 * 1024,
    )

    # Begin listening for client connections
    await server.start()

Step 4: Configuring the Aggregator
----------------------------------

An **aggregator** is a server component that combines model updates from clients to form a new global model. The aggregation strategy determines how these updates are combined, which can significantly impact the learning process.

Default Aggregation Strategy: Federated Averaging
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As of NanoFed version **0.1.4**, the library supports the **Federated Averaging (FedAvg)** strategy through the ``FedAvgAggregator`` class. This strategy:

1. Computes a weighted average of client model updates based on the number of samples each client processes.

2. Aggregates metrics from clients, such as accuracy or loss.

.. code-block:: python

    # Configure the aggregator
    aggregator = FedAvgAggregator()

.. tip::
    You might want to implement custom aggregation logic. NanoFed makes this easy by providing a ``BaseAggregator`` class.

    .. code-block:: python

        from typing import Sequence
        from nanofed.core import ModelProtocol, ModelUpdate
        from nanofed.server import AggregationResult, BaseAggregator


        class CustomAggregator(BaseAggregator[ModelProtocol]):
            def aggregate(self, model: ModelProtocol, updates: Sequence[ModelUpdate]) -> AggregationResult[ModelProtocol]:
                # Get weights for each client update
                weights = self._compute_weights(updates)

                new_state = {}  # Implement ustom aggregation logic for model parameters

                agg_metrics = {}  # Implement custom metric aggregation

                return AggregationResult(
                    model_state=new_state,
                    metrics=agg_metrics
                )

            def _compute_weights(self, updates: Sequence[ModelUpdate]) -> list[float]:
                # Custom weighting logic
                pass

Step 5: Defining the Coordinator Configuration
----------------------------------------------

The **Coordinator** is a central component in a FL workflow. It manages the orchestration of training rounds, including scheduling, communication, and validation of client updates. Before creating the Coordinator, you need to define its configuration.

Coordinator Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^

The ``CoordinatorConfig`` class specifies the parameters that govern the FL process. These include the number of training rounds, client participation criteria, timeout durations, and directories storing data.

.. code-block:: python

    coordinator_config = CoordinatorConfig(
        num_rounds=2,
        min_clients=3,
        min_completion_rate=0.5,
        round_timeout=300,
        base_dir=base_dir
    )

Key Configuration Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. ``num_rounds``: Specifies the total number of training rounds.
2. ``min_clients``: Minimum number of clients required to participate in each round.
3. ``min_completion_rate``: Minimum fraction of total clients that must complete their training updates in a round.
4. ``round_timeout``: Maximum time (in seconds) to wait for client updates during a training round.
5. ``base_dir``: Base directory for storing data, including metrics, model weights, and configuration files.

Step 6: Initializing the Coordinator
------------------------------------

In this step, you'll create an instance of the ``Coordinator`` class using the previously configured components.

.. code-block:: python

    coordinator = Coordinator(
        model_manager=model_manager,
        aggregator=aggregator,
        server=server,
        config=coordinator_config
    )

Step 7: Implementing a Federeated Client
----------------------------------------

In FL, **clients** are devices or nodes that perform local training on their data and send updates to the server. Each client operates independently.

Overview of a Federated Client's Workflow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. **Dataset Preparation**: Load the local dataset for the client, making sure it matches the expected input for the global model.
2. **Training**: Train the model locally using the client's dataset.
3. **Communication**: Fetch the global model from the server. Submit locally trained updates and metrics to the server.
4. **Iteration**: Repeat the process for each training round until the server signals completion.

Client Implementation
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    async def run_client(client_id: str, coordinator: Coordinator, num_samples: int):
        # Prepare the client's local dataset
        # Use any PyTorch DataLoader.

        # Use different subset sizes for each client to demonstrate FedAvg weighting.
        subset_fraction = num_samples / 60000  # MNIST has 60,000 samples

        train_loader = load_mnist_data(
            data_dir=coordinator.data_dir,
            batch_size=64,
            train=True,
            subset_fraction=subset_fraction
        )

        # Configure training hyperparameters
        training_config = TrainingConfig(
            epochs=2,
            batch_size=256,
            learning_rate=0.1,
            device="cpu",
            log_interval=10,
        )
        trainer = TorchTrainer(training_config)

        # Server URL for communication
        server_url = coordinator.server.url

        async with HTTPClient(server_url=server_url, client_id=client_id) as client:
            while True:
                try:
                    # Check if the server has completed training
                    if await client.check_server_status():
                        break

                    # Fetch the global model from the server
                    model_state, _ = await client.fetch_global_model()
                    model = MNISTModel()
                    model.load_state_dict(model_state) # Load global model parameters
                    model.to(training_config.device)

                    # Perform local training
                    optimizer = torch.optim.SGD(
                        model.parameters(),
                        lr=training_config.learning_rate
                    )
                    metrics = None
                    for epoch in range(training_config.epochs):
                        metrics = trainer.train_epoch(
                            model, train_loader, optimizer, epoch
                        )

                    # Submit the locally trained model and metrics to the server
                    if metrics:
                        success = await client.submit_update(model, metrics)
                        if not success:
                            print(f"Client {client_id}: Update submission failed.")
                            break

                except Exception as e:
                    print(f"Client {client_id} encountered an error: {e}")
                    break

The ``HTTPClient`` is a context manager that facilitates communication with the federated server. Using ``async with HTTPClient(...)`` makes sure that:

- The client session is properly opened and closed.

- Resources like network connections and memory are managed efficiently.

The loop continues until the server signals that the training is complete. The signal is checked using ``await client.check_server_status()``.

The client starts by fetching the current global model's parameters (``model_state``) from the server:

.. code-block:: python

    model_state, _ = await client.fetch_global_model()


The client uses the global model as a starting point for its local training:

.. code-block:: python

    model = MNISTModel()
    model.load_state_dict(model_state)

- A new model instance is created to avoid interference with the previous states.
- ``load_state_dict`` initializes the model with the parameters from the global model.

This model is then trained on the client's dataset:

.. code-block:: python

    for epoch in range(training_config.epochs):
        metrics = trainer.train_epoch(model, train_loader, optimizer, epoch)

- **Metrics** are computed during training and will be sent back to the server along with the updated model.

.. code-block:: python

    success = await client.submit_update(model, metrics)
    if not success:
        break

The server aggregates these updates from multiple clients to update the global model.

Step 8: Running the Federated Experiment
----------------------------------------

Now that the server, coordinator, and client functions are defined, you can run them concurrently to simulate the FL process.

.. code-block:: python

    await asyncio.gather(
        coordinate(coordinator),
        run_client("client_1", coordinator, num_samples=12000),
        run_client("client_2", coordinator, num_samples=8000),
        run_client("client_3", coordinator, num_samples=4000),
    )

After executing the federated learning experiment, NanoFed generates several outputs, organizes them into directories, and provides detailed logs and saved artifacts.

1. ``runs/models/``: Subdirectory for storing global model checkpoints.
    - ``configs/``: Stores metadata and configuration files for each saved model version.
    - ``models/``: Stores serialized PyTorch model files.
2. ``runs/metrics/``: Stores JSON files contianing aggregated metrics for each training round.
3. ``runs/data/``: (Optional) A subdirectory for client-specific datasets or any intermediate data.

Example Metrics Artifact
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: json

    {
        "round_id": 1,
        "start_time": "2024-12-12T05:28:58.396750+00:00",
        "end_time": "2024-12-12T05:28:59.774794+00:00",
        "num_clients": 1,
        "agg_metrics": {
            "loss": 0.25233903527259827,
            "accuracy": 0.9375,
            "samples_processed": 8000.0
        },
        "status": "COMPLETED",
        "client_metrics": [
            {
                "client_id": "client_2",
                "metrics": {
                    "loss": 0.25233903527259827,
                    "accuracy": 0.9375,
                    "samples_processed": 8000
                },
                "weight": 1.0
            }
        ]
    }

Top-Level Fields
""""""""""""""""

1. ``round_id``: Identifier for the training round (i.e., ``1`` for the second round).
2. ``start_time``/``end_time``: ISO 8601 timestamps marking the round's start and end.
3. ``num_clients``: Number of clients that successfully submitted updates (i.e., ``2``).
4. ``agg_metrics``: Weighted aggregation metrics across clients.
5. ``status``: Outcome of the round.

Client-Specific Metrics
"""""""""""""""""""""""

1. ``client_id``: Identifier for the client.
2. ``metrics``: Local metrics reported by the client's local training.
3. ``weight``: Proportional contribution of the client to the global model. In FedAvg, :math:`\text{weight} = \frac{\text{client samples}}{\text{total samples}}`


.. note::

    The field ``num_clients`` shows that only **1 client** contributed to the round. This behavior is determined by the ``min_completion_rate`` configuration, which controls the minimum number of clients required to submit updates for the round to complete successfully. More clients can contribute to a training round.

    As we specified ``min_clients`` to be ``3``, 3 clients must still participate in the training process, but since ``min_completion_rate`` is ``0.5`` in this example,

    .. math::

        \text{required clients}= \text{floor}(\text{min clients} \times \text{min completion rate}) = \text{floor}(3 \times 0.5)=1

    **1** client is required to submit an update.

Conclusion
----------

You have successfully completed a federated learning experiment using NanoFed. This tutorial demonstrated how to:

1. Set up the global model and federated server.
2. Configure the training coordinator and aggregation strategy.
3. Implement and manage federated clients.
4. Run the experiment and analyze the generated results.

Feel free to experiment with different configurations, such as:

- Changing the number of clients and completion rates.
- Extending the ``BaseAggregator`` to implement custom aggregation strategies.
