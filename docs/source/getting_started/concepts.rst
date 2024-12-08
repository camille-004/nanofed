Concepts Guide
==============

Core Architecture
-----------------

NanoFed is built around three main components that work together in an asynchronous environment:

.. md-mermaid::
    :name: NanoFed Architecture
    :class: align-center

    graph TB
        subgraph Client ["Client"]
            D[Local Dataset] --> T[Local Training]
            T --> U[Model Updates]
        end

        subgraph Server ["Server"]
            GM[Global Model] --> A[Aggregation]
            A --> GM
        end

        subgraph Coordinator ["Coordinator"]
            R[Round Management]
            M[Metrics Collection]
            C[Client Tracking]
        end

        U --> A
        GM --> T
        R --> |Controls| A
        A --> |Reports to| M
        T --> |Reports to| C


HTTP Communication Layer
------------------------

NanoFed uses HTTP for client-server communication.

Why HTTP?
~~~~~~~~~

HTTP provides several advances:

* **Stateless protocol**: Each request is independent, so error recovery is simpler
* **Widely supported**: Works everywhere Python runs
* **Firewall-friendly**: Usually allowed through corporate firewalls
* **Good tooling**: Extensive debugging and monitoring tools available

Implementation
~~~~~~~~~~~~~~

On a high level, here's how NanoFed implements HTTP communication:

.. code-block:: python

    class HTTPClient:
        """Asynchronous HTTP client for FL communication."""

        async def fetch_global_model(self) -> tuple[dict[str, torch.Tensor], int]:
            """Fetch current global model from server."""
            async with self._session.get(f"{self._server_url}/model") as response:
                data: GlobalModelResponse = await response.json()
                return self._process_model_response(data)

        async def submit_update(
            self,
            model: ModelProtocol,
            metrics: dict[str, float]
        ) -> bool:
            """Submit local model update to server."""
            update = self._prepare_update(model, metrics)
            async with self._session.post(
                f"{self._server_url}/update",
                json=update
            ) as response:
                return await self._process_update_response(response)

Key API Endpoints:

.. code-block:: text

    GET /model      # Get latest global model
    POST /update    # Submit model Updates
    GET /status     # Check training status

.. md-mermaid::
    :name: HTTP Communication Flow
    :class: align-center

    sequenceDiagram
        participant C as Client
        participant S as Server

        Note over C,S: Training Round Start

        C->>+S: GET /model
        Note right of S: Server checks:<br/>1. Training status<br/>2. Loads current version<br/>3. Returns GlobalModelResponse

        Note over C: Client Process:<br/>1. Converts lists to tensors<br/>2. Updates local model<br/>3. Performs training

        C->>+S: POST /update
        Note left of C: Client sends:<br/>ClientModelUpdateRequest<br/>- Model state<br/>- Training metrics<br/>- Round number

        Note right of S: Server Process:<br/>1. Validate round number<br/>2. Store ServerModelUpdateRequest<br/>3. Returns ModelUpdateResponse

        C->>+S: GET /status
        Note right of S: Server returns:<br/>- Current round<br/>- Updates received<br/>- Training status

Key Data Structures
-------------------

Base Response
~~~~~~~~~~~~~

.. code-block:: python

    class BaseResponse(TypedDict):
        status: Literal["success", "error"]
        message: str
        timestamp: str

Model Update Flow
~~~~~~~~~~~~~~~~~

1. Client -> Server (POST /update)

.. code-block:: python

    class ClientModelUpdateRequest(TypedDict):
        client_id: str
        round_number: int
        model_state: dict[str, list[float] | list[list[float]]]
        metrics: dict[str, float]
        timestamp: str

2. Server Processing

.. code-block:: python

    class ServerModelUpdateRequest(TypedDict, total=False):
        client_id: str
        round_number: int
        model_state: dict[str, list[float] | list[list[float]]]
        metrics: dict[str, float]
        timestamp: str
        status: Literal["success", "error"]
        message: str
        accepted: bool

3. Server -> Client Response

.. code-block:: python

    class ModelUpdateResponse(BaseResponse):
        update_id: str
        accepted: bool

Global Model Flow
~~~~~~~~~~~~~~~~~

Server -> Client (GET /model)

.. code-block:: python

    class GlobalModelResponse(BaseResponse):
        model_state: dict[str, list[float] | list[list[float]]]
        round_number: int
        version_id: str

Asynchronous Programming
------------------------

Federated leraning involves a lot of waiting - waiting for models to download, waiting for clients to train, waiting for updates to be sent back. Traditional synchronous programming would block (pause execution) during these operations, which is inefficient.

In federated learning, we have two main types of operations:

**I/O (Input/Output) Operations:**

- Network communication (sending/receiving models)

- HTTP requests/responses

- Reading/writing model checkpoints

- These operations spend most of their time *waiting*

**CPU-Bound Operations:**

- Local model training

- Gradient computations

- Model parameter aggregation

- These operations spend their time *computing*

.. md-mermaid::
    :name: Sync vs Async Comparison
    :class: align-center

    sequenceDiagram
        participant C1 as Client 1
        participant C2 as Client 2
        participant S as Server

        Note over C1,S: Synchronous Approach (Blocking)
        C1->>+S: Request Model
        Note right of S: Server waits
        S-->>-C1: Send Model
        C2->>+S: Request Model
        Note right of S: Server waits
        S-->>-C2: Send Model

        Note over C1,S: Asynchronous Approach (Non-blocking)
        par Parallel Request
            C1->>S: Request Model
            C2->>S: Request Model
        end
        par Parallel Responses
            S-->>C1: Send Model
            S-->>C2: Send Model
        end

Benefits
~~~~~~~~

1. **Concurrent Client Handling**

.. code-block:: python
    :caption: Server handling multiple clients:
    :emphasize-lines: 4,5

    async def _handle_get_model(self, request: web.Request) -> web.Request:
        """Handle request for global model."""
        try:
            # Can handle multiple clients requesting the model
            # simultaneously without blocking
            version = self._model_manager.current_version
            model_state = self._convert_model_state(version)
            return web.json_response(model_state)
        except Exception as e:
            return web.json_response({"error": str(e)})

2. **Efficient Resource Usage**

.. code-block:: python
    :caption: Client training process
    :emphasize-lines: 5,11

    async def run_training():
        async with HTTPClient(server_url, client_id) as client:
            while True:
                # Fetch model (I/O)
                model_state, round_num = await client.fetch_global_model()

                # CPU-bound local training runs synchronously
                metrics = trainer.train_epoch(model, data)

                # Submit update (I/O operation)
                await client.submit_update(model, metrics)

3. **Scalability**

The server can handle many clients simultaneously because it's not blocked waiting for:

- Model distribution

- Update collection

- Status checks

- Client synchronization

**Synchronous Approach**:

- Each client must wait for others to finish

- Network delays stack up

- Total round time = Sum of all client times

**Asynchronous Approach**:

- Clients operate independently

- Network operations overlap

- Total round time = Slowest client + Network overhead

Implementation Deep Dive
~~~~~~~~~~~~~~~~~~~~~~~~

1. **Async Context Managers**

.. code-block:: python
    :caption: Client session management

    async def __aenter__(self) -> "HTTPClient":
        """Initialize async resources."""
        self._session = aiohttp.ClientSession()
        return self

    async def __aexist__(self, exc_type, exc_val, exc_tb):
        """Clean up async resources."""
        if self._session:
            await self._session.close()

2. **Concurrent Client Updates**

.. code-block:: python
    :caption: Server update handling
    :emphasize-lines: 3

    async def _handle_submit_update(self, request: web.Request):
        """Handle model updates from clients."""
        async with self._lock:  # Protect shared resources
            # Process updates concurrently from multiple clients
            # while maintaining data consistency
            update = await request.json()
            self._updates[update["client_id"]] = update

3. **Round Management**

.. code-block:: python
    :caption: Training round coordination
    :emphasize-lines: 4

    async def wait_for_completion(self, poll_inverval: int = 10):
        """Poll server until training completes."""
        while not self._is_training_done:
            # Non-blocking sleep between status checks
            await asyncio.sleep(poll_interval)
            await self.check_server_status()


The Training Process
--------------------

A training round begins with the server distributing the latest global model to all patricipating clients. Each client trains the model locally on its dataset by processing data in batches over multiple epochs, performing forward and backward passes to update model parameters. Once local training in complete, clients submit their model updates and training metrics, such as accuracy and loss, back to the server. The server aggregates these updates, using algorithms like Federated Averaging (FedAvg), to create an improved global model. This updated model becomes the baseline for the next round, and the process repeats until the desire performance or a specified number of rounds is achieved.

.. md-mermaid::
    :name: Process of a Training Round
    :class: align-center

    sequenceDiagram
    participant S as 🌐 Server
    participant C1 as 🖥️ Client 1
    participant C2 as 🖥️ Client 2

    S->>+C1: Distribute Global Model
    S->>+C2: Distribute Global Model
    C1-->>S: Acknowledge Receipt
    C2-->>S: Acknowledge Receipt

    Note over C1, C2: Clients Perform Local Training

    loop For Each Epoch
        C1->>C1: Process Local Dataset
        C2->>C2: Process Local Dataset
        loop For Each Batch
            C1->>C1: Forward + Backward Pass
            C2->>C2: Forward + Backward Pass
            C1->>C1: Update Model Parameters
            C2->>C2: Update Model Parameters
        end
    end

    C1->>+S: Submit Model Update
    C2->>+S: Submit Model Update

    Note over S: Server Aggregates Updates

    S->>S: Update Global Model
    S->>S: Log Metrics


Round-Based Training
~~~~~~~~~~~~~~~~~~~~

Training happens in rounds, coordinated by the server:

1. **Round Initialization**

.. code-block:: python

    async def train_round(self) -> RoundMetrics:
        self._status = RoundStatus.IN_PROGRESS
        self._server._updates.clear()

        # Wait for minimum required clients
        if not await self._wait_for_clients(self._config.round_timeout):
            raise TimeoutError(f"Round {self._current_round} timed out")

2. **Local Training**

Each client runs independently:

.. code-block:: python
    :emphasize-lines: 11-16

    class TorchTrainer:
        def train_epoch(
            self,
            model: ModelProtocol,
            dataloader: DataLoader,
            optimizer: torch.optim.Optimizer
        ) -> dict[str, float]:
            model.train()
            total_loss = 0.0

            for batch in dataloader:
                optimizer.zero_grad()
                loss = self._train_step(model, batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

3. **Update Aggregation**

The server combines client updates using FedAvg, or any other aggregator:

.. code-block:: python

    def aggregate(self, updates: Sequence[ModelUpdate]) -> AggregationResult:
        weights = self._compute_weights(len(updates))
        state_agg: dict[str, torch.Tensor] = {}

        # Weighted average of parameters
        for update, weight in zip(updates, weights):
            for key, value in update.model_state.items():
                tensor = self._to_tensor(value)
                state_agg[key] += tensor * weight

Model Manager
--------------

The ``ModelManager`` is a component in NanoFed's server architecture that handles versioning, persistence, and distribution of models throughout federated learning. It acts as the source of truth for the global model state and maintains a complete history of model evolution throughout training.

.. md-mermaid::
    :name: Model Management Flow
    :class: align-center

    flowchart TB
        subgraph Server ["Server"]
            direction TB
            MM["ModelManager"] --> |"Loads/Saves"| MS[("Model Storage")]
            AGG["Aggregator"] --> |"Gets current model"| MM
            AGG --> |"Saves aggregated model"| MM
            MM --> |"Provides model"| SRV["HTTP Server"]
        end

        subgraph Clients ["Clients"]
            C1["Client 1"] --> |"GET /model"| SRV
            C2["Client 2"] --> |"GET /model"| SRV
            C3["Client 3"] --> |"GET /model"| SRV

            C1 --> |"POST /update"| AGG
            C2 --> |"POST /update"| AGG
            C3 --> |"POST /update"| AGG
        end

        subgraph Storage ["Storage"]
            MS --- Models["Models Directory (.pt)"]
            MS --- Configs["Configs Directory (.json)"]
        end

The ``ModelManager`` integrates with other server components in several ways:

1. **HTTP Server Integration**

    .. code-block:: python

        server = HTTPServer(
            host="0.0.0.0",
            port=8080,
            model_manager=model_manager,  # Provides models for client requests
            max_request_size=100 * 1024 * 1024,
        )

2. **Aggregator Interaction**

    - After each round of aggregation, the aggregator saves the new global model through the ``ModelManager``
    - The ModelManager assigns a new version ID and persists both model state and metadata
    - This new version becomes available for the next round of training

Version Control
~~~~~~~~~~~~~~~

NanoFed tracks model versions using a dedicated manager:

.. code-block:: python

    @dataclass(frozen=True)
    class ModelVersion:
        version_id: str
        timestamp: datetime
        config: dict[str, Any]
        path: Path

Aggregation Strategies
----------------------

A key component in federated learning is the aggregation strategy - how to combine model updates from multiple clients into a single improved global model.

.. md-mermaid::
    :name: Aggregation Flow
    :class: align-center

    flowchart TB
        subgraph Clients
            C1[Client 1 Update] --> A
            C2[Client 2 Update] --> A
            C3[Client 3 Update] --> A
        end

        subgraph Server
            A[Aggregator] --> GM[Global Model]
            GM --> |Next Round| Clients
        end

FedAvg: The Default Aggregator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

NanoFed implements Federated Averaging (FedAvg) as its default aggregation strategy. Given :math:`K` clients, each with local model parameters :math:`w_k` and dataset size :math:`n_k`, the global model parameters :math:`w` are computed as:

.. math::

    w = \sum_{k=1}^K \frac{n_k}{n} w_k

where :math:`n = \sum_{k=1}^K n_k` is the total number of samples across all clients.

Key Steps
~~~~~~~~~

1. **Weight Computation**

    For each client :math:`k`, its weight :math:`\alpha_k` is computed as:

    .. math::
        \alpha_k = \frac{n_k}{\sum_{i=1}^K n_i}

    These weights ensure that:
        - :math:`\sum_{k=1}^K \alpha_k = 1`
        - Clients with more data have proportionally more influence
        - The aggregation is unbiased

2. **Parameter Aggregation**

    For each layer :math:`l` in the neural network:

    .. math::

        w_l = \sum_{k=1}^K \alpha_k w_{k,l}

    where :math:`w_{k,l}` are the parameters of layer :math:`l` from client :math:`k`.

3. **Metrics Aggregation**

    For metrics like accuracy :math:`a_k` from each client, the weighted average is:

    .. math::

        a_{global} = \sum_{k=1}^K \alpha_k a_k

Custom Aggregation Strategies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To implement a custom strategy, extend the base aggregator:

.. code-block:: python

    class BaseAggregator(ABC, Generic[T]):
        """Base class for aggregation strategies."""

        @abstractmethod
        def aggregate(
            self, model: T, updates: Sequence[ModelUpdate]
        ) -> AggregationResult[T]:
            """Aggregate model updates."""
            pass
