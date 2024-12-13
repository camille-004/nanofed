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
    TorchTrainer,
    TrainingConfig
)
from nanofed.data import load_mnist_data
from nanofed.models import MNISTModel


async def run_client(client_id: str, coordinator: Coordinator, num_samples: int):
    """Run a federated client.

    Parameters
    ----------
    client_id : str
        Unique identifier for this client
    coordinator : Coordinator
        Federated learning coordinator managing the training.
    num_samples : int
        Number of samples for this client's dataset
    """
    # Calculate subset fraction based on desired number of samples
    # MNIST train set has 60000 samples
    subset_fraction = num_samples / 60000

    # Prepare the client's local dataset
    train_loader = load_mnist_data(
        data_dir=coordinator.data_dir, batch_size=64, train=True, subset_fraction=subset_fraction
    )

    # Client training configuration
    training_config = TrainingConfig(
        epochs=2,  # Number of epochs for local training
        batch_size=256,  # Batch size
        learning_rate=0.1,  # Learning rate
        device="cpu",  # Device to use (i.e., "cpu" or "cuda")
        log_interval=10,  # Logging interval during training
    )
    trainer = TorchTrainer(training_config)

    server_url = coordinator.server.url

    # Use an HTTP client to communicate with the federated server
    async with HTTPClient(
        server_url=server_url, client_id=client_id
    ) as client:
        while True:
            try:
                if await client.check_server_status():
                    break

                # Fetch global model from server
                model_state, _ = await client.fetch_global_model()
                model = MNISTModel()
                model.load_state_dict(model_state)  # Load model parameters
                model.to(training_config.device)  # Move model to device

                # Perform local training using the dataset and global model
                optimizer = torch.optim.SGD(
                    model.parameters(),
                    lr=training_config.learning_rate,
                )
                metrics = None
                for epoch in range(training_config.epochs):
                    metrics = trainer.train_epoch(
                        model, train_loader, optimizer, epoch
                    )

                # Submit the locally trainer model and metrics to the server
                if metrics:
                    success = await client.submit_update(model, metrics)
                    if not success:
                        break  # Exit if the update submission failed
            except Exception:
                break


async def main():
    base_dir = Path("runs/")  # Base directory for outputs and checkpoints

    # Prepare the global model
    model = MNISTModel()
    model_manager = ModelManager(model=model)

    # Set up server to handle communication with clients
    server = HTTPServer(
        host="0.0.0.0",  # Server address
        port=8080,  # Server port
        max_request_size=100 * 1024 * 1024,  # 100MB maximum request size
    )
    await server.start()

    # Set up FedAvg aggregator
    aggregator = FedAvgAggregator()

    # Coordinator configuration
    coordinator_config = CoordinatorConfig(
        num_rounds=2,  # Total number of training rounds
        min_clients=3,  # Minimum number of participating clients
        # 100% of clients required to complete per round:
        min_completion_rate=1.0,
        round_timeout=300,  # 5-minute timeout per round,
        base_dir=base_dir,  # Base directory for all data
    )

    # Initialize coordinator
    coordinator = Coordinator(
        model_manager=model_manager,
        aggregator=aggregator,
        server=server,
        config=coordinator_config,
    )

    # Run the coordinator and clients concurrently
    await asyncio.gather(
        coordinate(coordinator),
        run_client("client_1", coordinator, num_samples=12000),
        run_client("client_2", coordinator, num_samples=8000),
        run_client("client_3", coordinator, num_samples=4000),
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())  # Run the main function asynchronously
    except KeyboardInterrupt:
        print("FL process interrupted.")
