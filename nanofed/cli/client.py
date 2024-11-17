import asyncio
from pathlib import Path
from typing import Any

import aiohttp
import click
import torch
import yaml

from nanofed.communication.http.client import HTTPClient
from nanofed.data.mnist import load_mnist_data
from nanofed.models.mnist import MNISTModel
from nanofed.trainer.base import TrainingConfig
from nanofed.trainer.torch import TorchTrainer
from nanofed.utils.logger import Logger


async def run_client(
    config_path: Path, client_id: str, data_path: Path
) -> None:
    """Run client."""
    logger = Logger()

    with logger.context("client", client_id):
        try:
            with open(config_path) as f:
                config: dict[str, Any] = yaml.safe_load(f)

            train_loader = load_mnist_data(
                data_path,
                batch_size=config["training"]["batch_size"],
                train=True,
            )

            training_config = TrainingConfig(
                epochs=config["training"]["local_epochs"],
                batch_size=config["training"]["batch_size"],
                learning_rate=config["training"]["learning_rate"],
                device=config["training"].get("device", "cpu"),
                log_interval=config["training"].get("log_interval", 10),
            )

            trainer = TorchTrainer(training_config)

            async with HTTPClient(
                server_url=config["server"]["url"],
                client_id=client_id,
                timeout=config["client"].get("timeout", 300),
            ) as client:
                while True:
                    try:
                        async with aiohttp.ClientSession() as session:
                            async with session.get(
                                f"{config['server']['url']}/status"
                            ) as response:
                                if response.status != 200:
                                    logger.info(
                                        "Server is unreachable. Exiting client."  # noqa
                                    )
                                    break

                        # Fetch global model
                        (
                            model_params,
                            round_num,
                        ) = await client.fetch_global_model()

                        model = MNISTModel()
                        model.load_state_dict(model_params)
                        model.to(training_config.device)

                        # Train locally
                        optimizer = torch.optim.SGD(
                            model.parameters(),
                            lr=training_config.learning_rate,
                        )

                        metrics = None
                        for epoch in range(training_config.epochs):
                            metrics = trainer.train_epoch(
                                model, train_loader, optimizer, epoch
                            )
                            logger.info(
                                f"Epoch {epoch} completed: "
                                f"Loss={metrics.loss:.4f}, Accuracy={metrics.accuracy:.4f}"  # noqa
                            )

                        # Submit final metrics for last epoch
                        success = await client.submit_update(
                            model,
                            {
                                "loss": metrics.loss,
                                "accuracy": metrics.accuracy,
                                "client_id": client_id,
                                "samples_processed": metrics.samples_processed,
                            },
                        )

                        if success:
                            logger.info(
                                f"Round {round_num} completed: "
                                f"Loss={metrics.loss:.4f}, "
                                f"Accuracy={metrics.accuracy:.4f}"
                            )
                        else:
                            logger.error("Failed to submit update")

                    except aiohttp.ClientError:
                        logger.info("Server is unreachable. Exiting client.")
                        break
                    except Exception as e:
                        logger.error(f"Error in training round: {str(e)}")
                        await asyncio.sleep(5)

        except Exception as e:
            logger.error(f"Client error: {str(e)}")
            import traceback

            traceback.print_exc()


@click.command()
@click.option(
    "--config",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to client configuration file",
)
@click.option("--client_id", required=True, help="Client ID")
@click.option(
    "--data_path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to data directory",
)
def main(config: Path, client_id: str, data_path: Path) -> None:
    asyncio.run(run_client(config, client_id, data_path))


if __name__ == "__main__":
    main()
