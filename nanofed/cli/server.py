import asyncio
import sys
from pathlib import Path
from typing import Any

import click
import yaml

from nanofed.communication.http.server import HTTPServer
from nanofed.core.types import ModelConfig
from nanofed.models.mnist import MNISTModel
from nanofed.orchestration.coordinator import (
    Coordinator,
    CoordinatorConfig,
)
from nanofed.server.aggregator.fedavg import FedAvgAggregator
from nanofed.server.model_manager.manager import ModelManager
from nanofed.utils.logger import Logger


async def run_server(config_path: Path) -> None:
    """Run federated learning server."""
    logger = Logger()

    with logger.context("server"):
        try:
            logger.info(f"Loading config from: {config_path}")
            with open(config_path) as f:
                config_text = f.read()
                logger.info(f"Config content:\n{config_text}")
                config: dict[str, Any] = yaml.safe_load(config_text)

            min_clients = config["training"]["min_clients"]
            logger.info(f"Expecting {min_clients} clients")

            if min_clients < 1:
                raise ValueError("min_clients must be at least 1")

            base_dir = Path(config["paths"]["base_dir"])
            model_dir = base_dir / "models"
            checkpoint_dir = base_dir / "checkpoints"
            metrics_dir = base_dir / "metrics"

            for directory in [model_dir, checkpoint_dir, metrics_dir]:
                directory.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created directory: {directory}")

            # Initialize components
            logger.info("Initializing model...")
            model = MNISTModel()
            model_manager = ModelManager(model_dir, model)

            if not list(model_dir.glob("*.pt")):
                logger.info("Saving initial model...")
                initial_config = ModelConfig(
                    name="mnist", version="1.0", architecture={"type": "cnn"}
                )
                model_manager.save_model(initial_config)

            aggregator = FedAvgAggregator()
            server = HTTPServer(
                host=config["server"]["host"],
                port=config["server"]["port"],
                model_manager=model_manager,
                max_request_size=config["server"].get(
                    "max_request_size", 100 * 1024 * 1024
                ),
            )

            coordinator_config = CoordinatorConfig(
                num_rounds=config["training"]["num_rounds"],
                min_clients=min_clients,
                min_completion_rate=config["training"]["min_completion_rate"],
                round_timeout=config["training"]["round_timeout"],
                checkpoint_dir=checkpoint_dir,
                metrics_dir=metrics_dir,
            )

            coordinator = Coordinator(
                model, aggregator, server, coordinator_config
            )

            await server.start()
            logger.info(
                f"Server listening at http://{config['server']['host']}:{config['server']['port']}"
            )

            try:
                logger.info("Starting training...")
                async for metrics in coordinator.start_training():
                    logger.info(
                        f"Round {metrics.round_id} completed: "
                        f"Loss={metrics.agg_metrics['loss']:.4f}, "
                        f"Accuracy={metrics.agg_metrics['accuracy']:.4f}"
                    )
            except KeyboardInterrupt:
                logger.info("Server shutting down...")
            finally:
                logger.info("Stopping server...")
                await server.stop()

        except Exception as e:
            logger.error(f"Server error: {str(e)}")
            import traceback

            logger.error("Traceback:")
            for line in traceback.format_exc().splitlines():
                logger.error(f"  {line}")
            raise


@click.command()
@click.argument(
    "config",
    type=click.Path(exists=True, path_type=Path),
)
def main(config: Path) -> None:
    """Run federated learning server."""
    try:
        asyncio.run(run_server(config))
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
