from nanofed.orchestration import Coordinator
from nanofed.utils import Logger


async def run_coordinator(coordinator: Coordinator) -> None:
    """Run the coordinator and consume training metrics.

    This function continuously starts training rounds and processes
    the generated metrics.

    Parameters
    ----------
    coordinator : Coordinator
        The federated coordinator instance
    """
    logger = Logger()
    with logger.context("coordinator.run"):
        try:
            async for _ in coordinator.start_training():
                pass
        except Exception as e:
            logger.error(f"Error while running coordinator: {str(e)}")
            raise
        finally:
            logger.info("Coordinator run completed.")
