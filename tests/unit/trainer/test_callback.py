import json
from pathlib import Path
from unittest import mock

from nanofed.trainer.base import TrainingMetrics
from nanofed.trainer.callback import MetricsLogger


@mock.patch("nanofed.trainer.callback.open", new_callable=mock.mock_open)
def test_metrics_logger_epoch_end(mock_open):
    log_dir = Path("/tmp/logs")
    logger = MetricsLogger(log_dir=log_dir, experiment_name="test_experiment")

    metrics = TrainingMetrics(
        loss=0.5, accuracy=0.9, epoch=1, batch=0, samples_processed=100
    )
    logger.on_epoch_end(epoch=1, metrics=metrics)

    mock_open.assert_called_once_with(mock.ANY, "w")
    log_file_handle = mock_open()
    written_data = "".join(
        call.args[0] for call in log_file_handle.write.call_args_list
    )

    parsed_data = json.loads(written_data)
    assert parsed_data[-1]["type"] == "epoch"
    assert parsed_data[-1]["epoch"] == 1
    assert parsed_data[-1]["loss"] == 0.5
    assert parsed_data[-1]["accuracy"] == 0.9


def test_metrics_logger_batch_end():
    log_dir = Path("/tmp/logs")
    logger = MetricsLogger(log_dir=log_dir, experiment_name="test_experiment")

    metrics = TrainingMetrics(
        loss=0.2, accuracy=0.95, epoch=1, batch=10, samples_processed=40
    )

    logger.on_batch_end(batch=10, metrics=metrics)

    assert len(logger._metrics) == 1
    assert logger._metrics[-1]["type"] == "batch"
    assert logger._metrics[-1]["batch"] == 10
    assert logger._metrics[-1]["loss"] == 0.2
    assert logger._metrics[-1]["accuracy"] == 0.95
