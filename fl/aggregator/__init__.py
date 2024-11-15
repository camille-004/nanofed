from enum import Enum

from fl.aggregator.fed_avg import FedAvgAggregator
from fl.aggregator.fed_prox import FedProxAggregator
from fl.aggregator.personalized import PersonalizedFedAvgAggregator
from fl.core.exceptions import ValidationError
from fl.core.protocols import ModelAggregator


class AggregatorType(Enum):
    FED_AVG = "FedAvg"
    FED_PROX = "FedProx"
    PERSONALIZED_FED_AVG = "PersonalizedFedAvg"

    @staticmethod
    def from_string(name: str) -> "AggregatorType":
        name = name.replace("-", "_").replace(" ", "_").upper()
        return AggregatorType[name]


def get_aggregator(agg_type: AggregatorType) -> ModelAggregator:
    """Factory method to get desired aggregator."""
    match agg_type:
        case AggregatorType.FED_AVG:
            return FedAvgAggregator()
        case AggregatorType.FED_PROX:
            return FedProxAggregator()
        case AggregatorType.PERSONALIZED_FED_AVG:
            return PersonalizedFedAvgAggregator()
        case _:
            raise ValidationError(f"Unsupported aggregator type: {agg_type}")
