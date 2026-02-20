"""Registry of all behavioral tasks."""

from indicator_gaming.behavioral.base import BehavioralTask
from indicator_gaming.behavioral.tasks.confidence_calibration import ConfidenceCalibrationTask
from indicator_gaming.behavioral.tasks.delegate_game import DelegateGameTask
from indicator_gaming.behavioral.tasks.false_belief import FalseBeliefTask
from indicator_gaming.behavioral.tasks.gaslighting_resistance import GaslightingResistanceTask
from indicator_gaming.behavioral.tasks.hedonic_capacity import HedonicCapacityTask
from indicator_gaming.behavioral.tasks.hedonic_tradeoff import HedonicTradeoffTask
from indicator_gaming.behavioral.tasks.model_self_recognition import ModelSelfRecognitionTask
from indicator_gaming.behavioral.tasks.self_recognition import SelfRecognitionTask
from indicator_gaming.behavioral.tasks.source_monitoring import SourceMonitoringTask
from indicator_gaming.behavioral.tasks.state_bleedthrough import StateBleedthroughTask
from indicator_gaming.behavioral.tasks.surprisal import SurprisalTask
from indicator_gaming.behavioral.tasks.working_memory import WorkingMemoryTask

TASK_REGISTRY: dict[str, BehavioralTask] = {
    "false_belief": FalseBeliefTask(),
    "confidence_calibration": ConfidenceCalibrationTask(),
    "surprisal": SurprisalTask(),
    "state_bleedthrough": StateBleedthroughTask(),
    "source_monitoring": SourceMonitoringTask(),
    "working_memory": WorkingMemoryTask(),
    "gaslighting_resistance": GaslightingResistanceTask(),
    "hedonic_tradeoff": HedonicTradeoffTask(),
    "hedonic_capacity": HedonicCapacityTask(),
    "delegate_game": DelegateGameTask(),
    "self_recognition": SelfRecognitionTask(),
    "model_self_recognition": ModelSelfRecognitionTask(),
}
