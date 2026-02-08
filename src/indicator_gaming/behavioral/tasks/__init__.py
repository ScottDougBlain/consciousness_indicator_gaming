"""Registry of all behavioral tasks."""

from indicator_gaming.behavioral.base import BehavioralTask
from indicator_gaming.behavioral.tasks.confidence_calibration import ConfidenceCalibrationTask
from indicator_gaming.behavioral.tasks.false_belief import FalseBeliefTask
from indicator_gaming.behavioral.tasks.state_bleedthrough import StateBleedthroughTask
from indicator_gaming.behavioral.tasks.surprisal import SurprisalTask

TASK_REGISTRY: dict[str, BehavioralTask] = {
    "false_belief": FalseBeliefTask(),
    "confidence_calibration": ConfidenceCalibrationTask(),
    "surprisal": SurprisalTask(),
    "state_bleedthrough": StateBleedthroughTask(),
}
