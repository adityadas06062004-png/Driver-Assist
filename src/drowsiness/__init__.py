"""Driver Assist: driver drowsiness detection core (no UI dependencies)."""

from .alerts import AlertEvent, AlertStateMachine, Level, SoundPlayer
from .calibration import Calibration, Calibrator
from .camera import Camera
from .config import Config
from .engine import DrowsinessEngine, FrameResult
from .health import Health, HealthMonitor, Issue
from .monitor import Monitor, MonitorState
from .perf import FrameGovernor
from .quality import FrameStats
from .risk import RiskScorer, RiskState

__all__ = [
    "AlertEvent", "AlertStateMachine", "Level", "SoundPlayer",
    "Calibration", "Calibrator", "Camera", "Config", "DrowsinessEngine", "FrameResult",
    "FrameGovernor", "FrameStats", "Health", "HealthMonitor", "Issue",
    "Monitor", "MonitorState", "RiskScorer", "RiskState",
]
__version__ = "0.3.0"
