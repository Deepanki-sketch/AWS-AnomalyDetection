"""
WMO-No. 8 Standard Quality Control (QC) Engine
Implements deterministic plausibility, step-test, persistence, and physical consistency filters.
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import numpy as np
from src.physics import AtmosphericPhysics


@dataclass
class QCLimits:
    """Configurable thresholds conforming to WMO-No. 8 Quality Control guidelines."""
    # Physical plausibility limits
    temp_min: float = -40.0      # deg C
    temp_max: float = 60.0       # deg C
    pressure_min: float = 800.0  # hPa (covers stations up to ~2000m AMSL)
    pressure_max: float = 1085.0 # hPa (highest recorded sea level pressure ~1084.8 hPa)
    rh_min: float = 0.0          # %
    rh_max: float = 100.0        # %

    # Maximum permissible step changes (per minute equivalent)
    max_delta_temp_per_min: float = 2.0     # deg C/min
    max_delta_pressure_per_min: float = 1.5 # hPa/min
    max_delta_rh_per_min: float = 12.0      # %/min

    # Persistence / Flatline criteria
    min_stdev_window_len: int = 5           # Window length for variance check
    min_identical_run: int = 4              # Number of consecutive identical samples for rapid flatline check
    identical_eps: float = 1e-4             # Max range treated as bit-identical
    min_temp_stdev: float = 0.005           # deg C (realistic limit, catches deadbands with minor noise)
    min_pressure_stdev: float = 0.005       # hPa
    min_rh_stdev: float = 0.015             # %


@dataclass
class QCResult:
    """Results of deterministic quality control check on a single observation."""
    is_valid: bool
    is_anomaly: bool
    anomaly_type: str  # 'NORMAL', 'OUT_OF_BOUNDS', 'SPIKE', 'STUCK_SENSOR', 'PHYSICAL_INCONSISTENCY', 'MISSING'
    parameter_failed: Optional[str] = None
    severity: float = 0.0  # 0.0 (clean) to 1.0 (critical error)
    reason: str = "Passed all standard QC tests"
    metrics: Dict[str, Any] = field(default_factory=dict)


class QualityControlEngine:
    """
    Quality Control pipeline evaluating AWS sensor data against
    WMO meteorological bounds, rate-of-change dynamics, and persistence rules.
    """

    def __init__(self, limits: Optional[QCLimits] = None):
        self.limits = limits or QCLimits()

    def check_plausibility(self, temp: Optional[float], 
                          pressure: Optional[float], 
                          rh: Optional[float]) -> Tuple[bool, str, str, float]:
        """
        Verify that values fall within physically possible environmental bounds.
        Returns: (passed, failed_param, reason, severity)
        """
        if temp is None or np.isnan(temp):
            return False, "temperature", "Missing or NaN temperature observation", 1.0
        if pressure is None or np.isnan(pressure):
            return False, "pressure", "Missing or NaN pressure observation", 1.0
        if rh is None or np.isnan(rh):
            return False, "humidity", "Missing or NaN humidity observation", 1.0

        # Temperature range check
        if temp < self.limits.temp_min or temp > self.limits.temp_max:
            return False, "temperature", (
                f"Temperature {temp:.1f}°C outside physical bounds "
                f"[{self.limits.temp_min}, {self.limits.temp_max}]°C"
            ), 0.95

        # Pressure range check
        if pressure < self.limits.pressure_min or pressure > self.limits.pressure_max:
            return False, "pressure", (
                f"Pressure {pressure:.1f} hPa outside physical bounds "
                f"[{self.limits.pressure_min}, {self.limits.pressure_max}] hPa"
            ), 0.95

        # Humidity range check
        if rh < self.limits.rh_min or rh > self.limits.rh_max:
            return False, "humidity", (
                f"Relative Humidity {rh:.1f}% outside physical limits "
                f"[{self.limits.rh_min}, {self.limits.rh_max}]%"
            ), 0.90

        return True, "", "Plausibility check passed", 0.0

    def check_step_change(self, 
                          curr_temp: float, curr_pressure: float, curr_rh: float,
                          prev_temp: float, prev_pressure: float, prev_rh: float,
                          delta_minutes: float = 1.0) -> Tuple[bool, str, str, float]:
        """
        Check for impossible instant rate-of-change spikes across consecutive observations.
        Returns: (passed, failed_param, reason, severity)
        """
        dt_mins = max(1.0, delta_minutes)
        d_temp = abs(curr_temp - prev_temp) / dt_mins
        d_press = abs(curr_pressure - prev_pressure) / dt_mins
        d_rh = abs(curr_rh - prev_rh) / dt_mins

        if d_temp > self.limits.max_delta_temp_per_min:
            severity = min(1.0, 0.5 + (d_temp / self.limits.max_delta_temp_per_min) * 0.25)
            return False, "temperature", (
                f"Impossible Temperature rate of change: {abs(curr_temp - prev_temp):.2f}°C "
                f"in {delta_minutes:.1f} min (exceeds limit {self.limits.max_delta_temp_per_min * dt_mins:.1f}°C)"
            ), severity

        if d_press > self.limits.max_delta_pressure_per_min:
            severity = min(1.0, 0.5 + (d_press / self.limits.max_delta_pressure_per_min) * 0.25)
            return False, "pressure", (
                f"Impossible Pressure rate of change: {abs(curr_pressure - prev_pressure):.2f} hPa "
                f"in {delta_minutes:.1f} min (exceeds limit {self.limits.max_delta_pressure_per_min * dt_mins:.1f} hPa)"
            ), severity

        if d_rh > self.limits.max_delta_rh_per_min:
            severity = min(1.0, 0.5 + (d_rh / self.limits.max_delta_rh_per_min) * 0.25)
            return False, "humidity", (
                f"Impossible Relative Humidity rate of change: {abs(curr_rh - prev_rh):.1f}% "
                f"in {delta_minutes:.1f} min (exceeds limit {self.limits.max_delta_rh_per_min * dt_mins:.1f}%)"
            ), severity

        return True, "", "Rate of change within allowable limits", 0.0

    def check_persistence(self, 
                          temp_history: List[float], 
                          press_history: List[float], 
                          rh_history: List[float],
                          curr_temp: Optional[float] = None,
                          curr_pressure: Optional[float] = None,
                          curr_rh: Optional[float] = None) -> Tuple[bool, str, str, float]:
        """
        Check for frozen / stuck sensor reading (zero variance or deadband over sliding window).
        True atmospheric signals always exhibit microscopic turbulence and fluctuations.
        Includes both rapid consecutive-identical detection (N>=4) and low-variance deadband tests.
        """
        t_seq = temp_history + ([curr_temp] if curr_temp is not None else [])
        p_seq = press_history + ([curr_pressure] if curr_pressure is not None else [])
        rh_seq = rh_history + ([curr_rh] if curr_rh is not None else [])

        run_len = self.limits.min_identical_run
        window_len = self.limits.min_stdev_window_len

        sensors = [
            ("temperature", t_seq, self.limits.min_temp_stdev, "°C"),
            ("pressure", p_seq, self.limits.min_pressure_stdev, "hPa"),
            ("humidity", rh_seq, self.limits.min_rh_stdev, "%")
        ]

        for s_name, seq, min_std, unit in sensors:
            # 1. Rapid Flatline: consecutive identical / near-identical values (N >= 4)
            if len(seq) >= run_len:
                w_run = seq[-run_len:]
                if (max(w_run) - min(w_run)) < self.limits.identical_eps:
                    return False, s_name, (
                        f"{s_name.capitalize()} sensor flatline/frozen: {run_len} consecutive identical "
                        f"readings (stuck at {w_run[-1]:.2f}{unit})"
                    ), 0.90

            # 2. Low-variance deadband check over sliding window (N >= 5)
            if len(seq) >= window_len:
                w_var = seq[-window_len:]
                s_std = float(np.std(w_var))
                s_amp = max(w_var) - min(w_var)
                if s_std < min_std and s_amp < (min_std * 3.0):
                    return False, s_name, (
                        f"{s_name.capitalize()} sensor frozen deadband: std={s_std:.5f}{unit} "
                        f"over {window_len} readings (stuck at {w_var[-1]:.2f}{unit})"
                    ), 0.85

        return True, "", "Sensors exhibit expected natural variation", 0.0

    @staticmethod
    def _is_missing(value: Optional[float]) -> bool:
        return value is None or (isinstance(value, (float, np.floating, int)) and np.isnan(float(value)))

    def _window_is_deadband(self, window: List[float], min_std: float) -> bool:
        if any(self._is_missing(v) for v in window):
            return False
        s_std = float(np.std(window))
        s_amp = float(max(window) - min(window))
        return s_std < min_std and s_amp < (min_std * 3.0)

    def frozen_run_start(self, values: List[float], end_idx: int, min_std: float) -> int:
        """
        Find where a frozen run that is visible at `end_idx` actually began.

        Used only for offline/batch backfill. Live streaming still flags from the
        first sample that meets the persistence threshold (4 identical or 5 deadband).
        """
        if end_idx < 0 or end_idx >= len(values) or self._is_missing(values[end_idx]):
            return end_idx

        v_end = float(values[end_idx])
        run_len = self.limits.min_identical_run
        window_len = self.limits.min_stdev_window_len
        eps = self.limits.identical_eps

        start = end_idx
        while start > 0 and not self._is_missing(values[start - 1]):
            if abs(float(values[start - 1]) - v_end) >= eps:
                break
            start -= 1
        if (end_idx - start + 1) >= run_len:
            return start

        if (end_idx + 1) < window_len:
            return end_idx

        w0 = end_idx - window_len + 1
        triggering = [float(v) for v in values[w0:end_idx + 1]]
        if not self._window_is_deadband(triggering, min_std):
            return end_idx

        start = w0
        while start > 0 and not self._is_missing(values[start - 1]):
            candidate = [float(v) for v in values[start - 1:end_idx + 1]]
            if not self._window_is_deadband(candidate, min_std):
                break
            start -= 1
        return start

    def check_physical_consistency(self, temp: float, rh: float) -> Tuple[bool, str, str, float]:
        """
        Check thermodynamic consistency:
        - Dew point cannot exceed dry bulb temperature (T_d <= T + 0.1 deg C)
        - Ambient air cannot sustain extreme supersaturation (RH > 102%)
        - Absolute zero RH (0.00%) sustained in open atmosphere typically indicates circuit disconnect
        """
        if rh > 100.5:
            return False, "humidity", f"Relative Humidity supersaturation ({rh:.1f}%) exceeds physical limit", 0.80

        dew_point = AtmosphericPhysics.dew_point(temp, rh)
        if dew_point > temp + 0.2:
            return False, "multivariate", (
                f"Thermodynamic violation: Dew point ({dew_point:.2f}°C) exceeds "
                f"dry bulb temperature ({temp:.2f}°C)"
            ), 0.85

        return True, "", "Thermodynamic consistency verified", 0.0

    def evaluate_observation(self,
                             curr_temp: float,
                             curr_pressure: float,
                             curr_rh: float,
                             prev_temp: Optional[float] = None,
                             prev_pressure: Optional[float] = None,
                             prev_rh: Optional[float] = None,
                             temp_hist: Optional[List[float]] = None,
                             press_hist: Optional[List[float]] = None,
                             rh_hist: Optional[List[float]] = None,
                             delta_minutes: float = 1.0) -> QCResult:
        """
        Executes all deterministic Quality Control checks in order of precedence:
        1. Range / Plausibility (Out of bounds or Missing)
        2. Physical Consistency
        3. Step Change (Spikes)
        4. Persistence (Stuck Sensor)
        """
        # 1. Plausibility Check
        ok, param, reason, sev = self.check_plausibility(curr_temp, curr_pressure, curr_rh)
        if not ok:
            atype = "MISSING" if ("Missing" in reason or "NaN" in reason) else "OUT_OF_BOUNDS"
            return QCResult(is_valid=False, is_anomaly=True, anomaly_type=atype,
                            parameter_failed=param, severity=sev, reason=reason)

        # 2. Physical Consistency
        ok, param, reason, sev = self.check_physical_consistency(curr_temp, curr_rh)
        if not ok:
            return QCResult(is_valid=False, is_anomaly=True, anomaly_type="PHYSICAL_INCONSISTENCY",
                            parameter_failed=param, severity=sev, reason=reason)

        # 3. Step Change Check (Spikes)
        if prev_temp is not None and prev_pressure is not None and prev_rh is not None:
            ok, param, reason, sev = self.check_step_change(
                curr_temp, curr_pressure, curr_rh,
                prev_temp, prev_pressure, prev_rh,
                delta_minutes
            )
            if not ok:
                return QCResult(is_valid=False, is_anomaly=True, anomaly_type="SPIKE",
                                parameter_failed=param, severity=sev, reason=reason)

        # 4. Persistence Check (Stuck / Frozen Sensor)
        if temp_hist is not None and press_hist is not None and rh_hist is not None:
            ok, param, reason, sev = self.check_persistence(
                temp_hist, press_hist, rh_hist,
                curr_temp=curr_temp, curr_pressure=curr_pressure, curr_rh=curr_rh
            )
            if not ok:
                return QCResult(is_valid=False, is_anomaly=True, anomaly_type="STUCK_SENSOR",
                                parameter_failed=param, severity=sev, reason=reason)

        return QCResult(is_valid=True, is_anomaly=False, anomaly_type="NORMAL",
                        severity=0.0, reason="Passed all WMO standard quality checks")
