"""
r_skef_core/config.py -- RECONSTRUCTED RskefConfig (not the original file).

Evidence used: detector.py, density.py, metrics.py, rskef_comprehensive_optimization.py.
NOT seen: calibration.py, features.py, gmc.py (all receive cfg). Fields they read
may be missing here.

Default classes:
  [B] value set by the baseline systems (Config A/B) in the optimization script
  [C] inferred from usage or function-signature defaults; not historically verified
  [D] unknown -> UNVERIFIED sentinel. Using it raises, naming the field.
A [B]/[C] default is NOT proof that the original class had that default.
"""
from dataclasses import dataclass, field, fields


class _Unverified:
    """Default that could not be recovered. Fails loudly if the value is used."""
    def __init__(self, name: str):
        self._name = name

    def __repr__(self):
        return "UNVERIFIED"

    def _fail(self, *args, **kwargs):
        raise ValueError(
            f"RskefConfig.{self._name} is UNVERIFIED (original default not recovered). "
            "Set it explicitly or restore it in config.py."
        )

    __bool__ = __int__ = __float__ = __index__ = _fail
    __lt__ = __le__ = __gt__ = __ge__ = _fail
    __add__ = __radd__ = __sub__ = __rsub__ = _fail
    __mul__ = __rmul__ = __truediv__ = __rtruediv__ = __pow__ = _fail
    __neg__ = __abs__ = _fail


def _unverified(name: str):
    return field(default_factory=lambda: _Unverified(name))


@dataclass  # must stay non-frozen, non-slots: script uses copy.copy, attribute assignment, vars(cfg)
class RskefConfig:
    # ---- Video / input -------------------------------------------------------------
    fps: float = 30.0                      # [C] fallback when the video reports invalid FPS (metrics.py defaults to 30.0)
    resize_width: int = _unverified("resize_width")    # [D] frame width after cv2.resize
    resize_height: int = _unverified("resize_height")  # [D] frame height after cv2.resize

    # ---- Mass / energy representation ----------------------------------------------
    # "area_mass" | "csrnet_global" | "csrnet_active_mass"
    mass_representation: str = "area_mass"  # [B] Config A value
    csrnet_crowd_floor: float = 4.0         # [C] density.py signature default; used only by csrnet_active_mass. [optimization-era]

    # ---- Calibration ---------------------------------------------------------------
    # Generic detector/calibrator strategies. The UMN validation adapter sets
    # "adapter_fixed_90" as a metadata label and does not invoke BaselineCalibrator.
    calibration_strategy: str = "fixed_150"  # Reconstructed default; not a historical parameter
    calibration_frames: int = 150            # Reconstructed generic default; adapter uses its own 90-observation buffer
    min_sigma_floor: float = _unverified("min_sigma_floor")            # [D] floor on MAD-derived sigma
    min_threshold_floor: float = _unverified("min_threshold_floor")    # [D] floor on t_on; also in jerk denominator

    # ---- Global motion compensation (GMC) ------------------------------------------
    enable_gmc: bool = False                 # [B] baseline = off; True in systems 3-5 [optimization-era]
    gmc_max_features: int = _unverified("gmc_max_features")            # [D] systems 3-5 use 200 (not a baseline value)
    gmc_ransac_threshold: float = _unverified("gmc_ransac_threshold")  # [D] systems 3-5 use 3.0 (not a baseline value)

    # ---- Motion features (consumed by features.py, not seen) -----------------------
    beta_div: float = 0.40                   # [B] baseline value; systems 3-5 use 0.30
    div_scale: float = _unverified("div_scale")                        # [D] never set in the current script
    bidirectional_divergence: bool = _unverified("bidirectional_divergence")  # [D]
    use_directional_disorder: bool = False   # [B] False in every system
    disorder_weight: float = _unverified("disorder_weight")            # [D]

    # ---- Temporal jerk surge gate (detector.py) ------------------------------------
    jerk_delta_frames: int = _unverified("jerk_delta_frames")          # [D] int >= 1
    surge_hold_frames: int = _unverified("surge_hold_frames")          # [D] int >= 1 (deque maxlen)
    theta_jerk: float = _unverified("theta_jerk")                      # [D] > 0 (squared; in denominator)
    g_0: float = _unverified("g_0")                                    # [D] gate floor, presumably in [0, 1)

    # ---- Smoothing and hysteresis --------------------------------------------------
    ema_alpha: float = _unverified("ema_alpha")                        # [D] presumably in (0, 1]
    k_on: float = 3.2                        # [B] baseline value; systems 3-5 use 3.8
    k_off: float = 1.8                       # [B] baseline value; systems 3-5 use 2.0
    persistence_frames: int = 7              # [B] baseline value; systems 3-5 use 8
    max_alarm_unanchored_frames: int = _unverified("max_alarm_unanchored_frames")  # [D]

    def unresolved_fields(self):
        """Names of fields still holding the UNVERIFIED sentinel."""
        return [f.name for f in fields(self) if isinstance(getattr(self, f.name), _Unverified)]