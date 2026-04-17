"""
Probability calibration stub for cancer ML classification.

Current state: no calibration is applied (cv='prefit' on a single-fold model
introduces bias). This module is reserved for future isotonic/Platt calibration
using a held-out calibration set.
"""


def calibrate_model(model, X_cal, y_cal, method="isotonic"):
    """Placeholder — calibration not yet implemented.

    To implement: use CalibratedClassifierCV(model, cv='prefit', method=method)
    on a dedicated calibration fold that was held out before training.
    """
    raise NotImplementedError(
        "Calibration requires a dedicated held-out calibration fold. "
        "Not yet implemented — contributions welcome."
    )
