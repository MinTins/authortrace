"""AuthorTrace — гібридна нейромережева система детекції штучних текстів.

Швидкий старт:
    from src.detector_v2 import AuthorTraceDetectorV2
    detector = AuthorTraceDetectorV2(
        model_path="results/authortrace_model.pt",
        scaler_path="results/scaler.json",
        calibrator_path="results/calibrator.json",
    )
    result = detector.analyze("Текст для аналізу...")
"""

__version__ = "2.0.0"
