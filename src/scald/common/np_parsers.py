from typing import Any

import numpy as np

from scald.common.logger import get_logger

logger = get_logger()


def predictions_to_numpy(predictions: list[Any]) -> np.ndarray:
    """Convert predictions list from Actor to numpy array."""
    if not predictions:
        logger.warning("Empty predictions list provided")
        return np.array([])

    try:
        arr = np.array(predictions)
        logger.debug(f"Converted predictions to numpy array: shape={arr.shape}, dtype={arr.dtype}")
        return arr

    except Exception as e:
        logger.error(f"Failed to convert predictions to numpy array: {e}")

        # Try to handle string predictions
        try:
            if isinstance(predictions[0], str):
                logger.info("Attempting to convert string predictions to numeric values")
                numeric_predictions = []
                for pred in predictions:
                    try:
                        numeric_predictions.append(float(pred))
                    except ValueError:
                        numeric_predictions.append(pred)
                arr = np.array(numeric_predictions)
                logger.debug(f"Converted string predictions: shape={arr.shape}, dtype={arr.dtype}")
                return arr
            else:
                raise ValueError(f"Cannot convert predictions to numpy array: {e}") from e

        except Exception as e2:
            logger.error(f"Fallback conversion also failed: {e2}")
            raise ValueError(f"Cannot convert predictions to numpy array: {e}") from e
