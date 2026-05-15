from typing import Annotated, Dict, List

from pydantic import BaseModel, Field


class IrisFeatures(BaseModel):
    sepal_length: float = Field(..., ge=0, le=10, description="cm")
    sepal_width: float = Field(..., ge=0, le=10, description="cm")
    petal_length: float = Field(..., ge=0, le=10, description="cm")
    petal_width: float = Field(..., ge=0, le=10, description="cm")


class PredictionResponse(BaseModel):
    class_id: int
    class_name: str
    probability: float


SampleRow = Annotated[List[float], Field(min_length=4, max_length=4)]


class DriftRequest(BaseModel):
    """Батч даних для перевірки drift.

    samples – список спостережень, кожне з 4 числовими ознаками.
    """

    samples: Annotated[List[SampleRow], Field(min_length=10)]
    alpha: float = Field(default=0.05, ge=0.001, le=0.5, description="Поріг значущості для KS-тесту")


class FeatureDriftInfo(BaseModel):
    statistic: float
    p_value: float
    drift_detected: bool


class DriftResponse(BaseModel):
    drift_detected: bool
    n_drifted_features: int
    drifted_features: List[str]
    per_feature: Dict[str, FeatureDriftInfo]
    n_samples: int
    alpha: float
