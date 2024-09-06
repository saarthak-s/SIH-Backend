from typing import List

from pydantic import BaseModel


class ForecastData(BaseModel):
    time: List[int]
    values: List[float]
