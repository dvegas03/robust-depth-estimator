"""Abstract base class for sensor data streams."""

from abc import ABC, abstractmethod


class SensorStream(ABC):
    @abstractmethod
    def connect(self):
        ...

    @abstractmethod
    def get_frame(self) -> dict:
        ...

    @abstractmethod
    def close(self):
        ...
