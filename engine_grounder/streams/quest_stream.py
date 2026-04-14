"""Live Meta Quest Device Hub Passthrough API integration."""

from engine_grounder.streams.sensor_stream import SensorStream


class QuestStream(SensorStream):
    def connect(self):
        raise NotImplementedError

    def get_frame(self) -> dict:
        raise NotImplementedError

    def close(self):
        raise NotImplementedError
