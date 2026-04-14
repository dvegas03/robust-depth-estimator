"""
Engine Grounder — Vision-Language Grounded 3-D Spatial Reasoning Pipeline.

Core public API
---------------
>>> from engine_grounder import Pipeline, PipelineResult
>>> from engine_grounder import RobustDepthEstimator, MeshBuilder
>>> from engine_grounder import Projector
>>> from engine_grounder import ShapeEncoder, ShapeDescriptor, VLMAgent
>>> from engine_grounder import SensorStream, MockStream, BunnyStream
"""

from engine_grounder.pipeline import Pipeline, PipelineResult
from engine_grounder.geometry.depth_filter import RobustDepthEstimator
from engine_grounder.geometry.mesh_builder import MeshBuilder
from engine_grounder.spatial.projector import Projector
from engine_grounder.perception.shape_encoder import ShapeEncoder, PointNetEncoder
from engine_grounder.perception.shape_descriptor import ShapeDescriptor
from engine_grounder.perception.vlm_agent import VLMAgent
from engine_grounder.streams.sensor_stream import SensorStream
from engine_grounder.streams.mock_stream import MockStream, BunnyStream

__version__ = "0.1.0"
__author__ = "David Vegas"

__all__ = [
    # Pipeline
    "Pipeline",
    "PipelineResult",
    # Geometry
    "RobustDepthEstimator",
    "MeshBuilder",
    # Spatial
    "Projector",
    # Perception
    "ShapeEncoder",
    "PointNetEncoder",
    "ShapeDescriptor",
    "VLMAgent",
    # Streams
    "SensorStream",
    "MockStream",
    "BunnyStream",
]
