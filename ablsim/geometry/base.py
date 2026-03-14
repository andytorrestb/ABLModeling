from abc import ABC, abstractmethod

class GeometryBase(ABC):
    def __init__(self, config):
        """
        Initialize geometry with simulation configuration.
        config: SimulationConfig instance
        """
        self.config = config

    @abstractmethod
    def __call__(self, x, y, z):
        """
        Return a boolean mask where the geometry exists (solid).
        x, y, z: coordinate arrays (numpy)
        """
        pass
