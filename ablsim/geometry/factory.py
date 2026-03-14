# ablsim/geometry/factory.py
from .cube import CubeGeometry

class GeometryFactory:
    _registry = {
        'cube': CubeGeometry
    }

    @classmethod
    def register(cls, name, geometry_class):
        cls._registry[name] = geometry_class

    @staticmethod
    def get_geometry(config):
        """
        Factory method to select geometry based on config.
        """
        geo_type = config.geometry.get('type', 'cube')
        
        geometry_class = GeometryFactory._registry.get(geo_type)
        if geometry_class:
            return geometry_class(config)
        else:
            raise ValueError(f"Unknown geometry type: {geo_type}. Available: {list(GeometryFactory._registry.keys())}")
