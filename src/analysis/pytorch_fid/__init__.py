__version__ = '0.2.1'


from .fid_score_multi import get_activations,calculate_activation_statistics,calculate_frechet_distance

__all__ = [
    'get_activations',
    'calculate_activation_statistics',
    'calculate_frechet_distance'
]