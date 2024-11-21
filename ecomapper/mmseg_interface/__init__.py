"""
MMSegmentation wrapper classes and custom component registration.
"""

from .mask2former_config_wrapper import Mask2FormerConfigWrapper
from .mmseg_config_wrapper import MMSegConfigWrapper

# Import all config wrappers here so that they are registered as children
# of ``MMSegConfigWrapper``.
__all__ = ['MMSegConfigWrapper', 'Mask2FormerConfigWrapper']
