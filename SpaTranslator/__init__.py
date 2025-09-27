from .utils import *
from .aligner_train import train_spatial_aligner
from .spatranslator import SpaTranslator

__version__ = "1.0.8"

banner = r"""
    _____               ______                           __        __              
   / ___/ ____   ____ _/_  __/_____ ____ _ ____   _____ / /____ _ / /_ ____   _____
   \__ \ / __ \ / __ `/ / /  / ___// __ `// __ \ / ___// // __ `// __// __ \ / ___/
  ___/ // /_/ // /_/ / / /  / /   / /_/ // / / /(__  )/ // /_/ // /_ / /_/ // /    
 /____// .___/ \__,_/ /_/  /_/    \__,_//_/ /_//____//_/ \__,_/ \__/ \____//_/     
      /_/                                                                           

SpaTranslator v{version}     
""".format(version=__version__)

print(banner)
