# the original version, with FCU before and after
from functools import partial
from .trq3d import *


TRQ3D = partial(
    TRQ3D,
    Encoder=TRQ3DEncoder,
    Decoder=TRQ3DDecoder
)