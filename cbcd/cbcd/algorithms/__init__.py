"""Top-level algorithm composition."""

from cbcd.algorithms.cml import cml
from cbcd.algorithms.fci import anytime_fci, fci, rfci
from cbcd.algorithms.lmarvel import lmarvel
from cbcd.algorithms.locals_ import locals_
from cbcd.algorithms.marvel import marvel
from cbcd.algorithms.mmb_by_mmb import mmb_by_mmb
from cbcd.algorithms.pc import pc

__all__ = [
    "anytime_fci",
    "cml",
    "fci",
    "lmarvel",
    "locals_",
    "mmb_by_mmb",
    "marvel",
    "pc",
    "rfci",
]
