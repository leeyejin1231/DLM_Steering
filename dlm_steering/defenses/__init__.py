from .base import Defender, NullDefender
from .steering import Ours
from .recovery import V3
from .baselines import SelfReminder, DiffuGuard

DEFENDERS = {d.name: d for d in (NullDefender, Ours, SelfReminder, DiffuGuard)}
