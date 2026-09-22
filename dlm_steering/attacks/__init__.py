"""Attack registry used by the experiment CLI."""
from .base import AttackResult, Attacker, NoAttack, Prefix
from .dija import DIJA
from .pap import PAP
from .pair import PAIR

ATTACKERS = {a.name: a for a in (NoAttack, Prefix, DIJA, PAP, PAIR)}
