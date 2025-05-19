from .implementations.ackley import Ackley as _Ackley

Ackley: type[_Ackley] = _Ackley

__all__ = ["Ackley"]