"""Import-safe configuration and component composition.

The package root intentionally exports no runtime construction entrypoint.
Callers import the narrow config or registry surface they need; concrete model
and algorithm construction remains owned by :mod:`visual_rl.runtime`.
"""

__all__: tuple[str, ...] = ()
