"""The Jinja environment shared by every report template/partial.

Kept in its own module (rather than directly in ``__init__.py``) so
that `sections.py`/`render.py` can import ``jinja_env`` without
depending on the package `__init__` having finished executing.
"""

from functools import cache

from jinja2 import Environment, PackageLoader, select_autoescape


@cache
def jinja_env() -> Environment:
    return Environment(
        loader=PackageLoader("transect.report"),
        autoescape=select_autoescape(default=True),
    )
