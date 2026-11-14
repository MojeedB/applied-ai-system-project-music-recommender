"""
Tiny in-memory result cache with expiry and a size cap.

Plain Python (no Streamlit) so it can be tested on its own. The caller decides what is
worth storing; app.py only stores successful answers, never failures.
"""

import time
from typing import Any, Callable, Dict, Hashable, List, Optional, Tuple


class ResultCache:
    def __init__(self, ttl_seconds: float = 3600, max_size: int = 200,
                 clock: Callable[[], float] = time.monotonic):
        self.ttl = ttl_seconds
        self.max_size = max_size
        self._clock = clock
        self._items: Dict[Hashable, Tuple[float, Any]] = {}

    def get(self, key: Hashable) -> Optional[Any]:
        """Return the stored value, or None if missing or older than the TTL."""
        entry = self._items.get(key)
        if entry is None:
            return None
        stored_at, value = entry
        if self._clock() - stored_at > self.ttl:
            del self._items[key]
            return None
        return value

    def put(self, key: Hashable, value: Any) -> None:
        """Store a value, dropping the oldest entries if the cache is full."""
        self._items.pop(key, None)  # re-inserting moves it to the newest position
        self._items[key] = (self._clock(), value)
        while len(self._items) > self.max_size:
            del self._items[next(iter(self._items))]

    def entries(self) -> List[Tuple[Hashable, Any]]:
        """All live (key, value) pairs, newest first. Expired entries are dropped."""
        for key in list(self._items):
            self.get(key)  # get() removes the entry if it has expired
        return [(key, value) for key, (_, value) in reversed(list(self._items.items()))]

    def clear(self) -> None:
        self._items.clear()

    def __len__(self) -> int:
        return len(self._items)
