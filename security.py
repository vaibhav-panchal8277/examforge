"""Access control and usage limits for running ExamForge as a shared/public app."""
import hmac
import threading
import time
from collections import OrderedDict, deque


def password_matches(given: str, expected: str) -> bool:
    """Constant-time comparison, so response timing doesn't leak the password."""
    return hmac.compare_digest(given.encode("utf-8"), expected.encode("utf-8"))


class LoginThrottle:
    """Locks a session out for `lockout_seconds` after `max_failures` wrong passwords."""

    def __init__(self, max_failures: int = 5, lockout_seconds: int = 60, clock=time.monotonic):
        self.max_failures = max_failures
        self.lockout_seconds = lockout_seconds
        self.clock = clock
        self.failures = 0
        self.locked_until = 0.0

    def seconds_locked(self) -> float:
        return max(0.0, self.locked_until - self.clock())

    def record_failure(self) -> None:
        self.failures += 1
        if self.failures >= self.max_failures:
            self.locked_until = self.clock() + self.lockout_seconds
            self.failures = 0

    def record_success(self) -> None:
        self.failures = 0
        self.locked_until = 0.0


class KeyedRegistry:
    """
    Thread-safe map from a key (e.g. a user id or email) to a lazily created
    object. Holds at most `max_items`, evicting the least recently used, so
    requests with endless made-up keys can't exhaust memory.
    """

    def __init__(self, factory, max_items: int = 10_000):
        self._factory = factory
        self._max_items = max_items
        self._items = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key):
        with self._lock:
            if key in self._items:
                self._items.move_to_end(key)
            else:
                self._items[key] = self._factory()
                if len(self._items) > self._max_items:
                    self._items.popitem(last=False)
            return self._items[key]

    def __len__(self):
        return len(self._items)


class RateLimiter:
    """
    Thread-safe sliding-window limit on a shared budget of requests.

    One instance is shared by every session of the app, so it caps total
    usage (and cost) no matter how many people or browser tabs use it.
    """

    def __init__(self, max_requests: int, window_seconds: float = 3600, clock=time.monotonic):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.clock = clock
        self._stamps = deque()
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self.max_requests > 0

    def _expire(self, now: float) -> None:
        while self._stamps and self._stamps[0] <= now - self.window_seconds:
            self._stamps.popleft()

    def try_acquire(self, cost: int = 1) -> tuple:
        """
        Reserve `cost` requests. Returns (allowed, retry_after_seconds).

        A cost larger than the whole budget is clamped to the budget, so a
        single action is never permanently impossible.
        """
        if not self.enabled:
            return True, 0.0
        cost = max(1, min(cost, self.max_requests))
        with self._lock:
            now = self.clock()
            self._expire(now)
            if len(self._stamps) + cost <= self.max_requests:
                self._stamps.extend([now] * cost)
                return True, 0.0
            # Wait until enough of the oldest requests leave the window.
            needed = len(self._stamps) + cost - self.max_requests
            retry_after = self._stamps[needed - 1] + self.window_seconds - now
            return False, max(0.0, retry_after)

    def remaining(self) -> int:
        if not self.enabled:
            return -1
        with self._lock:
            self._expire(self.clock())
            return self.max_requests - len(self._stamps)
