import threading

from security import LoginThrottle, RateLimiter, password_matches


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def test_password_matches():
    assert password_matches("s3cret", "s3cret")
    assert not password_matches("s3cret ", "s3cret")
    assert not password_matches("", "s3cret")
    assert password_matches("pässwörd", "pässwörd")


def test_login_throttle_locks_after_failures():
    clock = FakeClock()
    t = LoginThrottle(max_failures=3, lockout_seconds=60, clock=clock)
    for _ in range(2):
        t.record_failure()
    assert t.seconds_locked() == 0
    t.record_failure()
    assert t.seconds_locked() == 60
    clock.now += 61
    assert t.seconds_locked() == 0


def test_login_throttle_success_resets():
    t = LoginThrottle(max_failures=2, clock=FakeClock())
    t.record_failure()
    t.record_success()
    t.record_failure()
    assert t.seconds_locked() == 0


def test_rate_limiter_allows_up_to_budget():
    clock = FakeClock()
    rl = RateLimiter(5, window_seconds=3600, clock=clock)
    assert rl.try_acquire(3) == (True, 0.0)
    assert rl.try_acquire(2) == (True, 0.0)
    allowed, retry = rl.try_acquire(1)
    assert not allowed and retry == 3600
    assert rl.remaining() == 0


def test_rate_limiter_window_slides():
    clock = FakeClock()
    rl = RateLimiter(2, window_seconds=100, clock=clock)
    rl.try_acquire(1)
    clock.now += 40
    rl.try_acquire(1)
    allowed, retry = rl.try_acquire(1)
    assert not allowed and retry == 60  # the first request leaves the window in 60s
    clock.now += 60
    assert rl.try_acquire(1)[0]


def test_rate_limiter_cost_is_clamped_to_budget():
    rl = RateLimiter(3, clock=FakeClock())
    assert rl.try_acquire(8)[0]  # an 8-part syllabus still works with a budget of 3
    assert not rl.try_acquire(1)[0]


def test_rate_limiter_disabled_when_zero():
    rl = RateLimiter(0, clock=FakeClock())
    assert all(rl.try_acquire(100)[0] for _ in range(50))
    assert rl.remaining() == -1


def test_rate_limiter_is_thread_safe():
    rl = RateLimiter(100)
    results = []

    def worker():
        for _ in range(50):
            results.append(rl.try_acquire(1)[0])

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results.count(True) == 100
