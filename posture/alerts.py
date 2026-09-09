"""Escalating alerts, and the hold-to-clear mechanic.

Pure logic: this module decides *what level of nagging is warranted right now*
and never touches a screen, a notification API or a clock it was not given. All
the awkward cases live here where they can be tested, and the effects layer
just does as it is told.

The escalation
--------------
Levels rise with how long posture has been bad, not with how bad it is. A
slight slump you hold for two minutes is worth more attention than a deep one
you correct in five seconds, and the detector has already applied a 60-second
window and hysteresis before anything reaches here -- so by the time this sees
``bad`` it is a real, sustained deviation, not a twitch.

    0  none      nothing is wrong
    1  subtle    a colour change, and nothing more
    2  notify    an OS notification, once
    3  overlay   a full-screen window you have to fix your posture to dismiss

Hold-to-clear
-------------
The overlay does not close because posture stopped being bad. It closes after
posture has been *verifiably good* for a continuous stretch. That distinction
matters: "not bad" includes "I cannot see you", and leaning out of frame must
not be a way to dismiss it. Only a confident good reading counts toward the
hold, and any bad reading resets it to zero.

Absence
-------
An empty chair cancels everything -- there is nobody to nag, and an alert that
survives you leaving the room would be waiting to ambush you when you sit back
down. An unreadable frame freezes the timers instead: we do not know whether
you fixed it, so we neither escalate nor give credit.

Snooze
------
Explicit and time-boxed, as specified. It suppresses escalation for a fixed
period and then resumes from nothing, so a snooze can never quietly become
"off forever". There is deliberately no one-click dismiss.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

LEVEL_NONE = 0
LEVEL_SUBTLE = 1
LEVEL_NOTIFY = 2
LEVEL_OVERLAY = 3

LEVEL_NAMES = {
    LEVEL_NONE: "none",
    LEVEL_SUBTLE: "subtle",
    LEVEL_NOTIFY: "notify",
    LEVEL_OVERLAY: "overlay",
}

# Detector states this module reasons about, kept as plain strings so alerts
# does not import the detector and can be tested against stand-ins.
GOOD, BAD, UNKNOWN, AWAY = "good", "bad", "unknown", "away"


def _span(seconds: float) -> str:
    minutes, secs = divmod(int(max(0.0, seconds)), 60)
    return f"{minutes}m {secs}s" if minutes else f"{secs}s"


def _join(names: list[str]) -> str:
    if len(names) <= 1:
        return "".join(names)
    return f"{', '.join(names[:-1])} and {names[-1]}"


@dataclass
class AlertSettings:
    """Every timing the escalation uses. All editable from the panel."""

    enabled: bool = True
    # Seconds of sustained bad posture before each level. Cumulative from the
    # moment posture went bad, so the defaults mean: colour immediately, a
    # notification at 30 s, the overlay at 90 s.
    subtle_after: float = 0.0
    notify_after: float = 30.0
    overlay_after: float = 60.0
    # Continuous *verified good* seconds needed to dismiss the overlay.
    clear_hold: float = 5.0
    # How long the view of you may be lost before the hold restarts. A dropped
    # landmark for a moment should not punish you; a real gap should, because
    # "continuously within tolerance" cannot be claimed across a stretch where
    # nothing was measured. Without this, three good seconds, twenty seconds
    # out of frame and two more good seconds dismissed a five-second hold.
    hold_grace: float = 1.5
    # How long a snooze lasts. Time-boxed on purpose.
    snooze_seconds: float = 600.0
    # The overlay covers the whole screen, so it has to stay escapable by doing
    # what it asks. If it has been up this long and the hold has never once
    # *started* -- not "never finished", never started, meaning not a single
    # good reading in all that time -- then what it is asking for is not being
    # achieved: a baseline nothing can satisfy, a camera that has been moved,
    # a metric reading off a landmark it should not trust. Continuing to block
    # the screen on that is not discipline, it is a bug with a nag attached.
    #
    # Deliberately not a general timeout, and deliberately long. Someone who
    # reaches a good posture even for a moment has shown the target is
    # reachable, and the overlay goes on asking them to hold it -- which is the
    # whole mechanic, and this must not become a way to wait out the nag. Two
    # things have to be true together: no good reading at all, and this long
    # about it. Ten minutes of a screen-blocking window that has never once
    # acknowledged the person is a claim about the app, not about them.
    # Zero disables it.
    overlay_giveup_after: float = 600.0
    # Re-notify if posture is still bad this long after the last notification.
    # Zero disables the repeat, which is the default: one notification per
    # episode, with the overlay as the escalation rather than nagging.
    renotify_after: float = 0.0

    def threshold(self, level: int) -> float:
        """Seconds of bad posture at which ``level`` is reached."""
        if level <= LEVEL_NONE:
            return 0.0
        if level == LEVEL_SUBTLE:
            return self.subtle_after
        if level == LEVEL_NOTIFY:
            return self.subtle_after + self.notify_after
        return self.subtle_after + self.notify_after + self.overlay_after


@dataclass(frozen=True)
class AlertState:
    """What the rest of the app should do about your posture right now."""

    level: int = LEVEL_NONE
    since: float = 0.0            # monotonic time the level was entered
    bad_for: float = 0.0
    good_for: float = 0.0
    hold_required: float = 0.0    # seconds of good needed to clear the overlay
    hold_remaining: float | None = None
    snoozed_until: float | None = None
    offenders: tuple[str, ...] = ()
    headline: str = ""
    detail: str = ""

    @property
    def name(self) -> str:
        return LEVEL_NAMES.get(self.level, str(self.level))

    @property
    def snoozed(self) -> bool:
        return self.snoozed_until is not None

    def to_dict(self, now: float | None = None) -> dict[str, Any]:
        now = time.monotonic() if now is None else now
        return {
            "level": self.level,
            "name": self.name,
            "active": self.level > LEVEL_NONE,
            "bad_for": round(self.bad_for, 1),
            "good_for": round(self.good_for, 1),
            "in_level_for": round(max(0.0, now - self.since), 1) if self.since else 0.0,
            "hold_required": round(self.hold_required, 1),
            "hold_remaining": (None if self.hold_remaining is None
                               else round(max(0.0, self.hold_remaining), 1)),
            "snoozed": self.snoozed,
            "snooze_remaining": (None if self.snoozed_until is None
                                 else round(max(0.0, self.snoozed_until - now), 1)),
            "offenders": list(self.offenders),
            "headline": self.headline,
            "detail": self.detail,
        }


@dataclass
class AlertEvent:
    """Something the effects layer should do once, not on every tick."""

    kind: str                 # "escalate" | "clear" | "snoozed" | "resumed"
    level: int
    state: AlertState
    previous: int = LEVEL_NONE


class AlertEngine:
    """Turns a stream of posture verdicts into an escalation level.

    Call :meth:`update` once per verdict. It returns the current state; any
    transitions worth acting on are appended to :attr:`events`, which the
    caller drains. Keeping effects out of here means the whole escalation can
    be exercised in microseconds against a fake clock.
    """

    def __init__(self, settings: AlertSettings | None = None) -> None:
        self.settings = settings or AlertSettings()
        self._level = LEVEL_NONE
        self._since = 0.0
        self._bad_for = 0.0
        self._good_for = 0.0
        self._unknown_for = 0.0
        # Longest continuous good stretch reached since the overlay went up.
        # Zero after two minutes of it means the hold has never once started.
        self._best_hold = 0.0
        self._last_t: float | None = None
        self._snoozed_until: float | None = None
        self._last_notify_at: float | None = None
        self._offenders: tuple[str, ...] = ()
        self.events: list[AlertEvent] = []

    # -- control -----------------------------------------------------------

    def set_settings(self, settings: AlertSettings) -> None:
        self.settings = settings
        if not settings.enabled:
            self._reset(self._last_t or 0.0)

    def snooze(self, seconds: float | None = None, now: float | None = None) -> AlertState:
        """Suppress escalation for a fixed stretch, then resume from nothing."""
        now = time.monotonic() if now is None else now
        span = self.settings.snooze_seconds if seconds is None else max(0.0, seconds)
        self._snoozed_until = now + span
        self._reset(now)
        self.events.append(AlertEvent("snoozed", LEVEL_NONE, self.state(now)))
        return self.state(now)

    def cancel_snooze(self, now: float | None = None) -> AlertState:
        now = time.monotonic() if now is None else now
        if self._snoozed_until is not None:
            self._snoozed_until = None
            self.events.append(AlertEvent("resumed", LEVEL_NONE, self.state(now)))
        return self.state(now)

    def drain(self) -> list[AlertEvent]:
        events, self.events = self.events, []
        return events

    # -- the loop ----------------------------------------------------------

    def update(self, verdict: Any, now: float | None = None) -> AlertState:
        now = time.monotonic() if now is None else now
        elapsed = 0.0 if self._last_t is None else max(0.0, now - self._last_t)
        self._last_t = now

        state = getattr(verdict, "state", UNKNOWN)
        self._offenders = tuple(getattr(verdict, "offenders", ()) or ())

        if self._snoozed_until is not None and now >= self._snoozed_until:
            self._snoozed_until = None
            self.events.append(AlertEvent("resumed", LEVEL_NONE, self.state(now)))

        if not self.settings.enabled or self._snoozed_until is not None:
            self._reset(now, keep_level=False)
            return self.state(now)

        if state == AWAY:
            # Nobody to nag, and an alert left running would ambush them when
            # they sit back down.
            if self._level > LEVEL_NONE:
                self._clear(now)
            self._bad_for = self._good_for = 0.0
            return self.state(now)

        if state == BAD:
            self._bad_for += elapsed
            self._good_for = 0.0
            self._unknown_for = 0.0
        elif state == GOOD:
            self._good_for += elapsed
            self._bad_for = 0.0
            self._unknown_for = 0.0
        else:
            # UNKNOWN: the bad/good timers freeze, because we cannot see whether
            # it was fixed. But a gap longer than the grace period voids any
            # part-completed hold: you cannot have been continuously within
            # tolerance through a stretch nobody measured.
            self._unknown_for += elapsed
            if self._unknown_for > self.settings.hold_grace:
                self._good_for = 0.0

        if self._level >= LEVEL_OVERLAY:
            # Only a verified good stretch dismisses the overlay. Leaning out
            # of frame must not count as fixing your posture.
            self._best_hold = max(self._best_hold, self._good_for)
            if self._good_for >= self.settings.clear_hold:
                self._clear(now)
            elif self._hold_is_not_happening(now):
                self._stand_down(now)
            return self.state(now)

        if state == GOOD:
            if self._level > LEVEL_NONE:
                self._clear(now)
            return self.state(now)

        if state != BAD:
            # UNKNOWN: hold the current level. We cannot see whether it was
            # fixed, so neither escalating nor clearing is honest.
            return self.state(now)

        # Only a bad reading can raise the level. Worth stating rather than
        # leaving to the timers: subtle_after defaults to zero, so a bad_for of
        # 0.0 already satisfies the level-1 threshold and any other state
        # falling through here would re-raise the alert instantly.
        target = self._level_for(self._bad_for)
        if target > self._level:
            self._escalate(target, now)
        elif (target >= LEVEL_NOTIFY and self.settings.renotify_after > 0
              and self._last_notify_at is not None
              and now - self._last_notify_at >= self.settings.renotify_after):
            self._last_notify_at = now
            self.events.append(AlertEvent("escalate", self._level, self.state(now),
                                          previous=self._level))
        return self.state(now)

    # -- internals ---------------------------------------------------------

    def _level_for(self, bad_for: float) -> int:
        level = LEVEL_NONE
        for candidate in (LEVEL_SUBTLE, LEVEL_NOTIFY, LEVEL_OVERLAY):
            if bad_for >= self.settings.threshold(candidate):
                level = candidate
        return level

    def _hold_is_not_happening(self, now: float) -> bool:
        """Whether the overlay is demanding something that is not being met.

        The test is on progress, not on time alone: a single good reading at
        any point says the target is reachable and the person is simply being
        asked to hold it, which is the mechanic working. Never having reached
        one is the case that cannot be waited out.
        """
        limit = self.settings.overlay_giveup_after
        return (limit > 0.0 and self._best_hold <= 0.0
                and now - self._since >= limit)

    def _stand_down(self, now: float) -> None:
        """Take the overlay down and snooze, because it cannot be satisfied.

        Snoozing rather than merely clearing, because the posture that raised
        it is still bad by the app's own reckoning, so anything less would
        re-escalate within seconds and put the window straight back up. This is
        the one path that lowers the level without a good reading, and it is
        not a dismissal: the person did not ask for it, the app concluded it
        was asking for the impossible and said so.
        """
        previous, self._level, self._since = self._level, LEVEL_NONE, now
        self._last_notify_at = None
        self._bad_for = self._good_for = self._best_hold = 0.0
        self._snoozed_until = now + self.settings.snooze_seconds
        self.events.append(
            AlertEvent("stood_down", LEVEL_NONE, self.state(now), previous))

    def _escalate(self, level: int, now: float) -> None:
        previous, self._level, self._since = self._level, level, now
        if level >= LEVEL_OVERLAY and previous < LEVEL_OVERLAY:
            self._best_hold = 0.0
        if level >= LEVEL_NOTIFY:
            self._last_notify_at = now
        self.events.append(AlertEvent("escalate", level, self.state(now), previous))

    def _clear(self, now: float) -> None:
        previous, self._level, self._since = self._level, LEVEL_NONE, now
        self._last_notify_at = None
        self._best_hold = 0.0
        if previous > LEVEL_NONE:
            self.events.append(AlertEvent("clear", LEVEL_NONE, self.state(now), previous))

    def _reset(self, now: float, keep_level: bool = False) -> None:
        if not keep_level and self._level > LEVEL_NONE:
            self._clear(now)
        self._bad_for = self._good_for = self._unknown_for = 0.0
        self._best_hold = 0.0

    # -- reporting ---------------------------------------------------------

    def state(self, now: float | None = None) -> AlertState:
        now = time.monotonic() if now is None else now
        hold_remaining = None
        if self._level >= LEVEL_OVERLAY:
            hold_remaining = max(0.0, self.settings.clear_hold - self._good_for)
        return AlertState(
            level=self._level, since=self._since,
            bad_for=self._bad_for, good_for=self._good_for,
            hold_required=self.settings.clear_hold,
            hold_remaining=hold_remaining,
            snoozed_until=self._snoozed_until,
            offenders=self._offenders,
            headline=self._headline(),
            detail=self._detail(now),
        )

    def _headline(self) -> str:
        """What is being reported, said as a measurement.

        The app measures you against a baseline you set. It does not know
        whether your posture is healthy, and phrasing that named a body part as
        a fault -- "fix your neck" -- claimed knowledge it does not have and
        put a diagnosis where an observation belongs. So it reports what
        changed and for how long, and leaves the conclusion to the person who
        can see the whole picture.
        """
        if self._level == LEVEL_NONE:
            return ""
        span = _span(self._bad_for)
        if not self._offenders:
            return f"Sustained deviation from baseline, {span}"
        pretty = [o.replace("_", " ") for o in self._offenders]
        if len(pretty) == 1:
            return f"{pretty[0].capitalize()} away from baseline for {span}"
        return f"{_join(pretty).capitalize()} away from baseline for {span}"

    def _detail(self, now: float) -> str:
        if self._level == LEVEL_NONE:
            return ""
        if self._level >= LEVEL_OVERLAY:
            left = max(0.0, self.settings.clear_hold - self._good_for)
            return (f"Measured past your calibrated tolerance for "
                    f"{_span(self._bad_for)}. Returning to baseline for "
                    f"{left:.0f}s closes this.")
        return f"Measured past your calibrated tolerance for {_span(self._bad_for)}."
