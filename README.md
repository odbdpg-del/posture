# posture

A local posture monitor. Watches you through one or more webcams while you
work, and escalates an interruption if you hold a bad posture too long.

**Status: complete — all five phases.** Capture and pose landmarks;
calibration against your own posture; rolling-window detection with hysteresis
and presence pausing; escalating alerts with hold-to-clear; adaptive sampling;
multi-camera with role assignment; a local web UI for all of it; a tray icon;
and SQLite stats.

## Privacy

Frames are held in memory, turned into numbers, and dropped. Nothing writes,
encodes or transmits an image, and only derived values (angles, ratios,
timestamps) are ever persisted. `tests/test_invariants.py` enforces both rules
by scanning the package source, so they survive future edits instead of
depending on discipline.

The one exception is deliberate and sits outside the package:
`scripts/fetch_model.py` downloads the pose model once. The app itself has no
networking code and cannot reach the network.

## Setup

Platform assumed: **Windows 11** (from the environment this was built on; the
brief left it blank). Nothing is Windows-specific except the camera backend
order in `capture.py`, which already falls through to a portable default.

Double-clicking `start-posture.bat` does the whole of this section and then
starts the app, so the two steps below are only worth running by hand if you
want the dev dependencies as well, or you are not on Windows.

```bash
python -m venv .venv && .venv/Scripts/python.exe -m pip install -r requirements-dev.txt
```

```bash
.venv/Scripts/python.exe scripts/fetch_model.py
```

## Run it

Double-click **`start-posture.bat`**. On a fresh machine the first run builds
the virtual environment and downloads the pose model, which takes a few
minutes; after that it goes straight to the app. It passes any flags through,
so `start-posture.bat --diagnose` works too, and it keeps its window open with
the reason if a step fails rather than closing before you can read it.

Or, equivalently:

```bash
.venv/Scripts/python.exe -m posture
```

That starts the cameras and opens the control panel on
<http://127.0.0.1:8760>. From there you can see what each camera is tracking,
assign roles, change settings, and stop the app. It exits on Ctrl-C or the
panel's stop button; nothing keeps running in the background afterwards.

The terminal modes are still there, because they are the fastest way to check
one thing without a browser:

```bash
.venv/Scripts/python.exe -m posture --list-cameras
```

```bash
.venv/Scripts/python.exe -m posture --diagnose
```

```bash
.venv/Scripts/python.exe -m posture --watch --role side --debug
```

The debug window closes with `q`, `Esc`, or its own X button, and Ctrl-C in the
terminal works too. If one ever does get stuck, it is an ordinary process:
`Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*-m posture*' }`
finds it and `Stop-Process -Id <pid>` ends it.

Other flags: `--camera N`, `--fps`, `--vis`, `--port`, `--no-browser`,
`--config`, `--verbose`, `--duration`.

## The control panel

A full-viewport dark instrumentation layout: persistent header, left
navigation, a large live camera workspace, a right-hand status column, and a
measurement strip beneath. Five sections — Live, History, Cameras, Calibration,
Settings.

The hierarchy is deliberate: **live person → current posture → what is wrong →
how long it has been wrong → historical trend → configuration**. Diagnostics
that used to dominate the screen (landmark confidences, backend details, scale
references, detector notes) still exist, under Settings.

Colour is rationed. Mint, amber and coral mean posture state and nothing else,
so any coloured thing in the interface carries information; navigation and
buttons stay neutral slate. Monitoring breathes gently; posture warnings never
flash, because the thing being reported is already a minute old by design.

**The camera workspace has two layouts and three preview modes.** Single shows
one large feed with the others as chips; Grid shows every camera at once, and
clicking one promotes it. The preview mode is a setting:

| mode | what you see | frames retained |
| --- | --- | --- |
| `video` | camera image with posture geometry over it | yes |
| `skeleton` | the tracked figure only, on a blank panel | **no** |
| `off` | neither | **no** |

The mode is in the camera panel header next to Single/Grid, and again under
Settings. It changes what that panel shows, so a copy of the control belongs
on it — the first version had it only in Settings, three screens away, which
made it effectively undiscoverable.

Skeleton mode is not a cosmetic filter. It draws from the landmark coordinates
already in the status payload, so in that mode no frame is kept in memory and
none is encoded — a real reduction in what leaves the process, for anyone who
would rather not watch themselves all day. One predicate (`config.retains_frames`)
answers it for both the capture side and the HTTP side, so they cannot
disagree, and a test asserts both consult it.

**In video mode the preview shows live video with the posture geometry over it.** It started as
a stick figure on the theory that joint positions answer the only question a
setup screen needs to answer. That was wrong in practice: aiming a camera you
cannot see through is guesswork, and "is it pointed at me" is not a question a
diagram answers well.

So frames are encoded and sent to your own browser over loopback, where the
panel draws the shoulder line, head and torso centrelines, the vertical
reference and the tracked joints as SVG — coloured by posture state. The rules
around sending pixels at all are narrow and tested:

- **No frame is ever written to disk.** Unchanged, absolute, enforced across the
  whole package.
- **Only `webui.py` may encode a frame**, so encoding cannot spread to
  somewhere that then writes the bytes out.
- **Nothing leaves the machine.** The server still refuses to bind anywhere but
  loopback and still rejects foreign `Host` headers.
- **The toggle actually stops it.** "Show camera video" off means workers do not
  retain frames at all, rather than the page merely hiding them — checked by a
  test, because a switch that only hides would be a lie.

## Tests

```bash
.venv/Scripts/python.exe -m pytest
```

446 tests, no camera required. The geometry and the metrics are checked against
synthetic landmark poses built from anatomy-shaped parameters
(`tests/synthetic.py`), so a threshold or sign change can be verified without
slouching in front of a webcam.

The invariance tests are the ones that matter: the same body must read the same
whether the camera is 4:3 or 16:9, near or far, on your left or your right, and
whether or not the preview is mirrored.

`tests/conftest.py` redirects the stats database and the config file into a
temporary directory for the whole session, and asserts per-test that nothing
reached the real ones. This was not paranoia: the alert tests had been writing
into `~/.posture/posture.db`, and their escalations showed up in the panel as
sixty alerts in forty seconds. A test suite that can reach your real data will
eventually corrupt your real data.

The control panel is tested against a real socket with a stubbed monitor, which
covers the settings rules, the loopback binding and the DNS-rebinding guard.

## How detection works

**Calibration.** Press Calibrate, sit the way you want to sit, and it watches
for ten seconds. The baseline is the *median* of each metric and the spread is
the *median absolute deviation* rather than mean and standard deviation: ten
unsupervised seconds always contain junk samples — you settling, a glance at
the door, a landmark flickering — and robust statistics need half the samples
to be bad before they move. Bad posture is then deviation from that baseline,
never from a textbook ideal.

**Tolerances are clamped at both ends.** A tolerance is `3 × your spread`, but
floored and capped by the metric. The floor stops a very still calibration
producing a hair-trigger. The ceiling stops a fidgety one producing a metric
that can never fire — which is the more dangerous failure, because it is
silent.

Note the direction, because it is easy to get backwards: **raising a ceiling
makes a metric less sensitive**, since it lets a wider learned tolerance
through. Lower it to make something complain sooner.

Reaching a clamp is normal and the panel just says so — "floored", "capped" or
"learned" — next to the value in use. Only a spread more than twice the ceiling
is treated as a bad capture worth warning about.

**You can override any of it.** The Sensitivity section lists every metric with
the tolerance in use, where it came from, and a box to set your own number.
Empty means "use my calibration"; a number overrides the clamps entirely.

**Implausible geometry is rejected, not scored.** The pose model will fit a
person to an empty chair with a coat on it, and report the guess confidently.
Two guards catch the results: a torso tilted more than 60° from vertical, and
an ear that is not above its shoulder. Both were found on real hardware, and
the second failed *flatteringly* — neck tilt only counts upward deviation as
bad, so a reading of −149° produced zero excess and a confident score of 100
over an empty chair.

**The front metrics need you to be facing the front camera.** All three divide
by a horizontal span, and turning away foreshortens it: the horizontal
separation of the shoulders collapses toward zero while the vertical offset
between them does not, so `arctan2(dy, |dx|)` swings toward ±90° on a body that
has not moved. Measuring the straight-line distance between the shoulders does
*not* catch this — that distance stays healthy precisely because of the vertical
offset causing the trouble. Found live: shoulder tilt −52° and head roll −59°
against a 4° tolerance, from someone who had turned to talk to somebody. So the
gate is on the horizontal spread as a fraction of the shoulder span, which is
the same number as the reported tilt angle, and it drops all three metrics at
once rather than only the angle that shows the problem worst. It bounds the
damage rather than detecting the turn: in a projection, "turned 80°" and
"shoulders genuinely tilted" are the same picture.

**Hips are optional.** This is a desk app, and at a desk the hips are usually
under the desk, below the frame, or behind an armrest. So the side role has two
tiers. Ear plus shoulder gives **neck tilt** — the ear-over-shoulder angle from
vertical, which is the forward-head signal and the thing desk posture is mostly
about. A visible hip adds neck flexion, forward head and torso lean on top. Only
the shoulder is ever required, and a bad hip discards the torso metrics without
touching neck tilt, which was measured from landmarks the hip never touched.

**A baseline has to be somewhere neutral posture can reach.** One-sided metrics
measure you against your own baseline, so calibrating in a posture you will not
hold makes the metric permanently angry — sitting normally reads as a deviation
and no amount of sitting up ever clears it. Found live: a torso lean baseline of
−15.9°, captured while reclining, put an upright torso 15.9° past baseline
against a 7° tolerance, so the score sat near zero all day and blamed torso lean
while the person sat straight. Nothing else can catch it — a reclining torso is
physically plausible, and the spread was tight, so the baseline looked like a
good one. Calibration now checks whether the metric's neutral value would itself
be out of tolerance against the baseline it just captured, and says so. The
detector applies the same test to baselines already stored, since the ones that
matter were captured before the check existed.

**Direction matters.** Neck flexion is only bad when it *drops*; forward head
and torso lean only when they *rise*. Shoulder tilt, head roll and lateral
offset are bad either way. Without this, sitting up straighter would count
against you.

**The window and the hysteresis.** Each `(camera, metric)` pair keeps a
60-second window; posture is bad once a metric is out of tolerance for more
than 70% of it. Once flagged, it is judged against a *tighter* tolerance
(70% of the entry one) until it clears, so recovering takes a real change in
posture rather than a lucky sample. Cameras are not synchronised — adaptive
sampling means they run at different rates — so each keeps its own window and
fusion happens at verdict time: a metric is flagged if any camera says so.

**Suspension.** An empty chair for 15 seconds suspends judgement entirely, and
a person whose hips are not visible reports "cannot see you properly" rather
than silently doing nothing. Those are distinct states throughout.

**Adaptive sampling.** The rate follows the verdict: 5 Hz when a metric is
within half a tolerance of its limit or posture is already bad, 2 Hz when
things are comfortably fine, 0.5 Hz at an empty chair. Calibration pins the
full rate regardless, since it is the capture the statistics depend on.

## Stack

Python + MediaPipe Tasks Vision + OpenCV, as briefed. One deviation, forced
rather than chosen: **MediaPipe 1.0 removed `mp.solutions.pose`**, which used
to ship its model inside the pip package. The Tasks Vision API that replaced it
needs an external `.task` bundle, hence the one-time fetch script.

Two decisions worth knowing about:

- **VIDEO running mode, not IMAGE.** VIDEO reuses the previous frame's region
  of interest and skips the whole-frame detector on most frames. Measured here:
  17 ms versus 33 ms per frame. Nearly half the CPU budget, for a constraint we
  can live with (timestamps must increase; one landmarker per camera).
- **Side metrics normalize by torso length, not shoulder width.** The brief
  said shoulder width for everything. From a side view the two shoulder
  landmarks project almost on top of each other, so measured shoulder width
  collapses toward zero and its relative noise explodes — exactly where the
  side camera matters most. Torso length (shoulder to hip) is the longest
  segment reliably visible from the side. Front metrics still use shoulder
  width, which is correct there. Each sample reports which reference it used.

## What phase 1 measured

**CPU.** At 5 Hz and 640x480 the full pipeline costs about **21% of one core**
on this machine — over the <10% target. The breakdown, measured rather than
guessed:

| stage | cost |
| --- | --- |
| grab + decode | ~2 ms/sample |
| inference, back-to-back | 13.2 ms wall, 13.3 ms CPU (single-threaded) |
| inference, paced at 5 Hz | ~29 ms CPU |

The gap between those last two rows is the whole story: idling 187 ms between
samples lets the CPU drop to a low P-state, so the same work costs roughly
twice the CPU-time. A low-duty-cycle background task is penalised for being
well behaved. No GPU delegate is available in this MediaPipe build
(`GPU processing is disabled`), so this is a CPU-only workload.

Capture itself was made much cheaper along the way. The conventional approach —
a thread calling `grab()` continuously to keep the driver buffer empty — cost
5.5% of a core, and a variant that blocks until it can prove it holds a
brand-new frame cost 8.3%. Measurement showed the driver keeps a bounded 2-3
frame ring and drops the oldest, so lag is capped at ~70 ms with no draining at
all, which is nothing next to a 5 Hz sample rate. Capture now sleeps until a
sample is due and takes one frame, and a probe every 30 s watches for a backend
whose backlog actually *grows* — the case that would genuinely hurt.

Options for closing the remaining gap, in preference order:

1. **Adaptive sampling** (fits naturally into phase 2): full 5 Hz when a metric
   is near its tolerance or an alert is live, 1-2 Hz when posture is
   comfortably good, near zero when the chair is empty. Full responsiveness
   when it matters, and the all-day average lands well under budget.
2. **Lower the fixed rate.** ~12.5% at 3 Hz, ~8.3% at 2 Hz. A 60 s window still
   holds 120 samples at 2 Hz.
3. **Capture at 320x240.** ~12% at 5 Hz. Not yet validated for landmark
   accuracy, and the shoulders are already marginal on this camera.

Option 1 is now implemented; see "Adaptive sampling" above.

**Camera placement.** The built-in webcam cannot support either role. It is
framed on your head: the model finds a person in 100% of frames, but the hips
every side metric needs are never in view, and the shoulders the front metrics
need are confident only a third of the time.

```
  left_hip             0.00       0%  <- never seen
  side   role: metrics available in   0.0% of frames  (not viable)
  front  role: metrics available in  26.2% of frames  (not viable)
```

An external camera set further back solves it outright — a Logitech BRIO at
desk distance reports every landmark between 0.91 and 1.00 and resolves all
three side metrics. That is the setup calibration should be done against.

**Camera discovery.** Brute-force index probing is unusable on Windows: opening
the same USB camera takes 1.1 s through DirectShow and 67.5 s through Media
Foundation, so scanning a handful of indices across backends looks like a hang.
Enumeration now goes through DirectShow, which is instant, never opens a device
(no privacy light, no fighting whatever is already using it), and returns real
product names. Opening only happens when something specifically asks whether a
camera works. The backend order was flipped to try DirectShow first for the
same reason.

## Layout

| file | what it holds |
| --- | --- |
| `posture/geometry.py` | pure 2D math; aspect correction and angle conventions |
| `posture/landmarks.py` | landmark indices and extraction, isolated from the rest |
| `posture/metrics.py` | per-role metrics, visibility gating, scale normalization |
| `posture/capture.py` | camera thread, backend selection, staleness handling |
| `posture/pose.py` | MediaPipe wrapper, one instance per camera |
| `posture/overlay.py` | debug window drawing |
| `posture/devices.py` | camera enumeration by name, without opening anything |
| `posture/monitor.py` | one worker thread per camera; publishes state for the UI |
| `posture/webui.py` | loopback HTTP control panel and its JSON API |
| `posture/static/index.html` | the panel shell (loads the modules below) |
| `posture/static/css/app.css` | the design system |
| `posture/static/js/store.js` | app state and the polling that feeds it |
| `posture/static/js/api.js` | every call the panel makes, in one place |
| `posture/static/js/components/` | AppShell, TopStatusBar, Sidebar, LiveCameraView, CameraTile, CameraSelector, PostureOverlay, PostureScore, MetricGauge, AlertPanel, PostureTimeline, CalibrationWizard, CameraManager, SensitivitySettings, DiagnosticsPanel |
| `posture/calibration.py` | baselines, robust statistics, tolerance derivation |
| `posture/score.py` | the single posture score, isolated so it can be retuned |
| `posture/history.py` | bounded score track and episode log, write-through to the store |
| `posture/store.py` | SQLite: samples, episodes, alerts, and day statistics |
| `posture/alerts.py` | escalation, hold-to-clear, snooze — pure logic |
| `posture/notify.py` | OS notification, with a Tk banner fallback |
| `posture/overlay_window.py` | the full-screen hold-to-clear window |
| `posture/tray.py` | the tray icon, its colours and its menu |
| `posture/detector.py` | rolling windows, hysteresis, presence, adaptive rate |
| `posture/config.py` | config dataclasses, JSON round-trip |
| `posture/cli.py` | entry points: panel, `--diagnose`, `--watch` |

## Alerts

Three steps, rising with *how long* posture has been bad rather than how bad it
is. By the time anything fires, the detector has already applied its 60-second
window and hysteresis, so this is never reacting to a twitch.

| step | at | what happens |
| --- | --- | --- |
| subtle | immediately | the tray icon turns amber; nothing interrupts you |
| notify | 30 s | one OS notification |
| overlay | 90 s | a full-screen window you have to fix your posture to close |

**Hold-to-clear is the mechanic.** The overlay does not close because posture
stopped being bad — it closes after posture has been *verifiably good* for a
continuous stretch (5 s by default), with a live countdown on screen. That
distinction matters: "not bad" includes "I cannot see you", so leaning out of
frame is not a way to dismiss it. A blind gap longer than `hold_grace` voids a
part-completed hold, because you cannot claim to have been continuously within
tolerance across a stretch nobody measured.

**The snooze is explicit and time-boxed.** There is no one-click dismiss — a
test asserts the engine exposes no such method. A snooze suppresses escalation
for a fixed period and then resumes *from nothing*, so it can never quietly
become "off forever".

**Absence cancels, blindness freezes.** An empty chair clears the alert
outright: there is nobody to nag, and an alert left running would ambush you
when you sat back down. An unreadable frame freezes the timers instead — we do
not know whether you fixed it, so we neither escalate nor hand out credit.

**The overlay shows you what it can see.** Freezing the timers is honest but,
on its own, invisible: a countdown that has stopped moving looks exactly like
one that has decided your posture is wrong, and sitting up straighter does
nothing to clear it. So the window draws the live figure — landmark positions
only, never a frame, because this window covers whatever is being screen-shared
at the time — and says which it is: *Holding, 3s to go*, *Sit back to your
calibrated posture*, or *Cannot see your right shoulder — the hold is paused
until you are back in view*. The figure is drawn from the camera whose role
owns the metric being complained about, and it keeps drawing from that camera
after it loses sight of you, since that is the one you have to get back in
front of.

**The overlay cannot trap you.** It takes itself down if nothing updates it for
20 seconds, so unplugging the camera while it is up cannot leave a screen-
blocking window that no posture reading is able to dismiss.

That was not enough, because it only covered the window going deaf. Reported
from a live session: *"very often this screen is almost impossible to get back
out of"*. It was not almost impossible, it was impossible — a torso lean
baseline of −15.9° demanded a torso reclined 8.9° past vertical before the
metric counted as in tolerance, one permanently-flagged metric pins the state to
`bad`, and the hold only advances on a `good` reading. Sitting up straight, the
thing the window was asking for, could never clear it. Snooze was the only exit.

So there are two defences now, because they fail differently. The detector
**refuses to judge you against a baseline no neutral posture can reach**, which
fixes the diagnosable case at its root and reports the metric as suspect instead
of silently dropping it — the calibration panel says which, and the only fix is
the button next to it. And the alert engine **stands the overlay down when the
hold has never once started**: not "never finished", never started, meaning not
a single good reading in ten minutes. Both conditions are needed together, and
the discriminator matters — someone who reaches a good posture even for a moment
has shown the target is reachable and is simply being asked to hold it, which is
the mechanic working. The valve is a claim about the app being wrong, never a
way to wait out the nag by sitting still and refusing.

All timings, the hold, and the snooze length live in the config and are
editable under Settings. Both effects can be switched off independently of the
escalation, for anyone who wants the panel to know without the machine
interrupting.

## The tray icon

Where the app lives once the browser tab is closed, and step one of the
escalation. Mint for good, amber for a deviation worth noticing, coral once the
overlay is up, grey for anything the app cannot or should not judge — the same
vocabulary as the panel. Nothing animates.

Its colour and its tooltip come from the same place, which took a correction:
the first version took the colour from the windowed detector state and the
words from the near-instantaneous score, and produced a mint icon whose tooltip
read "needs correction (0)". Both now follow the score band, with an active
alert overriding. A tray icon carries one pixel of information and cannot
afford to disagree with its own label.

Right-click gives Open panel, Snooze/Resume, Calibrate and Stop monitoring.
It degrades to nothing: if pystray or Pillow are missing, or the platform has
no tray, the app logs it and carries on — a missing icon must never take
posture monitoring down with it.

## Only one copy

The panel refuses to share its port. `socketserver` turns on `SO_REUSEADDR` so a
restart does not trip over a socket in `TIME_WAIT`; on Windows that same flag
lets a *second process* bind a port something else is already listening on, and
the two then split incoming connections at random. Two copies of this app did
end up on 8760 together — both driving the cameras, both writing samples to the
same database, both showing a tray icon — and `/api/stats` was answered by
whichever accepted first.

Starting twice now fails immediately with "Posture is probably already running",
and the port is claimed *before* the cameras are opened, so the second copy no
longer takes the webcams away from the first for fifteen seconds on its way out.

## One rendering rule

**Anything holding a control is built once and updated in place; only
non-interactive content is rebuilt.**

The store emits about twice a second. Components that called `mount()` on every
emit replaced their own nodes underneath the pointer, and a click starting
before an emit and finishing after it landed on an element that no longer
existed — the "button did nothing the first time" bug. The same mechanism
stranded the timeline tooltip: the hovered episode band was replaced, so its
`mouseleave` never fired.

`dom.js` exports `reconcile(container, signature, build)` for the middle case —
lists whose *shape* changes rarely but whose contents update constantly. The
sidebar, layout toggle, camera chips, camera rows and the timeline all use one
of these two approaches. Measured after the fix: the camera list rebuilds 0
times in 8 seconds, and the timeline 3 times in 12 rather than 24.

## The posture score

One number, derived from measurements the app already makes — never invented.
Per metric it blends *how far past tolerance you are right now* with *how much
of the rolling window you have spent out of it*, then takes the **worst** axis
rather than an average, matching the detector's own rule that one bad axis is
enough. Bands: 85+ excellent, 70+ good, 50+ fair, below that needs correction.

**The score may not contradict the verdict it comes from.** The blend alone did
not manage that: any ratio at or past the floor maxes the instantaneous term, so
a single wild reading cost 60 points whatever the window said. Found live — a
metric out of tolerance for half its window scored 19, *needs correction*, while
the detector's own state was `good` and no alert was firing, because the
detector will not call a metric bad until it has been out for `bad_fraction`
(70%) of the window. So the blend is capped by how much of the window supports
it, anchored to the bands the panel actually shows: with nothing in the window
behind it a deviation may dent the score but not push it out of *good*; at the
detector's own threshold it may reach the floor of *fair* but not cross into
*needs correction*; past that the detector agrees, the cap lifts, and the score
is free to bottom out. A metric the detector has flagged is never capped at all.
The cap makes the score plateau once a deviation outruns its evidence, which is
the point — how far out you are right now is worth something, but not more than
the window will vouch for.

There is deliberately **no score** when nobody is at the desk, when the
landmarks cannot be read, or before calibration. A confident 100 over an empty
chair would be a lie, so the server sends null and the panel shows why.

The whole formula lives in `posture/score.py` and nothing else has an opinion
about it, so it can be retuned without touching the pipeline.

## Posture history

Real samples only. The server keeps an hour of score track (one sample every
two seconds) and a log of episodes — continuous stretches the detector spent in
a bad state, with their offending metrics, peak deviation and duration.
Stretches under five seconds are dropped as movement rather than posture.

The ring is in memory and bounded to an hour; it is what the live timeline
polls, and a database query every two seconds would be waste. Durability is a
write-through to the store below. Gaps (away, unreadable) break the line rather
than being interpolated across.

## Stats

`posture/store.py` is a SQLite database at `~/.posture/posture.db` (WAL,
`synchronous=NORMAL`, so a write costs microseconds on the sampling thread).
Three tables:

| table | one row per |
| --- | --- |
| `samples` | score and state, every two seconds |
| `episodes` | a stretch spent in bad posture, with offenders and peak deviation |
| `alerts` | an escalation step or a clear |

Metrics only. There are no image columns and no BLOB columns anywhere, and a
test walks the live schema to keep it that way — the database arrived after
"frames never reach disk", and it must not be the thing that breaks it.

**Time is attributed across gaps, not through them.** A day is measured by
summing the interval between consecutive samples, and any interval longer than
six seconds is credited to nobody:

```
09:00 good ─ 09:10 good        →  10 minutes of good posture
09:10 good ─ 15:00 good        →  nothing; the app was shut
```

Without that rule, closing the laptop at five and opening it at nine would read
as sixteen hours of excellent posture. The same rule breaks streaks, so twenty
good minutes either side of a lunch break is not a forty-minute streak.

A day gives you good/bad/away/unknown seconds, the good percentage of *measured*
time, the longest good streak, alert and episode counts, an average score, and a
96-bucket timeline. Empty buckets are still emitted so the shape of the day
survives: one busy hour should be a bar in the morning, not a bar filling the
chart. A day with no data reports nulls rather than zeros — "no reading" and
"zero percent" are different claims.

Retention is `stats.keep_days` (90 by default); `prune()` runs at startup.
Setting it to 0 keeps everything. Deleting the file loses your history and
nothing else.

## Roadmap

1. ✅ Capture, landmarks, live metrics, debug overlay
2. ✅ Calibration, rolling-window detection, hysteresis, presence pausing
3. ✅ Alert escalation and the hold-to-clear overlay
4. ✅ Multi-camera and role assignment
5. ✅ Local web UI, config editing, tray icon, SQLite stats

Everything in the brief is built. The camera placement caveat in **Camera
setup** is the one thing the software cannot fix for you.
