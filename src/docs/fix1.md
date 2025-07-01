# ========== DRONE PROJECT REFACTOR BRIEF ==========

#

# Goal

# ----

# 1. Align the live repo with the architecture specified in prompts.md.

# 2. Eliminate the macOS keyboard-input failure.

# 3. Harden thread-safety for Hub snapshots.

#

# -------------------------------------------------------------------

# SECTION 1 · PACKAGE & DIRECTORY SHIFTS

# -------------------------------------------------------------------

#

# 1.1 Convert src/ into an installable top-level package:

# - Rename src/ → src/drone_sim/

# - Add src/drone_sim/**init**.py (can be empty)

# - Update each intra-repo import:

# FROM: `from physics.rigid_body import RigidBody`

# TO: `from drone_sim.physics.rigid_body import RigidBody`

#

# 1.2 Preserve legacy code but keep it non-importable:

# - Rename drone_project/legacy/ → drone_project/archive/

# - Remove any **init**.py files inside archive/

#

# 1.3 Create placeholders if missing:

# - src/drone_sim/ui/ (keep empty **init**.py)

# - src/drone_sim/drone/ai/ (empty **init**.py)

#

# -------------------------------------------------------------------

# SECTION 2 · SETTINGS CONSOLIDATION

# -------------------------------------------------------------------

#

# 2.1 Add src/drone_sim/config/settings.py with:

# ```python

# from pathlib import Path

# import yaml

# from dataclasses import dataclass

#

# \_CFG_DIR = Path(**file**).with_suffix('')

#

# @dataclass(frozen=True)

# class Settings:

# simulation: bool

# poll_hz: int

# # ... add every key you need

#

# def load(mode: str = 'default') -> Settings:

# with open(\_CFG_DIR / f'{mode}.yaml') as f:

# data = yaml.safe_load(f)

# return Settings(\*\*data)

# ```

#

# 2.2 Replace all direct `import yaml` reads with:

# ```python

# from drone_sim.config.settings import load

# settings = load()

# ```

#

# -------------------------------------------------------------------

# SECTION 3 · HUB & POLLER THREAD-SAFETY

# -------------------------------------------------------------------

#

# 3.1 Modify src/drone_sim/input/hub.py

# - Wrap the shared snapshot in a threading.Lock().

# - Use an immutable @dataclass called SensorFrame:

# ```python

# @dataclass(frozen=True)

# class SensorFrame:

# t: float

# controller: ControllerState

# gps: GPSFix | None

# ...

# ```

# - Hub stores ONLY the latest SensorFrame; swap pointer under lock.

#

# 3.2 Modify src/drone_sim/input/poller.py

# - Each Poller subclass **must not** sleep in .poll().

# - Poller thread loop:

# ```python

# while not stop_evt.is_set():

# frame = self.poll()

# hub.update(frame)

# stop_evt.wait(dt) # dt = 1/settings.poll_hz

# ```

#

# -------------------------------------------------------------------

# SECTION 4 · KEYBOARD INPUT (MAC FIX)

# -------------------------------------------------------------------

#

# 4.1 Replace keyboard backend:

# - Delete src/drone_sim/input/controllers/keyboard.py

# - New file src/drone_sim/input/controllers/keyboard.py :

# ```python

# from pynput import keyboard

# class KeyboardController:

# def **init**(self):

# self.\_state = set()

# listener = keyboard.Listener(

# on_press=self.\_on, on_release=self.\_off)

# listener.start() # NON-BLOCKING, main thread

# def \_on(self, key): self.\_state.add(key)

# def \_off(self, key): self.\_state.discard(key)

# def poll(self): return frozenset(self.\_state)

# ```

#

# 4.2 macOS requires Accessibility perms; add a warning log:

# ```python

# import platform, logging, sys

# if platform.system() == 'Darwin':

# logging.warning(

# "Grant Accessibility permission to %s for keyboard capture",

# sys.argv[0]

# )

# ```

#

# 4.3 Ensure keyboard listener starts in **main** thread.

# In simulator.py before any threads:

# ```python

# from drone_sim.input.controllers.keyboard import KeyboardController

# controller = KeyboardController()

# ```

#

# -------------------------------------------------------------------

# SECTION 5 · DRONE MODULE REFACTOR

# -------------------------------------------------------------------

#

# 5.1 Split src/drone_sim/drone/main.py into:

# - controller.py (position → attitude PID stack)

# - mixer.py (torque → per-motor throttle)

# - planner.py (task / mission manager)

#

# 5.2 Each sub-file exposes a class with `.step(sensor_frame, dt)`.

#

# -------------------------------------------------------------------

# SECTION 6 · UNIT TEST GUARD

# -------------------------------------------------------------------

#

# 6.1 Update run_tests.py to:

# - `pytest -q src/drone_sim`

# - Ensure at least one test verifies KeyboardController.poll()

#

# -------------------------------------------------------------------

# Done. The tool can now implement the edits file-by-file.

#

# ================================================================
