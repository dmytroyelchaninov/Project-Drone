Whole project is separated by drone_project/legacy/ and drone_project/src/
archive/ is older version and longer updated, but it contains useful scripts and files which could be partially reworked and reused in src/.
src/ contains actual project directory.

_Structure of src/:_
setup.py
simulator.py
real.py
run_tests.py

1. cfg/
   settings.yaml
   default.yaml
   user.yaml
   emulation.yaml
2. docs/
   project documentation
3. input/
   hub.py
   poller.py
   3.1. controllers/
   joystick.py
   keyboard.py
   3.2 sensors/
   all sensor obj like Gps, Barometer, etc
4. physics/
   all physics module related files
5. drone/
   main.py - handles transformation of unified input from sensors/devices (listens HUB) and writes the output voltage to HUB. Depending on mode, task, go, may be interrupted by AI (so input.device will be ignored) and utilizing sensors, complete task, or even over take the control of device by AI in case of danger situation. Might be directory instead of single file, because there are a lot of components.
   5.1. ai/
   ai related stuff to avoid obstacles, travel, follow and other situations
6. logs/
   logs from different sources
7. tests/
8. ui/
   transmission.py
   emulator.py
   etc

_*BLOCKS*_ (either separate directories or files, depending on how big block is)

**Physics**
SINGLETON
Module containing classes and methods which helps to calculate something accordingly to physics law. Utilized both by Transmission and Drone's AI (to predict some effects based on that).

**Hub**
SINGLETON
Run in separate thread, so it can do its job without interferring with other processes, and if some process fails - hub still works.
imports settings.
Initializes with:
self.simulation=settings.SIMULATION (true/false)
self.frequency=settings.GENERAL.POLL_FREQUENCY, integer
self.device=settings.CURRENT_DEVICE, dictionary
self.sensors=settings.AVAILABLE_SENSORS, list of dictionaries
self.input={
gps: list, [lat, long, height]
barometer: float, in pascals
gyroscope: list, [angular_velocity, axes_of_rotation]
temperature: float, in Kelvins
anemometer: list, [wind_speed, wind_direction]
compass: ,
cameras: list, [camera_1_obj, ... ]
},
self.output=[], list for voltage for each engine.
self.task=(None, take_off, land, follow, back_to_base, projectile - available presets), - will talk about them later
self.mode=(ai, hybrid, manual - available choices),
self.go=(off, operate, idle, float - available choices).
Typical CURRENT_DEVICE = {
type: 'keyboard'
obj: each device object (currently joystick and keyboard planned) has it's own polling function logic, but any must have .poll() method. Connects to keyboard. If joystick device type, connects to joystick. Obj can change if programm running in real or simulator mode.
}
Typical AVAILABLE_SENSORS = [
'gyroscope': {
fake: bool,
obj: Obj, - each sensor object has own polling function, like device. In simulator settings contain obj which will fake info based on some logic (also taken from Hub)
...
},
'temperature': {
....
},
'gps':{
...
}
...
]

Hub serves as independent data storage, That's our database. Better to keep all "running" logic outside from this object.
Collects input data from external things (device, sensors) and output data from drone. Loggs them in debug mode.
The input has two big parts, regularly polled infos about device(controller), sensors.
Output from drone normally contains list of voltage for each engine. Drone considers "first" engine is the left, from 1st view sight and then clockwise.
Hub is SINGLETON and created only once initially, and everywhere always updated in place so each consumer of Hub data receives updates instantly.

**Environment(Hub)**
SINGLETON

import Map
Instantiated for both real and simulation options.
Init:
self.drone_position = self.get_position(hub.input.sensors.get(gps), hub.input.sensors.get(barometer))
self.map = Map.get_map(drone_position) - obj, has both offline map or online api method. Both methods receive the self.drone_position. Caches some square around the drone. Map should have the attitude above the sea level as well. This will allow to calculate height above ground using drone_position[2] and the current ground attitude in the point where drone is.
core methods:
.get_position(gps, barometer)- usually gps attitude is very inaccurate so we take only lat long from gps, height is absolute by barometer (height above the sea level). list with [lat, long, height].
.get_map(position) - obj, has both offline map or online api method. Both methods receive the self.drone_position. Caches some square around the drone. Map should have the attitude above the sea level as well. This will allow to calculate height above ground using drone_position[2] and the current ground attitude in the point where drone is.
.scan_obstacles(cameras=[] from hub.input.sensors.camera) - obj. Depends on settings.SIMULATION. Anyway, simulation or not, it should return preprocessed 3 dimension tensor (matrix), which contains obstacles info. Matrix can only contain numbers between 0 and 1. Where 0 means at this point 100% no obstacle, 1 - obstacle part for sure (have you played minesweeper?). This will allow to reduce storage of the environment - we can cut from memory all 0 (like we do with massive data matrices for machine learning). Scan obstacles() utilizes live recognition of objects arround using hub.input.sensors.cameras live data from camera, (during simulation we feed as camera some fake data), classifying them as well.

**Poller(Hub)**
Poller might need methods .start(), .stop(), .force_poll(), .test()
Utilizes Hub.device.obj.poll() and for each item in Hub.sensors - item.obj.poll() to get information. Updates Hub.input.
Takes information from device polling and finds "mode", "go" and updates Hub with that. User can change mode and go from device (keyboard/joystick).
ANY Hub updates are done "in place" and async in real time, accordingly to frequency.
Has separate force_poll to avoid queue.
On error writes critical log, updates Hub.mode='ai', Hub.go='float', Hub.task=None and iteratively tries poll again then once 20ms for 3 minutes. If still no response - writes to Hub: mode='ai', go='operate', task='back_to_base'.

**Drone(Hub, Environment)**
Whole Drone object idea - take some Hub info (input and it's attributes), and write to Hub output list of voltage for each engine.
Here is how it does this.
Drone always listens Hub.input, Hub.task, Hub.mode, Hub.go, if anything changes, it changes it's parameters or even logic. During logic may ask environment.scan_obstacles to get matrix of surrounding obstacles, or .get_position or other methods.
Let's break down the main trees.

**Transmission(Hub, Environment, Physics)**
Init:
self.drone_cfg = settings.DRONE
Immediatory between Hub and Emulator. Used mostly by emulator to create 3d view from first sight (drone's sight) including drone motion, render environment. Hardly utilizes Physics module to make motion correct. Takes data from Hub.output, which is just list of voltage levels per each engine. Then using drone_cfg finds the real rpm, thrust etc. So all physics happens here. But sometimes transmission is used by AI.
Core methods:
.render_environment()
.render_drone()

**Emulator(Transmission)**
(settings.SIMULATION=True)
Through transmission renders drone, ground, obstacles, useful elements for UI (eg drone level sensor info, drone trajectory, tasks etc)

**RealEngines(hub)**
(settings.SIMULATION=False)
Reads from hub.output and changes voltage for each real engine hardware accordingly.

_*EXECUTABLES*_

**Simulator** (src/simulator)
One of the main executable files. (along with src/real)
python src/simulator --training (or without training flag).
Training flag will be later implemented for AI training - it will launch simulation but without UI and with training settings, so AI can utilize full power of videochip to train model (used in hybrid and AI modes)
Flow without training flag:

1. Creates basic UI, suggests user to set up initial settings. All settings have some default mode. Such as connect joystick (default is keyboard), select Drone config (propellers, battery, engines, cargo, sizes, sensors), select Environment config (obstacles, gravity value). If user clicks start - set settings (also set settings.SIMULATION=True), and starts loading.
   Loading Phase:
   Instantiates: Hub, Environment(Hub), Poller(hub), Drone(hub, environment), Emulator(transmission), Transmission(hub, environment, Physics),
   Tests Poller if it's receiving data from sensors/device. Just one time poll, to check if everything is correct.
