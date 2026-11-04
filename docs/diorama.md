# Diorama

Diorama shows the drive as a live 3-D miniature: the road around the car, the
lane lines and road edges, the planned path, and the car ahead, the way the
driving model sees them. It is all computer-drawn; no camera footage appears in
it.

<a href="images/diorama.webp"><img src="images/diorama.webp" width="100%" alt="A recorded drive in the Diorama viewer, each moment beside its road camera: above, the road view from behind the car at 18 mph, the planned path ending at the lead car in blue 15 m ahead; below, the bird's-eye view at 8 mph in traffic, labelled BIRD'S-EYE · TRAFFIC, the lead 11 m ahead"></a>

You can watch it:

- **in a browser,** from a Jetlink server on Linux or a Mac;
- **in the Mac app,** in the **Diorama** view;
- **on an iPhone,** in the **Diorama** tab, from a Mac nearby or the phone's own server;
- **for a recorded drive,** on any computer, beside the drive's own footage.

The view follows behind the car while it cruises, and pans up to a bird's-eye
view when it stops, turns, nears a crosswalk or sits in traffic. The label in
the corner says which, and why. The road view is drawn from the narrow camera,
which sees furthest ahead; bird's-eye from the wide camera, which sees the
road all around the car. Everywhere Diorama runs, live or from a recording,
it takes the cameras' calibration from the comma: until the comma has
calibrated (its first drive), Diorama waits and says so.

Diorama needs a segmentation model, about 180 MB (90 MB as the Mac's CoreML
version). It runs on the GPU only while someone watches, and steps aside
whenever the driving model needs the GPU.

**Contents:** [Linux](#linux) · [Mac](#mac) · [iPhone](#iphone) ·
[A recorded drive](#a-recorded-drive) · [Set the camera](#set-the-camera) ·
[Options](#options) ·
[Known limits](#known-limits) · [Credits](#credits)

## Linux

An NVIDIA GPU with TensorRT, as for Jetlink itself ([Linux PC](linux-pc.md),
[Jetson](jetson.md)).

### Set up

Once. This needs [uv](https://docs.astral.sh/uv/), `git` and `ffmpeg`, and a
checkout of Jetlink, which the replay and the commands below run from:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh     # uv, if you don't have it
sudo apt install git ffmpeg
git clone https://github.com/zoompilot/jetlink ~/jetlink && cd ~/jetlink
uv venv && uv pip install --python .venv -e . onnx onnxruntime opencv-python pycapnp zstandard
```

Then fetch the model:

```bash
uvx --from huggingface_hub hf download ernestchu/jetlink-diorama \
  --include diorama_seg.onnx --include diorama.json --local-dir ~/diorama-models
```

### Start and stop the server

**If Jetlink is installed** (the [installer](linux-pc.md)), run it in a
terminal with Diorama added. This pauses the background service while it runs:

```bash
jetlink run --listen --status-port 8080 --diorama ~/diorama-models
```

- **Stop:** Ctrl-C. Then `jetlink start` brings the background service back.
- `--listen` also takes a replay over the network; leave it out to serve only
  the comma on USB.

**Without the installer,** build the server from the checkout once (this needs
Docker), and on a PC fetch TensorRT 11.3's libraries from NVIDIA's wheel:

```bash
cd ~/jetlink
scripts/build-linux.sh linux-x86_64          # linux-aarch64 on a Jetson
tar -xzf dist/jetlink-server-*-linux-x86_64.tar.gz
wheel=tensorrt_cu13_libs-11.3.0.99-py3-none-manylinux_2_28_x86_64.whl   # not on a Jetson
curl -fLO "https://pypi.nvidia.com/tensorrt-cu13-libs/$wheel"
unzip -j "$wheel" 'tensorrt_libs/lib*.so*' -x '*_win_*' -d ~/tensorrt-11.3
```

Then run it:

```bash
cd ~/jetlink
jetlink-server-*-linux-x86_64/bin/jetlink-server --listen --status-port 8080 \
  --backend trt --tensorrt-libs ~/tensorrt-11.3 --diorama ~/diorama-models
```

- **Stop:** Ctrl-C.
- `--tensorrt-libs` names TensorRT 11.3's libraries (an installed PC has them
  in `/opt/jetlink/tensorrt/11.3.0.99`). On a Jetson, leave it out: TensorRT
  comes with JetPack. The wheel is 3.8 GB and its libraries 2.6 GB; delete the
  wheel once they're out.
- The server keeps the models and their prepared engines in
  `~/.cache/jetlink`, about 2.4 GB with Diorama.

**The first time** a page opens Diorama, the server prepares the model for
your GPU: a few minutes on an RTX 4090, longer on a Jetson. Until then the page
shows the road and path without the model's detail.

### Feed it a replay

To try it at a desk without the car, play a recorded drive into the running
server as the comma would. You need the drive's segment folders (each with
`fcamera.hevc`, `ecamera.hevc` and `rlog.zst`, from the comma's
`/data/media/0/realdata`) and a driving model ONNX ([catalog](models.md)). In a second terminal, from the
checkout:

```bash
cd ~/jetlink
.venv/bin/python -m diorama replay ~/drives/<route>--{3,4,5} \
  --model ~/models/big_driving_supercombo.onnx --host 127.0.0.1 --loop
```

- **Stop:** Ctrl-C. Run it again to start over.
- **The first time,** the server prepares the driving model and Diorama's
  before either shows anything, so a short first replay can end before the
  page fills in. `--loop` keeps it going; or run it again.
- **Which part:** list the segment folders to play, in order.
  `--start 20 --seconds 30` plays 30 s from 20 s in.
- **Once:** leave out `--loop` to stop at the end.
- **Another computer:** give its address as `--host`. Its server needs `--listen`.
- **Calibration:** each segment's comes from its `rlog.zst`, as the comma
  logged it: nothing is estimated or assumed. A segment without one, or from
  before the comma had calibrated, is refused, saying which.

### Watch it

Open **`http://<host>:8080/diorama/`**: `http://localhost:8080/diorama/` on
the same computer. From a laptop, you can tunnel to a headless box first with
`ssh -L 8080:localhost:8080 <host>`, then open the localhost address.

- **Model input,** beside the timings, shows the camera frame the model sees,
  with what it found coloured over it.
- **Perception** is how often the model runs (10 Hz, or 20 with
  `--diorama-every-frame`).
- With a [web password](control-protocol.md#the-web-page) set, the page asks
  for it, as the status page does.

## Mac

A Mac with Apple silicon and Xcode 26.

### Set up

Once. This needs [Homebrew](https://brew.sh), and builds the app from a
checkout of Jetlink, which the replay also runs from:

```bash
brew install uv xcodegen ffmpeg
git clone https://github.com/zoompilot/jetlink ~/jetlink && cd ~/jetlink
make -C macos app                            # builds macos/build/Jetlink.app
uv venv && uv pip install --python .venv -e . onnx onnxruntime opencv-python pycapnp zstandard
```

Copy `macos/build/Jetlink.app` to **Applications** if you like; the commands
below use it where it was built.

### In the app

1. Open **Jetlink** (`open ~/jetlink/macos/build/Jetlink.app`). Its server
   starts with the app.
2. Click **Diorama** in the sidebar (⌘5), then **Download Models…** and
   **Agree and Download**. It fetches about 90 MB once, and the server picks
   it up without a restart.
3. With the comma connected, **Diorama** shows the drive.

**Stop:** quit Jetlink. **Restart Server**, in **Settings**, restarts it.

### The camera

**Settings > General > Diorama > Set Up Camera…** opens the **Diorama Camera**
window. It sets one view at a time: pick **Road view** or **Bird's-eye** at the
top.

The camera is set over a scene of its own, the same whether or not a comma is
connected or a drive plays: the ego car, a lead 10 m ahead, the planned path
stopping 2 m short of it, and the ego lane's two lines. Nothing from the
segmentation.

- **Zoom:** pinch on the trackpad, or turn the mouse wheel.
- **Pitch,** the camera up and down over the car: click and drag, down to see
  it from higher.
- **Forward,** the camera along the road: two fingers on the trackpad, or drag
  with the right button. Down moves it forward, up back.

**Reset View** brings back the view's own camera. **Save** keeps it for every
Diorama view the Mac draws, and **Cancel** drops it. The Mac keeps a camera of
its own, apart from the [web page's](#set-the-camera).

### Feed the app a replay

The app serves the comma over USB. To play a recorded drive into it instead,
quit Jetlink and open it listening on the network:

```bash
~/jetlink/macos/build/Jetlink.app/Contents/MacOS/Jetlink -developerTransport tcp -tcpPort 5599
```

You need the drive's segment folders (each with `fcamera.hevc`,
`ecamera.hevc` and `rlog.zst`, from the comma's `/data/media/0/realdata`) and a
driving model ONNX ([catalog](models.md)). In a second terminal, from the checkout, play the
drive into the app and watch it in **Diorama**:

```bash
cd ~/jetlink
.venv/bin/python -m diorama replay ~/drives/<route>--{3,4,5} \
  --model ~/models/big_driving_supercombo.onnx --host 127.0.0.1 --port 5599 --loop
```

- **Stop:** Ctrl-C stops the replay. Quit Jetlink (Ctrl-C in its terminal) and
  open it normally to go back to USB.
- `--start`, `--seconds` and `--loop` work as on [Linux](#feed-it-a-replay),
  and the calibration comes from the drive's `rlog.zst` the same way.

### From a terminal

The server alone, without the app, watched in a browser. From the checkout:

```bash
cd ~/jetlink
swift build -c release --package-path JetlinkKit --product jetlink-server
uvx --from huggingface_hub hf download ernestchu/jetlink-diorama \
  --include "diorama_seg.mlpackage/*" --include diorama.json --local-dir ~/diorama-models
JetlinkKit/.build/release/jetlink-server --listen --status-port 8080 --diorama ~/diorama-models
```

In a second terminal, play a drive into it, on the server's own port:

```bash
cd ~/jetlink
.venv/bin/python -m diorama replay ~/drives/<route>--{3,4,5} \
  --model ~/models/big_driving_supercombo.onnx --host 127.0.0.1 --loop
```

Then open `http://localhost:8080/diorama/`. **Stop:** Ctrl-C in each terminal.

## iPhone

The iPhone app's **Diorama** tab can show the drive two ways, picked at its top:

- **Mac:** a Mac nearby runs Jetlink, the comma and the model; the iPhone only
  shows the drive. This is the one for a car mount.
  1. On the Mac, turn on **Settings > General > Diorama > Show on iPhones nearby**.
  2. On the iPhone, open **Diorama** and pick **Mac**. It finds the Mac by
     itself, with no network needed between them: both just need Wi-Fi on. The
     iPhone asks once for Local Network access.

  The Mac's Diorama toolbar shows who is watching, and can disconnect them.
- **This iPhone:** the phone's own server, with the comma on its cable, runs
  the model too. Tap **Turn On** in the tab to download it (90 MB). It uses
  the phone's GPU beside the driving model.

The screen stays on while the tab shows the drive.

### The camera

**Settings > Diorama > Set Up Camera…** sets the camera as on a
[Mac](#the-camera), over the same scene of its own, whether or not the tab
shows a drive.

- **Zoom:** pinch.
- **Pitch:** drag with one finger.
- **Forward:** drag with two fingers, down to move forward.

On an iPad with a trackpad or mouse, two fingers scrolling and a drag with the
right button move it forward and back, and a mouse wheel zooms. An iPhone watching a Mac
draws with its own camera, not the Mac's.

## A recorded drive

To look at a drive beside its own footage, on any computer and without a
server, process it once and open the viewer. From the checkout, set up as for
[Linux](#set-up) or a [Mac](#set-up-1):

```bash
cd ~/jetlink
uv pip uninstall --python .venv onnxruntime && uv pip install --python .venv "onnxruntime-gpu[cuda,cudnn]"   # once, on Linux only
.venv/bin/python -m diorama run ~/drives/<route>--{3,4,5} --model ~/models/big_driving_supercombo.onnx \
  --models ~/diorama-models --out ~/diorama-drive
.venv/bin/python -m diorama view ~/diorama-drive
```

The viewer opens in your browser, with play, pause and seek (Space plays and
pauses). The segment folders need their `rlog.zst`, as for a replay.
`--start` and `--seconds` pick a part of the drive, and `--device cuda`,
`coreml` or `cpu` where it runs. `run` segments both cameras, so the viewer
switches between them with the view, as live.

On a Mac, `run` takes the ONNX model, not the app's CoreML one. Fetch it as on
Linux:

```bash
uvx --from huggingface_hub hf download ernestchu/jetlink-diorama \
  --include diorama_seg.onnx --include diorama.json --local-dir ~/diorama-models
```

## Set the camera

Each view, the road view and bird's-eye, has a camera you can set in the
browser, saved to a file. From the checkout, set up as for [Linux](#set-up):

```bash
.venv/bin/python -m diorama camera
```

The page opens with a panel for the camera. Pick **Road view** or
**Bird's-eye**, and the page holds that view while you set it. It draws a
scene of its own, with no drive or footage needed: the ego car, a lead 10 m
ahead, the planned path stopping 2 m short of it, and the ego lane's two lines.
Nothing from the segmentation.

| | Trackpad | Touchscreen | Mouse |
| --- | --- | --- | --- |
| **Zoom** | Pinch | Pinch | Wheel |
| **Pitch,** the car tilting toward you | Click and drag | Drag | Drag with the left button |
| **Forward,** the camera along the road | Two fingers | Two fingers | Drag with the right button |

- Drag down to see the car from higher. Pitch turns the camera about the car,
  and zoom scales the picture about it, so neither moves the car on screen.
- Two fingers or the right button down move the camera forward, up back, as
  if pulling the road under it: level with the road, at any pitch.
- **Reset view** brings back the view's own camera. **Save** writes both views
  to `~/.config/jetlink/diorama-camera.json` (`--file` for another).
- `view` draws with the saved camera. `camera --show` prints it, and
  `camera --reset` deletes it.
- **The live page** uses it when the server starts with
  `--diorama-camera ~/.config/jetlink/diorama-camera.json`. Reopen the page
  after a save.
- The file is the web page's. The Mac and iPhone apps keep their own camera,
  set in their settings ([Mac](#the-camera), [iPhone](#the-camera-1)).

## Options

| Option | What it does |
| --- | --- |
| `--diorama DIR` | Turns Diorama on, with the model in DIR. |
| `--diorama-every-frame` | Runs the model on every frame (20 Hz) instead of every other frame, for about twice the GPU time. An RTX 4090 does it easily. |
| `--diorama-in-flight N` | Runs N at once; only for a GPU too slow to keep up with every frame alone. |
| `--diorama-pause-ms MS` | Pauses Diorama for 3 s after a driving frame slower than this (40 by default). |
| `--diorama-camera FILE` | Draws the page with the camera `python -m diorama camera` saved ([above](#set-the-camera)). |
| Mac: **Settings > General > Diorama** | The model's folder, **Segment every frame (20 Hz)**, **Show on iPhones nearby**, and **Set Up Camera…**. |

## Known limits

- **Only the car ahead** is drawn, always as the same car, facing along its
  lane. It can jitter, and blink on and off when the car is far or the driving
  model is unsure.
- **When the car ahead changes lanes,** it disappears for a second or two,
  until the model settles on the next car.
- **Around a switch of view,** a segmentation from the other camera may be
  dropped, a tenth of a second without new detail.
- All lane lines are white.
- **Setting the camera in a browser,** a mouse whose wheel scrolls smoothly,
  with no notches, moves the camera forward instead of zooming. Pinch
  instead.

## Credits

- **Segmentation:** Mask2Former by Meta, ResNet-50, from its model zoo (model
  49189528_1). Its code is MIT; its weights are under
  [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/), and were
  trained on Mapillary Vistas, whose terms allow non-commercial use only.
- **The car:** ["2021 Tesla Model Y"](https://sketchfab.com/3d-models/2021-tesla-model-y-59e2ead369984b1a85c800ff6cf6789d)
  by tonielpro520, under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/),
  recoloured, without its badges, interior and brakes.

How it works, its costs and its messages: [Diorama internals](diorama-internals.md).
