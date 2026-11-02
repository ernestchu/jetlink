# Diorama

Diorama shows the drive as a live 3-D miniature: the road around the car, the
lane lines and road edges, the planned path, and the car ahead, the way the
driving model sees them. It is all computer-drawn; no camera footage appears in
it.

<a href="images/diorama.webp"><img src="images/diorama.webp" width="100%" alt="A recorded drive in the Diorama viewer: the road camera on the left; on the right, Diorama's road view from behind the car, the planned path ending at the lead car in blue 15 m ahead, with the lane lines, road edges and sidewalks"></a>

You can watch it:

- **in a browser,** from a Jetlink server on Linux or a Mac;
- **in the Mac app,** in the **Diorama** view;
- **on an iPhone,** in the **Diorama** tab, from a Mac nearby or the phone's own server;
- **for a recorded drive,** on any computer, beside the drive's own footage.

The view follows behind the car while it cruises, and pans up to a bird's-eye
view when it stops, turns, nears a crosswalk or sits in traffic. The label in
the corner says which, and why.

Diorama needs a segmentation model, about 900 MB (510 MB as the Mac's CoreML
version). It runs on the GPU only while someone watches, and steps aside
whenever the driving model needs the GPU.

**Contents:** [Linux](#linux) · [Mac](#mac) · [iPhone](#iphone) ·
[A recorded drive](#a-recorded-drive) · [Options](#options) ·
[Known limits](#known-limits) · [Credits](#credits)

## Linux

An NVIDIA GPU with TensorRT, as for Jetlink itself ([Linux PC](linux-pc.md),
[Jetson](jetson.md)).

### Get the model

Once. This needs Python and the Hugging Face CLI:

```bash
python3 -m pip install -U "huggingface_hub[cli]"
hf download ernestchu/jetlink-diorama --include diorama_seg.onnx --include diorama.json \
  --local-dir ~/diorama-models
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

**From a checkout,** build the server once, then run it:

```bash
scripts/build-linux.sh linux-x86_64          # linux-aarch64 on a Jetson
tar -xzf dist/jetlink-server-*-linux-x86_64.tar.gz
jetlink-server-*-linux-x86_64/bin/jetlink-server --listen --status-port 8080 \
  --backend trt --tensorrt-libs /opt/jetlink/tensorrt/11.3.0.99 --diorama ~/diorama-models
```

- **Stop:** Ctrl-C.
- `--tensorrt-libs` names TensorRT 11.3's libraries (an installed PC has them
  there; see [installing by hand](installation-reference.md#installing-by-hand)
  otherwise). On a Jetson, leave it out.

**The first time** a page opens Diorama, the server prepares the model for
your GPU: a few minutes on an RTX 4090, longer on a Jetson. Until then the page
shows the road and path without the model's detail.

### Feed it a replay

To try it at a desk without the car, play a recorded drive into the server as
the comma would. You need a checkout, `ffmpeg`, the drive's segment folders
(each with `fcamera.hevc` and `ecamera.hevc`, from the comma's
`/data/media/0/realdata`) and a driving model ONNX
([catalog](models.md)). In the checkout's top folder:

```bash
python3 -m venv .venv && source .venv/bin/activate
python3 -m pip install -e . onnx onnxruntime opencv-python      # once
python3 -m diorama replay ~/drives/<route>--{3,4,5} \
  --model ~/models/big_driving_supercombo.onnx --host 127.0.0.1 --loop
```

- **Stop:** Ctrl-C. Run it again to start over.
- **Which part:** list the segment folders to play, in order.
  `--start 20 --seconds 30` plays 30 s from 20 s in.
- **Once:** leave out `--loop` to stop at the end.
- **Another computer:** give its address as `--host`. Its server needs `--listen`.
- **Calibration:** the replay estimates the camera's from the drive's first
  minute. `--rpy ROLL PITCH YAW` (radians) sets it instead, and starts sooner.

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

### In the app

1. Open **Jetlink**. Its server starts with the app.
2. Click **Diorama** in the sidebar (⌘5), then **Download Models…** and
   **Agree and Download**. It fetches about 510 MB once, and the server picks
   it up without a restart.
3. With the comma connected, **Diorama** shows the drive.

**Stop:** quit Jetlink. **Restart Server**, in **Settings**, restarts it.

### Feed the app a replay

The app serves the comma over USB. To play a recorded drive into it instead,
quit Jetlink and open it listening on the network:

```bash
/Applications/Jetlink.app/Contents/MacOS/Jetlink -developerTransport tcp -tcpPort 5599
```

Then, from a checkout ([Linux](#feed-it-a-replay) has the setup), play the
drive into it and watch it in **Diorama**:

```bash
python3 -m diorama replay ~/drives/<route>--{3,4,5} \
  --model ~/models/big_driving_supercombo.onnx --host 127.0.0.1 --port 5599 --loop
```

**Stop:** Ctrl-C stops the replay. Quit Jetlink and open it normally to go
back to USB.

### From a terminal

The server alone, watched in a browser:

```bash
swift build -c release --package-path JetlinkKit --product jetlink-server
hf download ernestchu/jetlink-diorama --include "diorama_seg.mlpackage/*" --include diorama.json \
  --local-dir ~/diorama-models
JetlinkKit/.build/release/jetlink-server --listen --status-port 8080 --diorama ~/diorama-models
```

Feed it a replay as on [Linux](#feed-it-a-replay), and open
`http://localhost:8080/diorama/`. **Stop:** Ctrl-C.

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
  the model too. Tap **Turn On** in the tab to download it (510 MB). It uses
  the phone's GPU beside the driving model.

The screen stays on while the tab shows the drive.

## A recorded drive

To look at a drive beside its own footage, on any computer and without a
server, process it once and open the viewer. In the checkout's top folder, with
the [replay's setup](#feed-it-a-replay):

```bash
python3 -m pip install onnxruntime-gpu      # once; on a Mac, the onnxruntime you have
python3 -m diorama run ~/drives/<route>--{3,4,5} --model ~/models/big_driving_supercombo.onnx \
  --models ~/diorama-models --out ~/diorama-drive
python3 -m diorama view ~/diorama-drive
```

The viewer opens in your browser, with play, pause and seek (Space plays and
pauses). `--start` and `--seconds` pick a part of the drive, and `--device cuda`,
`coreml` or `cpu` where it runs.

## Options

| Option | What it does |
| --- | --- |
| `--diorama DIR` | Turns Diorama on, with the model in DIR. |
| `--diorama-every-frame` | Runs the model on every frame (20 Hz) instead of every other frame, for about twice the GPU time. An RTX 4090 does it easily. |
| `--diorama-in-flight N` | Runs N at once; only for a GPU too slow to keep up with every frame alone. |
| `--diorama-pause-ms MS` | Pauses Diorama for 3 s after a driving frame slower than this (40 by default). |
| Mac: **Settings > General > Diorama** | The model's folder, **Segment every frame (20 Hz)**, and **Show on iPhones nearby**. |

## Known limits

- **Only the car ahead** is drawn, always as the same car, facing along its
  lane. It can jitter, and blink on and off when the car is far or the driving
  model is unsure.
- **When the car ahead changes lanes,** it disappears for a second or two,
  until the model settles on the next car.
- **A narrow view:** the model sees only the narrow camera, so curbs, sidewalks
  and crosswalks to the side fill in patchily.
- All lane lines are white.

## Credits

- **Segmentation:** Mask2Former by Meta (MIT). Its weights were trained on
  Mapillary Vistas, whose terms allow non-commercial use only.
- **The car:** ["2021 Tesla Model Y"](https://sketchfab.com/3d-models/2021-tesla-model-y-59e2ead369984b1a85c800ff6cf6789d)
  by tonielpro520, under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/),
  recoloured, without its badges, interior and brakes.

How it works, its costs and its messages: [Diorama internals](diorama-internals.md).
