# Diorama internals

How [Diorama](diorama.md) works, what it costs, how to make and publish its
model, and the messages the server, the page and the apps share. For using it,
see [Diorama](diorama.md).

## How it works

The comma already sends the server two things each frame: the warped narrow
and wide camera images and the driving model's inputs. The server answers with
the model's outputs. Diorama adds one model that runs on a warped image:
Mask2Former (Swin-L, Mapillary Vistas panoptic), on every other frame (10 Hz).
It segments the road surface, paint, crosswalks, curbs and sidewalks.

It segments the narrow image in the road view, which sees furthest ahead, and
the wide image in bird's-eye, which sees the road close around the car: the
crossings and corners of an intersection, and the sidewalks either side. The
server keeps the same scene the watchers make, to know which view they show,
and the scene uses a segmentation only in its camera's view. The wide image's
pixel covers as much road as the narrow's twice as far, so it fills the road
map to 20 m ahead rather than 40.

The geometry is the comma's own, never estimated or assumed. Its warp has
undone the camera's calibration (`rpyCalib`) for both images, but the wide
camera sits at angles of its own from the device, fixed by how the unit was
built, and the road is as far below as the camera's height. calibrationd
keeps both (`wideFromDeviceEuler` and `height`, saved in
`CalibrationParams`), and the comma's hello carries them as `camera`. Until
it does (a comma not calibrated yet), the server's Diorama waits and says so.
A replay and `diorama run` read the same from each segment's `rlog.zst`
(`diorama/rlog.capnp` is the slice of cereal's schema it needs), warp each
segment with its logged `rpyCalib`, and refuse a segment without one.
`diorama run` segments both images every other frame, and the page takes the
one for its view, as live.

The page does the rest, in the browser, and the apps do the same natively:

- It builds a road map around the car from the segmentation.
- It draws the lead car from each driving-model frame on its own: a car with
  its rear at the model's lead distance and offset, facing along its lane,
  with brake lights when the model predicts it braking. It is drawn only when
  both agree there is one:
  - the model's lead probability is over 50%, and
  - the newest segmentation, at most half a second old, has a vehicle at the
    lead's spot in the frame, standing on the road within 25% (or 3 m) of the
    model's distance.

  Without a segmentation (while it is prepared, or paused after a slow
  driving frame) no lead is drawn. Nothing is tracked or smoothed between
  frames, and no other cars are drawn.
- It draws the scene in WebGL, and the apps draw it in Metal
  ([below](#the-apps-draw-it-natively)).

The view sits behind the car while it cruises, and pans to a bird's-eye view
when the scene asks for a look around. The page names the reason beside the
view:

- **Stopped**, below 3 mph, or **slow**, below 12 mph.
- **Turn:** the planned path bends more than 20° within the next 40 m and the
  plan slows below 22 mph for it, or bends more than 45° (below 45 mph either
  way). The view pans as the plan starts slowing for the corner, and stays on
  the road through a ramp or bend taken at speed.
- **Crossing:** a crosswalk ahead within 2.5 s (at least 10 m), going below
  20 mph or with the plan slowing below 15 mph before it: not one driven
  straight over.
- **Traffic:** a lead within 15 m, below 25 mph.

A cue has to hold for 0.4 s before the view pans, and the road view comes back
only after 3 s without one and above 18 mph. Either view stays for 3 s at
least, so it never flickers between the two. `VIEW` in diorama-engine.js holds
the numbers.

**The comma comes first:**

- The server hands Diorama the frame only after the reply has gone to the comma.
- The models run only while a page, the Mac view or an app's Diorama tab is open.
- They pause for 3 s after any driving frame slower than 40 ms
  (`--diorama-pause-ms`), since they share the GPU with the driving model.
- They segment every other frame (10 Hz). `--diorama-every-frame` segments
  every frame (20 Hz), for about twice the GPU time; `--diorama-in-flight 2`
  runs two at once, for a GPU too slow to do one per frame alone. The Mac
  app's **Segment every frame (20 Hz)** setting is both.

## Export the model

Once, on a computer with Python 3.10+, PyTorch, `transformers` and `onnx`
(a GPU is not needed):

```bash
python3 -m pip install torch transformers onnx
python3 -m diorama export --out ~/diorama-models
```

This writes `diorama_seg.onnx` (910 MB) and `diorama.json`.

**Or download it** from the export the Mac app uses, without PyTorch. A
Jetson or Linux PC needs only the ONNX graph, not the CoreML package, and each
name takes its own `--include`:

```bash
hf download ernestchu/jetlink-diorama --include diorama_seg.onnx --include diorama.json --local-dir ~/diorama-models
```

A folder from an older export that also has DD3D still works; its DD3D graph
is not run.

**For a Mac,** add `--coreml`. It also writes the model as a native CoreML
package, `diorama_seg.mlpackage` (513 MB), which the Mac app and
`jetlink-server` on a Mac run instead of the ONNX graph: about 3.5 times as
fast. It needs `coremltools` and the newest
PyTorch it supports:

```bash
python3 -m pip install "torch==2.7.0" "torchvision==0.22.0" transformers onnx coremltools
python3 -m diorama export --out ~/diorama-models --coreml
```

**License:**

Mask2Former's code is MIT. The Mapillary Vistas weights come from Meta through
Hugging Face, and the dataset they were trained on has its own terms. Check
them before you share the exported files.

**The car:** the page draws every car as
["2021 Tesla Model Y"](https://sketchfab.com/3d-models/2021-tesla-model-y-59e2ead369984b1a85c800ff6cf6789d)
by tonielpro520, licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Diorama recolours
it in the page's flat colours and leaves out its badges, interior, brakes and
the frames hidden behind its body (`diorama/assets.py`, rebuilt with
`python -m diorama assets <model.obj>`).

**Rebuilding the car:** the car is `diorama-assets.bin`, made from the model
in [Credits](diorama.md#credits) by `diorama/assets.py`: download its OBJ, then
`python -m diorama assets <model.obj>` (needs scipy).

## Publish the model for the app

The Mac app downloads what `ernestchu/jetlink-diorama` holds: `diorama.json`
and the segmentation it names, from the repo's `main`. After an export with
`--coreml`, upload the
folder as it is, with the [credits](diorama.md#credits) on the model card:

```bash
hf upload ernestchu/jetlink-diorama ~/diorama-models . --repo-type model
```

The app reads each graph's SHA-256 and size from the repo's file listing, so
nothing else needs updating. A new export replaces the graphs on the next
download; a Mac that has the old ones keeps them until its folder is emptied.

## The Mac

The app's **Download Models…** fetches the CoreML package, not the ONNX graph,
from [`ernestchu/jetlink-diorama`](https://huggingface.co/ernestchu/jetlink-diorama)
into **Settings > General > Diorama**'s folder. Each large file is checked
against the repo's SHA-256 as it arrives, and a download stopped part way keeps
the files it finished. To use a model you exported yourself, choose its folder
there and restart the server.

The model runs as a native CoreML program on the GPU, so the Neural Engine
stays the driving model's. CoreML compiles it the
first time the view opens, in a few seconds, and keeps the compile in the cache
folder's `diorama/`. A folder without the CoreML package runs the ONNX graph
through onnxruntime's CoreML provider instead: slower, and its first compile
takes a few minutes.

**Segment every frame (20 Hz)**, in **Settings > General > Diorama**, runs
the segmentation on every frame instead of every other one. It is off by
default, and takes effect at once. One run takes about 56 ms on an M5 Pro, too
long for every frame, but two at once on the GPU, each on an engine of its own,
finish one every 39 ms (`Limits.everyFrame`). Each frame's own run then takes
about 80 ms, so the **Segmentation** timing reads about 80 ms while
**Perception** reads 20 Hz. The driving model, on the Neural Engine, kept its
13.6 ms. The GPU stays nearly fully busy while Diorama is shown, and the model
takes twice the memory.

## An iPhone watching a Mac

The two need no network in common: they find each other over Wi-Fi peer to
peer (AWDL), the way AirDrop does. An iPhone using **wireless CarPlay** has
its Wi-Fi joined to the car, with no internet and no Mac on it, and still
finds the Mac this way. Both need Wi-Fi turned on, and the iPhone asks once
for Local Network access.

Checked on an iPhone 15 Pro watching an M5 Pro Mac, with the iPhone's Wi-Fi
on but joined to no network: it found the Mac over AWDL and drew the drive
smoothly, with the Mac's send queue empty. An iPhone on a USB cable to the
Mac (for Xcode) connects over the cable's network instead, so test AWDL with
the cable out. Wireless CarPlay sharing the radio has not been tried yet.

- **The design:**
  - The Mac's app runs an `NWListener` that advertises
    `_jetlink-diorama._tcp` over Bonjour, with `includePeerToPeer`, so it is
    found over AWDL as well as on any network the two share. It listens
    whether or not the server runs.
  - The iPhone runs an `NWBrowser` for the same service, connects to the Mac
    it found, and connects again a second after any drop. It looks only while
    the tab is in front: it stops when the app goes to the background, and
    looks again when it comes back.
  - Each viewer gets a watcher of its own on the stage, from the meta on, just
    as the Mac's own view does; the models run while any watcher is there. A
    new stage, when the Mac's server restarts, keeps viewers connected and
    starts them over from its meta.
- **What crosses:** only the [messages](#messages) the scene is drawn from: `M`
  meta, `D` drive and `S` and `W` seg, each after its u32 little-endian length
  exactly as a `drive.diorama` holds them. A replayed drive measured about
  180 KB/s (20 drive messages and 10 segmentations a second), well within
  what AWDL carries beside CarPlay.
- **What never leaves the Mac:** `I`, the camera frames the model sees, and
  `T`, what the model is doing. The iPhone's tab has no **Model input**
  toggle when it watches a Mac, and the Mac refuses the frames to a viewer
  that asks for them anyway.
- **Back the other way:** what the iPhone sends, in the same framing: `hello:`
  and the iPhone's name, then `ready`, sent again on each new connection.
- **A slow iPhone:** neither end queues more than it can hand on. The iPhone
  reads again only once its scene has taken the last batch, and the Mac sends
  again only once the last send has left. A viewer that falls behind backs up
  into its watcher, which the stage drops past 256 messages; it is watched
  again from a fresh meta, which the scene starts over from.
- **Who can watch:** any iPhone nearby running Jetlink, while the Mac's
  setting is on. It is off by default, and the Mac's toolbar lists who is
  watching. What it sees is the drawing's inputs (the driving model's
  outputs and the segmentation), never the camera.

## The model on an iPhone

With **This iPhone** turned on, the tab downloads the CoreML package, about
510 MB, after the same licence sheet as the Mac's, into the app's Application
Support (out of iCloud backups). The phone's server then runs it on the GPU
on the frames it serves, while the tab is open.

The driving model splits the iPhone's Neural Engine and GPU, so the two share
the GPU. As well as pausing after a slow driving frame, the stage takes at
most half the GPU's time on an iPhone (`Limits.duty`): after a segmentation
of t ms, it rests another t ms. Its latency, heat and the driving model's
frame times beside it have not been measured on an iPhone yet.

## What it costs

| Host | Segmentation | Notes |
| --- | ---: | --- |
| RTX 4090, TensorRT fp16 | 5.0 ms | outputs match onnxruntime fp32 on 99.5% of pixels (98.2% at worst over 8 frames) |
| RTX 4090, TensorRT 11 in `jetlink-server` | 9.8 ms | the server's strongly typed build keeps the graph's fp32; the driving model's round trip stayed at 2.8 ms p50 with the page open |
| RTX 4090, TensorRT 11 in `jetlink-server`, `--diorama-every-frame` | 10.0 ms a run, every frame (20.0 Hz) | one in flight is enough; the driving model's round trip stayed at 3.0 ms p50, 3.5 ms p99 (3.1 and 3.7 at 10 Hz). `--diorama-in-flight 2` loads a second engine from the same plan (1 GB more GPU memory) and changes nothing here |
| RTX 4090, onnxruntime CUDA | 23 ms | `python -m diorama run` |
| CPU (onnxruntime) | 685 ms | works, at a few frames a second |
| M5 Pro Mac, native CoreML on the GPU | 53 ms | the app (`--coreml`). Query map matches the ONNX graph on 99.9% of pixels |
| M5 Pro Mac, native CoreML on the GPU, two at once | 38.9 ms a frame (25.7 Hz), each run about 80 ms | the app's **Segment every frame (20 Hz)** |
| M5 Pro Mac, onnxruntime CoreML on the GPU | 185 ms | the ONNX graph: about 5 Hz. First compile about 2 minutes |
| M5 Pro Mac, onnxruntime CPU | 275 ms | |
| Jetson Orin | not measured yet | |
| iPhone, native CoreML on the GPU | not measured yet | at most half the GPU's time |

The Mac's and the 4090's driving-model latencies were measured with DD3D
running too, before it was dropped.

The page's engine takes about 3 ms of a browser frame (5–7 ms at the 95th
percentile, 22 ms at most when it re-cuts crosswalk stripes).

## The apps draw it natively

The Mac and iPhone apps draw Diorama in Metal, not in a web view. What they
draw is the page's drawing, step for step:

- **The scene:** `DioramaScene.swift` (JetlinkDiorama) is diorama-engine.js in
  Swift: the same smoothing, road map, lead check and crosswalk stripes, in
  the same order. `JetlinkDioramaTests` runs the page's own
  diorama-engine.js in JavaScriptCore beside it over a synthetic drive and
  checks that every frame's path, lanes, lead, stripes, pose and road map
  agree. On a recorded 90-second drive the two agreed on every frame, road
  map bytes included.
- **The drawing:** `DioramaRenderer.swift` (JetlinkDioramaUI) is
  diorama-render.js's shaders, camera and geometry in Metal: the road map as
  two mipmapped textures updated where they changed, the model's road and
  lane lines rasterised into a mask each frame, decals, and the car from
  diorama-assets.bin. The HUD, the timings and **Model input** are SwiftUI.
- **When it draws:** once per batch of messages, at most 20 times a second,
  and when the view's size or the light or dark appearance changes. Nothing
  draws while no drive comes.
- **Speed:** a frame of the scene takes at most about 4 ms on an M5 Pro. Both
  targets are optimised in a debug build too, since unoptimised the road map
  alone runs slower than the drive.

A change to diorama-engine.js or diorama-render.js needs the same change in
Swift; the parity test catches one made to the engine alone.

## Messages

The server's `/diorama/events` stream, a recorded `drive.diorama`, the Mac
app and a Mac's iPhone viewers all carry the same messages. Each one is a kind
byte, then its fields, little-endian. In a file and to an iPhone, each message
follows its u32 length; on the event stream, each is one base64 `data:` line.

| Kind | Fields |
| --- | --- |
| `M` meta | JSON: the frame's camera (`frame`), the wide frame's (`wide`, its `from_device` the wide camera's roll, pitch and yaw from the device), the camera's `height`, the drive fields and their float counts, the segmentation's classes. `height` and `from_device` are the comma's calibration; without them nothing is placed, or no wide frame used |
| `D` drive | u32 frame, f32 seconds, f32 model ms, f32[] the drive fields' raw outputs |
| `S` seg | u32 frame, f32 ms, u16 width, u16 height, u16 queries, u8[queries] class (255 dropped), u8[queries] score × 255, u32 runs, then (u8 query, u8 length − 1) per run |
| `W` wide seg | as `S`, of the wide frame: live, the server sends it in bird's-eye; `diorama run` writes both for every segmented frame. The page uses `S` in the road view and `W` in bird's-eye |
| `T` status | UTF-8: what the segmentation is doing (`idle`, `preparing …`, `running`, `paused: …`, `off: ` and why). Live only, and not to an iPhone watching a Mac: the page and the apps show it beside the timings when it is not running |
| `I` input | u32 frame, u16 width, u16 height, u8 planes, then the packed narrow frame the models see (four Y planes, U, V). Live only, and only to a page that asks: the web page's **Model input** toggle on `/diorama/events/frames`, or the Mac view's, or the iPhone tab's on its own server. Never to an iPhone watching a Mac |

`diorama/packets.py` writes them; diorama-engine.js and, in the apps,
`DioramaScene.swift` read them.
`DioramaPackets.swift` matches the Python byte for byte, which
`JetlinkDioramaTests` checks.
