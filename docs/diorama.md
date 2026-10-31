# Diorama

Diorama draws the drive as a live 3-D miniature, from what the driving model
sees. You get the planned path, lane lines, road edges and the lead car, along
with the cars, people and road around them. It is all computer-drawn: no camera
footage appears in it.

<a href="images/diorama.webp"><img src="images/diorama.webp" width="100%" alt="A recorded drive in the Diorama viewer: the road camera on the left, Diorama's bird's-eye view on the right with the lead car in blue, parked cars along the road and the planned path; below, the driving model's and the networks' timings"></a>

It runs in three places, all with the same page:

| Where | How to open it |
| --- | --- |
| Jetson or Linux PC | `http://<host>:<status port>/diorama/` on the web page, with `--diorama` ([below](#live-on-a-jetson-or-linux-pc)) |
| Mac app | **Diorama** in the sidebar (⌘5) |
| A recorded drive, on any computer | `python -m diorama view` ([below](#try-it-on-your-own-footage)), beside the drive's own footage |

## How it works

The comma already sends the server two things each frame: the warped narrow
camera image and the driving model's inputs. The server answers with the model's
outputs. Diorama adds two networks that run on the same warped image:

- **Segmentation:** Mask2Former (Swin-L, Mapillary Vistas panoptic), on every
  other frame (10 Hz). It finds the road surface, paint, crosswalks, curbs and
  sidewalks, and the cars, trucks, bikes and people on them.
- **Headings:** DD3D (VoVNet-99, KITTI), on every fourth frame. It gives each
  car's heading, including which way it faces.

The page does the rest, in the browser:

- It ranges each car and person where they touch the predicted road surface,
  fused with the model's own lead distance.
- It tracks them between looks.
- It builds a road map around the car from the segmentation.
- It draws the scene in WebGL.

The view sits behind the car at speed and pans to a bird's-eye view below
20 mph.

**The comma comes first:**

- The server hands Diorama the frame only after the reply has gone to the comma.
- The networks run only while a page or the Mac view is open.
- They pause for 3 s after any driving frame slower than 40 ms
  (`--diorama-pause-ms`), since they share the GPU with the driving model.

## Export the networks

Once, on a computer with Python 3.10+, PyTorch, `transformers` and `onnx`
(a GPU is not needed):

```bash
python3 -m pip install torch transformers onnx
python3 -m diorama export --out ~/diorama-networks
```

This writes three files, about 1.3 GB:

- `diorama_seg.onnx` (910 MB)
- `diorama_dd3d.onnx` (400 MB)
- `diorama.json`

DD3D's weights are downloaded from TRI's release; `--dd3d-weights FILE` uses a
copy you have. `--no-dd3d` leaves DD3D out, and cars then face along their lane.

**Licenses:**

- DD3D's code and weights are MIT (Toyota Research Institute).
- Mask2Former's code is MIT. The Mapillary Vistas weights come from Meta
  through Hugging Face, and the dataset they were trained on has its own terms.
  Check them before you share the exported files.

## Try it on your own footage

`python -m diorama run` takes a drive's segment folders from the comma's
`/data/media/0/realdata`, with `fcamera.hevc` and `ecamera.hevc` in each. It
runs the drive through a driving model and the networks exactly as a server
would see it, then plays the result in your browser beside the footage:

```bash
python3 -m pip install onnxruntime-gpu opencv-python numpy   # onnxruntime on a Mac; ffmpeg on the PATH
python3 -m diorama run ~/drives/<route>--{3,4,5} \
  --model ~/models/big_driving_supercombo.onnx --nets ~/diorama-networks \
  --out ~/diorama-drive --start 30 --seconds 120
python3 -m diorama view ~/diorama-drive
```

- **`--model`:** a driving model ONNX that Jetlink serves. A Jetson's or Mac's
  cache has them, or download one from the [catalog](models.md).
- **Calibration:** estimated from the model's own pose over the first minute,
  as openpilot's calibrationd does it. `--rpy ROLL PITCH YAW` (radians) sets it
  instead.
- **`--device`:** `cuda`, `coreml` or `cpu`. On an RTX 4090 with
  onnxruntime-gpu a drive runs at about twice real time.
- **Output:** `drive.diorama` (the messages a live server would have sent,
  about 11 MB a minute) and `footage.mp4`. The viewer has play, pause and seek,
  and Space toggles play.

### Replay a drive into a running server

`python -m diorama replay` sends a recorded drive to a server over TCP, frame
by frame at 20 Hz, exactly as a comma would. The server's live Diorama page then
shows it, so you can try the live path at a desk:

```bash
jetlink-server --listen --status-port 8080 --diorama ~/diorama-networks
python3 -m diorama replay ~/drives/<route>--3 --model ~/models/big_driving_supercombo.onnx --host 127.0.0.1
# open http://127.0.0.1:8080/diorama/
```

## Live on a Jetson or Linux PC

Copy the exported folder to the host, then add `--diorama` to the server's
options. For the installed service, put the folder in `/etc/jetlink/server.env`
and the option on the service's command line:

```bash
jetlink-server --usb --backend trt --status-port 5600 --diorama /mnt/data/diorama-networks
```

- **First run:** the first time a page opens Diorama, the server builds the two
  networks' TensorRT plans into the cache's `diorama/`. That takes about four
  minutes on an RTX 4090, and longer on a Jetson. Until then the page draws the
  model's road and lead without the networks.
- **Address:** `http://<host>:5600/diorama/`.
- **Sign-in:** with a [web password](control-protocol.md#the-web-page), the page
  needs the sign-in, as the status page does.

## Mac

1. Export the networks somewhere ([above](#export-the-networks)).
2. Choose that folder in **Settings > General > Diorama**.
3. Restart the server.

**Diorama** in the sidebar (⌘5) then shows the drive. The networks run through
CoreML on the GPU, so the Neural Engine stays the driving model's. CoreML
compiles them the first time the view opens, which takes a few minutes; the
compile is kept in the cache folder's `diorama/`.

## What it costs

| Host | Segmentation | DD3D | Notes |
| --- | ---: | ---: | --- |
| RTX 4090, TensorRT fp16 | 5.0 ms | 3.0 ms | outputs match onnxruntime fp32 on 99.2% of pixels; boxes within 8 cm |
| RTX 4090, onnxruntime CUDA | 23 ms | 13 ms | `python -m diorama run` |
| CPU (onnxruntime) | 685 ms | 322 ms | works, at a few frames a second |
| Jetson Orin, Mac | not measured yet | | |

The page's engine takes about 3 ms of a browser frame (5–7 ms at the 95th
percentile, 22 ms at most when it re-cuts crosswalk stripes).

## Known limits

- **A narrow view:** the frame the comma sends is the driving model's 512×256
  crop of the narrow camera, about ±16° across. Curbs and sidewalks to the side
  are seen only from 20 m or more ahead, so they fill in patchily, and
  crosswalks often too little for their stripes.
- **Far cars:** past about 40 m a car's range from where it touches the road
  shifts with every small error in the predicted road height. Diorama fuses in
  the model's lead distance, but other far cars can sit too close for a moment.
- **Lane colour:** all lane lines are drawn white.

## Messages

The server's `/diorama/events` stream, a recorded `drive.diorama`, and the Mac
app all carry the same messages. Each one is a kind byte, then its fields,
little-endian. In a file, each message follows its u32 length; on the event
stream, each is one base64 `data:` line.

| Kind | Fields |
| --- | --- |
| `M` meta | JSON: the frame's camera, the drive fields and their float counts, the networks' classes |
| `D` drive | u32 frame, f32 seconds, f32 model ms, f32[] the drive fields' raw outputs |
| `S` seg | u32 frame, f32 ms, u16 width, u16 height, u16 queries, u8[queries] class (255 dropped), u8[queries] score × 255, u32 runs, then (u8 query, u8 length − 1) per run |
| `B` boxes | u32 frame, f32 ms, u16 rows, f32[rows × 16] DD3D rows (`diorama/export.py` names the columns) |

`diorama/packets.py` writes them and diorama-engine.js reads them.
`DioramaPackets.swift` matches the Python byte for byte, which
`JetlinkDioramaTests` checks.
