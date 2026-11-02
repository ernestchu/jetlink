import Foundation
import JetlinkLog
import JetlinkServer
import Synchronization

/// Diorama on a running server: a live 3-D miniature of the drive, built
/// from the frames the comma sends anyway.
///
/// Every frame the server answers, the stage takes the driving model's
/// outputs and, every other frame, a warped image (`FrameObserver`): the
/// narrow camera's in the road view, the wide camera's in bird's-eye, which
/// shows the road close around the car the narrow one sees only ahead. The
/// stage keeps the scene the watchers make, to know which view they show.
/// It places what the cameras saw with the comma's own calibration, from its
/// hello (`cameraCalibrated`), and waits for one: nothing is assumed.
/// On threads of its own it runs the segmentation on that image, and hands
/// the messages of docs/diorama-internals.md to whoever watches: the web page's event stream, the Mac app's view. The page does
/// the rest in the browser.
///
/// The comma comes first. Nothing runs while nobody watches; the frame hook
/// only tries its lock and copies at most 100 KB; and a driving frame slower
/// than `Limits.slowFrameUs` stops the models for `Limits.backOff`, since
/// they share the accelerator with the driving model.
public final class DioramaStage: FrameObserver, @unchecked Sendable {
  public struct Limits: Sendable {
    /// Segment every this many frames (10 Hz).
    public var segEvery: UInt32 = 2
    /// Segmentations run at once, each on an engine of its own. One leaves
    /// the GPU idle between runs; two keep a Mac's busy enough to segment
    /// every frame (20 Hz), at twice the memory.
    public var inFlight = 1
    /// Every frame, two at once: 20 Hz where the GPU keeps up.
    public static let everyFrame: (segEvery: UInt32, inFlight: Int) = (1, 2)
    /// A driving frame this slow pauses the models.
    public var slowFrameUs: UInt32 = 40_000
    public var backOff: TimeInterval = 3
    /// Messages a watcher may fall behind by before it is dropped.
    public var behind = 256
    /// The share of the time the models may run: after a run of t, they
    /// rest t × (1 / duty − 1). An iPhone, whose GPU has the driving model's
    /// second half, asks for less than all of it.
    public var duty: Double = 1
    public init() {}
  }

  /// The model `python -m diorama export` wrote, as diorama.json describes it.
  public struct Models: @unchecked Sendable {
    public let folder: URL
    /// diorama.json, as JSONSerialization read it.
    public let meta: [String: Any]
    /// The segmentation's ONNX graph, or with `coreml` its CoreML package (a `.mlpackage`).
    public let seg: URL

    /// `coreml`: the CoreML package diorama.json names where it names one
    /// (`export --coreml`), the ONNX graph otherwise. A Mac asks for it.
    /// Other graphs it names, such as an older export's DD3D, are not run.
    public init(folder: URL, coreml: Bool = false) throws {
      let url = folder.appending(path: "diorama.json")
      guard let data = try? Data(contentsOf: url), let meta = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] else {
        throw DioramaError("no diorama.json in \(folder.path); export the models with: python -m diorama export --out \(folder.path)")
      }
      let file = { (model: String) -> String? in
        let entry = meta[model] as? [String: Any]
        return (coreml ? entry?["coreml"] as? String : nil) ?? entry?["file"] as? String
      }
      guard let seg = file("seg") else { throw DioramaError("\(url.path) names no segmentation graph") }
      self.folder = folder
      self.meta = meta
      self.seg = folder.appending(path: seg)
      // a missing graph is found here, not minutes later behind the backend's error
      if !FileManager.default.fileExists(atPath: self.seg.path) {
        throw DioramaError("\(url.path) names \(self.seg.lastPathComponent), which is not in \(folder.path)")
      }
    }
  }

  /// Builds or loads an engine for one of the graphs: the host's backend
  /// decides how (TensorRT, CoreML on the GPU, the CPU). Called on the
  /// stage's thread; the first time may take minutes.
  public typealias EngineFactory = @Sendable (_ graph: URL, _ name: String, _ report: @escaping ProgressFn) throws -> any Engine

  /// One page or view watching: its own queue of messages.
  public final class Watcher: @unchecked Sendable {
    fileprivate var queue: [Data] = []
    fileprivate var dropped = false
    fileprivate var closed = false
    /// It also wants the frames the models see (`input` messages): a page's Model input view.
    fileprivate var input = false
  }

  public let models: Models
  /// As the stage was made; `segment(every:inFlight:)` changes the rate after.
  public let limits: Limits
  private let makeEngine: EngineFactory
  private let log = ServerLog(category: "diorama")

  // What the frames hand over, behind a lock the session thread only tries.
  private let frameLock = NSLock()
  private var watching = 0
  private var drives: [(frame: UInt32, values: [Float], ms: Float)] = []
  private var image: [UInt8] = []
  private var imageFrame: UInt32?
  private var imageWide = false
  private var modelSha = ""
  private var fields: [(name: String, count: Int)] = []
  private var ranges: [Range<Int>] = []
  private var fieldsChanged = false
  private let wake = DispatchSemaphore(value: 0)
  /// Until when the models rest, in uptime nanoseconds: outside the lock, so a slow frame is never missed.
  private let pausedUntil = Atomic<UInt64>(0)

  // The watchers' queues.
  private let watchLock = NSCondition()
  private var watchers: [ObjectIdentifier: Watcher] = [:]
  private var stopped = false
  private var status = DioramaStage.uncalibrated

  // The models, prepared on a thread of their own: a first TensorRT build
  // takes minutes, and the drive messages keep flowing meanwhile.
  private let engineLock = NSLock()
  private var engines: [any Engine] = []
  private var preparing = false
  private var loadFailed = false
  private let half: [UInt16] = (0..<256).map { DioramaStage.float16Bits(Float($0)) }
  /// Until when the models rest after a run, by `Limits.duty`, in uptime nanoseconds.
  private let restUntil = Atomic<UInt64>(0)
  /// The rate now: segment every this many frames, this many at once.
  private let segEvery: Atomic<UInt32>
  private let inFlight: Atomic<Int>

  // The runners, one thread per engine, and the frame waiting for one.
  private let runLock = NSCondition()
  private var waiting: (frame: [UInt8], id: UInt32, wide: Bool)?
  private var busy = 0
  /// The newest frame whose segmentation went out: one that finishes after a newer one is dropped.
  private var lastSeg: UInt32?

  // The scene the watchers make of the messages, kept here too: while it is in bird's-eye, the wide frame is segmented.
  private let sceneLock = NSLock()
  private let scene = DioramaScene()
  private let birdsEye = Atomic<Bool>(false)
  /// The comma's calibration of its cameras, as its hello gave it.
  private var calibration: CameraCalibration?

  public init(models: Models, limits: Limits = Limits(), makeEngine: @escaping EngineFactory) {
    self.models = models
    self.limits = limits
    self.makeEngine = makeEngine
    segEvery = Atomic(max(limits.segEvery, 1))
    inFlight = Atomic(max(limits.inFlight, 1))
  }

  /// Segment every `every` frames, `inFlight` at once: `Limits.everyFrame`
  /// for 20 Hz, or back to 10 Hz. Engines for more at once are made as
  /// needed; ones no longer needed stay loaded until the stage stops.
  public func segment(every: UInt32, inFlight: Int) {
    segEvery.store(max(every, 1), ordering: .relaxed)
    self.inFlight.store(max(inFlight, 1), ordering: .relaxed)
  }

  public func start() {
    let thread = Thread { [weak self] in self?.loop() }
    thread.name = "jetlink-diorama"
    thread.qualityOfService = .utility
    thread.start()
  }

  public func stop() {
    watchLock.lock()
    stopped = true
    for w in watchers.values { w.closed = true }
    watchLock.broadcast()
    watchLock.unlock()
    wake.signal()
    runLock.lock()
    runLock.broadcast()
    runLock.unlock()
  }

  /// What the stage is doing, for a page or a view.
  public var state: String {
    watchLock.lock()
    defer { watchLock.unlock() }
    return status
  }

  // MARK: watchers

  /// A new watcher, its queue starting with the meta message and the models'
  /// status. `input`: the frames the models see as well, the packed narrow
  /// frame, 196 KB ten times a second, so only for a debug view showing it.
  public func watch(input: Bool = false) -> Watcher {
    let w = Watcher()
    w.input = input
    let meta = DioramaPackets.meta(metaInfo())
    watchLock.lock()
    watchers[ObjectIdentifier(w)] = w
    w.queue.append(meta)
    w.queue.append(DioramaPackets.status(status))
    watchLock.unlock()
    frameLock.lock()
    watching += 1
    frameLock.unlock()
    wake.signal()
    return w
  }

  public func unwatch(_ w: Watcher) {
    watchLock.lock()
    let was = watchers.removeValue(forKey: ObjectIdentifier(w)) != nil
    w.closed = true
    watchLock.broadcast()
    watchLock.unlock()
    if was {
      frameLock.lock()
      watching -= 1
      frameLock.unlock()
    }
  }

  /// The messages for `w` since the last call, waiting up to `timeout` for
  /// one: empty on a quiet timeout, nil once the watcher is closed or fell
  /// too far behind (it should watch again, from a fresh meta).
  public func next(_ w: Watcher, timeout: TimeInterval) -> [Data]? {
    let deadline = Date(timeIntervalSinceNow: timeout)
    watchLock.lock()
    defer { watchLock.unlock() }
    while w.queue.isEmpty && !w.closed && !w.dropped && !stopped {
      if !watchLock.wait(until: deadline) { break }
    }
    if w.closed || w.dropped || stopped { return nil }
    let out = w.queue
    w.queue.removeAll(keepingCapacity: true)
    return out
  }

  // MARK: FrameObserver

  public func frameServed(
    frameID: UInt32, narrow: UnsafeRawBufferPointer, wide: UnsafeRawBufferPointer, outputs: UnsafeBufferPointer<Float>, spec: ModelSpec,
    totalUs: UInt32
  ) {
    if totalUs > limits.slowFrameUs {
      pausedUntil.store(DispatchTime.now().uptimeNanoseconds + UInt64(limits.backOff * 1e9), ordering: .relaxed)
    }
    // a try, not a wait: a frame never queues behind the stage (it may miss this one's image)
    guard frameLock.try() else { return }
    defer { frameLock.unlock() }
    guard watching > 0, calibration != nil else { return }
    if spec.sha256 != modelSha {
      modelSha = spec.sha256
      let found = DioramaPackets.driveFields.compactMap { name in spec.outputSlices.first { $0.name == name }.map { (name, $0.range(in: outputs.count)) } }
      fields = found.map { ($0.0, $0.1.count) }
      ranges = found.map(\.1)
      fieldsChanged = true
    }
    var values = [Float]()
    values.reserveCapacity(ranges.reduce(0) { $0 + $1.count })
    for r in ranges { values.append(contentsOf: outputs[r]) }
    drives.append((frameID, values, Float(totalUs) / 1000))
    if drives.count > 40 { drives.removeFirst(drives.count - 40) }
    if frameID % segEvery.load(ordering: .relaxed) == 0 {
      imageWide = birdsEye.load(ordering: .relaxed)
      image = Array((imageWide ? wide : narrow).bindMemory(to: UInt8.self))
      imageFrame = frameID
    }
    wake.signal()
  }

  public func cameraCalibrated(_ calibration: CameraCalibration?) {
    frameLock.lock()
    guard calibration != self.calibration else {
      frameLock.unlock()
      return
    }
    self.calibration = calibration
    // the meta again, with it; before the first frame, a watcher's own meta has it
    if !fields.isEmpty { fieldsChanged = true }
    frameLock.unlock()
    if calibration == nil {
      setStatus(Self.uncalibrated)
    } else if state == Self.uncalibrated {
      setStatus("idle")
    }
  }

  static let uncalibrated = "waiting: the comma sent no camera calibration (not calibrated yet, or older jetlink on it)"

  // MARK: the stage's thread

  private func loop() {
    while true {
      _ = wake.wait(timeout: .now() + 1)
      watchLock.lock()
      let done = stopped
      watchLock.unlock()
      if done { return }

      frameLock.lock()
      let pending = drives
      drives.removeAll(keepingCapacity: true)
      let paused = DispatchTime.now().uptimeNanoseconds < pausedUntil.load(ordering: .relaxed)
      let seen: [UInt8]? = imageFrame != nil ? image : nil
      let frame = paused ? nil : seen
      let frameID = imageFrame ?? 0, wide = imageWide
      imageFrame = nil
      let announce = fieldsChanged
      fieldsChanged = false
      let anyone = watching > 0
      frameLock.unlock()
      guard anyone else { continue }

      if announce {
        let meta = DioramaPackets.meta(metaInfo())
        broadcast(meta)
        follow(meta)
      }
      for d in pending {
        let drive = DioramaPackets.drive(frame: d.frame, seconds: Float(d.frame) / 20, modelMs: d.ms, values: d.values)
        broadcast(drive)
        follow(drive)
      }
      // what the models see, paused or not, for a debug view that asked
      if let seen { showSeen(seen, frameID) }
      // a pause does not stop a build, and must not hide why the models never run
      if paused && !preparingOrFailed { setStatus("paused: the driving model ran slow") }
      guard let frame, DispatchTime.now().uptimeNanoseconds >= restUntil.load(ordering: .relaxed) else { continue }
      let ready = enginesReady()
      guard ready > 0 else { continue }
      // to a runner that is free, within the rate's number at once; with none free, the frame is missed
      runLock.lock()
      if busy < min(ready, inFlight.load(ordering: .relaxed)) {
        waiting = (frame, frameID, wide)
        busy += 1
        runLock.signal()
      }
      runLock.unlock()
    }
  }

  /// One runner, on engine `index`: segments the frames handed to it until the stage stops.
  private func runner(_ index: Int) {
    while true {
      runLock.lock()
      while waiting == nil && !isStopped { runLock.wait() }
      guard let (frame, frameID, wide) = waiting else {
        runLock.unlock()
        return
      }
      waiting = nil
      runLock.unlock()

      engineLock.lock()
      let engine = engines[index]
      engineLock.unlock()
      setStatus("running")
      let started = DispatchTime.now().uptimeNanoseconds
      let message = run(engine, frame, { segMessage($0, frameID, $1, wide: wide) })
      if limits.duty < 1 {
        let took = DispatchTime.now().uptimeNanoseconds - started
        restUntil.store(DispatchTime.now().uptimeNanoseconds + UInt64(Double(took) * (1 / max(limits.duty, 0.05) - 1)), ordering: .relaxed)
      }
      runLock.lock()
      busy -= 1
      // newer, or a new session's frames counting from 0 again (as the scene, a jump back of more than 40)
      let newest = lastSeg.map { frameID > $0 || frameID + 40 < $0 } ?? true
      if newest && message != nil { lastSeg = frameID }
      runLock.unlock()
      if newest, let message {
        broadcast(message)
        follow(message)
      }
    }
  }

  private var isStopped: Bool {
    watchLock.lock()
    defer { watchLock.unlock() }
    return stopped
  }

  /// How many engines are ready; the first call, and one asking for more at once than there are, starts
  /// preparing them.
  private func enginesReady() -> Int {
    engineLock.lock()
    defer { engineLock.unlock() }
    if engines.count < inFlight.load(ordering: .relaxed) && !preparing && !loadFailed {
      preparing = true
      let thread = Thread { [weak self] in self?.prepare() }
      thread.name = "jetlink-diorama-prepare"
      thread.qualityOfService = .utility
      thread.start()
    }
    return engines.count
  }

  private func prepare() {
    let log = self.log
    let progress: (String) -> ProgressFn = { name in
      { stage, frac, msg in if frac == 0 || frac == 1 { log.info("\(name): \(stage), \(msg)") } }
    }
    do {
      // engines until there are as many as the rate runs at once; a second is the first's compile, loaded again
      while true {
        engineLock.lock()
        let have = engines.count
        engineLock.unlock()
        if have >= inFlight.load(ordering: .relaxed) || isStopped { break }
        if have == 0 { setStatus("preparing the segmentation model; the first time takes minutes") }
        let seg = try makeEngine(models.seg, "seg", progress("segmentation"))
        engineLock.lock()
        engines.append(seg)
        engineLock.unlock()
        let thread = Thread { [weak self] in self?.runner(have) }
        thread.name = "jetlink-diorama-run-\(have)"
        thread.qualityOfService = .utility
        thread.start()
        log.info(have == 0 ? "segmentation ready" : "segmentation engine \(have + 1) ready, for \(have + 1) at once")
      }
      engineLock.lock()
      preparing = false
      engineLock.unlock()
    } catch {
      engineLock.lock()
      loadFailed = true
      preparing = false
      let have = engines.count
      engineLock.unlock()
      if have > 0 {
        // the ones there are keep running, fewer at once
        log.warning("no segmentation engine \(have + 1), so \(have) at once: \(error)")
        return
      }
      log.error("cannot run the models: \(error)")
      // the page has one line for it: the graphs by name, the folder is in the log
      setStatus("off: " + "\(error)".replacingOccurrences(of: models.folder.path + "/", with: ""))
    }
  }

  /// Stages the packed frame as the graph's input, runs it, and makes its message.
  private func run(_ engine: any Engine, _ frame: [UInt8], _ message: (any Engine, Float) -> Data?) -> Data? {
    guard let input = engine.inputs["frame"], let dst = engine.hostInput("frame"), input.count == frame.count else {
      log.error("the graph's input is not the packed \(frame.count)-byte frame")
      return nil
    }
    switch input.type {
    case .float16:
      let out = dst.assumingMemoryBound(to: UInt16.self)
      for i in 0..<frame.count { out[i] = half[Int(frame[i])] }
    case .float:
      let out = dst.assumingMemoryBound(to: Float.self)
      for i in 0..<frame.count { out[i] = Float(frame[i]) }
    case .uint8:
      frame.withUnsafeBytes { dst.copyMemory(from: $0.baseAddress!, byteCount: frame.count) }
    default:
      log.error("cannot stage a \(input.type.name) input")
      return nil
    }
    let started = DispatchTime.now().uptimeNanoseconds
    do {
      try engine.run()
    } catch {
      log.error("\(error)")
      return nil
    }
    return message(engine, Float(DispatchTime.now().uptimeNanoseconds - started) / 1e6)
  }

  private func floats(_ engine: any Engine, _ name: String) -> UnsafeBufferPointer<Float>? {
    guard let spec = engine.outputs[name], spec.type == .float, let p = engine.output(name) else { return nil }
    return UnsafeBufferPointer(start: p.assumingMemoryBound(to: Float.self), count: spec.count)
  }

  private func segMessage(_ engine: any Engine, _ frame: UInt32, _ ms: Float, wide: Bool) -> Data? {
    guard let qmap = floats(engine, "qmap"), let label = floats(engine, "label"), let score = floats(engine, "score"),
      let shape = engine.outputs["qmap"]?.shape, shape.count >= 2
    else {
      log.error("the segmentation graph has no float qmap, label and score")
      return nil
    }
    return DioramaPackets.seg(
      frame: frame, ms: ms, qmap: qmap, width: shape[shape.count - 1], height: shape[shape.count - 2], label: label, score: score, wide: wide)
  }

  /// The model input to the watchers that want it. A watcher this far
  /// behind skips the frame rather than queueing more of them.
  private func showSeen(_ packed: [UInt8], _ frame: UInt32) {
    watchLock.lock()
    defer { watchLock.unlock() }
    let wanting = watchers.values.filter { $0.input && !$0.closed && !$0.dropped && $0.queue.count < 8 }
    guard !wanting.isEmpty else { return }
    let message = DioramaPackets.input(frame: frame, packed: packed)
    for w in wanting { w.queue.append(message) }
    watchLock.broadcast()
  }

  /// A message the watchers were sent, into the stage's own scene: is it in bird's-eye now?
  private func follow(_ message: Data) {
    sceneLock.lock()
    defer { sceneLock.unlock() }
    scene.push(DioramaMessage(message))
    birdsEye.store(scene.state.map { !$0.viewWhy.isEmpty } ?? false, ordering: .relaxed)
  }

  private func broadcast(_ message: Data) {
    watchLock.lock()
    for w in watchers.values where !w.closed && !w.dropped {
      if w.queue.count >= limits.behind {
        w.dropped = true
        continue
      }
      w.queue.append(message)
    }
    watchLock.broadcast()
    watchLock.unlock()
  }

  private var preparingOrFailed: Bool {
    engineLock.lock()
    defer { engineLock.unlock() }
    return preparing || loadFailed
  }

  /// The status, and to every watcher when it changes.
  private func setStatus(_ s: String) {
    watchLock.lock()
    defer { watchLock.unlock() }
    guard s != status else { return }
    status = s
    let message = DioramaPackets.status(s)
    for w in watchers.values where !w.closed && !w.dropped && w.queue.count < limits.behind { w.queue.append(message) }
    watchLock.broadcast()
  }

  /// The meta message: the frame's camera and the models' classes from
  /// diorama.json, and the drive fields as the loaded model lays them out.
  private func metaInfo() -> [String: Any] {
    frameLock.lock()
    let drive = fields.map { [$0.name, $0.count] as [Any] }
    let calibration = self.calibration
    frameLock.unlock()
    var info: [String: Any] = ["version": 1, "fps": 20, "live": true, "drive": drive]
    if let calibration { info["height"] = calibration.height }
    for key in ["frame", "seg"] { if let v = models.meta[key] { info[key] = v } }
    // the wide frame as the comma warps it, openpilot's sbigmodel: half the narrow's focal length, its horizon
    // halfway down from the narrow's (packets.wide_frame); and the wide camera's angles from the device
    if let calibration, let f = models.meta["frame"] as? [String: Any], let fl = (f["f"] as? NSNumber)?.doubleValue,
      let h = (f["h"] as? NSNumber)?.doubleValue, let cy = (f["cy"] as? NSNumber)?.doubleValue
    {
      info["wide"] = f.merging(["f": fl / 2, "cy": (h + cy) / 2, "from_device": calibration.wideFromDevice]) { $1 }
    }
    return info
  }

  /// IEEE half precision of a float that is exactly representable in it: every pixel value is.
  static func float16Bits(_ value: Float) -> UInt16 {
    let bits = value.bitPattern
    let sign = UInt16((bits >> 16) & 0x8000)
    if value == 0 { return sign }
    let exponent = Int((bits >> 23) & 0xff) - 127 + 15
    return sign | UInt16(exponent << 10) | UInt16((bits >> 13) & 0x3ff)
  }
}

public struct DioramaError: Error, CustomStringConvertible {
  public let description: String
  public init(_ description: String) { self.description = description }
}
