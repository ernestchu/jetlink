import Foundation
import JetlinkServer
import Synchronization
import Testing

@testable import JetlinkDiorama

/// The messages of a recorded drive: each after its u32 length.
func framed(_ data: Data) -> [Data] {
  var out: [Data] = []
  var at = data.startIndex
  while at + 4 <= data.endIndex {
    let n = data[at..<at + 4].enumerated().reduce(0) { $0 | Int($1.element) << (8 * $1.offset) }
    out.append(data[at + 4..<at + 4 + n])
    at += 4 + n
  }
  return out
}

@Suite("Diorama's messages")
struct DioramaPacketsTests {
  @Test("Drive and seg messages are diorama/packets.py's bytes")
  func matchesPython() throws {
    let url = try #require(Bundle.module.url(forResource: "packets", withExtension: "bin", subdirectory: "Fixtures"))
    let expected = framed(try Data(contentsOf: url))
    try #require(expected.count == 2)

    // the inputs JetlinkKit/Scripts/make_diorama_fixtures.py writes them from
    let values: [Float] = [0.5, -1.25, 3.0, 1e-3, 250.0]
    var qmap = [Float](repeating: 255, count: 3 * 400)
    for c in 10..<20 { qmap[400 + c] = 7 }
    for c in 0..<400 { qmap[800 + c] = 3 }
    let label: [Float] = [12, 40, 7, 3, 0]
    let score: [Float] = [0.0, 0.5, 0.9, 1.0, 0.25]

    #expect(DioramaPackets.drive(frame: 7, seconds: 0.35, modelMs: 3.5, values: values) == expected[0])
    let seg = qmap.withUnsafeBufferPointer { q in
      label.withUnsafeBufferPointer { l in
        score.withUnsafeBufferPointer { s in DioramaPackets.seg(frame: 8, ms: 21.5, qmap: q, width: 400, height: 3, label: l, score: s) }
      }
    }
    #expect(seg == expected[1])
  }

  @Test("Half precision of every pixel value is exact")
  func halves() {
    #expect(DioramaStage.float16Bits(0) == 0)
    #expect(DioramaStage.float16Bits(1) == 0x3C00)
    #expect(DioramaStage.float16Bits(128) == 0x5800)
    #expect(DioramaStage.float16Bits(255) == 0x5BF8)
  }

  @Test("The page's files are in the bundle")
  func page() throws {
    for name in DioramaPage.files.keys {
      let file = try #require(DioramaPage.file(name), "\(name)")
      #expect(!file.data.isEmpty)
    }
    #expect(DioramaPage.file("../Package.swift") == nil)
  }
}

/// Stands in for the segmentation: the packed frame's size in, outputs of the shapes the exported graph has.
final class FakeModel: EngineCore, @unchecked Sendable {
  private(set) var runs = 0
  private(set) var lastFirstPixel: UInt16 = 0

  init() throws {
    let input = TensorSpec(name: "frame", type: .float16, shape: [1, 6, 128, 256])
    let outputs = [
      TensorSpec(name: "qmap", type: .float, shape: [1, 128, 256]), TensorSpec(name: "label", type: .float, shape: [200]),
      TensorSpec(name: "score", type: .float, shape: [200]),
    ]
    try super.init(inputs: ["frame": input], outputs: Dictionary(uniqueKeysWithValues: outputs.map { ($0.name, $0) }))
  }

  override func run() throws {
    runs += 1
    lastFirstPixel = hostInput("frame")!.assumingMemoryBound(to: UInt16.self)[0]
    if let qmap = output("qmap") {
      let q = UnsafeMutablePointer(mutating: qmap.assumingMemoryBound(to: Float.self))
      for i in 0..<(128 * 256) { q[i] = i < 1000 ? 4 : 255 }
    }
  }
}

@Suite("Diorama's stage")
struct DioramaStageTests {
  let spec = ModelSpec(
    sha256: "ab", nbytes: 1, frameSkip: 4, inputShapes: [], outputShapes: [NamedShape("outputs", [1, 20])],
    outputSlices: [NamedSlice("plan", 0..<4), NamedSlice("pose", 4..<6), NamedSlice("hidden_state", 6..<20)], checkpoint: nil)

  /// `olderExport`: diorama.json also names DD3D, as exports before it was dropped did, without its graph here.
  func models(olderExport: Bool = false) throws -> DioramaStage.Models {
    let dir = FileManager.default.temporaryDirectory.appending(path: "diorama-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    var meta: [String: Any] = [
      "frame": ["w": 512, "h": 256, "f": 910, "cx": 256, "cy": 47.6],
      "seg": ["file": "diorama_seg.onnx", "labels": ["Road"], "road_class": ["Road": 1], "things": [:], "ds": 2],
    ]
    if olderExport { meta["dd3d"] = ["file": "diorama_dd3d.onnx", "classes": ["Car"], "columns": []] }
    try JSONSerialization.data(withJSONObject: meta).write(to: dir.appending(path: "diorama.json"))
    try Data().write(to: dir.appending(path: "diorama_seg.onnx"))
    return try DioramaStage.Models(folder: dir)
  }

  @Test("A folder without the graph diorama.json names is refused, naming it")
  func missingGraph() throws {
    let dir = try models().folder
    try FileManager.default.removeItem(at: dir.appending(path: "diorama_seg.onnx"))
    #expect {
      try DioramaStage.Models(folder: dir)
    } throws: { "\($0)".contains("diorama_seg.onnx, which is not in") }
  }

  @Test("An older export that also names DD3D loads, and only the segmentation is built")
  func olderExport() throws {
    let built = Mutex<[String]>([])
    let seg = try FakeModel()
    let stage = DioramaStage(models: try models(olderExport: true)) { _, name, _ in
      built.withLock { $0.append(name) }
      return seg
    }
    stage.start()
    defer { stage.stop() }
    let w = stage.watch()
    for f in UInt32(0)..<4 {
      serve(stage, frame: f)
      Thread.sleep(forTimeInterval: 0.05)
    }
    let got = collect(stage, w, 8)
    #expect(built.withLock { $0 } == ["seg"])
    #expect(got.contains { $0.first == UInt8(ascii: "S") })
    // the page is told of the segmentation's classes only
    let meta = try #require(got.first { $0.first == UInt8(ascii: "M") })
    let info = try #require(try JSONSerialization.jsonObject(with: meta.dropFirst()) as? [String: Any])
    #expect(info["seg"] != nil && info["dd3d"] == nil)
  }

  func serve(_ stage: DioramaStage, frame: UInt32, pixel: UInt8 = 9, totalUs: UInt32 = 20_000) {
    let narrow = [UInt8](repeating: pixel, count: 6 * 128 * 256)
    let outputs = (0..<20).map(Float.init)
    narrow.withUnsafeBytes { n in
      outputs.withUnsafeBufferPointer { o in stage.frameServed(frameID: frame, narrow: n, outputs: o, spec: spec, totalUs: totalUs) }
    }
  }

  /// Messages until `count` have come, or `done` says so, or two seconds have passed.
  func collect(_ stage: DioramaStage, _ w: DioramaStage.Watcher, _ count: Int = .max, until done: ([Data]) -> Bool = { _ in false }) -> [Data] {
    var got: [Data] = []
    let deadline = Date(timeIntervalSinceNow: 2)
    while got.count < count && !done(got) && Date() < deadline, let more = stage.next(w, timeout: 0.1) { got += more }
    return got
  }

  /// The status messages among `got`, as text.
  func statuses(_ got: [Data]) -> [String] {
    got.filter { $0.first == UInt8(ascii: "T") }.map { String(decoding: $0.dropFirst(), as: UTF8.self) }
  }

  @Test("Nobody watching: frames are not even copied, and nothing runs")
  func idle() throws {
    let seg = try FakeModel()
    let stage = DioramaStage(models: try models()) { _, _, _ in seg }
    stage.start()
    defer { stage.stop() }
    serve(stage, frame: 0)
    Thread.sleep(forTimeInterval: 0.2)
    #expect(seg.runs == 0)
  }

  @Test("A watcher hears the meta, every frame's outputs and the segmentation of every other frame")
  func watched() throws {
    let seg = try FakeModel()
    let stage = DioramaStage(models: try models()) { _, _, _ in seg }
    stage.start()
    defer { stage.stop() }
    let w = stage.watch()
    for f in UInt32(0)..<4 {
      serve(stage, frame: f)
      Thread.sleep(forTimeInterval: 0.05)
    }
    let got = collect(stage, w, 9)
    let kinds = got.map { Character(UnicodeScalar($0.first!)) }
    #expect(kinds.first == "M")
    #expect(kinds.filter { $0 == "D" }.count == 4)
    #expect(kinds.contains("S"))
    #expect(seg.lastFirstPixel == DioramaStage.float16Bits(9))
    // the meta that follows the first frame names the drive fields with this model's counts, hidden state left out
    let meta = try #require(got.dropFirst().first { $0.first == UInt8(ascii: "M") })
    let info = try #require(try JSONSerialization.jsonObject(with: meta.dropFirst()) as? [String: Any])
    let drive = try #require(info["drive"] as? [[Any]])
    #expect(drive.map { $0[0] as? String } == ["plan", "pose"])
    #expect(drive.map { ($0[1] as? NSNumber)?.intValue } == [4, 2])
    // a drive message carries those 6 floats: kind, frame, seconds, ms, then the values
    let d = try #require(got.first { $0.first == UInt8(ascii: "D") })
    #expect(d.count == 13 + 6 * 4)
    stage.unwatch(w)
  }

  @Test("The drive keeps flowing while the models are prepared")
  func preparing() throws {
    let seg = try FakeModel()
    let stage = DioramaStage(models: try models()) { _, _, _ in
      Thread.sleep(forTimeInterval: 1)  // a first build
      return seg
    }
    stage.start()
    defer { stage.stop() }
    let w = stage.watch()
    for f in UInt32(0)..<6 {
      serve(stage, frame: f)
      Thread.sleep(forTimeInterval: 0.05)
    }
    let got = collect(stage, w, 8)
    #expect(got.filter { $0.first == UInt8(ascii: "D") }.count == 6)
    #expect(!got.contains { $0.first == UInt8(ascii: "S") })
    #expect(stage.state.hasPrefix("preparing"))
  }

  @Test("A watcher hears why the models are off, and a pause does not hide it")
  func offSaysWhy() throws {
    struct Missing: Error, CustomStringConvertible { var description: String { "File doesn't exist" } }
    let stage = DioramaStage(models: try models()) { _, _, _ in throw Missing() }
    stage.start()
    defer { stage.stop() }
    let w = stage.watch()
    // apart, since a frame the stage's thread is taking the last one's from is
    // skipped, as a comma's would be
    for (f, us) in [(UInt32(0), UInt32(20_000)), (2, 55_000), (4, 20_000)] {
      serve(stage, frame: f, totalUs: us)
      Thread.sleep(forTimeInterval: f == 0 ? 0.2 : 0.05)
    }
    let got = collect(stage, w) { got in
      got.filter { $0.first == UInt8(ascii: "D") }.count == 3 && statuses(got).last == "off: File doesn't exist"
    }
    #expect(statuses(got).first == "idle")
    #expect(statuses(got).last == "off: File doesn't exist")
    #expect(stage.state == "off: File doesn't exist")
    #expect(got.filter { $0.first == UInt8(ascii: "D") }.count == 3)
    // a page that opens later hears it straight away
    let later = stage.watch()
    let first = collect(stage, later, 2)
    #expect(first.last.map { String(decoding: $0.dropFirst(), as: UTF8.self) } == "off: File doesn't exist")
  }

  @Test("Only a watcher that asks gets the frames the models see, paused or not")
  func input() throws {
    var limits = DioramaStage.Limits()
    limits.backOff = 30
    let stage = DioramaStage(models: try models(), limits: limits) { _, _, _ in try FakeModel() }
    stage.start()
    defer { stage.stop() }
    let plain = stage.watch(), debug = stage.watch(input: true)
    Thread.sleep(forTimeInterval: 0.05)  // past the wake a new watcher gives the stage's thread, which would skip the frame
    serve(stage, frame: 0, pixel: 77, totalUs: 55_000)  // slow: the models pause, the view still sees the frame
    let shown = collect(stage, debug) { $0.contains { $0.first == UInt8(ascii: "I") } }.filter { $0.first == UInt8(ascii: "I") }
    let message = try #require(shown.first)
    #expect(message.count == 10 + 6 * 128 * 256)
    #expect(Array(message[1..<10]) == [0, 0, 0, 0, 0, 1, 128, 0, 6])  // frame 0, 256 by 128, 6 planes
    #expect(message.dropFirst(10).allSatisfy { $0 == 77 })
    #expect(!collect(stage, plain, 4).contains { $0.first == UInt8(ascii: "I") })
  }

  @Test("A slow driving frame pauses the models")
  func backsOff() throws {
    let seg = try FakeModel()
    var limits = DioramaStage.Limits()
    limits.backOff = 30
    let stage = DioramaStage(models: try models(), limits: limits) { _, _, _ in seg }
    stage.start()
    defer { stage.stop() }
    let w = stage.watch()
    serve(stage, frame: 0, totalUs: 55_000)
    Thread.sleep(forTimeInterval: 0.2)
    serve(stage, frame: 2)
    _ = collect(stage, w, 3)
    #expect(seg.runs == 0)
    #expect(stage.state.hasPrefix("paused"))
  }
  @Test("Every frame, two at once: a second engine, every frame segmented, never more than two running, in frame order")
  func everyFrame() throws {
    let running = RunCount(), built = Mutex(0)
    var limits = DioramaStage.Limits()
    (limits.segEvery, limits.inFlight) = DioramaStage.Limits.everyFrame
    let stage = DioramaStage(models: try models(), limits: limits) { _, _, _ in
      built.withLock { $0 += 1 }
      return try SlowModel(seconds: 0.08, running: running)
    }
    stage.start()
    defer { stage.stop() }
    let w = stage.watch()
    // a frame every 50 ms, with each run taking 80: one engine alone could not keep up
    var got: [Data] = []
    for f in UInt32(0)..<40 {
      serve(stage, frame: f)
      Thread.sleep(forTimeInterval: 0.05)
      got += stage.next(w, timeout: 0) ?? []
    }
    got += collect(stage, w, until: { _ in false })
    let segs = got.filter { $0.first == UInt8(ascii: "S") }.map { $0[1...4].withUnsafeBytes { $0.loadUnaligned(as: UInt32.self) } }
    #expect(built.withLock { $0 } == 2)
    #expect(running.most == 2)
    #expect(segs == segs.sorted() && Set(segs).count == segs.count)
    // the first frames go by while the engines are made
    #expect(segs.count >= 25)
  }

  @Test("Asked for every frame while running, the stage makes the second engine then")
  func switches() throws {
    let running = RunCount(), built = Mutex(0)
    let stage = DioramaStage(models: try models()) { _, _, _ in
      built.withLock { $0 += 1 }
      return try SlowModel(seconds: 0.08, running: running)
    }
    stage.start()
    defer { stage.stop() }
    let w = stage.watch()
    for f in UInt32(0)..<10 {
      serve(stage, frame: f)
      Thread.sleep(forTimeInterval: 0.05)
    }
    #expect(built.withLock { $0 } == 1 && running.most == 1)
    stage.segment(every: DioramaStage.Limits.everyFrame.segEvery, inFlight: DioramaStage.Limits.everyFrame.inFlight)
    for f in UInt32(10)..<30 {
      serve(stage, frame: f)
      Thread.sleep(forTimeInterval: 0.05)
    }
    _ = collect(stage, w)
    #expect(built.withLock { $0 } == 2)
    #expect(running.most == 2)
  }
}

/// How many segmentations run at once, and the most there were.
final class RunCount: Sendable {
  private let state = Mutex((now: 0, most: 0))
  var most: Int { state.withLock { $0.most } }
  func begin() {
    state.withLock {
      $0.now += 1
      $0.most = max($0.most, $0.now)
    }
  }
  func end() { state.withLock { $0.now -= 1 } }
}

/// A segmentation that takes its time, counting how many run at once.
final class SlowModel: EngineCore, @unchecked Sendable {
  let seconds: Double
  let running: RunCount

  init(seconds: Double, running: RunCount) throws {
    self.seconds = seconds
    self.running = running
    let input = TensorSpec(name: "frame", type: .float16, shape: [1, 6, 128, 256])
    let outputs = [
      TensorSpec(name: "qmap", type: .float, shape: [1, 128, 256]), TensorSpec(name: "label", type: .float, shape: [200]),
      TensorSpec(name: "score", type: .float, shape: [200]),
    ]
    try super.init(inputs: ["frame": input], outputs: Dictionary(uniqueKeysWithValues: outputs.map { ($0.name, $0) }))
  }

  override func run() throws {
    running.begin()
    Thread.sleep(forTimeInterval: seconds)
    running.end()
  }
}

#if canImport(CoreML)
  import CoreML

  @Suite("Diorama's CoreML engine")
  struct CoreMLEngineTests {
    /// Fixtures/tiny.mlpackage: `export_coreml` of a module whose qmap is the frame's first two planes added.
    func package() throws -> URL {
      try #require(Bundle.module.url(forResource: "tiny", withExtension: "mlpackage", subdirectory: "Fixtures"))
    }

    @Test("The packed fp16 frame in, float32 outputs out, from the host buffers")
    func runs() throws {
      let cache = FileManager.default.temporaryDirectory.appending(path: "coreml-\(UUID().uuidString)")
      defer { try? FileManager.default.removeItem(at: cache) }
      let engine = try CoreMLEngine(package: try package(), cache: cache)
      #expect(engine.inputs["frame"]?.type == .float16)
      #expect(engine.inputs["frame"]?.shape == [1, 6, 128, 256])
      #expect(engine.outputs["qmap"]?.type == .float)
      let frame = try #require(engine.hostInput("frame")).assumingMemoryBound(to: UInt16.self)
      let plane = 128 * 256
      for i in 0..<(6 * plane) { frame[i] = DioramaStage.float16Bits(Float(i / plane + 1)) }  // plane k holds k + 1
      frame[7] = DioramaStage.float16Bits(100)  // one pixel of the first plane
      try engine.run()
      let qmap = try #require(engine.output("qmap")).assumingMemoryBound(to: Float.self)
      #expect(qmap[0] == 3)
      #expect(qmap[7] == 102)
      #expect(qmap[plane - 1] == 3)
    }

    @Test("The compile is kept, and made again when the package changes")
    func cachesTheCompile() throws {
      let cache = FileManager.default.temporaryDirectory.appending(path: "coreml-\(UUID().uuidString)")
      defer { try? FileManager.default.removeItem(at: cache) }
      let first = try CoreMLEngine.compiled(try package(), cache: cache)
      let made = try FileManager.default.attributesOfItem(atPath: first.path)[.creationDate] as? Date
      Thread.sleep(forTimeInterval: 0.05)
      let again = try CoreMLEngine.compiled(try package(), cache: cache)
      #expect(again == first)
      #expect(try FileManager.default.attributesOfItem(atPath: again.path)[.creationDate] as? Date == made)
      try "stale".write(to: cache.appending(path: "tiny.stamp"), atomically: true, encoding: .utf8)
      _ = try CoreMLEngine.compiled(try package(), cache: cache)
      #expect(try FileManager.default.attributesOfItem(atPath: first.path)[.creationDate] as? Date != made)
    }
  }
#endif
