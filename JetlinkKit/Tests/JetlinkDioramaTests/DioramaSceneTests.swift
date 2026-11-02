import Foundation
import Testing

@testable import JetlinkDiorama

/// A synthetic drive, made here so the tests need no recording: a curving road
/// with four lane lines, a lead that comes and goes and brakes now and then,
/// a speed that crosses the views' threshold both ways, and a segmentation
/// with the road, its paint, a crosswalk passed on the way and the lead's car.
enum SyntheticDrive {
  static let fields: [(String, Int)] = [
    ("plan", 990), ("lane_lines", 528), ("lane_lines_prob", 8), ("road_edges", 264), ("lead", 144), ("lead_prob", 3), ("pose", 12),
  ]
  static let labels = ["Road", "Lane Marking - General", "Lane Marking - Crosswalk", "Car", "Sidewalk", "Curb", "Sky"]
  /// The queries the segmentation answers with, one per label.
  static let road: UInt8 = 0, paint: UInt8 = 1, crosswalk: UInt8 = 2, car: UInt8 = 3, sidewalk: UInt8 = 4, curb: UInt8 = 5

  static var meta: Data {
    let json: [String: Any] = [
      "version": 1, "fps": 20, "live": true, "height": 1.22,
      "frame": ["w": 512, "h": 256, "f": 910, "cx": 256, "cy": 47.6],
      "wide": ["w": 512, "h": 256, "f": 455, "cx": 256, "cy": 151.8, "from_device": [0, 0.02, -0.016]],
      "drive": fields.map { [$0.0, $0.1] as [Any] },
      "seg": [
        "ds": 2, "labels": labels, "things": ["Car": "car"],
        "road_class": ["Road": 1, "Lane Marking - General": 2, "Lane Marking - Crosswalk": 3, "Curb": 4, "Sidewalk": 5],
      ],
    ]
    return Data("M".utf8) + (try! JSONSerialization.data(withJSONObject: json))
  }

  static func speed(_ t: Double) -> Double { 12 + 7 * sin(t / 3) }
  static func curve(_ t: Double) -> Double { 0.0015 * sin(t / 5) }
  static func leadX(_ t: Double) -> Double { 20 + 3 * sin(t / 2) }
  /// A little deterministic noise, for the smoother to smooth.
  static func noise(_ k: Int, _ i: Int) -> Double { 0.05 * sin(Double(k * 31 + i * 17)) }

  static func drive(_ k: Int) -> Data {
    let t = Double(k) / 20, v = speed(t), c = curve(t)
    var values: [Float] = []
    // plan, 33 x 15 and its log stds
    var plan = [Double](repeating: 0, count: 495)
    for i in 0..<33 {
      let ahead = 10 * pow(Double(i) / 32, 2), x = v * ahead
      plan[i * 15] = x + noise(k, i)
      plan[i * 15 + 1] = c * x * x
      plan[i * 15 + 2] = 0.01 * x
      plan[i * 15 + 3] = v
      plan[i * 15 + 6] = 1.5 * cos(t / 3)
    }
    values += plan.map(Float.init) + [Float](repeating: -1, count: 495)
    // lane lines, 4 x 33 x (y, z), then road edges, 2 x 33 x 2
    func lines(_ offsets: [Double]) -> [Float] {
      var out: [Float] = []
      for o in offsets { for i in 0..<33 { out += [Float(o + c * Diorama.xIdxs[i] * Diorama.xIdxs[i] + noise(k, i)), 0] } }
      return out + [Float](repeating: -1.5, count: out.count)
    }
    values += lines([-5.4, -1.8, 1.8, 5.4])
    values += [-2, 3, -2, 3, -2, 3, -2, -0.5]
    values += lines([-7.2, 7.2])
    // lead, 3 x 6 x (x, y, v, a), then stds
    var lead = [Float](repeating: 0, count: 72)
    for s in 0..<3 {
      for i in 0..<6 {
        let at = (s * 6 + i) * 4
        lead[at] = Float(leadX(t) + Double(i))
        lead[at + 1] = 0.2
        lead[at + 2] = Float(v)
        lead[at + 3] = Float(-1.5 * sin(t))
      }
    }
    values += lead + [Float](repeating: -1, count: 72)
    values += [t > 8 && t < 9.5 ? -2 : 2, 0, 0]
    values += [Float(v), 0, 0, 0, 0, Float(0.02 * sin(t / 4))] + [Float](repeating: -2, count: 6)
    var data = Data("D".utf8)
    append(&data, UInt32(k))
    append(&data, Float(t))
    append(&data, Float(12.5))
    for x in values { append(&data, x) }
    return data
  }

  /// The segmentation of frame `k`, seen from where the car is then: `x` metres along the road from the start.
  static func seg(_ k: Int, at x0: Double) -> Data {
    let w = 256, h = 128, t = Double(k) / 20
    var qmap = [UInt8](repeating: 255, count: w * h)
    for (i, hit) in ground.enumerated() {
      guard let (x, y) = hit else { continue }
      let world = x0 + x, lane = abs(abs(y) - 1.8)
      let q: UInt8
      if x - leadX(t) > 0 && x - leadX(t) < 20 && abs(y - 0.2) < 0.9 {
        q = car
      } else if world > 120 && world < 124 && abs(y) < 7 {
        q = crosswalk
      } else if lane < 0.12 {
        q = paint
      } else if abs(y) < 7.2 {
        q = road
      } else if abs(y) < 7.5 {
        q = curb
      } else if abs(y) < 10 {
        q = sidewalk
      } else {
        q = 255
      }
      qmap[i] = q
    }
    var runs: [(UInt8, Int)] = []
    for q in qmap {
      if let last = runs.last, last.0 == q, last.1 < 256 { runs[runs.count - 1].1 += 1 } else { runs.append((q, 1)) }
    }
    var data = Data("S".utf8)
    append(&data, UInt32(k))
    append(&data, Float(40))
    append(&data, UInt16(w))
    append(&data, UInt16(h))
    append(&data, UInt16(labels.count))
    data += Data((0..<labels.count).map { UInt8($0) })
    data += Data([UInt8](repeating: 230, count: labels.count))
    append(&data, UInt32(runs.count))
    for (q, n) in runs { data += Data([q, UInt8(n - 1)]) }
    return data
  }

  /// Where each query-map pixel's ray meets the road, ahead of the car, if it does.
  static let ground: [(Double, Double)?] = {
    let w = 256, h = 128, ds = 2.0
    let inverse = Diorama.m3inv(Diorama.m3mul([910, 0, 256, 0, 910, 47.6, 0, 0, 1], Diorama.viewFromDevice))
    return (0..<w * h).map { i in
      let d = Diorama.m3vec(inverse, [(Double(i % w) + 0.5) * ds, (Double(i / w) + 0.5) * ds, 1])
      guard d[2] > 1e-6 else { return nil }
      let s = 1.22 / d[2]
      return (d[0] * s, d[1] * s)
    }
  }()

  /// The drive's messages in the order a server sends them: the meta, then each frame's outputs, the segmentation
  /// of the frame before every other frame coming just ahead of them.
  static func messages(frames: Int) -> [Data] {
    var out = [meta], x = 0.0
    var along: [Int: Double] = [:]
    for k in 0..<frames {
      along[k] = x
      if k % 2 == 0, k > 0, let seen = along[k - 1] { out.append(seg(k - 1, at: seen)) }
      out.append(drive(k))
      x += speed(Double(k) / 20) / 20
    }
    return out
  }

  static func append<T: FixedWidthInteger>(_ data: inout Data, _ v: T) { withUnsafeBytes(of: v.littleEndian) { data.append(contentsOf: $0) } }
  static func append(_ data: inout Data, _ v: Float) { append(&data, v.bitPattern) }
}

/// What the tests compare of a frame's scene: the same numbers from the page's engine and from DioramaScene.
struct SceneSummary: Equatable, CustomStringConvertible {
  let frame: Int
  let values: [Double]
  /// The road map's bytes, weighted, every 50 frames: exact.
  let map: Int

  init(frame: Int, values: [Double], map: Int) {
    self.frame = frame
    self.values = values
    self.map = map
  }

  init(_ scene: DioramaScene) {
    let s = scene.state!
    var values = [s.px[32], s.py[20], s.out.laneLines[40], s.out.roadEdgesStds[0], s.view, s.speed, s.accel]
    values += [s.lead?.x ?? .nan, s.lead?.heading ?? .nan, s.lead.map { $0.braking ? 1 : 0 } ?? .nan]
    values += [Double(s.stripes.count), s.stripes.first?[0].x ?? .nan, s.stripes.last?[2].y ?? .nan, s.pose.x, s.pose.y, s.pose.z]
    var map = 0
    if s.frame % 50 == 0, let m = scene.roadmap {
      for i in 0..<m.a.count { map += Int(m.a[i]) * ((i % 7) + 1) + Int(m.b[i]) }
    }
    self.init(frame: s.frame, values: values, map: map)
  }

  static func == (a: Self, b: Self) -> Bool {
    a.frame == b.frame && a.map == b.map && a.values.count == b.values.count
      && zip(a.values, b.values).allSatisfy { ($0.isNaN && $1.isNaN) || abs($0 - $1) <= 1e-9 * max(1, abs($0)) }
  }

  var description: String { "frame \(frame): \(values), map \(map)" }
}

@Suite("The scene")
struct DioramaSceneTests {
  @Test("Messages decode: a drive's floats, a segmentation's runs past 256, a meta's camera")
  func decodes() throws {
    let messages = SyntheticDrive.messages(frames: 3)
    guard case .meta(let meta) = DioramaMessage(messages[0]) else { Issue.record("no meta"); return }
    #expect(meta.frame?.f == 910)
    #expect(meta.drive.map(\.count) == [990, 528, 8, 264, 144, 3, 12])
    #expect(meta.seg?.roadClass["Curb"] == 4)
    guard case .drive(let frame, _, let ms, let values) = DioramaMessage(messages[1]) else { Issue.record("no drive"); return }
    #expect(frame == 0 && ms == 12.5 && values.count == 1949)
    guard case .seg(let seg) = DioramaMessage(messages[3]) else { Issue.record("no seg"); return }
    #expect(seg.frame == 1 && seg.w == 256 && seg.h == 128 && seg.qmap.count == 256 * 128)
    #expect(seg.qmap[0] == 255 && seg.qmap[seg.qmap.count - 128] == SyntheticDrive.road)
    #expect(!seg.wide)
    // the same segmentation of the wide frame
    var wide = messages[3]
    wide[wide.startIndex] = UInt8(ascii: "W")
    guard case .seg(let w) = DioramaMessage(wide) else { Issue.record("no wide seg"); return }
    #expect(w.wide && w.frame == 1 && w.qmap == seg.qmap)
  }

  @Test("A segmentation counts only in its camera's view: the narrow frame's in the road view, the wide frame's in bird's-eye")
  func camerasByView() {
    for wide in [false, true] {
      let scene = DioramaScene()
      var used = 0, birdsEye = false
      for var m in SyntheticDrive.messages(frames: 400) {
        if m.first == UInt8(ascii: "S") {
          if wide { m[m.startIndex] = UInt8(ascii: "W") }
          birdsEye = scene.state.map { !$0.viewWhy.isEmpty } ?? false
          let before = scene.lastSeg?.frame
          scene.push(DioramaMessage(m))
          guard case .seg(let seg) = DioramaMessage(m) else { continue }
          scene.push(DioramaMessage(SyntheticDrive.drive(seg.frame + 1)))
          let took = scene.lastSeg?.frame == seg.frame && before != seg.frame
          #expect(took == (birdsEye == wide), "frame \(seg.frame): wide \(wide), bird's-eye \(birdsEye)")
          if took { used += 1 }
          continue
        }
        scene.push(DioramaMessage(m))
      }
      #expect(used > 20)
    }
  }

  @Test("Without the comma's camera height nothing is placed, and without its wide calibration no wide frame is used")
  func uncalibrated() throws {
    let json = try #require(try JSONSerialization.jsonObject(with: SyntheticDrive.meta.dropFirst()) as? [String: Any])
    for drop in ["height", "wide"] {
      var j = json
      j.removeValue(forKey: drop)
      let scene = DioramaScene()
      scene.push(DioramaMessage(Data("M".utf8) + (try JSONSerialization.data(withJSONObject: j))))
      #expect((scene.cam == nil) == (drop == "height"))
      #expect(scene.wideCam == nil)
    }
  }

  @Test("The scene follows the drive: the road view at speed, bird's-eye when slow or at a crosswalk, a lead and a crosswalk")
  func steps() {
    let scene = DioramaScene()
    var views: [Double] = [], leads = 0, stripes = 0
    for m in SyntheticDrive.messages(frames: 400) {
      scene.push(DioramaMessage(m))
      if case .drive = DioramaMessage(m), let s = scene.state {
        views.append(s.view)
        if s.lead != nil { leads += 1 }
        if !s.stripes.isEmpty { stripes += 1 }
      }
    }
    #expect(views.count == 400)
    #expect(views.contains(0) && views.contains(1))
    #expect(leads > 100)
    #expect(stripes > 10)
    #expect(scene.roadmap?.a.contains { $0 > 200 } == true)
  }

  @Test("A frame from before the last starts the scene over")
  func restarts() {
    let scene = DioramaScene()
    let messages = SyntheticDrive.messages(frames: 40)
    for m in messages { scene.push(DioramaMessage(m)) }
    #expect(scene.frame == 39)
    scene.push(DioramaMessage(SyntheticDrive.drive(5)))
    #expect(scene.frame == 5)
    #expect(scene.state?.pose == SIMD3(0, 0, 0))
  }
}

#if canImport(JavaScriptCore)
  import JavaScriptCore

  @Suite("The scene against the page's engine")
  struct DioramaSceneParityTests {
    /// diorama-engine.js in JavaScriptCore, fed the same bytes.
    final class PageEngine {
      let context = JSContext()!

      init() throws {
        let source = try #require(DioramaPage.file("diorama-engine.js")).data
        // JavaScriptCore has no TextDecoder; the engine only decodes UTF-8 JSON with it
        context.evaluateScript(
          """
          globalThis.TextDecoder = class { decode(u8) { let s = ''; for (const b of u8) s += String.fromCharCode(b); return decodeURIComponent(escape(s)); } };
          """)
        context.evaluateScript(String(decoding: source, as: UTF8.self))
        context.evaluateScript(
          """
          const E = DioramaEngine, engine = new E.Engine();
          function take(bytes) {
            const msg = E.decode(new Uint8Array(bytes));
            engine.push(msg);
            if (msg.kind !== 'D' || !engine.state) return null;
            const s = engine.state, l = s.lead, map = engine.roadmap;
            let sum = 0;
            if (s.frame % 50 === 0) for (let i = 0; i < map.a.length; i++) sum += map.a[i] * ((i % 7) + 1) + map.b[i];
            const st = s.stripes, nan = NaN;
            return [s.frame, sum, s.px[32], s.py[20], s.out.lane_lines[40], s.out.road_edges_stds[0], s.view, s.hud.v, s.hud.a,
              l ? l.x : nan, l ? l.heading : nan, l ? (l.braking ? 1 : 0) : nan,
              st.length, st.length ? st[0][0][0] : nan, st.length ? st[st.length - 1][2][1] : nan, s.pose[0], s.pose[1], s.pose[2]];
          }
          """)
        if let e = context.exception { throw ParityError.script(e.toString()) }
      }

      func take(_ data: Data) -> SceneSummary? {
        let out = context.objectForKeyedSubscript("take").call(withArguments: [[UInt8](data)])!
        guard out.isArray, let v = out.toArray() as? [NSNumber] else { return nil }
        let d = v.map(\.doubleValue)
        return SceneSummary(frame: Int(d[0]), values: Array(d[2...]), map: Int(d[1]))
      }
    }

    enum ParityError: Error { case script(String) }

    @Test("Frame by frame, the Swift scene is the page's: path, lanes, lead, crosswalk, pose and road map")
    func matchesThePage() throws {
      let page = try PageEngine()
      let scene = DioramaScene()
      var compared = 0, leads = 0, stripes = 0
      for m in SyntheticDrive.messages(frames: 400) {
        let expected = page.take(m)
        scene.push(DioramaMessage(m))
        guard let expected else { continue }
        let got = SceneSummary(scene)
        #expect(got == expected)
        if got != expected { break }
        compared += 1
        if !got.values[7].isNaN { leads += 1 }
        if got.values[10] > 0 { stripes += 1 }
      }
      #expect(compared == 400)
      #expect(leads > 100 && stripes > 10)
    }
  }
#endif
