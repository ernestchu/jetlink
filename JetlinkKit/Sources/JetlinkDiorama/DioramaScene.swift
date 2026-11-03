import Foundation

/// Diorama's scene from the server's messages (docs/diorama-internals.md), for the apps
/// to draw natively: diorama-engine.js in Swift, step for step, so the page
/// and the apps draw the same drive. The driving model's outputs give the
/// path, lanes, road edges, lead and ego motion; the segmentation gives the
/// road surface, which a road map holds around the car. The lead is drawn
/// from each frame's own prediction when the newest segmentation has a
/// vehicle where it is.
///
/// Road frame throughout: x forward, y right, z down, origin at the camera,
/// the road at z = height.
public enum Diorama {
  public static let fps = 20.0, dt = 1 / 20.0, mph = 2.23694
  public static let idxN = 33
  public static let xIdxs: [Double] = (0..<33).map { 192 * pow(Double($0) / 32, 2) }
  /// The ego car's bumpers, from the camera.
  public static let egoFront = 1.6, egoRear = -3.1
  static let planWidth = 15
  static let leadMhpN = 2, leadSel = 3, leadLen = 6, leadWidth = 4
  static let occluder: UInt8 = 255
  /// Frames: a segmentation older than this (0.5 s) confirms no lead.
  static let segFresh = 10
  /// A lead the segmentation alone shows (DioramaSeg.leadAhead): at most this far, metres
  public static let segLeadMax = 15.0
  static let segLeadHalf = 1.0, segLeadWidth = 0.5

  // MARK: small math

  @inline(__always) public static func clamp(_ x: Double, _ a: Double, _ b: Double) -> Double { min(max(x, a), b) }
  public static func smoothstep(_ e0: Double, _ e1: Double, _ x: Double) -> Double {
    let t = clamp((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)
  }
  static func sexp(_ x: Double) -> Double { exp(min(x, 11)) }
  /// JavaScript's Math.round: halves round up, -2.5 to -2.
  @inline(__always) static func jsRound(_ x: Double) -> Int { Int((x + 0.5).rounded(.down)) }
  static func sigmoid(_ x: Double) -> Double { 1 / (1 + sexp(-x)) }

  public static func interp(_ x: Double, _ xs: [Double], _ ys: [Double]) -> Double {
    let n = xs.count
    if x <= xs[0] { return ys[0] }
    if x >= xs[n - 1] { return ys[n - 1] }
    var lo = 0, hi = n - 1
    while hi - lo > 1 {
      let m = (lo + hi) >> 1
      if xs[m] <= x { lo = m } else { hi = m }
    }
    let d = xs[hi] - xs[lo]
    let t = (x - xs[lo]) / (d == 0 ? 1 : d)
    return ys[lo] + t * (ys[hi] - ys[lo])
  }

  /// numpy's default (linear) percentile of an unsorted array.
  static func percentile(_ arr: [Double], _ q: Double) -> Double {
    if arr.isEmpty { return .nan }
    let a = arr.sorted()
    let r = q / 100 * Double(a.count - 1), lo = Int(r.rounded(.down)), hi = Int(r.rounded(.up))
    return a[lo] + (a[hi] - a[lo]) * (r - Double(lo))
  }

  // 3x3 row-major
  public static func m3mul(_ m: [Double], _ n: [Double]) -> [Double] {
    var out = [Double](repeating: 0, count: 9)
    for i in 0..<3 { for j in 0..<3 { out[3 * i + j] = m[3 * i] * n[j] + m[3 * i + 1] * n[3 + j] + m[3 * i + 2] * n[6 + j] } }
    return out
  }
  public static func m3vec(_ m: [Double], _ v: [Double]) -> [Double] {
    [m[0] * v[0] + m[1] * v[1] + m[2] * v[2], m[3] * v[0] + m[4] * v[1] + m[5] * v[2], m[6] * v[0] + m[7] * v[1] + m[8] * v[2]]
  }
  static func m3inv(_ m: [Double]) -> [Double] {
    let (a, b, c, d, e, f, g, h, i) = (m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8])
    let co = [e * i - f * h, -(d * i - f * g), d * h - e * g]
    let det = a * co[0] + b * co[1] + c * co[2]
    return [co[0], -(b * i - c * h), b * f - c * e, co[1], a * i - c * g, -(a * f - c * d), co[2], -(a * h - b * g), a * e - b * d].map { $0 / det }
  }
  public static let viewFromDevice: [Double] = [0, 1, 0, 0, 0, 1, 1, 0, 0]
  /// openpilot's rot_from_euler: roll about x, then pitch about y, then yaw about z.
  static func rotFromEuler(_ r: Double, _ p: Double, _ y: Double) -> [Double] {
    let (cr, sr, cp, sp, cy, sy) = (cos(r), sin(r), cos(p), sin(p), cos(y), sin(y))
    return m3mul([cy, -sy, 0, sy, cy, 0, 0, 0, 1], m3mul([cp, 0, sp, 0, 1, 0, -sp, 0, cp], [1, 0, 0, 0, cr, -sr, 0, sr, cr]))
  }
}

// MARK: - messages

/// One of the server's messages, decoded.
public enum DioramaMessage: Sendable {
  case meta(DioramaMeta)
  case status(String)
  case input(frame: Int, width: Int, height: Int, packed: [UInt8])
  case drive(frame: Int, seconds: Double, ms: Double, values: [Float])
  case seg(DioramaSegMessage)
  case other

  public init(_ data: Data) {
    let b = [UInt8](data)
    guard let kind = b.first else {
      self = .other
      return
    }
    func u16(_ at: Int) -> Int { Int(b[at]) | Int(b[at + 1]) << 8 }
    func u32(_ at: Int) -> Int { u16(at) | u16(at + 2) << 16 }
    func f32(_ at: Int) -> Float { Float(bitPattern: UInt32(u32(at))) }
    switch kind {
    case UInt8(ascii: "M"):
      self = (try? JSONSerialization.jsonObject(with: Data(b[1...]))).flatMap { ($0 as? [String: Any]).map { .meta(DioramaMeta($0)) } } ?? .other
    case UInt8(ascii: "T"):
      self = .status(String(decoding: b[1...], as: UTF8.self))
    case UInt8(ascii: "I") where b.count >= 10:
      let w = u16(5), h = u16(7), planes = Int(b[9])
      self = .input(frame: u32(1), width: w, height: h, packed: Array(b[10..<min(b.count, 10 + w * h * planes)]))
    case UInt8(ascii: "D") where b.count >= 13:
      let n = (b.count - 13) / 4
      self = .drive(frame: u32(1), seconds: Double(f32(5)), ms: Double(f32(9)), values: (0..<n).map { f32(13 + 4 * $0) })
    case UInt8(ascii: "S") where b.count >= 15, UInt8(ascii: "W") where b.count >= 15:
      let w = u16(9), h = u16(11), nq = u16(13)
      var at = 15
      let label = Array(b[at..<at + nq])
      at += nq
      let score = Array(b[at..<at + nq])
      at += nq
      let runs = u32(at)
      at += 4
      var qmap = [UInt8](repeating: 0, count: w * h)
      var p = 0
      for _ in 0..<runs {
        let v = b[at], len = Int(b[at + 1]) + 1
        let end = min(p + len, qmap.count)
        if p < end { for i in p..<end { qmap[i] = v } }
        p += len
        at += 2
      }
      self = .seg(DioramaSegMessage(frame: u32(1), ms: Double(f32(5)), w: w, h: h, label: label, score: score, qmap: qmap, wide: kind == UInt8(ascii: "W")))
    default:
      self = .other
    }
  }

  /// A recorded drive: messages one after another, each after its u32 length.
  public static func file(_ data: Data) -> [Data] {
    let b = [UInt8](data)
    var out: [Data] = [], at = 0
    while at + 4 <= b.count {
      let n = Int(b[at]) | Int(b[at + 1]) << 8 | Int(b[at + 2]) << 16 | Int(b[at + 3]) << 24
      guard at + 4 + n <= b.count else { break }
      out.append(Data(b[at + 4..<at + 4 + n]))
      at += 4 + n
    }
    return out
  }
}

public struct DioramaSegMessage: Sendable {
  public let frame: Int, ms: Double, w: Int, h: Int
  public let label: [UInt8], score: [UInt8], qmap: [UInt8]
  /// Of the wide frame, not the narrow.
  public var wide = false
}

/// The meta message: the frame's camera (and the wide frame's, live), the driving model's heads, and the segmentation's classes.
public struct DioramaMeta: Sendable {
  public struct Frame: Sendable { public let w: Double, h: Double, f: Double, cx: Double, cy: Double }
  public struct Seg: Sendable {
    public let labels: [String], things: [String: String], roadClass: [String: Int], ds: Double
  }
  public let frame: Frame?, wide: Frame?
  /// The wide camera's roll, pitch and yaw from the device, as the comma calibrated them, where it said.
  public let wideFromDevice: [Double]?
  /// The camera's height above the road, as the comma calibrated it; nothing is placed without it.
  public let height: Double?
  public let drive: [(name: String, count: Int)]
  public let seg: Seg?
  public let route: String?

  public init(_ json: [String: Any]) {
    func num(_ v: Any?) -> Double? { (v as? NSNumber)?.doubleValue }
    func camera(_ v: Any?) -> Frame? {
      guard let f = v as? [String: Any], let w = num(f["w"]), let h = num(f["h"]), let fl = num(f["f"]), let cx = num(f["cx"]), let cy = num(f["cy"]) else {
        return nil
      }
      return Frame(w: w, h: h, f: fl, cx: cx, cy: cy)
    }
    frame = camera(json["frame"])
    wide = camera(json["wide"])
    wideFromDevice = ((json["wide"] as? [String: Any])?["from_device"] as? [Any]).flatMap { a in
      let v = a.compactMap { num($0) }
      return v.count == 3 ? v : nil
    }
    height = num(json["height"]).flatMap { $0 > 0 ? $0 : nil }
    drive = (json["drive"] as? [[Any]] ?? []).compactMap { pair in
      guard pair.count == 2, let name = pair[0] as? String, let n = num(pair[1]) else { return nil }
      return (name, Int(n))
    }
    if let s = json["seg"] as? [String: Any] {
      seg = Seg(
        labels: s["labels"] as? [String] ?? [], things: s["things"] as? [String: String] ?? [:],
        roadClass: (s["road_class"] as? [String: Any] ?? [:]).compactMapValues { num($0).map(Int.init) }, ds: num(s["ds"]) ?? 1)
    } else {
      seg = nil
    }
    route = (json["source"] as? [String: Any])?["route"] as? String
  }
}

// MARK: - driving model outputs

/// The heads openpilot's Parser makes of them (diorama/outputs.py does the same), flat, in openpilot's shapes.
public struct DioramaOutputs: Sendable {
  public var plan: [Double] = [], planStds: [Double] = []  // (33, 15): 0..2 position, 3..5 velocity, 6..8 acceleration
  public var laneLines: [Double] = [], laneLinesStds: [Double] = []  // (4, 33, 2): y, z
  public var roadEdges: [Double] = [], roadEdgesStds: [Double] = []  // (2, 33, 2)
  public var pose: [Double] = [], poseStds: [Double] = []
  public var laneLinesProb: [Double] = [], leadProb: [Double] = []
  public var lead: [Double] = [], leadStds: [Double] = []  // (3, 6, 4): x, y, v, a

  public func plan(_ i: Int, _ c: Int) -> Double { plan[i * Diorama.planWidth + c] }
  public static func line(_ l: [Double], _ line: Int, _ i: Int, _ c: Int) -> Double { l[(line * Diorama.idxN + i) * 2 + c] }
  public func lead(_ sel: Int, _ t: Int, _ c: Int) -> Double { lead[(sel * Diorama.leadLen + t) * Diorama.leadWidth + c] }

  init(values: [Float], fields: [(name: String, count: Int)]) {
    var raw: [String: ArraySlice<Float>] = [:]
    var at = 0
    for (name, n) in fields {
      raw[name] = values[min(at, values.count)..<min(at + n, values.count)]
      at += n
    }
    func mdn(_ r: ArraySlice<Float>) -> ([Double], [Double]) {
      let n = r.count / 2, s = r.startIndex
      return (r[s..<s + n].map(Double.init), r[s + n..<s + 2 * n].map { Diorama.sexp(Double($0)) })
    }
    if let r = raw["plan"] { (plan, planStds) = mdn(r) }
    if let r = raw["lane_lines"] { (laneLines, laneLinesStds) = mdn(r) }
    if let r = raw["road_edges"] { (roadEdges, roadEdgesStds) = mdn(r) }
    if let r = raw["pose"] { (pose, poseStds) = mdn(r) }
    if let r = raw["lane_lines_prob"] { laneLinesProb = r.map { Diorama.sigmoid(Double($0)) } }
    if let r = raw["lead_prob"] { leadProb = r.map { Diorama.sigmoid(Double($0)) } }
    if let r = raw["lead"].map(Array.init) {
      let sel = Diorama.leadSel
      if r.count != 2 * sel * Diorama.leadLen * Diorama.leadWidth {
        let per = r.count / Diorama.leadMhpN, n = (per - sel) / 2
        lead = [Double](repeating: 0, count: sel * n)
        leadStds = [Double](repeating: 0, count: sel * n)
        for s in 0..<sel {
          // the hypothesis the softmax over hypotheses weights most, for this selection
          var best = 0
          for hyp in 1..<Diorama.leadMhpN where r[hyp * per + per - sel + s] > r[best * per + per - sel + s] { best = hyp }
          for i in 0..<n {
            lead[s * n + i] = Double(r[best * per + i])
            leadStds[s * n + i] = Diorama.sexp(Double(r[best * per + n + i]))
          }
        }
      } else {
        (lead, leadStds) = mdn(r[...])
      }
    }
  }
}

/// Element-wise random walk Kalman filter: R from the model's std, Q = (k std)^2.
struct KalmanFilter {
  let k: Double
  var x: [Double]? = nil, cov: [Double] = []

  mutating func update(_ z: [Double], _ std: [Double]) -> [Double] {
    guard var x, x.count == z.count else {
      x = z
      cov = std.map { pow(max($0, 1e-3), 2) }
      return z
    }
    for i in 0..<z.count {
      let r = pow(max(std[i], 1e-3), 2), prior = cov[i] + k * k * r, gain = prior / (prior + r)
      x[i] += gain * (z[i] - x[i])
      cov[i] = (1 - gain) * prior
    }
    self.x = x
    return x
  }
}

struct OutputSmoother {
  var plan, laneLines, roadEdges, laneProb, edgeStd: KalmanFilter
  init(k: Double) { (plan, laneLines, roadEdges, laneProb, edgeStd) = (.init(k: k), .init(k: k), .init(k: k), .init(k: k), .init(k: k)) }

  mutating func smooth(_ o: DioramaOutputs) -> DioramaOutputs {
    var s = o
    let tenth = { (a: [Double]) in [Double](repeating: 0.1, count: a.count) }
    s.plan = plan.update(o.plan, o.planStds)
    s.laneLines = laneLines.update(o.laneLines, o.laneLinesStds)
    s.roadEdges = roadEdges.update(o.roadEdges, o.roadEdgesStds)
    s.laneLinesProb = laneProb.update(o.laneLinesProb, tenth(o.laneLinesProb))
    s.roadEdgesStds = edgeStd.update(o.roadEdgesStds, tenth(o.roadEdgesStds))
    return s
  }
}

// MARK: - the camera that saw the frame

/// The driving model's frames are already calibrated: road point -> pixel is K view_from_device, K the narrow
/// frame's or the wide's. The comma warps the wide camera with the narrow one's calibration, so the wide frame is
/// turned by the wide camera's angles from the device too (`rotation`, the model's wide_from_device_euler).
public struct DioramaFrameCamera: Sendable {
  public let w: Double, h: Double, f: Double, height: Double
  let proj: [Double], inverse: [Double]

  init(_ meta: DioramaMeta, frame f: DioramaMeta.Frame, rotation: [Double]? = nil) {
    w = f.w
    h = f.h
    self.f = f.f
    height = meta.height ?? 0
    let view = rotation.map { Diorama.m3mul(Diorama.viewFromDevice, $0) } ?? Diorama.viewFromDevice
    proj = Diorama.m3mul([f.f, 0, f.cx, 0, f.f, f.cy, 0, 0, 1], view)
    inverse = Diorama.m3inv(proj)
  }

  func project(_ x: Double, _ y: Double, _ z: Double) -> (u: Double, v: Double, w: Double) {
    let w = proj[6] * x + proj[7] * y + proj[8] * z
    return ((proj[0] * x + proj[1] * y + proj[2] * z) / w, (proj[3] * x + proj[4] * y + proj[5] * z) / w, w)
  }

  /// The road point where pixel (u, v)'s ray meets the plane z = gz, or nil.
  func groundHit(_ u: Double, _ v: Double, _ gz: Double) -> (Double, Double, Double)? {
    let d = Diorama.m3vec(inverse, [u, v, 1])
    guard d[2] > 1e-6 else { return nil }
    let t = gz / d[2]
    return t > 0 ? (d[0] * t, d[1] * t, d[2] * t) : nil
  }
}

// MARK: - the ego car's path through the world

/// pos_k + R(yaw_k) p_ego, integrated from the model's pose (velocity, yaw rate). Kept for the last few seconds.
struct EgoTrail {
  /// World x, y and yaw by frame, the newest last.
  private(set) var poses: [Int: SIMDPose] = [:]
  private var order: [Int] = []
  private var last: Int? = nil

  typealias SIMDPose = (x: Double, y: Double, yaw: Double)

  mutating func add(_ frame: Int, _ pose: [Double]) {
    guard let last, let p = poses[last] else {
      poses[frame] = (0, 0, 0)
      order.append(frame)
      self.last = frame
      return
    }
    var (x, y, yaw) = p
    let steps = Int(Diorama.clamp(Double(frame - last), 1, 20))  // a gap in a live feed: assume the car kept going
    for _ in 0..<steps {
      let t0 = pose[0] * Diorama.dt, t1 = pose[1] * Diorama.dt, c = cos(yaw), s = sin(yaw)
      x += c * t0 - s * t1
      y += s * t0 + c * t1
      yaw += pose[5] * Diorama.dt
    }
    poses[frame] = (x, y, yaw)
    order.append(frame)
    self.last = frame
    while let first = order.first, first < frame - 200 {
      poses[first] = nil
      order.removeFirst()
    }
  }

  func pose(_ frame: Int) -> SIMDPose? { poses[frame] }
}

// MARK: - segmentation

/// The query map as the road map's classes (0 other, 1 road, 2 lane paint, 3 crosswalk, 4 curb, 5 sidewalk,
/// `occluder` under a thing) and the things, each by its kind.
public struct DioramaSeg: Sendable {
  public let w: Int, h: Int, ds: Double
  /// Of the wide frame, not the narrow.
  public let wide: Bool
  public let cls: [UInt8], qmap: [UInt8]
  public let qThing: [String?]

  public init(_ msg: DioramaSegMessage, _ seg: DioramaMeta.Seg) {
    var qClass = [UInt8](repeating: 0, count: 256)
    var qThing = [String?](repeating: nil, count: 256)
    for q in 0..<msg.label.count {
      let l = Int(msg.label[q])
      if l == 255 || l >= seg.labels.count { continue }
      let name = seg.labels[l]
      if let thing = seg.things[name] {
        qClass[q] = Diorama.occluder
        qThing[q] = thing
      } else {
        qClass[q] = UInt8(clamping: seg.roadClass[name] ?? 0)
      }
    }
    w = msg.w
    h = msg.h
    wide = msg.wide
    ds = seg.ds
    qmap = msg.qmap
    cls = msg.qmap.map { $0 == 255 ? 0 : qClass[Int($0)] }
    self.qThing = qThing
  }

  /// Does the segmentation have a vehicle where the model puts the lead? The rear of the lead, (x, y) on the road
  /// at road height `road` (z down), seen from 0.2 to 1.4 m above it, within 2 query-map pixels either side; and
  /// that vehicle stands on the road at about the lead's distance (within 25%, or 3 m), so a car further on, seen
  /// past empty road at the lead's distance, does not count. Where it stands is the lowest of the vehicle's pixels
  /// below the hit across a car's width (1.2 m either side: its wheels, where the road shows under its middle), each
  /// column's run down from the hit's row: a vehicle behind it, in the same region of a semantic segmentation, is
  /// higher and changes nothing. One whose bottom is cut off by the frame's edge counts.
  func seesVehicle(_ cam: DioramaFrameCamera, _ x: Double, _ y: Double, _ road: Double) -> Bool {
    var up = 0.2
    while up <= 1.41 {
      defer { up += 0.2 }
      let (u, v, depth) = cam.project(x, y, road - up)
      if depth <= 0 { continue }
      let cu = Diorama.jsRound(u / ds), cv = Int((v / ds).rounded(.down))
      if cv < 0 || cv >= h { continue }
      let lo = max(cu - 2, 0), hi = min(cu + 2, w - 1)
      if lo > hi { continue }
      guard
        let q = (lo...hi).lazy.map({ self.qmap[cv * self.w + $0] }).first(where: { $0 != 255 && self.qThing[Int($0)].map { $0 != "person" } == true })
      else { continue }
      // the vehicle's bottom across a car's width
      let half = Int((cam.f * 1.2 / depth / ds).rounded(.up))
      var bottom = -1, at = cu
      for c in stride(from: max(cu - half, 0), through: min(cu + half, w - 1), by: 1) where qmap[cv * w + c] == q {
        var r = cv
        while r + 1 < h && qmap[(r + 1) * w + c] == q { r += 1 }
        if r > bottom {
          bottom = r
          at = c
        }
      }
      if bottom >= h - 1 { return true }
      if let hit = cam.groundHit((Double(at) + 0.5) * ds, Double(bottom + 1) * ds, road), abs(hit.0 - x) <= max(3, 0.25 * x) { return true }
    }
    return false
  }

  /// The nearest vehicle the segmentation has in the ego car's path, where the model has no lead: the lowest row of
  /// vehicle pixels (not a person) on the road within segLeadHalf of the path (pathY, by distance), at
  /// least segLeadWidth wide there, between the ego car's front and segLeadMax ahead. Where it meets the road is
  /// its distance; one cut off by the frame's bottom is at most the frame's nearest road. The rows run up from the
  /// bottom, so a vehicle behind it, higher up, changes nothing. Its distance, or nil.
  func leadAhead(_ cam: DioramaFrameCamera, _ pathY: (Double) -> Double, _ road: Double) -> Double? {
    for r in stride(from: h - 1, through: 0, by: -1) {
      var near = false, width = 0.0, sum = 0.0, n = 0
      for c in 0..<w {
        guard let g = cam.groundHit((Double(c) + 0.5) * ds, Double(r + 1) * ds, road), g.0 <= Diorama.segLeadMax else { continue }
        near = true
        if g.0 < Diorama.egoFront || abs(g.1 - pathY(g.0)) > Diorama.segLeadHalf { continue }
        let q = qmap[r * w + c]
        guard q != 255, let thing = qThing[Int(q)], thing != "person" else { continue }
        width += ds * g.0 / cam.f
        sum += g.0
        n += 1
      }
      if width >= Diorama.segLeadWidth { return sum / Double(n) }
      if !near { return nil }  // above the rows that reach the road within segLeadMax
    }
    return nil
  }
}

// MARK: - the scene, frame by frame

/// The scene from the messages, as diorama-engine.js's Engine makes it. Not
/// thread safe: one thread pushes, and the renderer reads under the same lock.
public final class DioramaScene {
  /// The lead, as this frame predicts it: the distance to its rear and its offset, facing along its lane. `seg`:
  /// the segmentation alone shows it, the model having none there.
  public struct Lead: Sendable {
    public let x: Double, y: Double, heading: Double, braking: Bool, seg: Bool
  }

  /// One frame's scene.
  public struct State: Sendable {
    public let frame: Int, seconds: Double
    public let out: DioramaOutputs
    public let px: [Double], py: [Double], pz: [Double]
    public let lead: Lead?
    /// The ego car in the world: x, y, yaw.
    public let pose: SIMD3<Double>
    /// Crosswalk bars, in the world.
    public let stripes: [[SIMD2<Double>]]
    /// 0 the road view behind the car, 1 bird's-eye; and while bird's-eye, why (DioramaViewChooser's cue), or "".
    public let view: Double, viewWhy: String
    /// Speed (m/s), acceleration (m/s²), the lead's distance.
    public let speed: Double, accel: Double, leadDistance: Double?
  }

  public struct Timing: Sendable {
    public var model: [Double] = [], seg: [Double] = [], segFrames: [Int] = [], lag: [Int] = []
    public init() {}
  }

  public var panSeconds = 3.0
  /// What picks the road view or bird's-eye; its numbers can be changed before a drive starts.
  public var viewOptions = DioramaViewChooser.Options()

  public private(set) var meta: DioramaMeta?
  public private(set) var cam: DioramaFrameCamera?
  /// The wide frame's camera, turned by the wide camera's angles from the device as the comma calibrated them: only
  /// where the meta has both. Without, the wide frame's segmentations are not used.
  public private(set) var wideCam: DioramaFrameCamera?
  public private(set) var roadmap: DioramaRoadMap?
  public private(set) var state: State?
  public private(set) var frame = -1
  public private(set) var timing = Timing()
  /// The newest segmentation read, and the frame it is of.
  public private(set) var lastSeg: (frame: Int, seg: DioramaSeg)?

  private var smoother = OutputSmoother(k: 0.3)
  private var trail = EgoTrail()
  private var pendingSeg: DioramaSegMessage?
  private var stripes: [[SIMD2<Double>]] = [], stripesFrame = -1
  private var speed: Double?, view: Double?
  private var chooser = DioramaViewChooser()

  public init() {}

  public func reset() {
    // the camera's height is the comma's word (its calibration), never assumed: without it, nothing is placed
    cam = meta.flatMap { meta in meta.height == nil ? nil : meta.frame.map { DioramaFrameCamera(meta, frame: $0) } }
    wideCam = meta.flatMap { meta in
      guard cam != nil, let r = meta.wideFromDevice, let wide = meta.wide else { return nil }
      return DioramaFrameCamera(meta, frame: wide, rotation: Diorama.rotFromEuler(r[0], r[1], r[2]))
    }
    smoother = OutputSmoother(k: 0.3)
    trail = EgoTrail()
    roadmap = cam.map { DioramaRoadMap(cam: $0) }
    pendingSeg = nil
    lastSeg = nil
    stripes = []
    stripesFrame = -1
    speed = nil
    view = nil
    chooser = DioramaViewChooser(viewOptions)
    frame = -1
    state = nil
    timing = Timing()
  }

  public func push(_ msg: DioramaMessage) {
    switch msg {
    case .meta(let meta):
      self.meta = meta
      reset()
    case .drive(let frame, let seconds, let ms, let values) where meta != nil:
      step(frame: frame, seconds: seconds, ms: ms, values: values)
    case .seg(let seg) where meta != nil:
      // the narrow frame's in the road view, the wide frame's in bird's-eye; the other is not used
      guard seg.wide == (chooser.target == 1) else { break }
      pendingSeg = seg
      keepLast(&timing.seg, seg.ms, 50)
    default:
      break
    }
  }

  /// One 20 Hz frame of the driving model's outputs, and the newest segmentation that has arrived since the last.
  private func step(frame k: Int, seconds: Double, ms: Double, values: [Float]) {
    guard let meta, !meta.drive.isEmpty else { return }  // a live server's meta before its first frame
    if frame >= 0 && (k <= frame || k > frame + 40) { reset() }  // a seek, a restart, or a long gap
    guard let cam, let roadmap else { return }
    let raw = DioramaOutputs(values: values, fields: meta.drive)
    guard raw.plan.count >= Diorama.idxN * Diorama.planWidth, raw.pose.count >= 6 else { return }
    let out = smoother.smooth(raw)
    keepLast(&timing.model, ms, 50)
    trail.add(k, raw.pose)
    frame = k

    let n = Diorama.idxN, xs = Diorama.xIdxs
    let px = (0..<n).map { out.plan($0, 0) }, py = (0..<n).map { out.plan($0, 1) }, pz = (0..<n).map { out.plan($0, 2) }
    let pv = (0..<n).map { out.plan($0, 3) }
    let moving = px[n - 1] > px[0] + 1
    let zpath: (Double) -> Double = moving ? { Diorama.interp($0, px, pz) } : { _ in 0 }
    let laneY: [Double] = (0..<n).map { i -> Double in
      (DioramaOutputs.line(out.laneLines, 1, i, 0) + DioramaOutputs.line(out.laneLines, 2, i, 0)) / 2
    }
    let slopes: [Double] = (1..<n).map { i -> Double in (laneY[i] - laneY[i - 1]) / (xs[i] - xs[i - 1]) }
    let slopeXs = Array(xs[1...])
    let laneHeading: (Double) -> Double = { x in atan(Diorama.interp(x, slopeXs, slopes)) }

    // the newest segmentation, into the road map from the pose it was seen at
    let segMsg = pendingSeg
    pendingSeg = nil
    if let segMsg, let seen = trail.pose(segMsg.frame), let segInfo = meta.seg, let segCam = segMsg.wide ? wideCam : cam {
      keepLast(&timing.segFrames, segMsg.frame, 20)
      keepLast(&timing.lag, k - segMsg.frame, 20)
      let seg = DioramaSeg(segMsg, segInfo)
      roadmap.observe(seen, seg, zpath, segCam)
      lastSeg = (segMsg.frame, seg)
    }

    // crosswalk stripes, re-cut as the map fills in
    let pNow = trail.pose(k)!
    if segMsg != nil && k - stripesFrame >= 6 {
      stripes = roadmap.crosswalks(pNow, laneHeading)
      stripesFrame = k
    }

    // this frame's lead, as this frame predicts it: the distance to its rear and its offset, facing along the
    // ego lane there (its lines reach 192 m; the plan, laid out in time, can end short of a slow car's lead and
    // turn back at its end), braking when its predicted acceleration says so. Only where the newest segmentation,
    // at most segFresh frames old, has a vehicle. Without one, the vehicle that segmentation has in the planned
    // path within segLeadMax, if any (DioramaSeg.leadAhead): on the path, facing along the lane, never braking, marked seg.
    var lead: Lead? = nil
    let fresh = lastSeg.flatMap { k - $0.frame <= Diorama.segFresh ? $0.seg : nil }
    if let fresh, let segCam = fresh.wide ? wideCam : cam {
      if let p = raw.leadProb.first, p > 0.5, !raw.lead.isEmpty {
        let x = raw.lead(0, 0, 0), y = raw.lead(0, 0, 1)
        if fresh.seesVehicle(segCam, x, y, cam.height + zpath(x)) {
          lead = Lead(x: x, y: y, heading: laneHeading(x), braking: raw.lead(0, 0, 3) < -0.8, seg: false)
        }
      }
      let pathY: (Double) -> Double = moving ? { Diorama.interp($0, px, py) } : { _ in 0 }
      if lead == nil, let x = fresh.leadAhead(segCam, pathY, cam.height) {
        lead = Lead(x: x, y: pathY(x), heading: laneHeading(x), braking: false, seg: true)
      }
    }

    // road view or bird's-eye (DioramaViewChooser), panning between the two
    let v = out.plan(0, 3)
    let speed = self.speed.map { 0.9 * $0 + 0.1 * v } ?? v
    self.speed = speed
    let pose = SIMD3(pNow.x, pNow.y, pNow.yaw)
    let target = chooser.step(chooser.cue(speed, px, py, pv, stripes, pose, lead), speed)
    let stepS = Diorama.dt / panSeconds
    if let view {
      let sign: Double = target > view ? 1 : target < view ? -1 : 0
      self.view = Diorama.clamp(view + sign * stepS, min(view, target), max(view, target))
    } else {
      view = target
    }

    state = State(
      frame: k, seconds: seconds, out: out, px: px, py: py, pz: pz, lead: lead,
      pose: pose, stripes: stripes, view: Diorama.smoothstep(0, 1, view ?? 0), viewWhy: target == 1 ? chooser.why : "",
      speed: speed, accel: out.plan(0, 6), leadDistance: lead?.x)
  }
}

/// Road view or bird's-eye: bird's-eye while the scene asks for a look around, the road view while cruising, as
/// diorama-engine.js's ViewChooser, step for step. Cues, each a reason shown beside the view: stopped; a turn coming
/// (the planned path bending within the next 40 m, and slowing for it or sharply: a sweeping ramp or bend taken at
/// speed is not one); a crosswalk ahead within a few seconds, going slowly or slowing for it (not one driven straight
/// over); a close lead in slow traffic; or simply slow. A cue must hold a moment before the view pans down, the road
/// view comes back only after none for a while and at speed, and either view is held a few seconds at least.
public struct DioramaViewChooser: Sendable {
  public struct Options: Sendable {
    public var stopped = 3.0, slow = 12.0, roadAbove = 18.0  // mph
    // a bend of turnDeg within turnAhead m that the plan slows below turnSlow mph for, or one of turnSharp degrees;
    // either only below turnBelow mph
    public var turnDeg = 20.0, turnSlow = 22.0, turnSharp = 45.0, turnAhead = 40.0, turnBelow = 45.0
    // a crosswalk within crossSeconds (at least crossMin m) ahead, below crossBelow mph or with the plan slowing below
    // crossSlow mph before it
    public var crossSeconds = 2.5, crossMin = 10.0, crossBelow = 20.0, crossSlow = 15.0
    public var queueLead = 15.0, queueBelow = 25.0  // a lead within that many m in traffic below that many mph
    public var enter = 0.4, leave = 3.0, hold = 3.0  // seconds
    public init() {}
  }

  public let o: Options
  /// 1 bird's-eye, 0 the road view; nil before the first frame.
  public private(set) var target: Double?
  /// The cue that last asked for bird's-eye.
  public private(set) var why = ""
  private var since = 0.0, cueFor = 0.0, calmFor = 0.0

  public init(_ o: Options = Options()) { self.o = o }

  /// The cue in this frame, or "": v in m/s; px, py the plan, pv its speeds; stripes in the world, pose the ego's
  /// (x, y, yaw); lead or nil.
  public func cue(
    _ v: Double, _ px: [Double], _ py: [Double], _ pv: [Double], _ stripes: [[SIMD2<Double>]], _ pose: SIMD3<Double>,
    _ lead: DioramaScene.Lead?
  ) -> String {
    let mph = v * Diorama.mph
    if mph < o.stopped { return "stopped" }
    if mph < o.turnBelow {
      var j = 1
      while j < px.count - 1 && px[j] < o.turnAhead { j += 1 }
      let dx0 = px[1] - px[0], dx1 = px[j] - px[j - 1]
      let h0 = atan2(py[1] - py[0], dx0 != 0 ? dx0 : 1e-3), h1 = atan2(py[j] - py[j - 1], dx1 != 0 ? dx1 : 1e-3)
      let bend = abs(atan2(sin(h1 - h0), cos(h1 - h0))) * 180 / .pi
      let slowest = pv[0...j].min()! * Diorama.mph
      if px[j] - px[0] > 5 && (bend > o.turnSharp || (bend > o.turnDeg && slowest < o.turnSlow)) { return "turn" }
    }
    let reach = max(o.crossMin, v * o.crossSeconds)
    var r = 0
    while r < px.count - 1 && px[r] < reach { r += 1 }
    let slowing = pv[0...r].min()! * Diorama.mph < o.crossSlow
    if (mph < o.crossBelow || slowing) && !stripes.isEmpty {
      let c = cos(pose.z), s = sin(pose.z)
      for quad in stripes {
        for w in quad {
          let x = c * (w.x - pose.x) + s * (w.y - pose.y), y = -s * (w.x - pose.x) + c * (w.y - pose.y)
          if x > 0 && x < reach && abs(y) < 12 { return "crossing" }
        }
      }
    }
    if let lead, lead.x < o.queueLead && mph < o.queueBelow { return "traffic" }
    if mph < o.slow { return "slow" }
    return ""
  }

  /// 1 for bird's-eye, 0 for the road view, after this frame's cue.
  public mutating func step(_ cue: String, _ v: Double) -> Double {
    let mph = v * Diorama.mph
    since += Diorama.dt
    cueFor = cue.isEmpty ? 0 : cueFor + Diorama.dt
    calmFor = cue.isEmpty ? calmFor + Diorama.dt : 0
    if !cue.isEmpty { why = cue }
    guard let target else {
      self.target = cue.isEmpty ? 0 : 1
      since = 0
      return self.target!
    }
    if since < o.hold { return target }
    if target == 0 && cueFor >= o.enter {
      self.target = 1
      since = 0
    } else if target == 1 && calmFor >= o.leave && mph > o.roadAbove {
      self.target = 0
      since = 0
      why = ""
    }
    return self.target!
  }
}

private func keepLast<T>(_ arr: inout [T], _ v: T, _ n: Int) {
  arr.append(v)
  if arr.count > n { arr.removeFirst() }
}
