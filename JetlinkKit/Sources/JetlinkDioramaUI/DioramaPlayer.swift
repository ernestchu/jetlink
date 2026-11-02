#if canImport(SwiftUI) && canImport(CoreGraphics)
  import CoreGraphics
  import Foundation
  import JetlinkDiorama
  import Observation

  /// The scene behind a lock: stepped off the main thread as batches come,
  /// and read by the renderer on it.
  final class SceneBox: @unchecked Sendable {
    private let lock = NSLock()
    private let scene = DioramaScene()
    private var input = ModelInput()

    func with<T>(_ body: (DioramaScene) -> T) -> T {
      lock.lock()
      defer { lock.unlock() }
      return body(scene)
    }

    /// Steps the scene through a batch, and what the view shows of it.
    func take(_ batch: [Data], showInput: Bool) -> DioramaPlayer.Update {
      lock.lock()
      defer { lock.unlock() }
      var update = DioramaPlayer.Update()
      for data in batch {
        let msg = DioramaMessage(data)
        scene.push(msg)
        switch msg {
        case .drive: update.frames += 1
        case .status(let text): update.status = text
        case .meta(let meta): update.route = meta.route
        default: break
        }
        if showInput, let meta = scene.meta, let picture = input.take(msg, meta) { update.input = picture }
      }
      if let s = scene.state {
        update.hud = DioramaPlayer.HUD(
          speed: Int((s.speed * Diorama.mph).rounded()), lead: s.leadDistance.map { Int($0.rounded()) }, view: s.view, accel: s.accel)
      }
      update.timing = scene.timing
      return update
    }
  }

  /// What a view of the drive shows beside the scene: the HUD, the models'
  /// timings, their state, and the frame they see. It steps the scene as the
  /// feed delivers, and asks the renderer to draw after each batch.
  @MainActor
  @Observable
  public final class DioramaPlayer {
    public struct HUD: Equatable, Sendable {
      public var speed: Int, lead: Int?, view: Double, accel: Double
      public var mode: String { view < 0.02 ? "ROAD VIEW" : view > 0.98 ? "BIRD'S-EYE" : "PANNING \(Int((view * 100).rounded()))%" }
    }

    struct Update: Sendable {
      var frames = 0
      var hud: HUD?
      var status: String?
      var route: String?
      var input: InputPicture?
      var timing = DioramaScene.Timing()
    }

    public private(set) var hud: HUD?
    /// The models, when they are not simply running: preparing, paused, or off and why.
    public private(set) var models: String?
    public private(set) var input: InputPicture?
    public private(set) var timing = DioramaScene.Timing()
    /// When the last driving frames came, for the frame rate and the waiting note.
    public private(set) var frameTimes: [Date] = []
    @ObservationIgnored let box = SceneBox()
    @ObservationIgnored var redraw: (() -> Void)?
    @ObservationIgnored var showInput = false

    public init() {}

    func take(_ batch: [Data]) async {
      let box = box, showInput = showInput
      let update = await Task.detached(priority: .userInitiated) { box.take(batch, showInput: showInput) }.value
      let now = Date()
      frameTimes.removeAll { now.timeIntervalSince($0) > 2 }
      frameTimes += Array(repeating: now, count: update.frames)
      if update.hud != hud { hud = update.hud }
      if let status = update.status { models = status == "running" || status == "idle" || status.isEmpty ? nil : status }
      if let picture = update.input { input = picture }
      timing = update.timing
      redraw?()
    }

    func clearInput() { input = nil }

    /// Frames a second as they arrived, over the last two seconds.
    public func frameRate(at now: Date) -> Double { Double(frameTimes.filter { now.timeIntervalSince($0) < 2 }.count) / 2 }
    public func waiting(at now: Date) -> Bool { frameTimes.last.map { now.timeIntervalSince($0) >= 2 } ?? true }
  }

  /// The frame the models see, the packed narrow frame unpacked to 512x256 RGB, with what the segmentation made of
  /// it tinted over it and each thing named.
  public struct InputPicture: @unchecked Sendable {
    public struct Label: Sendable {
      public let text: String, x: Double, y: Double, rgb: SIMD3<UInt8>
    }
    public let image: CGImage
    public let labels: [Label]
    public let caption: String
  }

  /// The last few input frames, for the segmentation that comes after its frame.
  struct ModelInput {
    static let segTint: [UInt8: SIMD3<UInt8>] = [1: [128, 64, 255], 2: [255, 214, 10], 3: [255, 255, 255], 4: [255, 140, 0], 5: [255, 92, 170]]
    static let thingTint: [String: SIMD3<UInt8>] = [
      "car": [56, 132, 255], "truck": [0, 200, 220], "bus": [0, 200, 220], "person": [255, 64, 64], "bicycle": [70, 210, 110], "motorcycle": [70, 210, 110],
    ]
    var images: [(frame: Int, w: Int, h: Int, rgba: [UInt8])] = []
    var lastSeg: DioramaSegMessage?
    var shown = -1

    mutating func take(_ msg: DioramaMessage, _ meta: DioramaMeta) -> InputPicture? {
      switch msg {
      case .input(let frame, let w, let h, let packed):
        guard packed.count >= 6 * w * h else { return nil }
        images.append((frame, 2 * w, 2 * h, Self.unpack(w, h, packed)))
        if images.count > 6 { images.removeFirst() }
      case .seg(let seg):
        lastSeg = seg
      case .meta:
        images = []
        lastSeg = nil
        shown = -1
        return nil
      default:
        return nil
      }
      guard let newest = images.last else { return nil }
      // the frame the newest segmentation is of, which comes a little after the frame itself; the newest frame
      // bare while the models are not segmenting (preparing, paused, off)
      if let seg = lastSeg, let image = images.first(where: { $0.frame == seg.frame }) {
        return image.frame != shown ? picture(image, seg, meta) : nil
      } else if lastSeg == nil || newest.frame - lastSeg!.frame >= 8 {
        return newest.frame != shown ? picture(newest, nil, meta) : nil
      }
      return nil
    }

    /// The four Y planes are Y[0::2, 0::2], Y[1::2, 0::2], Y[0::2, 1::2], Y[1::2, 1::2]; BT.601, video range.
    static func unpack(_ w: Int, _ h: Int, _ packed: [UInt8]) -> [UInt8] {
      let W = 2 * w, H = 2 * h, plane = w * h
      var px = [UInt8](repeating: 255, count: W * H * 4)
      func byte(_ x: Double) -> UInt8 { UInt8(min(max(x.rounded(), 0), 255)) }
      for y in 0..<H {
        for x in 0..<W {
          let at = (y >> 1) * w + (x >> 1), c = (Double(packed[((y & 1) + 2 * (x & 1)) * plane + at]) - 16) * 1.164
          let u = Double(packed[4 * plane + at]) - 128, v = Double(packed[5 * plane + at]) - 128, o = 4 * (y * W + x)
          px[o] = byte(c + 1.596 * v)
          px[o + 1] = byte(c - 0.391 * u - 0.813 * v)
          px[o + 2] = byte(c + 2.018 * u)
        }
      }
      return px
    }

    mutating func picture(_ image: (frame: Int, w: Int, h: Int, rgba: [UInt8]), _ seg: DioramaSegMessage?, _ meta: DioramaMeta) -> InputPicture? {
      shown = image.frame
      let W = image.w, H = image.h
      guard let context = Self.context(W, H), let base = Self.image(image.rgba, W, H) else { return nil }
      context.draw(base, in: CGRect(x: 0, y: 0, width: W, height: H))
      var notes = ["frame \(image.frame)"], labels: [InputPicture.Label] = []
      if let seg, let segInfo = meta.seg {
        let info = DioramaSeg(seg, segInfo), w = info.w, h = info.h, qmap = info.qmap
        var tint = [UInt8](repeating: 0, count: w * h * 4)
        for i in 0..<w * h {
          let q = qmap[i]
          if q == 255 { continue }
          let thing = info.qThing[Int(q)]
          guard let c = thing.map({ Self.thingTint[$0] ?? Self.thingTint["car"]! }) ?? Self.segTint[info.cls[i]] else { continue }
          // a thing's edge (another query beside it) opaque, its inside and the road classes see-through
          let x = i % w
          let edge = thing != nil && ((x > 0 && qmap[i - 1] != q) || (x < w - 1 && qmap[i + 1] != q) || (i >= w && qmap[i - w] != q) || (i < w * (h - 1) && qmap[i + w] != q))
          let a: UInt8 = edge ? 255 : thing != nil ? 110 : 90
          // premultiplied, as the context draws it
          let o = 4 * i
          tint[o] = UInt8(Int(c.x) * Int(a) / 255)
          tint[o + 1] = UInt8(Int(c.y) * Int(a) / 255)
          tint[o + 2] = UInt8(Int(c.z) * Int(a) / 255)
          tint[o + 3] = a
        }
        if let overlay = Self.image(tint, w, h) {
          context.interpolationQuality = .none
          context.draw(overlay, in: CGRect(x: 0, y: 0, width: W, height: H))
        }
        // each thing's name and score, above it
        var tops: [UInt8: Int] = [:], area = [Int](repeating: 0, count: 256)
        for i in 0..<w * h {
          let q = qmap[i]
          if q == 255 || info.qThing[Int(q)] == nil { continue }
          area[Int(q)] += 1
          if tops[q] == nil { tops[q] = i }
        }
        let sx = Double(W) / Double(w), sy = Double(H) / Double(h)
        for (q, i) in tops.sorted(by: { $0.value < $1.value }) where area[Int(q)] >= 6 {
          let thing = info.qThing[Int(q)]!
          let score = Double(seg.score[Int(q)]) / 255
          labels.append(.init(text: "\(thing) \(String(format: "%.2f", score))", x: Double(i % w) * sx, y: Double(i / w) * sy, rgb: Self.thingTint[thing] ?? Self.thingTint["car"]!))
        }
        notes.append("seg \(Int(seg.ms.rounded())) ms")
      }
      guard let composed = context.makeImage() else { return nil }
      return InputPicture(image: composed, labels: labels, caption: "model input · \(W)×\(H) · \(notes.joined(separator: " · "))")
    }

    static func context(_ w: Int, _ h: Int) -> CGContext? {
      CGContext(
        data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4, space: CGColorSpace(name: CGColorSpace.sRGB)!,
        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
    }

    static func image(_ rgba: [UInt8], _ w: Int, _ h: Int) -> CGImage? {
      guard let provider = CGDataProvider(data: Data(rgba) as CFData) else { return nil }
      return CGImage(
        width: w, height: h, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: w * 4, space: CGColorSpace(name: CGColorSpace.sRGB)!,
        bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue), provider: provider, decode: nil, shouldInterpolate: false,
        intent: .defaultIntent)
    }
  }
#endif
