#if canImport(Metal)
  import Foundation
  import JetlinkDiorama
  import Metal
  import simd

  /// Diorama's renderer: the scene in Metal, as diorama-render.js draws it in
  /// WebGL2, so the apps and the page look the same. Pure CGI, nothing from a
  /// camera is drawn. A virtual camera follows the ego car from behind at
  /// speed and pans up to a bird's-eye view below it.
  final class DioramaRenderer {
    struct Theme {
      let skyTop, horizon, ground, road, sidewalk, curb, paint, edge, path, ambSky, ambGround, key, rim: SIMD3<Float>

      static let dark = Theme(
        skyTop: [0.07, 0.075, 0.085], horizon: [0.17, 0.18, 0.20], ground: [0.135, 0.14, 0.15], road: [0.235, 0.24, 0.255],
        sidewalk: [0.18, 0.185, 0.20], curb: [0.50, 0.51, 0.54], paint: [0.86, 0.86, 0.87], edge: [1.0, 0.22, 0.22],
        path: [0.22, 0.52, 1.0], ambSky: [0.58, 0.60, 0.65], ambGround: [0.26, 0.26, 0.28], key: [0.62, 0.61, 0.60], rim: [0.30, 0.33, 0.40])
      static let light = Theme(
        skyTop: [0.95, 0.955, 0.96], horizon: [0.88, 0.885, 0.89], ground: [0.83, 0.835, 0.84], road: [0.955, 0.955, 0.96],
        sidewalk: [0.89, 0.89, 0.895], curb: [0.72, 0.72, 0.74], paint: [0.60, 0.60, 0.62], edge: [1.0, 0.30, 0.30],
        path: [0.22, 0.52, 1.0], ambSky: [0.72, 0.73, 0.76], ambGround: [0.48, 0.48, 0.50], key: [0.40, 0.40, 0.40], rim: [0.20, 0.20, 0.22])
    }

    /// The car's paint (diorama-assets.bin marks it): the lead's under its tint, and the ego car's.
    static let leadPaint: SIMD3<Float> = [0.62, 0.62, 0.64], egoPaint: SIMD3<Float> = [0.66, 0.66, 0.68]
    static let sampleCount = 4
    static let colorFormat = MTLPixelFormat.bgra8Unorm, depthFormat = MTLPixelFormat.depth32Float
    /// The model's road and lane lines, rasterised in the ego frame each frame (x -30..90 m, y -30..30 m, 0.1 m).
    static let maskSize = (w: 600, h: 1200)

    let device: MTLDevice
    var theme = Theme.dark
    private let queue: MTLCommandQueue
    private let sky, ground, decal, mesh, maskRoad, maskPaint: MTLRenderPipelineState
    private let depthWrite, depthTestOnly, depthOff: MTLDepthStencilState
    private let mipSampler, linearSampler: MTLSamplerState
    private let maskTexture: MTLTexture
    private let blank: MTLTexture
    private var map: [MTLTexture]?
    private let car: Asset?
    private let lightDir = simd_normalize(SIMD3<Float>(0.35, -0.45, 0.82))  // travels forward-right and down

    struct Asset {
      let vertices: MTLBuffer, indices: MTLBuffer
      let parts: [(vertexOffset: Int, indexOffset: Int, count: Int)]
      let dims: [Double], span: Float
    }

    init?(device: MTLDevice) {
      self.device = device
      guard let queue = device.makeCommandQueue(), let library = try? device.makeLibrary(source: Self.shaders, options: nil) else { return nil }
      self.queue = queue
      func pipeline(
        _ v: String, _ f: String, blend: Bool = false, mask: MTLColorWriteMask = .all, color: MTLPixelFormat = DioramaRenderer.colorFormat,
        samples: Int = DioramaRenderer.sampleCount, depth: Bool = true
      ) -> MTLRenderPipelineState? {
        let d = MTLRenderPipelineDescriptor()
        d.vertexFunction = library.makeFunction(name: v)
        d.fragmentFunction = library.makeFunction(name: f)
        d.colorAttachments[0].pixelFormat = color
        d.colorAttachments[0].writeMask = mask
        if blend {
          d.colorAttachments[0].isBlendingEnabled = true
          d.colorAttachments[0].sourceRGBBlendFactor = .sourceAlpha
          d.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
          d.colorAttachments[0].sourceAlphaBlendFactor = .sourceAlpha
          d.colorAttachments[0].destinationAlphaBlendFactor = .oneMinusSourceAlpha
        }
        if depth { d.depthAttachmentPixelFormat = Self.depthFormat }
        d.rasterSampleCount = samples
        return try? device.makeRenderPipelineState(descriptor: d)
      }
      guard let sky = pipeline("sky_v", "sky_f"), let ground = pipeline("ground_v", "ground_f"),
        let decal = pipeline("decal_v", "decal_f", blend: true), let mesh = pipeline("mesh_v", "mesh_f"),
        let maskRoad = pipeline("mask_v", "mask_f", mask: .red, color: .rgba8Unorm, samples: 1, depth: false),
        let maskPaint = pipeline("mask_v", "mask_f", mask: .green, color: .rgba8Unorm, samples: 1, depth: false)
      else { return nil }
      (self.sky, self.ground, self.decal, self.mesh, self.maskRoad, self.maskPaint) = (sky, ground, decal, mesh, maskRoad, maskPaint)

      func depth(_ compare: MTLCompareFunction, write: Bool) -> MTLDepthStencilState? {
        let d = MTLDepthStencilDescriptor()
        d.depthCompareFunction = compare
        d.isDepthWriteEnabled = write
        return device.makeDepthStencilState(descriptor: d)
      }
      guard let depthWrite = depth(.less, write: true), let depthTestOnly = depth(.less, write: false), let depthOff = depth(.always, write: false) else {
        return nil
      }
      (self.depthWrite, self.depthTestOnly, self.depthOff) = (depthWrite, depthTestOnly, depthOff)

      func sampler(mip: Bool) -> MTLSamplerState? {
        let d = MTLSamplerDescriptor()
        d.minFilter = .linear
        d.magFilter = .linear
        d.mipFilter = mip ? .linear : .notMipmapped
        d.maxAnisotropy = mip ? 8 : 1
        d.sAddressMode = .clampToEdge
        d.tAddressMode = .clampToEdge
        return device.makeSamplerState(descriptor: d)
      }
      guard let mipSampler = sampler(mip: true), let linearSampler = sampler(mip: false) else { return nil }
      (self.mipSampler, self.linearSampler) = (mipSampler, linearSampler)

      let md = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm, width: Self.maskSize.w, height: Self.maskSize.h, mipmapped: false)
      md.usage = [.renderTarget, .shaderRead]
      md.storageMode = .private
      let bd = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm, width: 1, height: 1, mipmapped: false)
      guard let maskTexture = device.makeTexture(descriptor: md), let blank = device.makeTexture(descriptor: bd) else { return nil }
      blank.replace(region: MTLRegionMake2D(0, 0, 1, 1), mipmapLevel: 0, withBytes: [UInt8](repeating: 0, count: 4), bytesPerRow: 4)
      (self.maskTexture, self.blank) = (maskTexture, blank)
      car = DioramaPage.file("diorama-assets.bin").flatMap { Self.loadAssets($0.data, device: device)["car"] }
    }

    /// diorama-assets.bin (diorama/assets.py): u32 header length, JSON header, then 16-byte vertices (int16 position
    /// as a fraction of the asset's span, int8 normal and emission, uint8 colour) and uint16 indices, in parts.
    static func loadAssets(_ data: Data, device: MTLDevice) -> [String: Asset] {
      let bytes = [UInt8](data)
      guard bytes.count >= 4 else { return [:] }
      let hl = Int(bytes[0]) | Int(bytes[1]) << 8 | Int(bytes[2]) << 16 | Int(bytes[3]) << 24
      guard 4 + hl <= bytes.count, let head = (try? JSONSerialization.jsonObject(with: Data(bytes[4..<4 + hl]))) as? [String: [String: Any]] else { return [:] }
      func int(_ v: Any?) -> Int { (v as? NSNumber)?.intValue ?? 0 }
      let parts = head.mapValues { ($0["parts"] as? [[String: Any]]) ?? [] }
      let nv = parts.values.reduce(0) { $0 + $1.reduce(0) { $0 + int($1["vertices"]) } }
      let ni = parts.values.reduce(0) { $0 + $1.reduce(0) { $0 + int($1["indices"]) } }
      let vStart = 4 + hl, iStart = vStart + nv * 16
      guard iStart + ni * 2 <= bytes.count,
        let vertices = bytes.withUnsafeBytes({ device.makeBuffer(bytes: $0.baseAddress! + vStart, length: max(nv * 16, 16)) }),
        let indices = bytes.withUnsafeBytes({ device.makeBuffer(bytes: $0.baseAddress! + iStart, length: max(ni * 2, 16)) })
      else { return [:] }
      var out: [String: Asset] = [:]
      for (name, a) in head {
        out[name] = Asset(
          vertices: vertices, indices: indices,
          parts: (parts[name] ?? []).map { (int($0["vertexOffset"]), int($0["indexOffset"]), int($0["indices"])) },
          dims: (a["dims"] as? [NSNumber])?.map(\.doubleValue) ?? [4.7, 1.9, 1.6], span: (a["span"] as? NSNumber)?.floatValue ?? 1)
      }
      return out
    }

    // MARK: the road map's textures

    /// The road map's two layers, uploaded where they changed. Under the scene's lock.
    /// True when it uploaded, and the mipmaps need making.
    func syncMap(_ roadmap: DioramaRoadMap) -> Bool {
      let cells = DioramaRoadMap.n
      if map == nil {
        let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm, width: cells, height: cells, mipmapped: true)
        d.usage = .shaderRead
        #if os(macOS)
          d.storageMode = device.hasUnifiedMemory ? .shared : .managed
        #else
          d.storageMode = .shared
        #endif
        guard let a = device.makeTexture(descriptor: d), let b = device.makeTexture(descriptor: d) else { return false }
        map = [a, b]
        roadmap.dirty = (0, cells, 0, cells)
      }
      guard let map, let d = roadmap.dirty, d.r1 > d.r0, d.c1 > d.c0 else { return false }
      for (texture, layer) in [(map[0], roadmap.a), (map[1], roadmap.b)] {
        layer.withUnsafeBytes { raw in
          texture.replace(
            region: MTLRegionMake2D(d.c0, d.r0, d.c1 - d.c0, d.r1 - d.r0), mipmapLevel: 0,
            withBytes: raw.baseAddress! + (d.r0 * cells + d.c0) * 4, bytesPerRow: cells * 4)
        }
      }
      roadmap.dirty = nil
      return true
    }

    // MARK: the virtual camera

    /// The camera for a view between the chase camera (0) and the bird's-eye view (1).
    static func camera(_ s: Double, _ hgt: Double, _ w: Double, _ h: Double) -> (f: Double, cx: Double, cy: Double, rotation: [Double], center: [Double]) {
      let chaseAt = [Diorama.egoRear - 6.5, 0, hgt - 3.2], chaseTurn = lookAt(chaseAt, [18, 0, hgt])
      let birdAt = [-14, 0, hgt - 30], birdTurn = lookAt(birdAt, [10, 0, hgt])
      let rel = logRot(Diorama.m3mul(birdTurn, transpose3(chaseTurn)))
      let rotation = Diorama.m3mul(rodrigues(rel.map { $0 * s }), chaseTurn)
      // rise a bit faster than it moves back
      let center = [chaseAt[0] + (birdAt[0] - chaseAt[0]) * s, chaseAt[1] + (birdAt[1] - chaseAt[1]) * s, chaseAt[2] + (birdAt[2] - chaseAt[2]) * pow(s, 0.7)]
      return ((0.62 * (1 - s) + 0.69 * s) * w, w / 2, (0.42 * (1 - s) + 0.5 * s) * h, rotation, center)
    }

    static func rodrigues(_ r: [Double]) -> [Double] {
      let th = sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2])
      if th < 1e-9 { return [1, 0, 0, 0, 1, 0, 0, 0, 1] }
      let k = r.map { $0 / th }, s = sin(th), c = 1 - cos(th)
      let skew = [0, -k[2], k[1], k[2], 0, -k[0], -k[1], k[0], 0], skew2 = Diorama.m3mul(skew, skew)
      return (0..<9).map { (($0 % 4 == 0) ? 1 : 0) + s * skew[$0] + c * skew2[$0] }
    }

    static func logRot(_ rot: [Double]) -> [Double] {
      let th = acos(Diorama.clamp((rot[0] + rot[4] + rot[8] - 1) / 2, -1, 1))
      if th < 1e-9 { return [0, 0, 0] }
      let f = th / (2 * sin(th))
      return [f * (rot[7] - rot[5]), f * (rot[2] - rot[6]), f * (rot[3] - rot[1])]
    }

    /// device_from_road: rows forward, right, down.
    static func lookAt(_ eye: [Double], _ target: [Double]) -> [Double] {
      let f = [target[0] - eye[0], target[1] - eye[1], target[2] - eye[2]], fn = sqrt(f[0] * f[0] + f[1] * f[1] + f[2] * f[2])
      let forward = f.map { $0 / fn }
      var y = [-forward[1], forward[0], 0.0]  // [0, 0, 1] x F
      let yn = sqrt(y[0] * y[0] + y[1] * y[1])
      y = y.map { $0 / yn }
      let z = [forward[1] * y[2] - forward[2] * y[1], forward[2] * y[0] - forward[0] * y[2], forward[0] * y[1] - forward[1] * y[0]]
      return forward + y + z
    }

    static func transpose3(_ m: [Double]) -> [Double] { [m[0], m[3], m[6], m[1], m[4], m[7], m[2], m[5], m[8]] }

    /// A row-major 4x4, as simd's column-major float4x4.
    static func matrix(_ m: [Double]) -> float4x4 {
      float4x4(rows: (0..<4).map { i in SIMD4<Float>((0..<4).map { Float(m[4 * i + $0]) }) })
    }

    // MARK: scene geometry, as diorama-render.js builds it

    /// Decal vertices: x, y, z, r, g, b, a.
    typealias Decals = [Float]

    static func ribbon(_ out: inout Decals, _ pts: [SIMD3<Double>], _ halfW: Double, _ rgb: SIMD3<Float>, _ alphas: [Double], soft: Bool, lift: Double = 0.02) {
      let n = pts.count
      guard n >= 2 else { return }
      let nrm: [SIMD2<Double>] = (0..<n).map { i in
        let a = pts[max(i - 1, 0)], b = pts[min(i + 1, n - 1)]
        let dx = b.x - a.x, dy = b.y - a.y, l = max(hypot(dx, dy), 1e-9)
        return SIMD2(-dy / l, dx / l)
      }
      let cols: [(Double, Double)] = soft ? [(-1, 0), (-0.35, 1), (0.35, 1), (1, 0)] : [(-1, 1), (1, 1)]
      func vertex(_ ring: Int, _ i: Int) -> [Float] {
        let (o, a) = cols[ring], p = pts[i]
        let x = Float(p.x + nrm[i].x * halfW * o), y = Float(p.y + nrm[i].y * halfW * o)
        return [x, y, Float(p.z - lift), rgb.x, rgb.y, rgb.z, Float(alphas[i] * a)]
      }
      for r in 0..<cols.count - 1 {
        for i in 0..<n - 1 {
          let a = vertex(r, i), b = vertex(r, i + 1), c = vertex(r + 1, i), d = vertex(r + 1, i + 1)
          for v in [a, b, c, b, d, c] { out.append(contentsOf: v) }
        }
      }
    }

    static func polygon(_ out: inout Decals, _ quad: [SIMD2<Double>], _ z: Double, _ rgb: SIMD3<Float>, _ alpha: Float, lift: Double = 0.025) {
      let v = quad.map { q -> [Float] in [Float(q.x), Float(q.y), Float(z - lift), rgb.x, rgb.y, rgb.z, alpha] }
      for i in [0, 1, 2, 0, 2, 3] { out.append(contentsOf: v[i]) }
    }

    static func blobShadow(
      _ out: inout Decals, _ x: Double, _ y: Double, _ heading: Double, _ length: Double, _ width: Double, _ z: Double, alpha: Float = 0.38
    ) {
      let c = cos(heading), s = sin(heading)
      func rect(_ l: Double, _ w: Double, _ a: Float) -> [[Float]] {
        let corners: [(Double, Double)] = [(l, w), (l, -w), (-l, -w), (-l, w)]
        return corners.map { (px, py) -> [Float] in
          let cx = x + c * px / 2 - s * py / 2, cy = y + s * px / 2 + c * py / 2
          return [Float(cx), Float(cy), Float(z - 0.015), 0, 0, 0, a]
        }
      }
      let inner = rect(length * 0.85, width * 0.75, alpha), outer = rect(length + 0.6, width + 0.6, 0)
      for v in [inner[0], inner[1], inner[2], inner[0], inner[2], inner[3]] { out.append(contentsOf: v) }
      for i in 0..<4 {
        let j = (i + 1) % 4
        for v in [inner[i], outer[i], outer[j], inner[i], outer[j], inner[j]] { out.append(contentsOf: v) }
      }
    }

    /// A model line (sampled at xIdxs) as 3D points, extended straight back behind the ego car.
    static func modelLine(_ ys: [Double], _ hgt: Double, xMax: Double = 80, xBack: Double = -25) -> [SIMD3<Double>] {
      var pts = [SIMD3(xBack, ys[0], hgt)]
      for i in 0..<Diorama.idxN where Diorama.xIdxs[i] < xMax { pts.append(SIMD3(Diorama.xIdxs[i], ys[i], hgt)) }
      return pts
    }

    static func line(_ l: [Double], _ i: Int) -> [Double] { (0..<Diorama.idxN).map { DioramaOutputs.line(l, i, $0, 0) } }

    // MARK: drawing

    /// What a frame draws, taken from the scene under its lock.
    struct Frame {
      let state: DioramaScene.State?
      let height: Double
      let origin: (x: Double, y: Double)?
      let mapUploaded: Bool
    }

    /// Takes what the frame needs from the scene. Call under the scene's lock.
    func take(_ scene: DioramaScene) -> Frame {
      var uploaded = false
      if scene.state != nil, let roadmap = scene.roadmap, roadmap.origin != nil { uploaded = syncMap(roadmap) }
      return Frame(state: scene.state, height: scene.cam?.height ?? 1.22, origin: scene.roadmap?.origin, mapUploaded: uploaded)
    }

    struct FrameUniforms {
      var vp: float4x4
      var cam: SIMD4<Float>
      var horizon: SIMD4<Float>
      var fog: SIMD4<Float>
    }

    struct GroundUniforms {
      var ground, sidewalk, road, curb, paint: SIMD4<Float>
      var pose: SIMD4<Float>  // the ego car in the map window: x, y (m from the window's corner), cos yaw, sin yaw
      var size: SIMD4<Float>  // the window's side (m), whether there is a map
    }

    struct MeshUniforms {
      var mvp, model: float4x4
      var nmat: float3x3
      var lightDir, ambSky, ambGround, key, rim, tint, paint: SIMD4<Float>
      var params: SIMD4<Float>  // tint amount, specular, brake, span
    }

    func draw(_ frame: Frame, to pass: MTLRenderPassDescriptor, size: CGSize, then present: (MTLCommandBuffer) -> Void) {
      guard let commands = queue.makeCommandBuffer() else { return }
      let th = theme, hgt = frame.height
      let w = Double(size.width), h = Double(size.height)
      guard w > 0, h > 0 else { return }

      if frame.mapUploaded, let map, let blit = commands.makeBlitCommandEncoder() {
        for t in map { blit.generateMipmaps(for: t) }
        blit.endEncoding()
      }

      // the model's road and lane lines into the ego-frame mask
      let mp = MTLRenderPassDescriptor()
      mp.colorAttachments[0].texture = maskTexture
      mp.colorAttachments[0].loadAction = .clear
      mp.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
      mp.colorAttachments[0].storeAction = .store
      if let enc = commands.makeRenderCommandEncoder(descriptor: mp) {
        if let state = frame.state { drawMask(enc, state) }
        enc.endEncoding()
      }

      let camv = Self.camera(frame.state?.view ?? 0, hgt, w, h)
      let near = 0.3, far = 500.0
      let proj = [
        2 * camv.f / w, 0, 2 * camv.cx / w - 1, 0,
        0, -2 * camv.f / h, 1 - 2 * camv.cy / h, 0,
        0, 0, far / (far - near), -far * near / (far - near),
        0, 0, 1, 0,
      ]
      let rot = Diorama.m3mul(Diorama.viewFromDevice, camv.rotation), eye = camv.center
      let t = Diorama.m3vec(rot, eye)
      let view = [rot[0], rot[1], rot[2], -t[0], rot[3], rot[4], rot[5], -t[1], rot[6], rot[7], rot[8], -t[2], 0, 0, 0, 1]
      let vp = Self.matrix(proj) * Self.matrix(view)
      var uniforms = FrameUniforms(
        vp: vp, cam: SIMD4(Float(eye[0]), Float(eye[1]), Float(eye[2]), 0), horizon: SIMD4(th.horizon, 1), fog: [45, 190, 0, 0])

      pass.colorAttachments[0].clearColor = MTLClearColor(red: Double(th.horizon.x), green: Double(th.horizon.y), blue: Double(th.horizon.z), alpha: 1)
      pass.colorAttachments[0].loadAction = .clear
      pass.depthAttachment.clearDepth = 1
      pass.depthAttachment.loadAction = .clear
      guard let enc = commands.makeRenderCommandEncoder(descriptor: pass) else { return }

      // sky
      enc.setRenderPipelineState(sky)
      enc.setDepthStencilState(depthOff)
      var skyColors = [SIMD4(th.skyTop, 1), SIMD4(th.horizon, 1)]
      enc.setFragmentBytes(&skyColors, length: MemoryLayout<SIMD4<Float>>.stride * 2, index: 0)
      enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)

      // ground with the road map
      enc.setRenderPipelineState(ground)
      enc.setDepthStencilState(depthWrite)
      var g = GroundUniforms(
        ground: SIMD4(th.ground, 0), sidewalk: SIMD4(th.sidewalk, 0), road: SIMD4(th.road, 0), curb: SIMD4(th.curb, 0), paint: SIMD4(th.paint, 0),
        pose: .zero, size: [Float(Double(DioramaRoadMap.n) * DioramaRoadMap.res), 0, 0, 0])
      if let state = frame.state, let origin = frame.origin, map != nil {
        let p = state.pose
        g.pose = SIMD4(Float(p.x - origin.x), Float(p.y - origin.y), Float(cos(p.z)), Float(sin(p.z)))
        g.size.y = 1
      }
      let quad: [Float] = [[-120, -300], [400, -300], [400, 300], [-120, -300], [400, 300], [-120, 300]].flatMap { [Float($0[0]), Float($0[1]), Float(hgt)] }
      enc.setVertexBytes(quad, length: quad.count * 4, index: 0)
      enc.setVertexBytes(&uniforms, length: MemoryLayout<FrameUniforms>.stride, index: 1)
      enc.setFragmentBytes(&uniforms, length: MemoryLayout<FrameUniforms>.stride, index: 0)
      enc.setFragmentBytes(&g, length: MemoryLayout<GroundUniforms>.stride, index: 1)
      enc.setFragmentTexture(g.size.y > 0 ? map?[0] : blank, index: 0)
      enc.setFragmentTexture(g.size.y > 0 ? map?[1] : blank, index: 1)
      enc.setFragmentTexture(maskTexture, index: 2)
      enc.setFragmentSamplerState(mipSampler, index: 0)
      enc.setFragmentSamplerState(linearSampler, index: 1)
      enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 6)

      if let state = frame.state {
        let objs = objects(state)
        drawDecals(enc, state, objs, hgt, &uniforms)
        drawMeshes(enc, objs, hgt, vp, &uniforms)
      }
      enc.endEncoding()
      present(commands)
      commands.commit()
    }

    private func drawMask(_ enc: MTLRenderCommandEncoder, _ state: DioramaScene.State) {
      let out = state.out
      guard out.roadEdges.count >= 2 * Diorama.idxN * 2, out.laneLines.count >= 4 * Diorama.idxN * 2 else { return }
      let left = Self.modelLine(Self.line(out.roadEdges, 0), 0), right = Self.modelLine(Self.line(out.roadEdges, 1), 0)
      var tris: [Float] = []
      for i in 0..<left.count - 1 {
        let a = left[i], b = left[i + 1], c = right[i], d = right[i + 1]
        tris += [a.x, a.y, b.x, b.y, c.x, c.y, b.x, b.y, d.x, d.y, c.x, c.y].map(Float.init)
      }
      enc.setRenderPipelineState(maskRoad)
      drawFlat(enc, tris, stride: 2)
      // paint next to a model lane line is drawn by that line
      var lines: [Float] = []
      for i in 0..<4 where out.laneLinesProb.count > 2 * i + 1 && out.laneLinesProb[2 * i + 1] > 0.3 {
        let pts = Self.modelLine(Self.line(out.laneLines, i), 0)
        var rib: Decals = []
        Self.ribbon(&rib, pts, 0.6, .zero, pts.map { _ in 1 }, soft: false, lift: 0)
        for v in stride(from: 0, to: rib.count, by: 7) { lines += [rib[v], rib[v + 1]] }
      }
      if !lines.isEmpty {
        enc.setRenderPipelineState(maskPaint)
        drawFlat(enc, lines, stride: 2)
      }
    }

    /// Vertices from the CPU: small ones inline, larger in a buffer of their own.
    private func drawFlat(_ enc: MTLRenderCommandEncoder, _ data: [Float], stride: Int) {
      guard !data.isEmpty else { return }
      let bytes = data.count * 4
      if bytes <= 4096 {
        enc.setVertexBytes(data, length: bytes, index: 0)
      } else {
        guard let buffer = device.makeBuffer(bytes: data, length: bytes) else { return }
        enc.setVertexBuffer(buffer, offset: 0, index: 0)
      }
      enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: data.count / stride)
    }

    struct Object { let x, y, heading, length, width: Double, braking: Bool }

    /// Only the lead, from this frame's prediction, as the one car: its rear at the model's distance, facing along its lane.
    private func objects(_ state: DioramaScene.State) -> [Object] {
      guard let l = state.lead, let car else { return [] }
      let length = car.dims[0], width = car.dims[1]
      return [
        Object(
          x: l.x + 0.5 * length * cos(l.heading), y: l.y + 0.5 * length * sin(l.heading), heading: l.heading, length: length, width: width, braking: l.braking)
      ]
    }

    private func drawDecals(_ enc: MTLRenderCommandEncoder, _ state: DioramaScene.State, _ objs: [Object], _ hgt: Double, _ uniforms: inout FrameUniforms) {
      let th = theme, out = state.out
      var tris: Decals = []
      tris.reserveCapacity(40_000)
      for o in objs { Self.blobShadow(&tris, o.x, o.y, o.heading, o.length, o.width, hgt) }
      Self.blobShadow(&tris, (Diorama.egoFront + Diorama.egoRear) / 2, 0, 0, Diorama.egoFront - Diorama.egoRear, 1.85, hgt)
      // road edges: a soft red glow
      if out.roadEdges.count >= 2 * Diorama.idxN * 2 {
        for i in 0..<2 {
          let pts = Self.modelLine(Self.line(out.roadEdges, i), hgt)
          let a = Diorama.clamp(1 - out.roadEdgesStds[i * Diorama.idxN * 2], 0.3, 1) * 0.5
          Self.ribbon(&tris, pts, 0.4, th.edge, pts.map { a * (1 - Diorama.smoothstep(55, 85, $0.x)) }, soft: true)
        }
      }
      // lane lines, faded by probability and distance
      if out.laneLines.count >= 4 * Diorama.idxN * 2 {
        for i in 0..<4 where out.laneLinesProb.count > 2 * i + 1 {
          let p = Diorama.clamp(out.laneLinesProb[2 * i + 1], 0, 1)
          if p < 0.15 { continue }
          let pts = Self.modelLine(Self.line(out.laneLines, i), hgt)
          Self.ribbon(&tris, pts, 0.065, th.paint, pts.map { 0.95 * p * (1 - Diorama.smoothstep(60, 90, $0.x)) }, soft: false)
        }
      }
      // crosswalk stripes (held in the world, drawn in this frame)
      let pose = state.pose, c = cos(pose.z), s = sin(pose.z)
      for quad in state.stripes {
        let q = quad.map { w -> SIMD2<Double> in
          let dx = w.x - pose.x, dy = w.y - pose.y
          return SIMD2(c * dx + s * dy, -s * dx + c * dy)
        }
        Self.polygon(&tris, q, hgt, th.paint, 0.9)
      }
      // the planned path, cut before the lead
      let px = state.px, maxD = state.lead.map { max($0.x - 3, 0) } ?? 80
      var pts: [SIMD3<Double>] = []
      for i in 0..<Diorama.idxN {
        if px[i] >= maxD {
          if i > 0 {
            let f = (maxD - px[i - 1]) / max(px[i] - px[i - 1], 1e-3)
            pts.append(SIMD3(maxD, state.py[i - 1] + f * (state.py[i] - state.py[i - 1]), hgt))
          }
          break
        }
        pts.append(SIMD3(px[i], state.py[i], hgt))
      }
      if pts.count >= 2 {
        // resample every 0.5 m along the arc, from just ahead of the bumper
        var arc = [0.0]
        for i in 1..<pts.count { arc.append(arc[i - 1] + hypot(pts[i].x - pts[i - 1].x, pts[i].y - pts[i - 1].y)) }
        let total = arc[arc.count - 1]
        if total >= 1 {
          var at = Array(stride(from: 0.0, to: total, by: 0.5))
          at.append(total)
          let xs = pts.map(\.x), ys = pts.map(\.y), zs = pts.map(\.z)
          let res = at.map { SIMD3(Diorama.interp($0, arc, xs), Diorama.interp($0, arc, ys), Diorama.interp($0, arc, zs)) }
            .filter { $0.x > Diorama.egoFront - 0.5 }
          if res.count >= 2 {
            Self.ribbon(&tris, res, 0.85, th.path, res.map { 0.8 * (1 - Diorama.smoothstep(12, 70, $0.x)) }, soft: true, lift: 0.03)
          }
        }
      }
      guard !tris.isEmpty else { return }
      enc.setRenderPipelineState(decal)
      enc.setDepthStencilState(depthTestOnly)
      enc.setVertexBytes(&uniforms, length: MemoryLayout<FrameUniforms>.stride, index: 1)
      enc.setFragmentBytes(&uniforms, length: MemoryLayout<FrameUniforms>.stride, index: 0)
      drawFlat(enc, tris, stride: 7)
    }

    private func drawMeshes(_ enc: MTLRenderCommandEncoder, _ objs: [Object], _ hgt: Double, _ vp: float4x4, _ uniforms: inout FrameUniforms) {
      guard let car else { return }
      let th = theme
      enc.setRenderPipelineState(mesh)
      enc.setDepthStencilState(depthWrite)
      enc.setVertexBuffer(car.vertices, offset: 0, index: 0)
      enc.setFragmentBytes(&uniforms, length: MemoryLayout<FrameUniforms>.stride, index: 0)
      func draw(_ x: Double, _ y: Double, _ heading: Double, paint: SIMD3<Float>, tint: SIMD3<Float>?, tintAmt: Float, spec: Float, brake: Float) {
        let c = cos(heading), s = sin(heading)
        // local (x forward, y right, z up) -> road (z down)
        let model = Self.matrix([c, -s, 0, x, s, c, 0, y, 0, 0, -1, hgt, 0, 0, 0, 1])
        let fc = Float(c), fs = Float(s)
        var m = MeshUniforms(
          mvp: vp * model, model: model,
          // the normal matrix, (M^-1)^T for the rotation-and-flip part
          nmat: float3x3(columns: (SIMD3(fc, fs, 0), SIMD3(-fs, fc, 0), SIMD3(0, 0, -1))),
          lightDir: SIMD4(lightDir, 0), ambSky: SIMD4(th.ambSky, 0), ambGround: SIMD4(th.ambGround, 0), key: SIMD4(th.key, 0), rim: SIMD4(th.rim, 0),
          tint: SIMD4(tint ?? .zero, 0), paint: SIMD4(paint, 0), params: [tintAmt, spec, brake, car.span])
        enc.setVertexBytes(&m, length: MemoryLayout<MeshUniforms>.stride, index: 1)
        enc.setFragmentBytes(&m, length: MemoryLayout<MeshUniforms>.stride, index: 1)
        for part in car.parts where part.count > 0 {
          enc.setVertexBufferOffset(part.vertexOffset * 16, index: 0)
          enc.drawIndexedPrimitives(
            type: .triangle, indexCount: part.count, indexType: .uint16, indexBuffer: car.indices, indexBufferOffset: part.indexOffset * 2)
        }
      }
      for o in objs { draw(o.x, o.y, o.heading, paint: Self.leadPaint, tint: th.path, tintAmt: 0.55, spec: 0.06, brake: o.braking ? 1 : 0) }
      draw((Diorama.egoFront + Diorama.egoRear) / 2, 0, 0, paint: Self.egoPaint, tint: nil, tintAmt: 0, spec: 0.35, brake: 0)
    }

    // MARK: shaders, diorama-render.js's in Metal

    static let shaders = """
      #include <metal_stdlib>
      using namespace metal;

      struct Frame { float4x4 vp; float4 cam; float4 horizon; float4 fog; };
      float3 apply_fog(float3 c, float3 w, constant Frame& f) {
        float t = clamp((length(w.xy - f.cam.xy) - f.fog.x) / (f.fog.y - f.fog.x), 0.0, 1.0);
        return mix(c, f.horizon.rgb, t * t * (3.0 - 2.0 * t));
      }

      struct SkyOut { float4 pos [[position]]; float vy; };
      vertex SkyOut sky_v(uint vid [[vertex_id]]) {
        const float2 p[3] = { float2(-1, -1), float2(3, -1), float2(-1, 3) };
        SkyOut o; o.pos = float4(p[vid], 0.9999, 1.0); o.vy = p[vid].y; return o;
      }
      fragment float4 sky_f(SkyOut in [[stage_in]], constant float4* c [[buffer(0)]]) {
        return float4(mix(c[1].rgb, c[0].rgb, clamp(in.vy * 0.5 + 0.5, 0.0, 1.0)), 1.0);
      }

      struct WorldOut { float4 pos [[position]]; float3 w; };
      vertex WorldOut ground_v(uint vid [[vertex_id]], const device packed_float3* v [[buffer(0)]], constant Frame& f [[buffer(1)]]) {
        WorldOut o; o.w = float3(v[vid]); o.pos = f.vp * float4(o.w, 1.0); return o;
      }
      struct Ground { float4 ground, sidewalk, road, curb, paint, pose, size; };
      float2 map_uv(float2 e, constant Ground& g) {
        float2 w = g.pose.xy + float2(g.pose.z * e.x - g.pose.w * e.y, g.pose.w * e.x + g.pose.z * e.y);
        return float2(w.y, w.x) / g.size.x;
      }
      fragment float4 ground_f(WorldOut in [[stage_in]], constant Frame& f [[buffer(0)]], constant Ground& g [[buffer(1)]],
                               texture2d<float> map_a [[texture(0)]], texture2d<float> map_b [[texture(1)]], texture2d<float> mask [[texture(2)]],
                               sampler mip [[sampler(0)]], sampler lin [[sampler(1)]]) {
        float3 vw = in.w, c = g.ground.rgb;
        float2 muv = float2((vw.y + 30.0) / 60.0, (90.0 - vw.x) / 120.0);
        float4 m = mask.sample(lin, muv) * ((muv.x > 0.0 && muv.x < 1.0 && muv.y > 0.0 && muv.y < 1.0) ? 1.0 : 0.0);
        // every texture read in uniform control flow (they are mipmapped), masked after
        float2 uv = map_uv(vw.xy, g);
        float inside = (g.size.y > 0.5 && uv.x > 0.0 && uv.x < 1.0 && uv.y > 0.0 && uv.y < 1.0) ? 1.0 : 0.0;
        float4 a = map_a.sample(mip, uv) * inside, b = map_b.sample(mip, uv) * inside;
        float seen = b.g;
        // lane paint is thin: a painted blob wider than 0.9 m both ways is a misread (shadow, glare, hatching)
        float p1 = map_a.sample(mip, map_uv(vw.xy + float2(0.45, 0.0), g)).g, p2 = map_a.sample(mip, map_uv(vw.xy - float2(0.45, 0.0), g)).g;
        float p3 = map_a.sample(mip, map_uv(vw.xy + float2(0.0, 0.45), g)).g, p4 = map_a.sample(mip, map_uv(vw.xy - float2(0.0, 0.45), g)).g;
        float blob = step(0.5, a.g) * step(0.5, min(min(p1, p2), min(p3, p4)));
        c += (g.sidewalk.rgb - g.ground.rgb) * smoothstep(0.35, 0.6, b.r) * seen;
        c = mix(c, g.road.rgb, smoothstep(0.4, 0.6, max(min(a.r + a.g + a.b, 1.0), m.r)));
        c = mix(c, g.curb.rgb, smoothstep(0.3, 0.55, a.a) * seen);
        c = mix(c, g.paint.rgb, smoothstep(0.4, 0.6, a.g) * (1.0 - max(m.g, blob)) * seen);
        return float4(apply_fog(c, vw, f), 1.0);
      }

      // the ego-frame mask: x -30..90 m down the texture from its top row, y -30..30 m across
      vertex float4 mask_v(uint vid [[vertex_id]], const device float2* v [[buffer(0)]]) {
        float2 p = v[vid];
        return float4(p.y / 30.0, (p.x - 30.0) / 60.0, 0.0, 1.0);
      }
      fragment float4 mask_f() { return float4(1.0); }

      struct DecalV { packed_float3 p; packed_float4 c; };
      struct DecalOut { float4 pos [[position]]; float3 w; float4 c; };
      vertex DecalOut decal_v(uint vid [[vertex_id]], const device DecalV* v [[buffer(0)]], constant Frame& f [[buffer(1)]]) {
        DecalOut o; o.w = float3(v[vid].p); o.c = float4(v[vid].c); o.pos = f.vp * float4(o.w, 1.0); return o;
      }
      fragment float4 decal_f(DecalOut in [[stage_in]], constant Frame& f [[buffer(0)]]) {
        return float4(apply_fog(in.c.rgb, in.w, f), in.c.a);
      }

      struct MeshV { packed_short3 p; packed_char3 n; char e; packed_uchar3 c; uchar pad0, pad1, pad2; };
      struct Mesh { float4x4 mvp, model; float3x3 nmat; float4 light_dir, amb_sky, amb_ground, key, rim, tint, paint, params; };
      struct MeshOut { float4 pos [[position]]; float3 w; float3 n; float3 c; float e; };
      vertex MeshOut mesh_v(uint vid [[vertex_id]], const device MeshV* v [[buffer(0)]], constant Mesh& m [[buffer(1)]]) {
        MeshV x = v[vid];
        float4 p = float4(max(float3(short3(x.p)) / 32767.0, -1.0) * m.params.w, 1.0);  // stored as fractions of span metres
        MeshOut o;
        o.w = (m.model * p).xyz;
        o.n = m.nmat * max(float3(char3(x.n)) / 127.0, -1.0);
        o.c = float3(uchar3(x.c)) / 255.0;
        o.e = max(float(x.e) / 127.0, -1.0);
        o.pos = m.mvp * p;
        return o;
      }
      fragment float4 mesh_f(MeshOut in [[stage_in]], constant Frame& f [[buffer(0)]], constant Mesh& m [[buffer(1)]]) {
        float3 N = normalize(in.n), V = normalize(f.cam.xyz - in.w);
        if (dot(N, V) < 0.0) N = -N;
        float3 col = in.e < 0.0 ? m.paint.rgb : in.c;  // emission -1 marks the paint, coloured per car
        float3 base = mix(col, col * 0.35 + m.tint.rgb * 0.75, m.params.x);
        float3 amb = mix(m.amb_ground.rgb, m.amb_sky.rgb, 0.5 + 0.5 * dot(N, float3(0.0, 0.0, -1.0)));
        float diff = clamp((dot(N, -m.light_dir.xyz) + 0.25) / 1.25, 0.0, 1.0);
        float spec = pow(max(dot(N, normalize(-m.light_dir.xyz + V)), 0.0), 48.0) * m.params.y;
        float rim = pow(1.0 - max(dot(N, V), 0.0), 3.0);
        float3 c = base * (amb + diff * m.key.rgb) + spec + rim * m.rim.rgb * 0.6;
        if (in.e > 0.0) c += col * in.e * (0.6 + 2.5 * m.params.z);
        return float4(apply_fog(c, in.w, f), 1.0);
      }
      """
  }
#endif
