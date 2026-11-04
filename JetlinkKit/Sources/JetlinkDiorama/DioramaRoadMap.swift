import Foundation

/// The road surface around the car in a world window that moves with it: per
/// 0.1 m cell, running averages of the segmentation's road classes, and how
/// much of it has been seen. Two RGBA8 layers, for the renderer's textures:
///   a = (road, lane paint, crosswalk, curb), b = (sidewalk, seen, -, -)
public final class DioramaRoadMap {
  public static let res = 0.1
  public static let n = 1536  // 153.6 m
  static let obsX = (3.0, 40.0), obsY = 25.0
  static let alpha = 0.44

  let cam: DioramaFrameCamera
  public private(set) var a: [UInt8]
  public private(set) var b: [UInt8]
  /// World (x, y) of cell (0, 0)'s corner; rows run along world x, columns along world y.
  public private(set) var origin: (x: Double, y: Double)? = nil
  /// What changed since the renderer last took it: rows r0..<r1, columns c0..<c1.
  public var dirty: (r0: Int, r1: Int, c0: Int, c1: Int)? = nil
  var crosswalkSeen = 0

  init(cam: DioramaFrameCamera) {
    self.cam = cam
    a = [UInt8](repeating: 0, count: Self.n * Self.n * 4)
    b = a
  }

  private var all: (r0: Int, r1: Int, c0: Int, c1: Int) { (0, Self.n, 0, Self.n) }

  /// Keep the car within 30 m of the window's centre, moving the window (and what it holds) when it is not.
  func follow(_ p: EgoTrail.SIMDPose) {
    let cells = Self.n, half = Double(cells) * Self.res / 2
    guard let origin else {
      self.origin = (p.x - half, p.y - half)
      dirty = all
      return
    }
    let cx = origin.x + half, cy = origin.y + half
    if abs(p.x - cx) < 30 && abs(p.y - cy) < 30 { return }
    let dr = Diorama.jsRound((p.x - cx) / Self.res), dc = Diorama.jsRound((p.y - cy) / Self.res)
    func shift(_ m: inout [UInt8]) {
      let old = m
      m = [UInt8](repeating: 0, count: m.count)
      let c0 = max(0, -dc), c1 = min(cells, cells - dc)
      guard c1 > c0 else { return }
      m.withUnsafeMutableBufferPointer { dst in
        old.withUnsafeBufferPointer { src in
          for r in max(0, -dr)..<max(max(0, -dr), min(cells, cells - dr)) {
            let from = ((r + dr) * cells + c0 + dc) * 4, to = (r * cells + c0) * 4
            (dst.baseAddress! + to).update(from: src.baseAddress! + from, count: (c1 - c0) * 4)
          }
        }
      }
    }
    shift(&a)
    shift(&b)
    self.origin = (origin.x + Double(dr) * Self.res, origin.y + Double(dc) * Self.res)
    dirty = all
  }

  /// `seg` as `cam` saw it: the map's own camera (the narrow frame's) or the wide frame's, whose pixel covers as
  /// much road as the narrow's twice as far, so it reaches half as far and weighs as the narrow's would there.
  func observe(_ p: EgoTrail.SIMDPose, _ seg: DioramaSeg, _ zpath: (Double) -> Double, _ cam: DioramaFrameCamera) {
    follow(p)
    guard let origin else { return }
    let cell = Self.res, cells = Self.n, c = cos(p.yaw), s = sin(p.yaw)
    let fine = cam.f / self.cam.f, farX = Self.obsX.1 * min(fine, 1)
    // the observation window's cells (3..40 m ahead, 25 m either side) as world cells
    let corners = [(Self.obsX.0, -Self.obsY), (Self.obsX.0, Self.obsY), (farX, -Self.obsY), (farX, Self.obsY)]
      .map { (x, y) in (p.x + c * x - s * y, p.y + s * x + c * y) }
    let r0 = max(0, Int(((corners.map(\.0).min()! - origin.x) / cell).rounded(.down)))
    let r1 = min(cells, Int(((corners.map(\.0).max()! - origin.x) / cell).rounded(.down)) + 1)
    let c0 = max(0, Int(((corners.map(\.1).min()! - origin.y) / cell).rounded(.down)))
    let c1 = min(cells, Int(((corners.map(\.1).max()! - origin.y) / cell).rounded(.down)) + 1)
    if r1 <= r0 || c1 <= c0 { return }
    let proj = cam.proj, h = cam.height, sw = seg.w, sh = seg.h, ds = seg.ds, frameW = cam.w, frameH = cam.h
    // the road height along the path, tabulated per metre (zpath is the plan's, interpolated)
    let zTab = (0..<42).map { zpath(Double($0)) }
    var crosswalk = 0
    seg.cls.withUnsafeBufferPointer { cls in
      a.withUnsafeMutableBufferPointer { a in
        b.withUnsafeMutableBufferPointer { b in
          for r in r0..<r1 {
            let dx = origin.x + (Double(r) + 0.5) * cell - p.x
            for col in c0..<c1 {
              let dy = origin.y + (Double(col) + 0.5) * cell - p.y
              let ex = c * dx + s * dy, ey = -s * dx + c * dy
              if ex < Self.obsX.0 || ex > farX || abs(ey) > Self.obsY { continue }
              let i0 = Int(ex), ez = h + zTab[i0] + (zTab[i0 + 1] - zTab[i0]) * (ex - Double(i0))
              let z = max(proj[6] * ex + proj[7] * ey + proj[8] * ez, 1e-3)
              if z <= 1 { continue }
              let u = (proj[0] * ex + proj[1] * ey + proj[2] * ez) / z, v = (proj[3] * ex + proj[4] * ey + proj[5] * ez) / z
              if !(u >= 0 && u < frameW - 1 && v >= 0 && v < frameH) { continue }
              let k = cls[min(Int(v / ds), sh - 1) * sw + min(Int(u / ds), sw - 1)]
              if k == Diorama.occluder { continue }  // a car or a person covers the road here: no evidence either way
              // near observations are more accurate (one pixel covers less road), so they weigh more
              let alpha = Self.alpha * Diorama.clamp(1.4 - ex / fine / 30, 0.25, 1)
              let at = (r * cells + col) * 4
              // + 0.5: the bytes round rather than truncate
              @inline(__always) func blend(_ x: UInt8, _ on: Bool) -> UInt8 {
                let v = Double(x)
                return UInt8(v + alpha * ((on ? 255 : 0) - v) + 0.5)
              }
              a[at] = blend(a[at], k == 1)
              a[at + 1] = blend(a[at + 1], k == 2)
              a[at + 2] = blend(a[at + 2], k == 3)
              a[at + 3] = blend(a[at + 3], k == 4)
              b[at] = blend(b[at], k == 5)
              b[at + 1] = UInt8(min(Double(b[at + 1]) + alpha * 255 + 0.5, 255))
              if k == 3 { crosswalk += 1 }
            }
          }
        }
      }
    }
    crosswalkSeen = crosswalk > 20 ? 40 : max(crosswalkSeen - 1, 0)
    if let d = dirty {
      dirty = (min(d.r0, r0), max(d.r1, r1), min(d.c0, c0), max(d.c1, c1))
    } else {
      dirty = (r0, r1, c0, c1)
    }
  }

  /// Bilinear read of a layer channel at world (wx, wy), 0..1.
  @inline(__always)
  static func sample(_ layer: UnsafeBufferPointer<UInt8>, _ ch: Int, _ origin: (x: Double, y: Double), _ wx: Double, _ wy: Double) -> Double {
    let fr = (wx - origin.x) / res - 0.5, fc = (wy - origin.y) / res - 0.5
    let r = Int(fr.rounded(.down)), c = Int(fc.rounded(.down)), cells = n
    if r < 0 || c < 0 || r >= cells - 1 || c >= cells - 1 { return 0 }
    let tr = fr - Double(r), tc = fc - Double(c), at = (r * cells + c) * 4 + ch
    let top = Double(layer[at]) * (1 - tc) + Double(layer[at + 4]) * tc
    let bottom = Double(layer[at + 4 * cells]) * (1 - tc) + Double(layer[at + 4 * cells + 4]) * tc
    return (top * (1 - tr) + bottom * tr) / 255
  }

  /// Zebra stripes, as 4-point polygons in world coordinates (fsd_roadmap.RoadMap.crosswalks).
  ///
  /// Crosswalks at an intersection merge into one region, so the mask is split into straight bands: bands crossing
  /// our road and bands crossing a side street. The split runs in a frame aligned with the crosswalk grid itself
  /// (edge orientations mod 90 degrees, the axis nearer our lane taken as along), so it still works mid-turn.
  /// Every pixel goes to exactly one band (the direction its run is longer in), so the corners where two crossings
  /// meet are not striped twice. Bars run along the traffic of the road they cross, at most maxWidth long.
  func crosswalks(
    _ p: EgoTrail.SIMDPose, _ laneHeading: (Double) -> Double, minArea: Double = 2.0, bandLen: Double = 4.5, maxWidth: Double = 4.0
  ) -> [[SIMD2<Double>]] {
    guard crosswalkSeen > 0, let origin else { return [] }
    let res = 0.2, xMax = 90.0, xMin = -30.0, yMax = 30.0
    let gridW = Diorama.jsRound(2 * yMax / res), gridH = Diorama.jsRound((xMax - xMin) / res)
    let c = cos(p.yaw), s = sin(p.yaw)
    let k = Int(bandLen / res)
    // the crosswalk raster, ego frame at 0.2 m
    var full = [UInt8](repeating: 0, count: gridW * gridH)
    var rmin = gridH, rmax = -1, cmin = gridW, cmax = -1, count = 0
    a.withUnsafeBufferPointer { a in
      b.withUnsafeBufferPointer { b in
        for row in 0..<gridH {
          let x = xMax - (Double(row) + 0.5) * res
          for col in 0..<gridW {
            let y = -yMax + (Double(col) + 0.5) * res, wx = p.x + c * x - s * y, wy = p.y + s * x + c * y
            if Self.sample(a, 2, origin, wx, wy) > 0.5 && Self.sample(b, 1, origin, wx, wy) > 0.5 {
              full[row * gridW + col] = 1
              count += 1
              rmin = min(rmin, row)
              rmax = max(rmax, row)
              cmin = min(cmin, col)
              cmax = max(cmax, col)
            }
          }
        }
      }
    }
    if Double(count) * res * res < minArea { return [] }
    // everything after works in a square crop around it, large enough for the grid turned any way plus a band
    let half = Int(ceil(hypot(Double(rmax - rmin), Double(cmax - cmin)) / 2)) + k + 4
    let rowFrom = max((rmin + rmax) / 2 - half, 0), rowTo = min((rmin + rmax) / 2 + half, gridH)
    let colFrom = max((cmin + cmax) / 2 - half, 0), colTo = min((cmin + cmax) / 2 + half, gridW)
    let cropW = colTo - colFrom, cropH = rowTo - rowFrom
    func toXY(_ col: Int, _ row: Int) -> (Double, Double) { (xMax - (Double(row + rowFrom) + 0.5) * res, -yMax + (Double(col + colFrom) + 0.5) * res) }
    func toPx(_ x: Double, _ y: Double) -> (Double, Double) { ((y + yMax) / res - 0.5 - Double(colFrom), (xMax - x) / res - 0.5 - Double(rowFrom)) }
    var mask = [UInt8](repeating: 0, count: cropW * cropH)
    for row in 0..<cropH { for col in 0..<cropW { mask[row * cropW + col] = full[(row + rowFrom) * gridW + colFrom + col] } }
    mask = morph(mask, cropW, cropH, 3, 3, open: false)
    var mxs: [Double] = [], mys: [Double] = []
    var r0 = cropH, r1 = 0, q0 = cropW, q1 = 0
    for row in 0..<cropH {
      for col in 0..<cropW where mask[row * cropW + col] != 0 {
        let (x, y) = toXY(col, row)
        mxs.append(x)
        mys.append(y)
        r0 = min(r0, row)
        r1 = max(r1, row)
        q0 = min(q0, col)
        q1 = max(q1, col)
      }
    }
    let cx = Diorama.percentile(mxs, 50), cy = Diorama.percentile(mys, 50)

    // grid orientation from the mask's edges, 4x the angle so the two crossing directions vote together
    r0 = max(r0 - 4, 0)
    r1 = min(r1 + 5, cropH)
    q0 = max(q0 - 4, 0)
    q1 = min(q1 + 5, cropW)
    let cw = q1 - q0, ch = r1 - r0, f = blur5(mask, cropW, r0, r1, q0, q1).map(Double.init)
    var zr = 0.0, zi = 0.0, sumG = 0.0
    if ch > 2 && cw > 2 {
      for row in 1..<ch - 1 {
        for col in 1..<cw - 1 {
          let at = row * cw + col
          let gc = (f[at - cw + 1] + 2 * f[at + 1] + f[at + cw + 1]) - (f[at - cw - 1] + 2 * f[at - 1] + f[at + cw - 1])
          let gr = (f[at + cw - 1] + 2 * f[at + cw] + f[at + cw + 1]) - (f[at - cw - 1] + 2 * f[at - cw] + f[at - cw + 1])
          if gc * gc + gr * gr <= 1e-3 { continue }
          let gx = -gr, gy = gc, g2 = gx * gx + gy * gy, th = 4 * atan2(gy, gx)  // rows run along -x
          zr += g2 * cos(th)
          zi += g2 * sin(th)
          sumG += g2
        }
      }
    }
    let lane = laneHeading(cx)
    var phi = lane
    if hypot(zr, zi) > 0.2 * sumG {  // a clear grid; otherwise trust the lane direction
      let ang = atan2(zi, zr) / 4
      phi = lane + (jsMod(jsMod(ang - lane + .pi / 4, .pi / 2) + .pi / 2, .pi / 2) - .pi / 4)
    }
    let axisU = (cos(phi), sin(phi)), axisW = (-axisU.1, axisU.0)

    // the aligned frame: turned by -phi about the crosswalk's centre, so U runs along -rows
    var al = [UInt8](repeating: 0, count: cropW * cropH)
    var ar0 = cropH, ar1 = 0, ac0 = cropW, ac1 = 0
    for row in 0..<cropH {
      for col in 0..<cropW {
        let (ax, ay) = toXY(col, row), dx = ax - cx, dy = ay - cy
        let (ec, er) = toPx(cx + axisU.0 * dx + axisW.0 * dy, cy + axisU.1 * dx + axisW.1 * dy)
        let ic = Diorama.jsRound(ec), ir = Diorama.jsRound(er)
        if ic >= 0 && ir >= 0 && ic < cropW && ir < cropH && mask[ir * cropW + ic] != 0 {
          al[row * cropW + col] = 1
          ar0 = min(ar0, row)
          ar1 = max(ar1, row + 1)
          ac0 = min(ac0, col)
          ac1 = max(ac1, col + 1)
        }
      }
    }
    if ar1 <= ar0 { return [] }
    var across = [UInt8](repeating: 0, count: cropW * cropH), along = across
    let hrun = runs(al, cropW, ar0, ar1, ac0, ac1, horizontal: true), vrun = runs(al, cropW, ar0, ar1, ac0, ac1, horizontal: false)
    for row in ar0..<ar1 {
      for col in ac0..<ac1 {
        let at = row * cropW + col
        if al[at] != 0 {
          if hrun[at] >= vrun[at] { across[at] = 1 } else { along[at] = 1 }
        }
      }
    }
    var stripes: [[SIMD2<Double>]] = []
    for (isAcross, band) in [(true, morph(across, cropW, cropH, 1, k, open: true)), (false, morph(along, cropW, cropH, k, 1, open: true))] {
      // back to the ego raster
      var back = [UInt8](repeating: 0, count: cropW * cropH)
      for row in 0..<cropH {
        for col in 0..<cropW {
          let (x, y) = toXY(col, row), dx = x - cx, dy = y - cy
          let (ac, arow) = toPx(cx + axisU.0 * dx + axisU.1 * dy, cy + axisW.0 * dx + axisW.1 * dy)
          let ic = Diorama.jsRound(ac), ir = Diorama.jsRound(arow)
          if ic >= 0 && ir >= 0 && ic < cropW && ir < cropH && band[ir * cropW + ic] != 0 { back[row * cropW + col] = 1 }
        }
      }
      let (labels, comps) = components(back, cropW, cropH)
      for (ci, pix) in comps.enumerated() {
        if Double(pix.count) * res * res < minArea { continue }
        var pu: [Double] = [], pw: [Double] = []
        for at in pix {
          let (x, y) = toXY(at % cropW, at / cropW)
          pu.append(x * axisU.0 + y * axisU.1)
          pw.append(x * axisW.0 + y * axisW.1)
        }
        let extU = (Diorama.percentile(pu, 2), Diorama.percentile(pu, 98)), extW = (Diorama.percentile(pw, 2), Diorama.percentile(pw, 98))
        let (longAx, lext) = isAcross ? (axisW, extW) : (axisU, extU)
        let (shortAx, sp) = isAcross ? (axisU, pu) : (axisW, pw)
        var sext = isAcross ? extU : extW
        if sext.1 - sext.0 < 0.8 { continue }
        if sext.1 - sext.0 > maxWidth {
          let m = Diorama.percentile(sp, 50)
          sext = (Diorama.clamp(m - maxWidth / 2, sext.0, sext.1), Diorama.clamp(m + maxWidth / 2, sext.0, sext.1))
        }
        let bar = 0.5, gap = 0.6
        var o = lext.0
        while o <= lext.1 - bar + 1e-6 {
          defer { o += bar + gap }
          func pt(_ l: Double, _ sh: Double) -> (Double, Double) { (longAx.0 * l + shortAx.0 * sh, longAx.1 * l + shortAx.1 * sh) }
          let quad = [pt(o, sext.0), pt(o, sext.1), pt(o + bar, sext.1), pt(o + bar, sext.0)]
          // only bars whose centre lies on the band
          let (qcol, qrow) = toPx((quad[0].0 + quad[2].0) / 2, (quad[0].1 + quad[2].1) / 2)
          let ci0 = Diorama.jsRound(qcol), ri0 = Diorama.jsRound(qrow)
          var on = false
          var r = max(ri0 - 2, 0)
          while r < min(ri0 + 3, cropH) && !on {
            var cc = max(ci0 - 2, 0)
            while cc < min(ci0 + 3, cropW) {
              if labels[r * cropW + cc] == ci + 1 {
                on = true
                break
              }
              cc += 1
            }
            r += 1
          }
          if on { stripes.append(quad.map { (x, y) in SIMD2(p.x + c * x - s * y, p.y + s * x + c * y) }) }
        }
      }
    }
    return stripes
  }
}

/// JavaScript's %: the sign of the dividend.
private func jsMod(_ a: Double, _ b: Double) -> Double { fmod(a, b) }

/// Binary morphology with a kh x kw rectangle anchored at its centre, as cv2's (the image border never erodes).
func morph(_ m: [UInt8], _ width: Int, _ height: Int, _ kh: Int, _ kw: Int, open: Bool) -> [UInt8] {
  func pass(_ src: [UInt8], dilate: Bool, horizontal: Bool, _ k: Int) -> [UInt8] {
    if k <= 1 { return src }
    var out = [UInt8](repeating: 0, count: width * height)
    let lo = -(k >> 1), hi = k - 1 - (k >> 1)
    for row in 0..<height {
      for col in 0..<width {
        var v: UInt8 = dilate ? 0 : 1
        for d in lo...hi {
          let r = horizontal ? row : row + d, c = horizontal ? col + d : col
          let s = r < 0 || c < 0 || r >= height || c >= width ? (dilate ? 0 : 1) : src[r * width + c]
          if dilate ? s != 0 : s == 0 {
            v = dilate ? 1 : 0
            break
          }
        }
        out[row * width + col] = v
      }
    }
    return out
  }
  func erode(_ x: [UInt8]) -> [UInt8] { pass(pass(x, dilate: false, horizontal: true, kw), dilate: false, horizontal: false, kh) }
  func dilate(_ x: [UInt8]) -> [UInt8] { pass(pass(x, dilate: true, horizontal: true, kw), dilate: true, horizontal: false, kh) }
  return open ? dilate(erode(m)) : erode(dilate(m))
}

/// 5x5 Gaussian (sigma 1) of a crop of a binary mask, as float.
func blur5(_ m: [UInt8], _ width: Int, _ r0: Int, _ r1: Int, _ c0: Int, _ c1: Int) -> [Float] {
  let k = [0.0545, 0.2442, 0.4026, 0.2442, 0.0545], w = c1 - c0, h = r1 - r0
  guard w > 0, h > 0 else { return [] }
  var tmp = [Float](repeating: 0, count: w * h), out = tmp
  for row in 0..<h {
    for col in 0..<w {
      var s = 0.0
      for d in -2...2 { s += k[d + 2] * Double(m[(row + r0) * width + min(max(col + d, 0), w - 1) + c0]) }
      tmp[row * w + col] = Float(s)
    }
  }
  for row in 0..<h {
    for col in 0..<w {
      var s = 0.0
      for d in -2...2 { s += k[d + 2] * Double(tmp[min(max(row + d, 0), h - 1) * w + col]) }
      out[row * w + col] = Float(s)
    }
  }
  return out
}

/// The length of the horizontal (or vertical) run of ones through every pixel of a crop.
func runs(_ m: [UInt8], _ width: Int, _ r0: Int, _ r1: Int, _ c0: Int, _ c1: Int, horizontal: Bool) -> [Int32] {
  var out = [Int32](repeating: 0, count: m.count)
  if horizontal {
    for row in r0..<r1 {
      var col = c0
      while col < c1 {
        if m[row * width + col] == 0 {
          col += 1
          continue
        }
        var e = col
        while e < c1 && m[row * width + e] != 0 { e += 1 }
        for i in col..<e { out[row * width + i] = Int32(e - col) }
        col = e
      }
    }
  } else {
    for col in c0..<c1 {
      var row = r0
      while row < r1 {
        if m[row * width + col] == 0 {
          row += 1
          continue
        }
        var e = row
        while e < r1 && m[e * width + col] != 0 { e += 1 }
        for i in row..<e { out[i * width + col] = Int32(e - row) }
        row = e
      }
    }
  }
  return out
}

/// 8-connected components: labels (1-based) and each one's pixel indices.
func components(_ m: [UInt8], _ width: Int, _ height: Int) -> (labels: [Int32], comps: [[Int]]) {
  var labels = [Int32](repeating: 0, count: width * height), comps: [[Int]] = []
  var stack: [Int] = []
  for i in 0..<width * height {
    if m[i] == 0 || labels[i] != 0 { continue }
    let id = Int32(comps.count + 1)
    var pix: [Int] = []
    labels[i] = id
    stack.append(i)
    while let at = stack.popLast() {
      let r = at / width, c = at % width
      pix.append(at)
      for dr in -1...1 {
        for dc in -1...1 {
          let rr = r + dr, cc = c + dc
          if rr < 0 || cc < 0 || rr >= height || cc >= width { continue }
          let j = rr * width + cc
          if m[j] != 0 && labels[j] == 0 {
            labels[j] = id
            stack.append(j)
          }
        }
      }
    }
    comps.append(pix)
  }
  return (labels, comps)
}
