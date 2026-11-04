#if canImport(Metal) && canImport(SwiftUI)
  import Foundation
  import JetlinkDiorama
  import Testing

  @testable import JetlinkDioramaUI

  /// The camera set by hand: at its defaults it is the camera as it was before
  /// it could be set, and pitch and forward turn and move it about the ego car
  /// so the car stays on the same pixel.
  struct DioramaCameraTests {
    typealias Camera = (f: Double, cx: Double, cy: Double, rotation: [Double], center: [Double])
    static let w = 1280.0, h = 720.0
    static let heights = [0.0, 1.22, 1.5]
    static let views = [0.0, 0.3, 0.5, 1.0]

    /// The first cameras, as they were written before the camera could be set.
    static func first(_ s: Double, _ hgt: Double) -> Camera {
      let chaseAt = [Diorama.egoRear - 6.5, 0, hgt - 3.2], chaseTurn = DioramaRenderer.lookAt(chaseAt, [18, 0, hgt])
      let birdAt = [-14, 0, hgt - 30], birdTurn = DioramaRenderer.lookAt(birdAt, [10, 0, hgt])
      let rel = DioramaRenderer.logRot(Diorama.m3mul(birdTurn, DioramaRenderer.transpose3(chaseTurn)))
      let rotation = Diorama.m3mul(DioramaRenderer.rodrigues(rel.map { $0 * s }), chaseTurn)
      let center = [chaseAt[0] + (birdAt[0] - chaseAt[0]) * s, chaseAt[1] + (birdAt[1] - chaseAt[1]) * s, chaseAt[2] + (birdAt[2] - chaseAt[2]) * pow(s, 0.7)]
      return ((0.62 * (1 - s) + 0.69 * s) * w, w / 2, (0.42 * (1 - s) + 0.5 * s) * h, rotation, center)
    }

    /// Where a road point lands in the image.
    static func pixel(_ c: Camera, _ p: [Double]) -> (x: Double, y: Double) {
      let d = Diorama.m3vec(c.rotation, [p[0] - c.center[0], p[1] - c.center[1], p[2] - c.center[2]])
      return (c.f * d[1] / d[0] + c.cx, c.f * d[2] / d[0] + c.cy)
    }

    /// How far in front of the camera a road point is.
    static func depth(_ c: Camera, _ p: [Double]) -> Double {
      Diorama.m3vec(c.rotation, [p[0] - c.center[0], p[1] - c.center[1], p[2] - c.center[2]])[0]
    }

    static func close(_ a: [Double], _ b: [Double], _ tol: Double) -> Bool {
      a.count == b.count && zip(a, b).allSatisfy { abs($0 - $1) <= tol }
    }

    /// The defaults are the first cameras moved defaultBack times as far from the car,
    /// looking the same way, with as many times the zoom: the car the same size on the same pixel.
    @Test(arguments: heights, views)
    func defaultsAreTheFirstCamerasFurtherBack(hgt: Double, s: Double) {
      let now = DioramaRenderer.camera(s, hgt, Self.w, Self.h), old = Self.first(s, hgt), k = DioramaRenderer.defaultBack
      let pivot = [DioramaRenderer.pivotX, 0, hgt]
      #expect(abs(now.f - k * old.f) < 1e-9)
      #expect(Self.close(now.rotation, old.rotation, 1e-12))
      #expect(Self.close(now.center, (0..<3).map { pivot[$0] + k * (old.center[$0] - pivot[$0]) }, 1e-9))
      // in either view, the car where the first camera drew it
      if s == 0 || s == 1 {
        let p = Self.pixel(now, pivot), q = Self.pixel(old, pivot)
        #expect(abs(p.x - q.x) < 1e-6 && abs(p.y - q.y) < 1e-6)
      }
    }

    /// Between the views the car goes straight from its pixel in one to its pixel in the
    /// other, so it never leaves the picture, however the views are set.
    @Test(arguments: heights)
    func thePanKeepsTheCarOnALine(hgt: Double) {
      let pivot = [DioramaRenderer.pivotX, 0, hgt]
      var set = DioramaCamera.defaults
      set.road.pitch = 30
      DioramaCamera.moveForward(&set.road, -4, view: 0)
      DioramaCamera.zoom(&set.road, to: 1.8, view: 0)
      DioramaCamera.zoom(&set.bird, to: 1.1, view: 1)
      for setup in [DioramaCamera.defaults, set] {
        let a = Self.pixel(DioramaRenderer.camera(0, hgt, Self.w, Self.h, setup), pivot)
        let b = Self.pixel(DioramaRenderer.camera(1, hgt, Self.w, Self.h, setup), pivot)
        for s in [0.1, 0.3, 0.5, 0.7, 0.9] {
          let p = Self.pixel(DioramaRenderer.camera(s, hgt, Self.w, Self.h, setup), pivot)
          #expect(abs(p.x - (a.x + (b.x - a.x) * s)) < 1e-6 && abs(p.y - (a.y + (b.y - a.y) * s)) < 1e-6, "s \(s)")
          #expect(p.y >= 0 && p.y <= Self.h)
        }
      }
    }

    /// Views moved forward and back, and up as moving forward at another pitch leaves them.
    static let offsets: [(forward: Double, rise: Double)] = [(0, 0), (12, 0), (-30, 4), (5, -8)]

    @Test(arguments: heights)
    func pitchKeepsTheEgoCarsPixel(hgt: Double) {
      let pivot = [DioramaRenderer.pivotX, 0, hgt]
      for view in 0...1 {
        let s = Double(view)
        for o in Self.offsets {
          for zoom in [0.3, DioramaCamera.defaults[view].zoom, 2.0] {
            var setup = DioramaCamera.defaults
            setup[view].forward = o.forward
            setup[view].rise = o.rise
            setup[view].zoom = zoom
            let at = Self.pixel(DioramaRenderer.camera(s, hgt, Self.w, Self.h, setup), pivot)
            for pitch in [-15.0, 0, 5, 45, 70, 89, 120, 170] {
              setup[view].pitch = pitch
              let cam = DioramaRenderer.camera(s, hgt, Self.w, Self.h, setup)
              let p = Self.pixel(cam, pivot)
              #expect(abs(p.x - at.x) < 1e-6 && abs(p.y - at.y) < 1e-6, "view \(view), \(o), zoom \(zoom), pitch \(pitch)")
              // still a rotation
              let rot = cam.rotation, rrt = Diorama.m3mul(rot, DioramaRenderer.transpose3(rot))
              #expect(Self.close(rrt, [1, 0, 0, 0, 1, 0, 0, 0, 1], 1e-12))
            }
          }
        }
      }
    }

    /// Zoom, about the car, keeps it on its pixel: in either view, wherever forward and pitch put the camera.
    @Test(arguments: heights)
    func zoomKeepsTheEgoCarsPixel(hgt: Double) {
      let pivot = [DioramaRenderer.pivotX, 0, hgt]
      for view in 0...1 {
        let s = Double(view)
        for o in Self.offsets {
          for pitch in [10.0, 45, 80] {
            var setup = DioramaCamera.defaults
            setup[view].forward = o.forward
            setup[view].rise = o.rise
            setup[view].pitch = pitch
            let base = DioramaRenderer.camera(s, hgt, Self.w, Self.h, setup)
            // a camera moved past the car has it behind, with no pixel to keep
            guard Self.depth(base, pivot) > 0 else { continue }
            let at = Self.pixel(base, pivot)
            for zoom in [0.2, 0.5, 1.4, 3] {
              var v = setup[view]
              DioramaCamera.zoom(&v, to: zoom, view: view)
              var zoomed = setup
              zoomed[view] = v
              let cam = DioramaRenderer.camera(s, hgt, Self.w, Self.h, zoomed)
              let p = Self.pixel(cam, pivot)
              #expect(abs(p.x - at.x) < 1e-6 && abs(p.y - at.y) < 1e-6, "view \(view), \(o), pitch \(pitch), zoom \(zoom)")
              #expect(abs(cam.f - zoom * Self.w) < 1e-9)
            }
          }
        }
      }
    }

    @Test(arguments: heights)
    func forwardMovesTheCameraAlongTheRoad(hgt: Double) {
      for view in 0...1 {
        let s = Double(view)
        for pitch in [10.0, DioramaCamera.defaults[view].pitch, 45, 80] {
          for start in Self.offsets {
            var v = DioramaCamera.defaults[view]
            v.pitch = pitch
            v.forward = start.forward
            v.rise = start.rise
            var setup = DioramaCamera.defaults
            setup[view] = v
            let still = DioramaRenderer.camera(s, hgt, Self.w, Self.h, setup)
            for metres in [-20.0, -2.5, 7, 20] {
              var moved = v
              DioramaCamera.moveForward(&moved, metres, view: view)
              setup[view] = moved
              let cam = DioramaRenderer.camera(s, hgt, Self.w, Self.h, setup)
              // along x by the metres, level with the road, looking the same way, and the
              // picture otherwise the same: no zoom, no shift
              #expect(abs(cam.center[0] - still.center[0] - metres) < 1e-9, "view \(view), pitch \(pitch), \(start), \(metres)")
              #expect(abs(cam.center[1] - still.center[1]) < 1e-12 && abs(cam.center[2] - still.center[2]) < 1e-9)
              #expect(Self.close(cam.rotation, still.rotation, 1e-12))
              #expect(cam.f == still.f && cam.cx == still.cx && cam.cy == still.cy)
            }
          }
        }
      }
    }

    /// A drag moves the camera just so far that the road under the finger follows it.
    @Test func forwardDragsTheRoadUnderTheFinger() throws {
      let hgt = DioramaRenderer.preview.height, w = 402.0, h = 640.0
      var checked = 0
      for view in 0...1 {
        for pitch in [DioramaCamera.defaults[view].pitch, 35, 60] {
          for start in [0.0, 6, -8] {
            var v = DioramaCamera.defaults[view]
            v.pitch = pitch
            DioramaCamera.moveForward(&v, start, view: view)
            for (x, y) in [(201.0, 600.0), (120, 450), (300, 520)] {
              for dy in [3.0, -3] {
                var setup = DioramaCamera.defaults
                setup[view] = v
                let cam = DioramaRenderer.camera(Double(view), hgt, w, h, setup)
                // the road point under (x, y)
                let ray = Diorama.m3vec(DioramaRenderer.transpose3(cam.rotation), [1, (x - cam.cx) / cam.f, (y - cam.cy) / cam.f])
                guard ray[2] > 0, let m = DioramaCamera.metresFollowing(v, view: view, dy: dy, x: x, y: y, w: w, h: h) else { continue }
                let t = (hgt - cam.center[2]) / ray[2]
                let ground = (0..<3).map { cam.center[$0] + t * ray[$0] }
                var moved = v
                DioramaCamera.moveForward(&moved, m, view: view)
                setup[view] = moved
                let p = Self.pixel(DioramaRenderer.camera(Double(view), hgt, w, h, setup), ground)
                // down the screen by the drag, to well under a point over a few points' step (off
                // the middle the road also spreads sideways as the camera nears, which no
                // forward move can undo: only the drag's own direction is followed)
                #expect(abs(p.y - (y + dy)) < 0.05, "view \(view), pitch \(pitch), start \(start), (\(x), \(y)), dy \(dy)")
                checked += 1
              }
            }
          }
        }
      }
      #expect(checked > 20)
    }

    @Test func morePitchIsHigher() {
      var setup = DioramaCamera.defaults
      setup.road.pitch += 20
      let higher = DioramaRenderer.camera(0, 1.22, Self.w, Self.h, setup), base = DioramaRenderer.camera(0, 1.22, Self.w, Self.h)
      #expect(higher.center[2] < base.center[2])  // z down: higher up
    }

    @Test func keptAsTenNumbersWithNoLimits() {
      var setup = DioramaCamera.defaults
      setup.road = DioramaCamera.View(zoom: 1.3, pitch: 30, forward: -12, rise: 3.5, shift: -0.2)
      #expect(DioramaCamera(rawValue: setup.rawValue) == setup)
      #expect(DioramaCamera(rawValue: "") == nil)
      #expect(DioramaCamera(rawValue: "1 2 3 4 5 6 7 8") == nil)
      // anything a user sets is kept as it is
      let wild = DioramaCamera(rawValue: "9 -5 -100 70 9 0.01 135 900 -61 -9")
      #expect(wild?.road == DioramaCamera.View(zoom: 9, pitch: -5, forward: -100, rise: 70, shift: 9))
      #expect(wild?.bird == DioramaCamera.View(zoom: 0.01, pitch: 135, forward: 900, rise: -61, shift: -9))
      // but a focal length is above zero
      #expect(DioramaCamera(rawValue: "0 20 0 0 0 1 60 0 0 0") == nil)
      #expect(DioramaCamera(rawValue: "1 20 0 0 0 -1 60 0 0 0") == nil)
      var v = DioramaCamera.defaults.road
      DioramaCamera.zoom(&v, to: 0, view: 0)
      #expect(v == DioramaCamera.defaults.road)
    }

    /// The scene the camera is set over is its own, never a drive's: the ego car, a lead
    /// 10 m ahead of its bumper, the path stopping 2 m short of it, two straight lane
    /// lines, and nothing from the segmentation.
    @Test func thePreviewIsAStillSceneOfItsOwn() throws {
      let frame = DioramaRenderer.preview
      let state = try #require(frame.state)
      let lead = try #require(state.lead)
      #expect(abs(lead.x - Diorama.egoFront - 10) < 1e-12 && lead.y == 0 && lead.heading == 0 && !lead.seg)
      #expect(abs(lead.x - DioramaRenderer.previewPathShortOfLead - Diorama.egoFront - 8) < 1e-12)
      #expect(state.py.allSatisfy { $0 == 0 })
      // nothing from the segmentation: no road map, no crosswalk
      #expect(frame.origin == nil && !frame.mapUploaded && state.stripes.isEmpty)
      // the ego lane's two lines, straight
      let out = state.out
      let shown = (0..<4).filter { out.laneLinesProb[2 * $0 + 1] > 0.3 }
      #expect(shown == [1, 2])
      for i in shown {
        let ys = (0..<Diorama.idxN).map { DioramaOutputs.line(out.laneLines, i, $0, 0) }
        #expect(Set(ys).count == 1)
      }
    }

    @Test func gesturesGoTheWaysThePageGoes() {
      var v = DioramaCamera.defaults.road
      DioramaCameraGesture.pitch(&v, dragDown: 10)
      #expect(v.pitch > DioramaCamera.defaults.road.pitch)
      v = DioramaCamera.defaults.road
      DioramaCameraGesture.wheel(&v, up: 100, view: 0)
      #expect(v.zoom > DioramaCamera.defaults.road.zoom)
      v = DioramaCamera.defaults.road
      DioramaCameraGesture.pinch(&v, by: 1.5, view: 0)
      #expect(v.zoom > DioramaCamera.defaults.road.zoom)
      v = DioramaCamera.defaults.road
      DioramaCameraGesture.forward(&v, down: 50, view: 0)
      #expect(v.forward > 0)  // fingers down: forward
    }
  }
#endif
