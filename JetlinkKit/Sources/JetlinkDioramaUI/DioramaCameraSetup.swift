#if canImport(SwiftUI) && canImport(MetalKit)
  import JetlinkDiorama
  import MetalKit
  import SwiftUI

  /// Each Diorama view's camera, set by hand one view at a time over a still
  /// scene of its own, the same whether or not a comma is connected or a drive
  /// plays: pinch or the mouse wheel zooms, a drag pitches the camera about the
  /// car, and two fingers or a right-drag move it forward and back along the
  /// road. Saved to this app's own defaults, which every Diorama view it draws reads.
  public struct DioramaCameraSetup: View {
    @AppStorage(DioramaCamera.storageKey) private var saved = DioramaCamera.defaults
    @State private var draft: DioramaCamera?
    @State private var view = 0
    /// Never fed: the preview draws its still scene, not a drive.
    @State private var player = DioramaPlayer()
    @Environment(\.colorScheme) private var colorScheme
    @Environment(\.dismiss) private var dismiss

    public init() {}

    public var body: some View {
      let camera = draft ?? saved
      VStack(spacing: 0) {
        Picker("View", selection: $view) {
          Text("Road view").tag(0)
          Text("Bird's-eye").tag(1)
        }
        .pickerStyle(.segmented)
        .labelsHidden()
        .padding(12)
        ZStack {
          DioramaMetalView(player: player, dark: colorScheme == .dark, camera: camera, fixedView: Double(view), preview: true)
          DioramaCameraGestures(view: view) { change($0) }
        }
        .clipped()
        .frame(minHeight: 240)
        numbers(camera[view])
        HStack {
          Button("Reset View") { change { $0 = view == 0 ? DioramaCamera.defaults.road : DioramaCamera.defaults.bird } }
          Spacer()
          Button("Cancel", role: .cancel) { dismiss() }
            .keyboardShortcut(.cancelAction)
          Button("Save") {
            saved = camera
            dismiss()
          }
          .keyboardShortcut(.defaultAction)
          .buttonStyle(.borderedProminent)
        }
        .padding(12)
      }
      .onAppear { draft = saved }
    }

    /// The three numbers, kept on one line each whatever the width, and how to change them.
    private func numbers(_ v: DioramaCamera.View) -> some View {
      VStack(alignment: .leading, spacing: 6) {
        HStack(spacing: 18) {
          number("Zoom", String(format: "%.2f", v.zoom))
          number("Pitch", String(format: "%.1f°", v.pitch))
          number("Forward", String(format: "%+.1f m", v.forward))
          Spacer(minLength: 0)
        }
        .font(.callout.monospacedDigit())
        Text(hint)
          .font(.footnote)
          .foregroundStyle(.secondary)
          .fixedSize(horizontal: false, vertical: true)
      }
      .padding(.horizontal, 12)
      .padding(.top, 10)
    }

    private func number(_ name: String, _ value: String) -> some View {
      HStack(spacing: 6) {
        Text(name)
        Text(value).foregroundStyle(.secondary)
      }
      .lineLimit(1)
      .fixedSize()
    }

    private var hint: String {
      #if os(macOS)
        "Pinch or scroll the wheel to zoom, drag to pitch, two fingers or right-drag to move forward and back."
      #else
        "Pinch to zoom, drag to pitch, two fingers to move forward and back."
      #endif
    }

    /// One gesture's change to the view shown, as far as it goes.
    private func change(_ edit: (inout DioramaCamera.View) -> Void) {
      var camera = draft ?? saved
      edit(&camera[view])
      draft = camera
    }
  }


  /// A gesture's change to the camera: the three degrees of freedom, as the page's.
  enum DioramaCameraGesture {
    /// Drag down: the camera rises, 0.25° a point.
    static func pitch(_ v: inout DioramaCamera.View, dragDown dy: Double) { v.pitch += dy * 0.25 }
    /// Pinch out: zoom in by the magnification, about the car.
    static func pinch(_ v: inout DioramaCamera.View, by ratio: Double, view: Int) {
      if ratio.isFinite, ratio > 0 { DioramaCamera.zoom(&v, to: v.zoom * ratio, view: view) }
    }
    /// The wheel turned up `up` pixels: zoom in, about the car.
    static func wheel(_ v: inout DioramaCamera.View, up: Double, view: Int) { DioramaCamera.zoom(&v, to: v.zoom * exp(up * 0.0015), view: view) }
    /// Moved `dy` points down the screen: forward along the road, as the road is pulled
    /// back under the camera; up: back. Faster the further the view's camera sits from
    /// the car, about 0.05 m a point in the road view.
    static func forward(_ v: inout DioramaCamera.View, down dy: Double, view: Int) {
      DioramaCamera.moveForward(&v, dy * 0.005 * DioramaCamera.distance(view: view), view: view)
    }
  }

  /// The camera's gestures over the scene, from a trackpad, a mouse or a touchscreen.
  struct DioramaCameraGestures: PlatformViewRepresentable {
    /// The view being set: 0 the road view, 1 the bird's-eye.
    let view: Int
    let change: ((inout DioramaCamera.View) -> Void) -> Void

    #if os(macOS)
      func makeNSView(context: Context) -> GestureView {
        let view = GestureView()
        updateNSView(view, context: context)
        return view
      }
      func updateNSView(_ view: GestureView, context: Context) {
        view.change = change
        view.view = self.view
      }

      final class GestureView: NSView {
        var change: (((inout DioramaCamera.View) -> Void) -> Void)?
        var view = 0

        override var acceptsFirstResponder: Bool { true }
        override func acceptsFirstMouse(for event: NSEvent?) -> Bool { true }

        override func magnify(with event: NSEvent) {
          let ratio = 1 + Double(event.magnification), view = view
          change? { DioramaCameraGesture.pinch(&$0, by: ratio, view: view) }
        }

        /// A trackpad's two fingers move forward and back; a mouse's wheel zooms.
        override func scrollWheel(with event: NSEvent) {
          // A trackpad's two fingers come in phases, began to ended, then momentum after
          // they lift; a mouse's wheel has none, smooth-scrolling mice included.
          if !event.phase.isEmpty {
            // natural scrolling: fingers up give a negative delta, as the content moves up with them
            let fingersDown = Double(event.isDirectionInvertedFromDevice ? event.scrollingDeltaY : -event.scrollingDeltaY)
            let view = view
            change? { DioramaCameraGesture.forward(&$0, down: fingersDown, view: view) }
          } else if event.momentumPhase.isEmpty {
            // the wheel turned up, whichever way the system scrolls; a notch is about 40 pixels, as a browser takes it
            let up = Double(event.isDirectionInvertedFromDevice ? -event.scrollingDeltaY : event.scrollingDeltaY)
            let pixels = event.hasPreciseScrollingDeltas ? up : up * 40, view = view
            change? { DioramaCameraGesture.wheel(&$0, up: pixels, view: view) }
          }
        }

        override func mouseDown(with event: NSEvent) { window?.makeFirstResponder(self) }

        override func mouseDragged(with event: NSEvent) {
          let dy = Double(event.deltaY)  // down positive
          change? { DioramaCameraGesture.pitch(&$0, dragDown: dy) }
        }

        override func rightMouseDown(with event: NSEvent) {}

        override func rightMouseDragged(with event: NSEvent) {
          let dy = Double(event.deltaY)
          let view = view
          change? { DioramaCameraGesture.forward(&$0, down: dy, view: view) }
        }
      }
    #else
      func makeUIView(context: Context) -> UIView {
        let view = UIView()
        view.backgroundColor = .clear
        let c = context.coordinator
        let pinch = UIPinchGestureRecognizer(target: c, action: #selector(Coordinator.pinch(_:)))
        let pitch = UIPanGestureRecognizer(target: c, action: #selector(Coordinator.pitch(_:)))
        pitch.minimumNumberOfTouches = 1
        pitch.maximumNumberOfTouches = 1
        let forward = UIPanGestureRecognizer(target: c, action: #selector(Coordinator.forward(_:)))
        forward.minimumNumberOfTouches = 2
        forward.maximumNumberOfTouches = 2
        // a pointer: a trackpad's two fingers and a right-drag move forward and back, a mouse's wheel zooms
        // Each takes only what it is for: a touch on the glass is never one of these, and
        // a pan's button mask means nothing to a finger, which it would take as a drag.
        let fingers = UIPanGestureRecognizer(target: c, action: #selector(Coordinator.forward(_:)))
        fingers.allowedScrollTypesMask = .continuous
        fingers.allowedTouchTypes = []
        let wheel = UIPanGestureRecognizer(target: c, action: #selector(Coordinator.wheel(_:)))
        wheel.allowedScrollTypesMask = .discrete
        wheel.allowedTouchTypes = []
        let right = SecondaryPan(target: c, action: #selector(Coordinator.forward(_:)))
        right.allowedTouchTypes = [NSNumber(value: UITouch.TouchType.indirectPointer.rawValue)]
        for g in [pinch, pitch, forward, fingers, wheel, right] {
          g.delegate = c
          view.addGestureRecognizer(g)
        }
        c.together = [pinch, forward]
        return view
      }

      func updateUIView(_ view: UIView, context: Context) {
        context.coordinator.change = change
        context.coordinator.view = self.view
      }

      /// A drag with a pointer's secondary button: a pan's button mask is fixed on creation.
      final class SecondaryPan: UIPanGestureRecognizer {
        override var buttonMask: UIEvent.ButtonMask {
          get { .secondary }
          set {}
        }
      }

      func makeCoordinator() -> Coordinator { Coordinator(change: change) }

      @MainActor
      final class Coordinator: NSObject, UIGestureRecognizerDelegate {
        var change: ((inout DioramaCamera.View) -> Void) -> Void
        var view = 0
        var together: [UIGestureRecognizer] = []

        init(change: @escaping ((inout DioramaCamera.View) -> Void) -> Void) { self.change = change }

        @objc func pinch(_ g: UIPinchGestureRecognizer) {
          let ratio = Double(g.scale)
          g.scale = 1
          let view = view
          change { DioramaCameraGesture.pinch(&$0, by: ratio, view: view) }
        }

        @objc func pitch(_ g: UIPanGestureRecognizer) {
          let dy = Double(g.translation(in: g.view).y)
          g.setTranslation(.zero, in: g.view)
          change { DioramaCameraGesture.pitch(&$0, dragDown: dy) }
        }

        @objc func forward(_ g: UIPanGestureRecognizer) {
          let dy = Double(g.translation(in: g.view).y)
          g.setTranslation(.zero, in: g.view)
          let view = view
          change { DioramaCameraGesture.forward(&$0, down: dy, view: view) }
        }

        /// The wheel turned up scrolls the content down, a positive translation.
        @objc func wheel(_ g: UIPanGestureRecognizer) {
          let dy = Double(g.translation(in: g.view).y)
          g.setTranslation(.zero, in: g.view)
          let view = view
          change { DioramaCameraGesture.wheel(&$0, up: dy, view: view) }
        }

        func gestureRecognizer(_ a: UIGestureRecognizer, shouldRecognizeSimultaneouslyWith b: UIGestureRecognizer) -> Bool {
          together.contains(a) && together.contains(b)
        }
      }
    #endif
  }
#endif
