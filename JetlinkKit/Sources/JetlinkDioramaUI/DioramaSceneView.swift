#if canImport(SwiftUI) && canImport(MetalKit)
  import JetlinkDiorama
  import MetalKit
  import SwiftUI

  /// The drive as a live 3-D miniature, drawn natively in Metal from a feed's
  /// messages: the scene, the HUD over it, and the models' timings below. The
  /// same on a Mac and an iPhone, and the same drawing as the page's.
  public struct DioramaSceneView: View {
    let feed: any DioramaFeed
    /// The Model input toggle: off for a feed that never carries the frames.
    let modelInput: Bool
    @State private var player = DioramaPlayer()
    @AppStorage("diorama.input") private var showInput = false
    @AppStorage(DioramaCamera.storageKey) private var camera = DioramaCamera.defaults
    @Environment(\.colorScheme) private var colorScheme

    public init(feed: any DioramaFeed, modelInput: Bool = true) {
      self.feed = feed
      self.modelInput = modelInput
    }

    public var body: some View {
      VStack(spacing: 0) {
        ZStack(alignment: .topLeading) {
          DioramaMetalView(player: player, dark: colorScheme == .dark, camera: camera)
          if let hud = player.hud { DioramaHUD(hud: hud).padding(14) }
          if modelInput && showInput, let picture = player.input {
            DioramaInputView(picture: picture)
              .frame(maxWidth: 512)
              .containerRelativeFrame(.horizontal) { width, _ in min(width * 0.46, 512) }
              .padding(14)
              .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .bottomTrailing)
              .allowsHitTesting(false)
          }
          TimelineView(.periodic(from: .now, by: 0.5)) { context in
            if player.waiting(at: context.date) {
              Text("Waiting for the comma…")
                .font(.system(size: 15))
                .foregroundStyle(Color(white: 0.73))
                .multilineTextAlignment(.center)
                .padding(24)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .background(Color(red: 0.06, green: 0.063, blue: 0.07).opacity(0.6))
            }
          }
          .allowsHitTesting(false)
        }
        .clipped()
        DioramaStats(player: player, modelInput: modelInput, showInput: $showInput)
      }
      .onAppear { start() }
      .onDisappear { feed.stop() }
      .onChange(of: ObjectIdentifier(feed)) {
        // the server restarted, or the source changed: a new feed, and the scene starts over from its meta
        start()
      }
      .onChange(of: showInput) {
        if !showInput { player.clearInput() }
        start()
      }
    }

    private func start() {
      let input = modelInput && showInput
      player.showInput = input
      feed.start(input: input) { [player] batch in await player.take(batch) }
    }
  }

  /// The speed, the lead, the view, and the acceleration as a bar, as the page's HUD.
  struct DioramaHUD: View {
    let hud: DioramaPlayer.HUD

    var body: some View {
      VStack(alignment: .leading, spacing: 0) {
        Text("\(hud.speed)")
          .font(.system(size: 44, weight: .semibold).monospacedDigit())
          .foregroundStyle(Color(white: 0.95))
        Text("MPH").font(.system(size: 12)).kerning(1).foregroundStyle(Color(white: 0.66))
        if let lead = hud.lead {
          Text("lead \(lead) m").font(.system(size: 12)).foregroundStyle(Color(white: 0.77)).padding(.top, 6)
        }
        Text(hud.mode).font(.system(size: 12)).kerning(0.7).foregroundStyle(Color(white: 0.55)).padding(.top, 6)
      }
      .padding(.vertical, 12)
      .padding(.leading, 16)
      .padding(.trailing, 36)
      .frame(minWidth: 150, alignment: .leading)
      .overlay(alignment: .trailing) { accel.padding(14) }
      .background(Color(red: 0.086, green: 0.09, blue: 0.1).opacity(0.86), in: .rect(cornerRadius: 12))
      .allowsHitTesting(false)
    }

    /// Up from the middle accelerating, down braking.
    private var accel: some View {
      GeometryReader { geo in
        let a = Diorama.clamp(hud.accel, -3.2, 3.2), h = geo.size.height
        let top = (0.5 - max(a, 0) / 3.2 * 0.5) * h, bottom = (0.5 + min(a, 0) / 3.2 * 0.5) * h
        let color = a > 0.2 ? Color(red: 0.35, green: 0.82, blue: 0.48) : a < -0.5 ? Color(red: 1, green: 0.35, blue: 0.32) : Color(white: 0.78)
        ZStack(alignment: .top) {
          Capsule().fill(Color(red: 0.23, green: 0.24, blue: 0.26))
          Capsule().fill(color).frame(height: max(h - top - bottom, 0)).offset(y: top)
        }
      }
      .frame(width: 6)
    }
  }

  /// The models' timings, their state when not simply running, and the Model input toggle.
  struct DioramaStats: View {
    let player: DioramaPlayer
    let modelInput: Bool
    @Binding var showInput: Bool

    var body: some View {
      TimelineView(.periodic(from: .now, by: 0.5)) { context in
        let t = player.timing, model = median(t.model), seg = median(t.seg)
        ViewThatFits(in: .horizontal) {
          HStack(spacing: 18) { items(context.date, model, seg, t) }
          LazyVGrid(columns: [GridItem(.adaptive(minimum: 92), alignment: .leading)], alignment: .leading, spacing: 8) {
            items(context.date, model, seg, t)
          }
        }
      }
      .padding(.horizontal, 16)
      .padding(.vertical, 10)
      .frame(maxWidth: .infinity, alignment: .leading)
      .background(.bar)
    }

    @ViewBuilder
    private func items(_ now: Date, _ model: Double?, _ seg: Double?, _ t: DioramaScene.Timing) -> some View {
      stat(ms(model), "Driving model", over: (model ?? 0) > 50)
      stat("\(Int(player.frameRate(at: now).rounded())) Hz", "Frames")
      stat(ms(seg), "Segmentation")
      stat(segRate(t), "Perception")
      if let models = player.models {
        stat(models.prefix(1).uppercased() + models.dropFirst(), "Models", over: models.hasPrefix("off"))
          .lineLimit(1)
          .help(models)
      }
      if modelInput {
        Toggle("Model input", isOn: $showInput)
          .font(.system(size: 12))
          .foregroundStyle(.secondary)
          #if os(macOS)
            .toggleStyle(.checkbox)
          #endif
          .help("The frame the models see: the narrow camera, warped to 512×256")
          .fixedSize()
      }
    }

    private func stat(_ value: String, _ name: String, over: Bool = false) -> some View {
      VStack(alignment: .leading, spacing: 1) {
        Text(value).font(.system(size: 15, weight: .semibold).monospacedDigit()).foregroundStyle(over ? Color.red : Color.primary)
        Text(name.uppercased()).font(.system(size: 11)).kerning(0.6).foregroundStyle(.secondary)
      }
      .frame(minWidth: 92, alignment: .leading)
    }

    private func ms(_ v: Double?) -> String { v.map { String(format: "%.1f ms", $0) } ?? "–" }

    private func segRate(_ t: DioramaScene.Timing) -> String {
      let f = t.segFrames
      guard f.count > 1 else { return "–" }
      return String(format: "%.1f Hz", Diorama.fps * Double(f.count - 1) / Double(max(f[f.count - 1] - f[0], 1)))
    }

    private func median(_ a: [Double]) -> Double? { a.isEmpty ? nil : a.sorted()[a.count >> 1] }
  }

  /// The frame the models see, each thing named above it.
  struct DioramaInputView: View {
    let picture: InputPicture

    var body: some View {
      VStack(spacing: 0) {
        Image(decorative: picture.image, scale: 1)
          .resizable()
          .interpolation(.none)
          .aspectRatio(2, contentMode: .fit)
          .overlay {
            GeometryReader { geo in
              let s = geo.size.width / 512
              ZStack(alignment: .topLeading) {
                ForEach(Array(picture.labels.enumerated()), id: \.offset) { _, label in
                  Text(label.text)
                    .font(.system(size: 10 * s, weight: .semibold))
                    .foregroundStyle(.black)
                    .padding(.horizontal, 3 * s)
                    .frame(height: 13 * s)
                    .background(Color(red: Double(label.rgb.x) / 255, green: Double(label.rgb.y) / 255, blue: Double(label.rgb.z) / 255).opacity(0.85))
                    .fixedSize()
                    .offset(x: max(label.x, 0) * s, y: max(label.y - 13, 0) * s)
                }
              }
              .frame(width: geo.size.width, height: geo.size.height, alignment: .topLeading)
            }
          }
        Text(picture.caption)
          .font(.system(size: 11).monospacedDigit())
          .foregroundStyle(Color(white: 0.87))
          .lineLimit(1)
          .truncationMode(.tail)
          .padding(.horizontal, 8)
          .padding(.vertical, 3)
          .frame(maxWidth: .infinity, alignment: .leading)
          .background(.black)
      }
      .background(.black)
      .clipShape(.rect(cornerRadius: 8))
      .shadow(color: .black.opacity(0.35), radius: 6, y: 2)
    }
  }


  #if os(macOS)
    typealias PlatformViewRepresentable = NSViewRepresentable
  #else
    typealias PlatformViewRepresentable = UIViewRepresentable
  #endif

  /// The Metal view the renderer draws into: on demand, once per batch the
  /// player takes, and when its size or the theme changes.
  struct DioramaMetalView: PlatformViewRepresentable {
    let player: DioramaPlayer
    let dark: Bool
    var camera = DioramaCamera.defaults
    /// One view whatever the drive's view chooser says.
    var fixedView: Double?
    /// The still scene the camera is set over, instead of the player's drive.
    var preview = false

    func makeCoordinator() -> Coordinator { Coordinator(player: player) }

    #if os(macOS)
      func makeNSView(context: Context) -> MTKView { make(context) }
      func updateNSView(_ view: MTKView, context: Context) { update(view, context) }
    #else
      func makeUIView(context: Context) -> MTKView { make(context) }
      func updateUIView(_ view: MTKView, context: Context) { update(view, context) }
    #endif

    private func make(_ context: Context) -> MTKView {
      let view = MTKView(frame: .zero, device: context.coordinator.renderer?.device ?? MTLCreateSystemDefaultDevice())
      view.colorPixelFormat = DioramaRenderer.colorFormat
      view.depthStencilPixelFormat = DioramaRenderer.depthFormat
      view.sampleCount = DioramaRenderer.sampleCount
      view.enableSetNeedsDisplay = true
      view.isPaused = true
      view.delegate = context.coordinator
      context.coordinator.view = view
      update(view, context)
      return view
    }

    private func update(_ view: MTKView, _ context: Context) {
      let coordinator = context.coordinator
      coordinator.player = player
      player.redraw = { [weak coordinator] in coordinator?.setNeedsDisplay() }
      if coordinator.dark != dark {
        coordinator.dark = dark
        coordinator.renderer?.theme = dark ? .dark : .light
        coordinator.setNeedsDisplay()
      }
      if let renderer = coordinator.renderer, renderer.camera != camera || renderer.fixedView != fixedView || coordinator.preview != preview {
        renderer.camera = camera
        renderer.fixedView = fixedView
        renderer.pathShortOfLead = preview ? DioramaRenderer.previewPathShortOfLead : 3
        coordinator.preview = preview
        coordinator.setNeedsDisplay()
      }
    }

    @MainActor
    final class Coordinator: NSObject, MTKViewDelegate {
      var player: DioramaPlayer
      let renderer: DioramaRenderer?
      weak var view: MTKView?
      var dark: Bool?
      var preview = false

      init(player: DioramaPlayer) {
        self.player = player
        renderer = MTLCreateSystemDefaultDevice().flatMap { DioramaRenderer(device: $0) }
      }

      private var drawPending = false

      /// Draws once more, soon: once for however many asks come before it does.
      func setNeedsDisplay() {
        #if os(macOS)
          // An MTKView drawn on demand by needsDisplay stops drawing once its window is
          // closed and opened again, as the camera's window is: SwiftUI keeps the view,
          // and AppKit never asks it to draw again. Drawing it directly always does.
          guard !drawPending else { return }
          drawPending = true
          DispatchQueue.main.async { [weak self] in
            guard let self else { return }
            self.drawPending = false
            self.view?.draw()
          }
        #else
          view?.setNeedsDisplay()
        #endif
      }

      /// A new size draws again: with no drive playing, nothing else would.
      nonisolated func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        MainActor.assumeIsolated { setNeedsDisplay() }
      }

      nonisolated func draw(in view: MTKView) {
        MainActor.assumeIsolated {
          guard let renderer, let pass = view.currentRenderPassDescriptor else { return }
          let frame = preview ? DioramaRenderer.preview : player.box.with { renderer.take($0) }
          renderer.draw(frame, to: pass, size: view.drawableSize) { commands in
            if let drawable = view.currentDrawable { commands.present(drawable) }
          }
        }
      }
    }
  }
#endif
