#if canImport(Network) && canImport(SwiftUI)
  import Foundation
  import JetlinkDiorama
  import Network
  import Observation

  /// The drive from a Mac nearby, for a view: which Macs there are, which one
  /// is watched, and its messages for the page. It looks for Macs only
  /// between `activate()` and `deactivate()`, the time the view is on screen.
  @MainActor
  @Observable
  public final class DioramaRemote: DioramaFeed {
    public private(set) var state: DioramaWatch.State = .stopped
    /// The Macs found showing the drive, by name.
    public private(set) var macs: [String] = []
    /// The Mac to watch when it is found, kept by the app between launches.
    public var preferred: String? {
      didSet { if let preferred, preferred != oldValue { watch.choose(preferred) } }
    }

    @ObservationIgnored private let watch: DioramaWatch
    @ObservationIgnored private var deliver: (@MainActor @Sendable ([Data]) async -> Void)?
    @ObservationIgnored private var active = false

    /// `name`: how this device appears in the Mac's list of viewers.
    public init(name: String, preferred: String?) {
      watch = DioramaWatch(name: name)
      self.preferred = preferred
      watch.onState = { [weak self] state, macs in
        Task { @MainActor in
          self?.state = state
          self?.macs = macs
        }
      }
      watch.onMessages = { [weak self] messages, done in
        Task { @MainActor in
          if let deliver = self?.deliver { await deliver(messages) }
          done()
        }
      }
    }

    public func activate() {
      guard !active else { return }
      active = true
      watch.start(preferring: preferred)
    }

    public func deactivate() {
      guard active else { return }
      active = false
      watch.stop()
    }

    // MARK: DioramaFeed

    /// `input` is ignored: a Mac sends only what the page draws with, never the frames the models see.
    public func start(input: Bool, deliver: @escaping @MainActor @Sendable ([Data]) async -> Void) {
      self.deliver = deliver
      watch.send("ready")
    }

    public func stop() {
      deliver = nil
    }
  }
#endif
