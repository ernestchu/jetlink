#if canImport(SwiftUI)
  import Foundation
  import JetlinkDiorama

  /// Where the scene's messages come from: the stage on this device's own
  /// server (`StageFeed`), or a Mac nearby (`DioramaRemote`).
  @MainActor
  public protocol DioramaFeed: AnyObject {
    /// From the meta on, the frames the models see as well with `input`;
    /// `deliver` returns once the scene has taken a batch, and no more come
    /// before it does. A second start replaces the first.
    func start(input: Bool, deliver: @escaping @MainActor @Sendable ([Data]) async -> Void)
    func stop()
  }

  /// The stage on this device's server, a watcher of it while the view is up.
  @MainActor
  public final class StageFeed: DioramaFeed {
    public let stage: DioramaStage
    private var relay: Task<Void, Never>?

    public init(stage: DioramaStage) { self.stage = stage }

    public func start(input: Bool, deliver: @escaping @MainActor @Sendable ([Data]) async -> Void) {
      stop()
      let stage = stage
      relay = Task.detached(priority: .utility) {
        while !Task.isCancelled {
          // a watcher that fell behind is dropped: watch again, from a fresh meta the scene starts over from
          let watcher = stage.watch(input: input)
          while !Task.isCancelled, let batch = stage.next(watcher, timeout: 0.5) {
            if !batch.isEmpty { await deliver(batch) }
          }
          stage.unwatch(watcher)
          if !Task.isCancelled { try? await Task.sleep(for: .milliseconds(200)) }
        }
      }
    }

    public func stop() {
      relay?.cancel()
      relay = nil
    }
  }
#endif
