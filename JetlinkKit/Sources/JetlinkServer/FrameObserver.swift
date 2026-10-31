import Foundation

/// Hears every frame the server answered, after the reply has gone to the
/// comma: JetlinkDiorama's stage, which shows the drive. It runs on the
/// session's thread with the next frame due in under 50 ms, so it copies what
/// it wants and returns; it must never wait on anything a frame needs.
public protocol FrameObserver: AnyObject, Sendable {
  /// `narrow` is the frame's warped narrow camera, the first of the two
  /// (6, H, W) planes the comma sent; `outputs` the model's whole output
  /// vector, hidden state included, laid out as `spec.outputSlices` say.
  /// Both are valid only for the call.
  func frameServed(
    frameID: UInt32, narrow: UnsafeRawBufferPointer, outputs: UnsafeBufferPointer<Float>, spec: ModelSpec, totalUs: UInt32)
}

extension EngineHost {
  /// Who hears the frames, if anyone. Set and cleared at any time: the
  /// session reads it once a frame.
  public var frameObserver: (any FrameObserver)? {
    get { observerBox.get() }
    set { observerBox.set(newValue) }
  }
}

/// The observer behind a lock no frame waits on for long: a read is a pointer copy.
final class ObserverBox: @unchecked Sendable {
  private let lock = NSLock()
  private var observer: (any FrameObserver)?

  func get() -> (any FrameObserver)? {
    lock.lock()
    defer { lock.unlock() }
    return observer
  }

  func set(_ observer: (any FrameObserver)?) {
    lock.lock()
    self.observer = observer
    lock.unlock()
  }
}
