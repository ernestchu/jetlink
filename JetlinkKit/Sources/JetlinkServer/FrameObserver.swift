import Foundation

/// Hears every frame the server answered, after the reply has gone to the
/// comma: JetlinkDiorama's stage, which shows the drive. It runs on the
/// session's thread with the next frame due in under 50 ms, so it copies what
/// it wants and returns; it must never wait on anything a frame needs.
public protocol FrameObserver: AnyObject, Sendable {
  /// `narrow` and `wide` are the frame's warped cameras, the two (6, H, W)
  /// planes the comma sent; `outputs` the model's whole output vector,
  /// hidden state included, laid out as `spec.outputSlices` say. All are
  /// valid only for the call.
  func frameServed(
    frameID: UInt32, narrow: UnsafeRawBufferPointer, wide: UnsafeRawBufferPointer, outputs: UnsafeBufferPointer<Float>, spec: ModelSpec,
    totalUs: UInt32)

  /// A new client's word on its cameras, from its hello, or nil when it
  /// says nothing (a comma not calibrated yet, or older jetlink).
  func cameraCalibrated(_ calibration: CameraCalibration?)
}

extension FrameObserver {
  public func cameraCalibrated(_ calibration: CameraCalibration?) {}
}

/// What the comma's calibrationd says of its cameras: the wide camera's roll,
/// pitch and yaw from the device (wideFromDeviceEuler, radians), and the
/// camera's height above the road (metres). The warp has already undone the
/// rest of the calibration.
public struct CameraCalibration: Sendable, Equatable {
  public let wideFromDevice: [Double]
  public let height: Double

  public init(wideFromDevice: [Double], height: Double) {
    self.wideFromDevice = wideFromDevice
    self.height = height
  }

  /// A hello's `camera`, {wide_from_device_euler: [r, p, y], height: h}, or nil unless it has both.
  public init?(hello camera: Any?) {
    guard let camera = camera as? [String: Any],
      let wide = (camera["wide_from_device_euler"] as? [Any])?.compactMap({ ($0 as? NSNumber)?.doubleValue }), wide.count == 3,
      let height = (camera["height"] as? NSNumber)?.doubleValue, height > 0
    else { return nil }
    self.init(wideFromDevice: wide, height: height)
  }
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
