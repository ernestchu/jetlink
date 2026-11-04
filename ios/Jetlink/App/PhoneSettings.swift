import Foundation
import JetlinkORT
import Observation

/// The few things worth changing on a phone, kept in UserDefaults.
@MainActor
@Observable
final class PhoneSettings {
  private let defaults: UserDefaults

  /// The TCP port the server listens on while `developer` is on, where bench
  /// tools such as `bench_link.py --host` reach the phone over Wi-Fi. Over
  /// the cable the phone dials the comma instead.
  var port: UInt16 {
    didSet { defaults.set(Int(port), forKey: Keys.port) }
  }

  /// Developer settings shown, and the server listening on `port`. Off, the
  /// phone only dials the comma, which is all driving needs. Seven taps on
  /// the version turn it on or off.
  var developer: Bool {
    didSet { defaults.set(developer, forKey: Keys.developer) }
  }

  /// The trunk on the Neural Engine and the rest on the GPU (14 ms a frame on an
  /// iPhone 18 Pro; the whole model on the Neural Engine took 100 ms or failed),
  /// or the whole model on the GPU for when something else holds the Neural Engine.
  var device: OrtProfile {
    didSet { defaults.set(device.rawValue, forKey: Keys.device) }
  }

  /// A small GPU job between frames so the GPU does not clock down in the gaps.
  var keepGPUAwake: Bool {
    didSet { defaults.set(keepGPUAwake, forKey: Keys.keepGPUAwake) }
  }

  /// A CPU core kept busy between frames while the Neural Engine runs the
  /// model, so the next frame is not waiting on a core that went to sleep.
  var keepCPUWarm: Bool {
    didSet { defaults.set(keepCPUWarm, forKey: Keys.keepCPUWarm) }
  }

  /// The screen stays on while Jetlink is open: iOS suspends an app whose
  /// phone locks, and a suspended app serves nothing.
  var keepScreenOn: Bool {
    didSet { defaults.set(keepScreenOn, forKey: Keys.keepScreenOn) }
  }

  /// Diorama's model on this iPhone, beside the driving model, while the
  /// Diorama tab shows it. Off until the user turns it on: it costs heat,
  /// battery and GPU time the driving model shares.
  var dioramaOnPhone: Bool {
    didSet { defaults.set(dioramaOnPhone, forKey: Keys.dioramaOnPhone) }
  }

  /// Where the Diorama tab's drive comes from.
  var dioramaSource: DioramaSource {
    didSet { defaults.set(dioramaSource.rawValue, forKey: Keys.dioramaSource) }
  }

  /// The Mac the Diorama tab last watched, watched again when it is found.
  var dioramaMac: String? {
    didSet { defaults.set(dioramaMac, forKey: Keys.dioramaMac) }
  }

  init(defaults: UserDefaults = .standard) {
    self.defaults = defaults
    let port = defaults.integer(forKey: Keys.port)
    self.port = port > 0 && port < 65_536 ? UInt16(port) : 5599
    self.developer = defaults.bool(forKey: Keys.developer)
    self.device = defaults.string(forKey: Keys.device).flatMap(OrtProfile.init(rawValue:)) ?? .ane
    self.keepGPUAwake = defaults.object(forKey: Keys.keepGPUAwake) as? Bool ?? true
    self.keepCPUWarm = defaults.object(forKey: Keys.keepCPUWarm) as? Bool ?? true
    self.keepScreenOn = defaults.object(forKey: Keys.keepScreenOn) as? Bool ?? true
    self.dioramaOnPhone = defaults.bool(forKey: Keys.dioramaOnPhone)
    self.dioramaSource = defaults.string(forKey: Keys.dioramaSource).flatMap(DioramaSource.init(rawValue:)) ?? .mac
    self.dioramaMac = defaults.string(forKey: Keys.dioramaMac)
  }

  /// Diorama's models, out of iCloud backups: Application Support, as the cache is.
  static var dioramaFolder: URL {
    URL.applicationSupportDirectory.appending(path: "Jetlink/diorama", directoryHint: .isDirectory)
  }

  /// Where models and prepared engines live. Application Support, not Caches:
  /// iOS empties Caches when space runs low, and a missing engine costs the
  /// comma its big model on the next drive.
  static var cacheDirectory: URL {
    URL.applicationSupportDirectory.appending(path: "Jetlink/cache", directoryHint: .isDirectory)
  }

  private enum Keys {
    static let port = "port"
    static let developer = "developer"
    static let device = "device"
    static let keepGPUAwake = "keepGPUAwake"
    static let keepCPUWarm = "keepCPUWarm"
    static let keepScreenOn = "keepScreenOn"
    static let dioramaOnPhone = "dioramaOnPhone"
    static let dioramaSource = "dioramaSource"
    static let dioramaMac = "dioramaMac"
  }
}

/// The Diorama tab's two ways: watch a Mac that runs the comma and the
/// models, or run the models here on this iPhone's own server.
enum DioramaSource: String, CaseIterable {
  case mac, phone
}

extension OrtProfile {
  var title: String {
    switch self {
    case .ane: "Neural Engine + GPU"
    case .coreml: "GPU"
    case .cpu: "CPU"
    // the Mac's and Android's, never offered here
    case .aneWhole, .htp, .htpWhole, .gpu: rawValue
    }
  }
}
