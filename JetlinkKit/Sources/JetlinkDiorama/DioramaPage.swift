import Foundation

/// The Diorama page's files: the page, its engine and renderer, and the
/// models it draws. A resource bundle beside the executable or the app, looked
/// up by path rather than through `Bundle.module`, which ends the process
/// when the bundle is missing.
public enum DioramaPage {
  public static let files: [String: String] = [
    "diorama.html": "text/html; charset=utf-8",
    "diorama-engine.js": "text/javascript; charset=utf-8",
    "diorama-render.js": "text/javascript; charset=utf-8",
    "diorama-assets.bin": "application/octet-stream",
  ]

  /// The folder holding the files, or nil when the bundle is not there.
  public static var folder: URL? {
    for directory in bundleDirectories {
      if let bundle = Bundle(url: directory.appendingPathComponent(bundleName)),
        let url = bundle.url(forResource: "diorama", withExtension: "html", subdirectory: "Resources")
      {
        return url.deletingLastPathComponent()
      }
    }
    return nil
  }

  /// One of `files`, with its content type.
  public static func file(_ name: String) -> (data: Data, type: String)? {
    guard let type = files[name], let folder, let data = try? Data(contentsOf: folder.appending(path: name)) else { return nil }
    return (data, type)
  }

  #if canImport(Darwin)
    static let bundleName = "JetlinkKit_JetlinkDiorama.bundle"
  #else
    static let bundleName = "JetlinkKit_JetlinkDiorama.resources"
  #endif

  /// Beside the executable, inside an app's resources, and beside the test bundle.
  static var bundleDirectories: [URL] {
    let main = Bundle.main.bundleURL
    let code = Bundle(for: Marker.self).bundleURL
    var dirs = [main]
    if let resources = Bundle.main.resourceURL, resources != main { dirs.append(resources) }
    if code != main { dirs.append(code.deletingLastPathComponent()) }
    return dirs
  }

  private final class Marker {}
}
