import Foundation

#if canImport(FoundationNetworking)
  import FoundationNetworking
#endif

/// Diorama's networks, as `python -m diorama export` writes them, from the
/// Hugging Face repo they are published in: diorama.json and the graphs it
/// names. The repo's tree gives each graph's sha256 and size, and each is
/// fetched as a model is, to a .part that is hashed as it arrives.
/// diorama.json goes in last, so a folder is never a set of networks a
/// server would load until every graph it names is there and whole.
public enum DioramaNetworks {
  public static let repo = "ernestchu/jetlink-diorama"
  public static let manifest = "diorama.json"
  static let manifestMax = 1 << 20
  static let timeout: TimeInterval = 15

  public static let defaultHost = "https://huggingface.co"

  public static func treeURL(host: String = defaultHost) -> String { "\(host)/api/models/\(repo)/tree/main" }

  public static func fileURL(_ name: String, host: String = defaultHost) -> String {
    "\(host)/\(repo)/resolve/main/\(LFS.pythonQuote(name))"
  }

  /// What a fetch brings down: diorama.json and each graph it names.
  public struct Plan: Sendable {
    public let host: String
    public let manifest: Data
    public let graphs: [(name: String, pointer: Pointer)]
    /// The graphs' bytes, for the prompt before the download.
    public var bytes: Int64 { graphs.reduce(0) { $0 + $1.pointer.size } }
  }

  /// The repo's diorama.json and its graphs' pointers, without downloading
  /// them. `host` is Hugging Face, or a mirror with its layout.
  public static func plan(host: String = defaultHost, session: URLSession = .shared) async throws(RegistryError) -> Plan {
    let http = HTTP(session: session)
    let listing = try await http.getJSON(treeURL(host: host), timeout: timeout, shape: .array)
    let entries = (listing.array ?? []).compactMap(\.object)
    let manifest = try await http.get(fileURL(manifest, host: host), timeout: timeout, limit: manifestMax)
    guard let meta = (try? JSON.parse(manifest))?.object else {
      throw .registry("\(repo)'s \(self.manifest) is not a JSON object")
    }
    let names = ["seg", "dd3d"].compactMap { meta[$0]?.object?["file"]?.string }
    guard meta["seg"]?.object?["file"]?.string != nil else {
      throw .registry("\(repo)'s \(self.manifest) names no segmentation graph")
    }
    var graphs: [(name: String, pointer: Pointer)] = []
    for name in names {
      // a bare file name: diorama.json says what to fetch, never where to write
      guard !name.isEmpty, !name.contains("/"), name != ".", name != ".." else {
        throw .registry("\(repo)'s \(self.manifest) names \(name), which is not a file name")
      }
      let lfs = entries.first { $0["path"]?.string == name }?["lfs"]?.object
      guard let oid = lfs?["oid"]?.string, CacheLayout.isSHA256(oid), let size = lfs?["size"]?.pythonIntValue, size > 0 else {
        throw .registry("\(repo) has no \(name)")
      }
      graphs.append((name, Pointer(oid: oid, size: size)))
    }
    return Plan(host: host, manifest: manifest, graphs: graphs)
  }

  /// Brings `plan` down into `folder`. A graph already there at its size is
  /// kept. `progress` runs from 0 to 1 over every graph's bytes; `shouldStop`
  /// is asked between chunks.
  public static func fetch(
    _ plan: Plan, into folder: URL, session: URLSession = .shared,
    progress: @escaping @Sendable (Double) -> Void = { _ in },
    shouldStop: @escaping @Sendable () -> Bool = { false }
  ) async throws(RegistryError) {
    do {
      try Files.makeDirectory(folder)
    } catch {
      throw .registry("cannot make \(folder.path(percentEncoded: false)): \(error)")
    }
    let total = Double(max(plan.bytes, 1))
    var done: Int64 = 0
    for (name, pointer) in plan.graphs {
      let dest = folder.appending(path: name)
      if let status = Files.status(dest), status.isFile, status.size == pointer.size {
        done += pointer.size
        progress(Double(done) / total)
        continue
      }
      let before = done
      _ = try await LFS.download(
        href: fileURL(name, host: plan.host), pointer: pointer, dest: dest, session: session,
        progress: { progress((Double(before) + $0 * Double(pointer.size)) / total) }, shouldStop: shouldStop)
      done += pointer.size
    }
    let dest = folder.appending(path: manifest)
    let part = folder.appending(path: manifest + ".part")
    do {
      try plan.manifest.write(to: part)
      try Files.replace(part, with: dest)
    } catch {
      try? Files.removeFile(part)
      throw .registry("cannot write \(dest.path(percentEncoded: false)): \(error)")
    }
    progress(1)
  }
}
