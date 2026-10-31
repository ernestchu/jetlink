import Foundation

#if canImport(FoundationNetworking)
  import FoundationNetworking
#endif

/// Diorama's models, as `python -m diorama export` writes them, from the
/// Hugging Face repo they are published in: diorama.json and the models it
/// names, each an ONNX graph or, for a Mac, a CoreML package (a folder). The
/// repo's tree gives each large file's sha256 and size, and each is fetched
/// as a driving model is, to a .part that is hashed as it arrives; a small
/// file is checked by its size. diorama.json goes in last, so a folder is
/// never a set of models a server would load until every file is there and
/// whole.
public enum DioramaModels {
  public static let repo = "ernestchu/jetlink-diorama"
  public static let manifest = "diorama.json"
  static let manifestMax = 1 << 20
  /// A file the repo keeps outside LFS: a package's manifest and program.
  static let smallMax: Int64 = 64 << 20
  static let timeout: TimeInterval = 15

  public static let defaultHost = "https://huggingface.co"

  public static func treeURL(host: String = defaultHost) -> String { "\(host)/api/models/\(repo)/tree/main?recursive=true" }

  public static func fileURL(_ name: String, host: String = defaultHost) -> String {
    "\(host)/\(repo)/resolve/main/\(LFS.pythonQuote(name))"
  }

  /// One file to fetch: its path in the repo and the folder, and its LFS
  /// pointer when the repo keeps it in LFS.
  public struct File: Sendable {
    public let path: String
    public let size: Int64
    public let pointer: Pointer?
  }

  /// What a fetch brings down: diorama.json and every file of the models it names.
  public struct Plan: Sendable {
    public let host: String
    public let manifest: Data
    public let files: [File]
    /// The models' bytes, for the prompt before the download.
    public var bytes: Int64 { files.reduce(0) { $0 + $1.size } }
  }

  /// The repo's diorama.json and its models' files, without downloading
  /// them. `coreml`: each model's CoreML package where diorama.json names one,
  /// its ONNX graph otherwise. `host` is Hugging Face, or a mirror with its layout.
  public static func plan(host: String = defaultHost, coreml: Bool = false, session: URLSession = .shared) async throws(RegistryError)
    -> Plan
  {
    let http = HTTP(session: session)
    let listing = try await http.getJSON(treeURL(host: host), timeout: timeout, shape: .array)
    let entries = (listing.array ?? []).compactMap(\.object).filter { $0["type"]?.string != "directory" }
    let manifest = try await http.get(fileURL(manifest, host: host), timeout: timeout, limit: manifestMax)
    guard let meta = (try? JSON.parse(manifest))?.object else {
      throw .registry("\(repo)'s \(self.manifest) is not a JSON object")
    }
    guard meta["seg"]?.object?["file"]?.string != nil else {
      throw .registry("\(repo)'s \(self.manifest) names no segmentation graph")
    }
    var files: [File] = []
    for model in ["seg", "dd3d"] {
      guard let entry = meta[model]?.object else { continue }
      guard let name = (coreml ? entry["coreml"]?.string : nil) ?? entry["file"]?.string else { continue }
      // a name in the folder: diorama.json says what to fetch, never where else to write
      guard !name.isEmpty, !name.hasPrefix("/"), !name.split(separator: "/").contains(where: { $0 == ".." || $0 == "." }) else {
        throw .registry("\(repo)'s \(self.manifest) names \(name), which is not a name in the folder")
      }
      // the file itself, or every file of the package it is
      let found = entries.filter { guard let path = $0["path"]?.string else { return false }; return path == name || path.hasPrefix(name + "/") }
      guard !found.isEmpty else { throw .registry("\(repo) has no \(name)") }
      for item in found {
        let path = item["path"]?.string ?? ""
        if let lfs = item["lfs"]?.object {
          guard let oid = lfs["oid"]?.string, CacheLayout.isSHA256(oid), let size = lfs["size"]?.pythonIntValue, size > 0 else {
            throw .registry("\(repo)'s \(path) is not an lfs object")
          }
          files.append(File(path: path, size: size, pointer: Pointer(oid: oid, size: size)))
        } else {
          guard let size = item["size"]?.pythonIntValue, size >= 0, size <= smallMax else {
            throw .registry("\(repo)'s \(path) has no size")
          }
          files.append(File(path: path, size: size, pointer: nil))
        }
      }
    }
    return Plan(host: host, manifest: manifest, files: files)
  }

  /// Brings `plan` down into `folder`. A file already there at its size is
  /// kept. `progress` runs from 0 to 1 over every file's bytes; `shouldStop`
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
    let http = HTTP(session: session)
    let total = Double(max(plan.bytes, 1))
    var done: Int64 = 0
    for file in plan.files {
      let dest = folder.appending(path: file.path)
      if let status = Files.status(dest), status.isFile, status.size == file.size {
        done += file.size
        progress(Double(done) / total)
        continue
      }
      try? Files.makeDirectory(dest.deletingLastPathComponent())
      if let pointer = file.pointer {
        let before = done
        _ = try await LFS.download(
          href: fileURL(file.path, host: plan.host), pointer: pointer, dest: dest, session: session,
          progress: { progress((Double(before) + $0 * Double(pointer.size)) / total) }, shouldStop: shouldStop)
      } else {
        if shouldStop() { throw .cancelled("stopped") }
        let body = try await http.get(fileURL(file.path, host: plan.host), timeout: timeout, limit: Int(smallMax))
        guard body.count == file.size else { throw .verify("\(file.path) is \(body.count) bytes, expected \(file.size)") }
        try write(body, to: dest)
      }
      done += file.size
      progress(Double(done) / total)
    }
    try write(plan.manifest, to: folder.appending(path: manifest))
    progress(1)
  }

  /// To a .part beside it, then renamed into place.
  static func write(_ data: Data, to dest: URL) throws(RegistryError) {
    let part = dest.deletingLastPathComponent().appending(path: dest.lastPathComponent + ".part")
    do {
      try data.write(to: part)
      try Files.replace(part, with: dest)
    } catch {
      try? Files.removeFile(part)
      throw .registry("cannot write \(dest.path(percentEncoded: false)): \(error)")
    }
  }
}
