import Foundation
import JetlinkTestSupport
import Testing

@testable import JetlinkRegistry

/// Diorama's models from their Hugging Face repo: the tree's pointers, the
/// graphs hashed as they arrive, and diorama.json only once they are all there.
struct DioramaModelsTests {
  let seg = Data((0..<5000).map { UInt8(truncatingIfNeeded: $0) })
  let dd3d = Data((0..<3000).map { UInt8(truncatingIfNeeded: $0 &* 7) })

  func manifest(dd3d: Bool = true, seg name: String = "diorama_seg.onnx") -> Data {
    var meta: [String: Any] = ["version": 1, "seg": ["file": name]]
    if dd3d { meta["dd3d"] = ["file": "diorama_dd3d.onnx"] }
    return try! JSONSerialization.data(withJSONObject: meta, options: [.sortedKeys])
  }

  func tree(segOid: String? = nil) -> Data {
    let entry = { (path: String, data: Data, oid: String?) -> [String: Any] in
      ["type": "file", "path": path, "size": data.count, "lfs": ["oid": oid ?? sha256Hex(data), "size": data.count, "pointerSize": 134]]
    }
    let listing: [[String: Any]] = [
      ["type": "file", "path": "diorama.json", "size": 200],
      entry("diorama_seg.onnx", seg, segOid),
      entry("diorama_dd3d.onnx", dd3d, nil),
    ]
    return try! JSONSerialization.data(withJSONObject: listing)
  }

  func routes(segOid: String? = nil, manifest: Data? = nil) -> [String: MockNet.Reply] {
    [
      DioramaModels.treeURL(): .body(tree(segOid: segOid)),
      DioramaModels.fileURL("diorama.json"): .body(manifest ?? self.manifest()),
      DioramaModels.fileURL("diorama_seg.onnx"): .chunks([seg.prefix(2500), seg.suffix(2500)]),
      DioramaModels.fileURL("diorama_dd3d.onnx"): .body(dd3d),
    ]
  }

  @Test func fetchesEveryGraphThenTheManifest() async throws {
    let tmp = try TemporaryDirectory()
    let net = MockNet(routes())
    let folder = tmp.url.appending(path: "diorama")
    let plan = try await DioramaModels.plan(session: net.session)
    #expect(plan.graphs.map(\.name) == ["diorama_seg.onnx", "diorama_dd3d.onnx"])
    #expect(plan.bytes == 8000)

    let seen = ProgressLog()
    try await DioramaModels.fetch(plan, into: folder, session: net.session, progress: seen.callback)
    #expect(try Data(contentsOf: folder.appending(path: "diorama_seg.onnx")) == seg)
    #expect(try Data(contentsOf: folder.appending(path: "diorama_dd3d.onnx")) == dd3d)
    #expect(try Data(contentsOf: folder.appending(path: "diorama.json")) == manifest())
    #expect(seen.all.last == 1)
    #expect(seen.all == seen.all.sorted(), "progress over both graphs only grows")
    #expect(try FileManager.default.contentsOfDirectory(atPath: folder.path).sorted() == ["diorama.json", "diorama_dd3d.onnx", "diorama_seg.onnx"])
  }

  @Test func aGraphAlreadyThereIsKept() async throws {
    let tmp = try TemporaryDirectory()
    let folder = tmp.url.appending(path: "diorama")
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    try seg.write(to: folder.appending(path: "diorama_seg.onnx"))
    let net = MockNet(routes())
    try await DioramaModels.fetch(try await DioramaModels.plan(session: net.session), into: folder, session: net.session)
    #expect(net.count(DioramaModels.fileURL("diorama_seg.onnx")) == 0)
    #expect(net.count(DioramaModels.fileURL("diorama_dd3d.onnx")) == 1)
  }

  @Test func aBadGraphLeavesNoManifest() async throws {
    let tmp = try TemporaryDirectory()
    let folder = tmp.url.appending(path: "diorama")
    let net = MockNet(routes(segOid: String(repeating: "b", count: 64)))
    let error = await #expect(throws: RegistryError.self) {
      try await DioramaModels.fetch(try await DioramaModels.plan(session: net.session), into: folder, session: net.session)
    }
    #expect(error?.kind == .verify)
    // nothing a server would load: no diorama.json, and no half-written graph
    let names = (try? FileManager.default.contentsOfDirectory(atPath: folder.path)) ?? []
    #expect(!names.contains("diorama.json"))
    #expect(!names.contains { $0.hasSuffix(".part") })
  }

  @Test func aManifestNamesFilesNotPaths() async throws {
    let net = MockNet(routes(manifest: manifest(seg: "../evil.onnx")))
    let error = await #expect(throws: RegistryError.self) { try await DioramaModels.plan(session: net.session) }
    #expect(error?.message.contains("not a file name") == true)
  }

  @Test func aGraphTheRepoLacksIsNamed() async throws {
    var routes = routes()
    routes[DioramaModels.treeURL()] = .body(Data("[]".utf8))
    let error = await #expect(throws: RegistryError.self) { try await DioramaModels.plan(session: MockNet(routes).session) }
    #expect(error?.message.contains("has no diorama_seg.onnx") == true)
  }
}
