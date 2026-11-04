import Foundation
import JetlinkTestSupport
import Testing

@testable import JetlinkRegistry

/// Diorama's models from their Hugging Face repo: the tree's pointers, the
/// graphs hashed as they arrive, and diorama.json only once they are all there.
struct DioramaModelsTests {
  let seg = Data((0..<5000).map { UInt8(truncatingIfNeeded: $0) })
  let dd3d = Data((0..<3000).map { UInt8(truncatingIfNeeded: $0 &* 7) })

  func manifest(dd3d: Bool = true, seg name: String = "diorama_seg.onnx", segCoreML: String? = nil) -> Data {
    var seg: [String: Any] = ["file": name]
    if let segCoreML { seg["coreml"] = segCoreML }
    var meta: [String: Any] = ["version": 1, "seg": seg]
    if dd3d { meta["dd3d"] = ["file": "diorama_dd3d.onnx"] }
    return try! JSONSerialization.data(withJSONObject: meta, options: [.sortedKeys])
  }

  /// A CoreML package's files: its program outside LFS, its weights in it.
  let program = Data("program".utf8)
  let weights = Data((0..<4000).map { UInt8(truncatingIfNeeded: $0 &* 3) })

  func tree(segOid: String? = nil) -> Data {
    let entry = { (path: String, data: Data, oid: String?) -> [String: Any] in
      ["type": "file", "path": path, "size": data.count, "lfs": ["oid": oid ?? sha256Hex(data), "size": data.count, "pointerSize": 134]]
    }
    let listing: [[String: Any]] = [
      ["type": "file", "path": "diorama.json", "size": 200],
      entry("diorama_seg.onnx", seg, segOid),
      entry("diorama_dd3d.onnx", dd3d, nil),
      ["type": "directory", "path": "diorama_seg.mlpackage"],
      ["type": "file", "path": "diorama_seg.mlpackage/Data/model.mlmodel", "size": program.count],
      entry("diorama_seg.mlpackage/Data/weights/weight.bin", weights, nil),
    ]
    return try! JSONSerialization.data(withJSONObject: listing)
  }

  func routes(segOid: String? = nil, manifest: Data? = nil) -> [String: MockNet.Reply] {
    [
      DioramaModels.treeURL(): .body(tree(segOid: segOid)),
      DioramaModels.fileURL("diorama.json"): .body(manifest ?? self.manifest()),
      DioramaModels.fileURL("diorama_seg.onnx"): .chunks([seg.prefix(2500), seg.suffix(2500)]),
      DioramaModels.fileURL("diorama_dd3d.onnx"): .body(dd3d),
      DioramaModels.fileURL("diorama_seg.mlpackage/Data/model.mlmodel"): .body(program),
      DioramaModels.fileURL("diorama_seg.mlpackage/Data/weights/weight.bin"): .body(weights),
    ]
  }

  @Test func fetchesTheSegmentationThenTheManifest() async throws {
    let tmp = try TemporaryDirectory()
    let net = MockNet(routes())
    let folder = tmp.url.appending(path: "diorama")
    let plan = try await DioramaModels.plan(session: net.session)
    // the manifest still names DD3D, as an older export's does: it is not fetched
    #expect(plan.files.map(\.path) == ["diorama_seg.onnx"])
    #expect(plan.bytes == 5000)

    let seen = ProgressLog()
    try await DioramaModels.fetch(plan, into: folder, session: net.session, progress: seen.callback)
    #expect(try Data(contentsOf: folder.appending(path: "diorama_seg.onnx")) == seg)
    #expect(try Data(contentsOf: folder.appending(path: "diorama.json")) == manifest())
    #expect(seen.all.last == 1)
    #expect(seen.all == seen.all.sorted(), "progress only grows")
    #expect(net.count(DioramaModels.fileURL("diorama_dd3d.onnx")) == 0)
    #expect(try FileManager.default.contentsOfDirectory(atPath: folder.path).sorted() == ["diorama.json", "diorama_seg.onnx"])
  }

  @Test func aMacFetchesTheCoreMLPackageInsteadOfTheGraph() async throws {
    let tmp = try TemporaryDirectory()
    let folder = tmp.url.appending(path: "diorama")
    let net = MockNet(routes(manifest: manifest(segCoreML: "diorama_seg.mlpackage")))
    let plan = try await DioramaModels.plan(coreml: true, session: net.session)
    #expect(plan.files.map(\.path) == ["diorama_seg.mlpackage/Data/model.mlmodel", "diorama_seg.mlpackage/Data/weights/weight.bin"])
    try await DioramaModels.fetch(plan, into: folder, session: net.session)
    #expect(try Data(contentsOf: folder.appending(path: "diorama_seg.mlpackage/Data/model.mlmodel")) == program)
    #expect(try Data(contentsOf: folder.appending(path: "diorama_seg.mlpackage/Data/weights/weight.bin")) == weights)
    #expect(net.count(DioramaModels.fileURL("diorama_seg.onnx")) == 0, "the ONNX graph a Mac does not run")
  }

  @Test func aGraphAlreadyThereIsKept() async throws {
    let tmp = try TemporaryDirectory()
    let folder = tmp.url.appending(path: "diorama")
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    try seg.write(to: folder.appending(path: "diorama_seg.onnx"))
    let net = MockNet(routes())
    try await DioramaModels.fetch(try await DioramaModels.plan(session: net.session), into: folder, session: net.session)
    #expect(net.count(DioramaModels.fileURL("diorama_seg.onnx")) == 0)
    #expect(try FileManager.default.contentsOfDirectory(atPath: folder.path).sorted() == ["diorama.json", "diorama_seg.onnx"])
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
    let net = MockNet(routes(manifest: manifest(seg: "diorama_seg.mlpackage/../../evil.onnx")))
    let error = await #expect(throws: RegistryError.self) { try await DioramaModels.plan(session: net.session) }
    #expect(error?.message.contains("not a name in the folder") == true)
  }

  @Test func aGraphTheRepoLacksIsNamed() async throws {
    var routes = routes()
    routes[DioramaModels.treeURL()] = .body(Data("[]".utf8))
    let error = await #expect(throws: RegistryError.self) { try await DioramaModels.plan(session: MockNet(routes).session) }
    #expect(error?.message.contains("has no diorama_seg.onnx") == true)
  }
}
