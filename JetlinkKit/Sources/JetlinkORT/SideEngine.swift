import Foundation
import JetlinkServer

extension OrtBackend {
  /// A graph that runs beside the driving model, as Diorama's networks do:
  /// one session, on the GPU through CoreML where there is one (the Neural
  /// Engine stays the driving model's), else onnxruntime's CPU provider.
  /// CoreML keeps its compile in `cache`, under the graph's COREML_CACHE_KEY,
  /// so only the first load compiles.
  public static func sideEngine(model: URL, cache: URL) throws -> OrtEngine {
    #if canImport(Metal)
      try FileManager.default.createDirectory(at: cache, withIntermediateDirectories: true)
      let plan = SessionPlan(model: model, computeUnits: "CPUAndGPU", cacheDirectory: cache)
    #else
      try OrtRuntime.load()
      let plan = SessionPlan(model: model, provider: nil, threads: HostChip.cpuThreads, label: "CPU")
    #endif
    return try OrtEngine(plans: [plan], device: "diorama", keepAlive: false, keepCPUWarm: false)
  }
}
