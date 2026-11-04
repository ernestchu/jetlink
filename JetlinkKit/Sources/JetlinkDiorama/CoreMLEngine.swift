#if canImport(CoreML)
  import CoreML
  import Foundation
  import JetlinkServer

  /// A Diorama model as a native CoreML program, as `python -m diorama export
  /// --coreml` writes it: the whole graph one program, in fp16, where
  /// onnxruntime's CoreML provider ran Mask2Former in 31 pieces with a CPU hop
  /// between each (53 ms a frame on an M5 Pro's GPU against 185 ms).
  ///
  /// The package is compiled once into `cache` and compiled again only when
  /// it changes. The input is the host buffer itself; the outputs, which the
  /// export asks for as float32, are copied out after each run.
  public final class CoreMLEngine: EngineCore, @unchecked Sendable {
    private let model: MLModel
    /// The host input buffer as CoreML reads it: made once the buffer exists.
    private var input: MLMultiArray!
    private let inputName: String

    /// `computeUnits` defaults to the GPU and the CPU: the Neural Engine is the driving model's.
    public init(package: URL, cache: URL, computeUnits: MLComputeUnits = .cpuAndGPU) throws {
      let configuration = MLModelConfiguration()
      configuration.computeUnits = computeUnits
      model = try MLModel(contentsOf: try CoreMLEngine.compiled(package, cache: cache), configuration: configuration)
      let description = model.modelDescription
      guard let (name, feature) = description.inputDescriptionsByName.first, description.inputDescriptionsByName.count == 1,
        let constraint = feature.multiArrayConstraint, constraint.dataType == .float16
      else {
        throw DioramaError("\(package.lastPathComponent) does not take one float16 frame")
      }
      let shape = constraint.shape.map(\.intValue)
      var outputs: [String: TensorSpec] = [:]
      for (name, feature) in description.outputDescriptionsByName {
        guard let shape = feature.multiArrayConstraint?.shape.map(\.intValue) else {
          throw DioramaError("\(package.lastPathComponent)'s \(name) is not an array")
        }
        outputs[name] = TensorSpec(name: name, type: .float, shape: shape)
      }
      inputName = name
      try super.init(inputs: [name: TensorSpec(name: name, type: .float16, shape: shape)], outputs: outputs)
      guard let host = hostInput(name) else { throw DioramaError("no host buffer for \(name)") }
      // row-major strides over the host buffer, which the stage fills before each run
      let strides = shape.indices.map { i in shape[(i + 1)...].reduce(1, *) }
      input = try MLMultiArray(
        dataPointer: host, shape: shape.map(NSNumber.init), dataType: .float16, strides: strides.map(NSNumber.init), deallocator: nil)
    }

    public override func execute() throws {
      let result = try model.prediction(from: try MLDictionaryFeatureProvider(dictionary: [inputName: input]))
      for (name, spec) in outputs {
        guard let array = result.featureValue(for: name)?.multiArrayValue, let dst = output(name) else { continue }
        // MLShapedArray reads any strides and either precision
        let floats = MLShapedArray<Float>(converting: array)
        floats.withUnsafeShapedBufferPointer { src, _, _ in
          UnsafeMutableRawPointer(mutating: dst).copyMemory(from: src.baseAddress!, byteCount: min(src.count, spec.count) * 4)
        }
      }
    }

    /// The package compiled into `cache`, compiled again when the package's
    /// files change: a first compile of Mask2Former takes several seconds.
    static func compiled(_ package: URL, cache: URL) throws -> URL {
      let fm = FileManager.default
      let compiled = cache.appending(path: package.deletingPathExtension().lastPathComponent + ".mlmodelc")
      let stampURL = cache.appending(path: package.deletingPathExtension().lastPathComponent + ".stamp")
      let stamp = try packageStamp(package)
      if fm.fileExists(atPath: compiled.path), (try? String(contentsOf: stampURL, encoding: .utf8)) == stamp {
        return compiled
      }
      try fm.createDirectory(at: cache, withIntermediateDirectories: true)
      let fresh = try MLModel.compileModel(at: package)
      try? fm.removeItem(at: compiled)
      try fm.moveItem(at: fresh, to: compiled)
      try stamp.write(to: stampURL, atomically: true, encoding: .utf8)
      return compiled
    }

    /// Every file in the package with its size and modification time.
    static func packageStamp(_ package: URL) throws -> String {
      let keys: [URLResourceKey] = [.fileSizeKey, .contentModificationDateKey, .isRegularFileKey]
      guard let walk = FileManager.default.enumerator(at: package, includingPropertiesForKeys: keys) else {
        throw DioramaError("cannot read \(package.path)")
      }
      var lines: [String] = []
      for case let url as URL in walk {
        let values = try url.resourceValues(forKeys: Set(keys))
        guard values.isRegularFile == true else { continue }
        lines.append("\(url.path.dropFirst(package.path.count)) \(values.fileSize ?? 0) \(values.contentModificationDate?.timeIntervalSince1970 ?? 0)")
      }
      return lines.sorted().joined(separator: "\n")
    }
  }

  extension DioramaStage {
    /// What an Apple host hands the stage: a CoreML package (`Models(coreml:
    /// true)`) runs through `CoreMLEngine` on the GPU, its compile kept in
    /// `cache`; an ONNX graph goes to `other`.
    public static func coreMLFactory(cache: URL, other: @escaping EngineFactory) -> EngineFactory {
      { graph, name, report in
        if graph.pathExtension == "mlpackage" {
          return try CoreMLEngine(package: graph, cache: cache.appending(path: "coreml-native", directoryHint: .isDirectory))
        }
        return try other(graph, name, report)
      }
    }
  }
#endif
