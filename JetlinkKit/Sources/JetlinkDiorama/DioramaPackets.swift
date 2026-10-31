import Foundation

/// Diorama's messages (docs/diorama.md), byte for byte what `diorama/packets.py`
/// writes into a recorded drive and diorama-engine.js reads: one byte of
/// kind, then the payload, little-endian.
public enum DioramaPackets {
  /// The driving model's heads the page draws from, in the order a drive message carries them.
  public static let driveFields = ["plan", "lane_lines", "lane_lines_prob", "road_edges", "lead", "lead_prob", "pose"]
  /// DD3D rows with a lower 3D score are not sent.
  public static let boxMinScore: Float = 0.1
  public static let boxColumns = 16

  public static func meta(_ info: [String: Any]) -> Data {
    var data = Data("M".utf8)
    data.append((try? JSONSerialization.data(withJSONObject: info, options: [.sortedKeys])) ?? Data("{}".utf8))
    return data
  }

  /// u32 frame, f32 seconds, f32 model ms, then the floats.
  public static func drive(frame: UInt32, seconds: Float, modelMs: Float, values: [Float]) -> Data {
    var data = Data(capacity: 13 + 4 * values.count)
    data.append(UInt8(ascii: "D"))
    data.appendLE(frame)
    data.appendLE(seconds.bitPattern)
    data.appendLE(modelMs.bitPattern)
    for v in values { data.appendLE(v.bitPattern) }
    return data
  }

  /// The seg graph's outputs: the query map (one float per pixel, 255 for
  /// none) as run-length bytes, and each query's class (255 when dropped) and score.
  public static func seg(
    frame: UInt32, ms: Float, qmap: UnsafeBufferPointer<Float>, width: Int, height: Int, label: UnsafeBufferPointer<Float>, score: UnsafeBufferPointer<Float>
  ) -> Data {
    var data = Data(capacity: 15 + 2 * label.count + 4096)
    data.append(UInt8(ascii: "S"))
    data.appendLE(frame)
    data.appendLE(ms.bitPattern)
    data.appendLE(UInt16(width))
    data.appendLE(UInt16(height))
    data.appendLE(UInt16(label.count))
    for i in 0..<label.count {
      data.append(score[i] > 0 ? UInt8(clamping: Int(label[i])) : 255)
    }
    for i in 0..<score.count {
      data.append(UInt8(clamping: Int((min(max(score[i], 0), 1) * 255).rounded(.toNearestOrEven))))
    }
    // runs of equal bytes, row-major, each at most 256 long
    var runs = Data(capacity: 4096)
    var count: UInt32 = 0
    var i = 0
    let n = width * height
    while i < n {
      let v = UInt8(clamping: Int(qmap[i]))
      var j = i + 1
      while j < n && j - i < 256 && UInt8(clamping: Int(qmap[j])) == v { j += 1 }
      runs.append(v)
      runs.append(UInt8(j - i - 1))
      count += 1
      i = j
    }
    data.appendLE(count)
    data.append(runs)
    return data
  }

  /// DD3D's rows with a 3D score worth sending.
  public static func boxes(frame: UInt32, ms: Float, rows: UnsafeBufferPointer<Float>) -> Data {
    let n = rows.count / boxColumns
    let keep = (0..<n).filter { rows[$0 * boxColumns + 6] >= boxMinScore }
    var data = Data(capacity: 11 + keep.count * boxColumns * 4)
    data.append(UInt8(ascii: "B"))
    data.appendLE(frame)
    data.appendLE(ms.bitPattern)
    data.appendLE(UInt16(keep.count))
    for r in keep {
      for c in 0..<boxColumns { data.appendLE(rows[r * boxColumns + c].bitPattern) }
    }
    return data
  }
}

extension Data {
  mutating func appendLE<T: FixedWidthInteger>(_ value: T) {
    Swift.withUnsafeBytes(of: value.littleEndian) { append(contentsOf: $0) }
  }
}
