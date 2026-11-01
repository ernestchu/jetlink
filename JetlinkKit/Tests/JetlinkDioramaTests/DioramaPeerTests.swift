#if canImport(Network)
  import Foundation
  import JetlinkServer
  import Synchronization
  import Testing

  @testable import JetlinkDiorama

  @Suite("Diorama between a Mac and an iPhone")
  struct DioramaPeerTests {
    @Test("Messages arrive whole however the stream is cut, as a drive.diorama frames them")
    func framing() throws {
      let messages = [Data("M{}".utf8), Data(), Data(repeating: 7, count: 300)]
      let bytes = DioramaPeer.frame(messages)
      #expect(bytes.count == 3 * 4 + 3 + 300)
      #expect(Array(bytes.prefix(4)) == [3, 0, 0, 0])
      for cut in [1, 3, 4, 5, 9, 100, bytes.count] {
        var reader = DioramaPeer.Reader(limit: 1000)
        var got: [Data] = []
        var at = 0
        while at < bytes.count {
          got += try reader.take(bytes.subdata(in: at..<min(at + cut, bytes.count)))
          at += cut
        }
        #expect(got == messages)
      }
    }

    @Test("A message over the limit ends the stream")
    func overLimit() {
      var reader = DioramaPeer.Reader(limit: 10)
      #expect(throws: DioramaError.self) { try reader.take(DioramaPeer.frame([Data(count: 11)])) }
    }

    /// A broadcaster on a stage over a fake segmentation, and a viewer dialing its port.
    final class Rig: @unchecked Sendable {
      let stage: DioramaStage
      let broadcaster = DioramaBroadcaster(advertise: false)
      let watch = DioramaWatch(name: "Test iPhone")
      let got = Mutex<[Data]>([])
      let viewers = Mutex<[DioramaBroadcaster.Viewer]>([])
      let states = Mutex<[DioramaWatch.State]>([])
      let helper = DioramaStageTests()

      init() throws {
        let seg = try FakeModel()
        stage = DioramaStage(models: try helper.models()) { _, _, _ in seg }
        stage.start()
        broadcaster.onChange = { [weak self] list in self?.viewers.withLock { $0 = list } }
        try broadcaster.start()
        broadcaster.setStage(stage)
        watch.onMessages = { [weak self] messages, done in
          self?.got.withLock { $0 += messages }
          done()
        }
        watch.onState = { [weak self] state, _ in self?.states.withLock { $0.append(state) } }
      }

      func connect() throws {
        watch.start(host: "127.0.0.1", port: try #require(broadcaster.port))
      }

      func serve(_ frames: Range<UInt32>) {
        for f in frames {
          helper.serve(stage, frame: f)
          Thread.sleep(forTimeInterval: 0.03)
        }
      }

      func kinds() -> [Character] { got.withLock { $0.map { Character(UnicodeScalar($0.first ?? 0)) } } }

      func wait(_ seconds: TimeInterval = 3, until: () -> Bool) -> Bool {
        let deadline = Date(timeIntervalSinceNow: seconds)
        while !until() && Date() < deadline { Thread.sleep(forTimeInterval: 0.02) }
        return until()
      }

      func stop() {
        watch.stop()
        broadcaster.stop()
        stage.stop()
      }
    }

    @Test("A viewer that is ready gets the meta, then the drive and the segmentation and nothing else, and is listed by name")
    func watching() throws {
      let rig = try Rig()
      defer { rig.stop() }
      try rig.connect()
      #expect(rig.wait { rig.viewers.withLock { $0.first?.name } == "Test iPhone" })
      #expect(rig.wait { rig.kinds().first == "M" })
      rig.serve(0..<6)
      #expect(rig.wait { rig.kinds().filter { $0 == "D" }.count == 6 && rig.kinds().contains("S") })
      // the models' status is not for drawing: it stays on the Mac
      #expect(Set(rig.kinds()) == ["M", "D", "S"])
      #expect(rig.states.withLock { $0 }.contains(.watching("test")))
    }

    @Test("A viewer that asks for the frames the models see is never sent them")
    func noInput() throws {
      let rig = try Rig()
      defer { rig.stop() }
      try rig.connect()
      #expect(rig.wait { rig.kinds().first == "M" })
      // what an older viewer's Model input toggle sends: a plain ready, from a fresh meta
      rig.got.withLock { $0.removeAll() }
      rig.watch.send("input:on")
      #expect(rig.wait { rig.kinds().first == "M" })
      rig.serve(0..<6)
      #expect(rig.wait { rig.kinds().filter { $0 == "S" }.count >= 2 })
      #expect(!rig.kinds().contains("I"))
      #expect(!rig.kinds().contains("T"))
    }

    @Test("A new stage: the viewer stays connected and starts over from its meta")
    func newStage() throws {
      let rig = try Rig()
      defer { rig.stop() }
      try rig.connect()
      #expect(rig.wait { rig.kinds().first == "M" })
      rig.broadcaster.setStage(nil)
      rig.got.withLock { $0.removeAll() }
      let seg = try FakeModel()
      let next = DioramaStage(models: try rig.helper.models()) { _, _, _ in seg }
      next.start()
      defer { next.stop() }
      rig.broadcaster.setStage(next)
      #expect(rig.wait { rig.kinds().first == "M" })
      #expect(rig.viewers.withLock { $0.count } == 1)
    }

    @Test("A viewer that goes away is let go, and the models stop with nobody watching")
    func leaving() throws {
      let rig = try Rig()
      defer { rig.stop() }
      try rig.connect()
      #expect(rig.wait { rig.viewers.withLock { $0.count } == 1 })
      rig.watch.stop()
      #expect(rig.wait { rig.viewers.withLock { $0.isEmpty } })
    }

    @Test("A viewer whose Mac restarts its listener connects again by itself")
    func reconnect() throws {
      let rig = try Rig()
      defer { rig.stop() }
      try rig.connect()
      #expect(rig.wait { rig.viewers.withLock { $0.count } == 1 })
      // the Mac drops every viewer: the viewer looks again, and the same port is still listening
      rig.broadcaster.dropViewers()
      #expect(rig.wait { rig.states.withLock { $0 }.contains(.searching) })
      rig.got.withLock { $0.removeAll() }
      #expect(rig.wait(5) { rig.kinds().first == "M" && rig.viewers.withLock { $0.count } == 1 })
    }
  }
#endif
