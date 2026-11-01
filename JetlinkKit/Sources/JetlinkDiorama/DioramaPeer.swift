#if canImport(Network)
  import Foundation
  import JetlinkServer
  import Network

  /// Diorama from one Apple device to another with no network in common: the
  /// Mac that runs the server and the models, and an iPhone in a car mount
  /// that only shows the drive, its Wi-Fi joined to the car for CarPlay.
  ///
  /// The Mac advertises `serviceType` over Bonjour, and peer to peer (AWDL)
  /// as well as on any network the two share; the iPhone browses for it. What
  /// crosses is only what the page draws with (`rendering`): the meta, the
  /// driving model's outputs and the segmentation, never the camera's frames
  /// nor the models' status. Each is after its u32 little-endian length,
  /// exactly as a drive.diorama holds them, about 180 KB a second. The
  /// viewer sends the same framing back, `hello:` and its name and then
  /// `ready`, and gets a watcher of its own from `ready` on.
  ///
  /// Neither end queues more than it can hand on: the viewer reads again
  /// only once its page has taken the last batch, and the Mac sends again
  /// only once the last send left. A slow viewer backs up into its watcher,
  /// which the stage drops past `Limits.behind`; it is watched again from a
  /// fresh meta, which the page starts over from.
  public enum DioramaPeer {
    public static let serviceType = "_jetlink-diorama._tcp"
    /// A viewer's line, and a message: anything longer ends the connection.
    static let maxControl = 4096
    static let maxMessage = 8 << 20

    /// The messages a viewer is sent: `M` meta, `D` drive and `S` seg, which
    /// the page draws the drive from. `I`, the frames the models see, and
    /// `T`, what they are doing, stay on the Mac.
    public static let rendering: Set<UInt8> = [UInt8(ascii: "M"), UInt8(ascii: "D"), UInt8(ascii: "S")]

    /// TCP over AWDL as well as infrastructure Wi-Fi, without Nagle, and with
    /// keepalives short enough that a phone that drove off is noticed in seconds.
    public static func parameters() -> NWParameters {
      let tcp = NWProtocolTCP.Options()
      tcp.noDelay = true
      tcp.enableKeepalive = true
      tcp.keepaliveIdle = 2
      tcp.keepaliveInterval = 1
      tcp.keepaliveCount = 3
      let parameters = NWParameters(tls: nil, tcp: tcp)
      parameters.includePeerToPeer = true
      parameters.serviceClass = .responsiveData
      return parameters
    }

    /// The messages, each after its u32 little-endian length.
    public static func frame(_ messages: [Data]) -> Data {
      var out = Data(capacity: messages.reduce(0) { $0 + 4 + $1.count })
      for m in messages {
        withUnsafeBytes(of: UInt32(m.count).littleEndian) { out.append(contentsOf: $0) }
        out.append(m)
      }
      return out
    }

    /// Whole messages split off a stream that arrives in pieces.
    public struct Reader: Sendable {
      private var buffer = Data()
      private let limit: Int

      public init(limit: Int) { self.limit = limit }

      /// The messages `bytes` completes; throws on one longer than the limit.
      public mutating func take(_ bytes: Data) throws -> [Data] {
        buffer.append(bytes)
        var out: [Data] = []
        var at = buffer.startIndex
        while buffer.endIndex - at >= 4 {
          let n = buffer[at..<at + 4].enumerated().reduce(0) { $0 | Int($1.element) << (8 * $1.offset) }
          guard n <= limit else { throw DioramaError("a \(n)-byte message, over the \(limit)-byte limit") }
          guard buffer.endIndex - at - 4 >= n else { break }
          out.append(Data(buffer[at + 4..<at + 4 + n]))
          at += 4 + n
        }
        buffer.removeSubrange(buffer.startIndex..<at)
        return out
      }
    }
  }

  // MARK: the Mac

  /// The Mac's end: a listener advertised over Bonjour, and a watcher of the
  /// stage for each viewer that asks. The stage may come and go with the
  /// server; viewers stay connected meanwhile and are watched again from the
  /// new stage's meta.
  public final class DioramaBroadcaster: @unchecked Sendable {
    /// A connected viewer, for a list in the app.
    public struct Viewer: Sendable, Hashable, Identifiable {
      public let id: Int
      /// What it said it is (`hello:`), or its address.
      public let name: String
      public let address: String
    }

    /// Called on the broadcaster's queue whenever a viewer comes, goes or changes.
    public var onChange: (@Sendable ([Viewer]) -> Void)?

    private let queue = DispatchQueue(label: "jetlink-diorama-peer")
    private let advertise: Bool
    private let log = ServerLog(category: "diorama")
    private var listener: NWListener?
    private var links: [Int: Link] = [:]
    private var nextID = 0
    private var stage: DioramaStage?

    /// `advertise` false listens without Bonjour, for tests that dial the port.
    public init(advertise: Bool = true) { self.advertise = advertise }

    /// The port it listens on once ready.
    public var port: UInt16? { queue.sync { listener?.port?.rawValue } }

    public func start() throws {
      let listener = try NWListener(using: DioramaPeer.parameters())
      // the Mac's own name, as the Finder's sidebar shows it
      if advertise { listener.service = NWListener.Service(type: DioramaPeer.serviceType) }
      listener.newConnectionHandler = { [weak self] connection in self?.accept(connection) }
      // start() waits for the first of these; later ones only signal a semaphore nobody waits on
      let ready = DispatchSemaphore(value: 0)
      listener.stateUpdateHandler = { [weak self] state in
        switch state {
        case .ready, .cancelled: ready.signal()
        case .failed(let error):
          self?.log.error("cannot show the drive on iPhones nearby: \(error)")
          ready.signal()
        default: break
        }
      }
      queue.sync { self.listener = listener }
      listener.start(queue: queue)
      _ = ready.wait(timeout: .now() + 5)
      if case .failed(let error) = listener.state { throw error }
    }

    public func stop() {
      queue.sync {
        listener?.cancel()
        listener = nil
        for link in links.values { link.close() }
        links.removeAll()
      }
      onChange?([])
    }

    /// Ends every viewer's connection, still listening: each looks again by itself.
    public func dropViewers() {
      let all = queue.sync { Array(links.values) }
      for link in all { link.close() }
    }

    /// The stage the viewers watch, nil while there is none: every viewer
    /// that asked is watched again from this one's meta.
    public func setStage(_ stage: DioramaStage?) {
      queue.sync {
        self.stage = stage
        for link in links.values { link.setStage(stage) }
      }
    }

    private func accept(_ connection: NWConnection) {
      nextID += 1
      let id = nextID
      let link = Link(id: id, connection: connection, stage: stage, queue: queue) { [weak self] in
        self?.changed()
      } ended: { [weak self] in
        guard let self else { return }
        self.links[id] = nil
        self.changed()
      }
      links[id] = link
      log.info("a viewer connected from \(link.address)")
      link.start()
      changed()
    }

    private func changed() {
      let viewers = links.values.map(\.viewer).sorted { $0.id < $1.id }
      onChange?(viewers)
    }

    /// One viewer: its controls in on the queue, and the stage's messages out
    /// from a thread of its own, one send at a time.
    final class Link: @unchecked Sendable {
      let id: Int
      let address: String
      private let connection: NWConnection
      private let queue: DispatchQueue
      private let changed: @Sendable () -> Void
      private let ended: @Sendable () -> Void
      private var reader = DioramaPeer.Reader(limit: DioramaPeer.maxControl)

      // Shared with the relay thread.
      private let lock = NSLock()
      private let wake = DispatchSemaphore(value: 0)
      private var stage: DioramaStage?
      private var watcher: DioramaStage.Watcher?
      private var name: String?
      private var asked = false
      private var closed = false

      init(
        id: Int, connection: NWConnection, stage: DioramaStage?, queue: DispatchQueue, changed: @escaping @Sendable () -> Void,
        ended: @escaping @Sendable () -> Void
      ) {
        self.id = id
        self.connection = connection
        self.stage = stage
        self.queue = queue
        self.changed = changed
        self.ended = ended
        switch connection.endpoint {
        case .hostPort(let host, _): address = "\(host)".split(separator: "%").first.map(String.init) ?? "\(host)"
        default: address = "\(connection.endpoint)"
        }
      }

      var viewer: DioramaBroadcaster.Viewer {
        lock.lock()
        defer { lock.unlock() }
        return .init(id: id, name: name ?? address, address: address)
      }

      func start() {
        connection.stateUpdateHandler = { [weak self] state in
          switch state {
          case .failed, .cancelled: self?.end()
          default: break
          }
        }
        connection.start(queue: queue)
        receive()
        let thread = Thread { [weak self] in self?.relay() }
        thread.name = "jetlink-diorama-viewer"
        thread.qualityOfService = .utility
        thread.start()
      }

      func close() {
        connection.cancel()
        end()
      }

      func setStage(_ stage: DioramaStage?) {
        lock.lock()
        if let watcher, let old = self.stage { old.unwatch(watcher) }
        watcher = nil
        self.stage = stage
        lock.unlock()
        wake.signal()
      }

      private func end() {
        lock.lock()
        let first = !closed
        closed = true
        if let watcher, let stage { stage.unwatch(watcher) }
        watcher = nil
        lock.unlock()
        wake.signal()
        if first { ended() }
      }

      private func receive() {
        connection.receive(minimumIncompleteLength: 1, maximumLength: 16 << 10) { [weak self] data, _, done, error in
          guard let self else { return }
          if let data, !data.isEmpty {
            do {
              for line in try self.reader.take(data) { self.control(String(decoding: line, as: UTF8.self)) }
            } catch {
              self.close()
              return
            }
          }
          if done || error != nil { self.close() } else { self.receive() }
        }
      }

      /// What the viewer's page posted: from `ready` on, a watcher, made
      /// again from a fresh meta when the page asks again (it reloaded).
      /// An older viewer's `input:on` is a plain `ready`: the frames the
      /// models see never leave the Mac.
      private func control(_ line: String) {
        lock.lock()
        if line.hasPrefix("hello:") {
          name = String(line.dropFirst(6).prefix(64))
        } else if line == "ready" || line == "input:on" || line == "input:off" {
          asked = true
          if let watcher, let stage { stage.unwatch(watcher) }
          watcher = nil
        }
        lock.unlock()
        wake.signal()
        changed()
      }

      private func relay() {
        while true {
          lock.lock()
          if closed {
            lock.unlock()
            return
          }
          guard asked, let stage else {
            lock.unlock()
            _ = wake.wait(timeout: .now() + 1)
            continue
          }
          let w = watcher ?? stage.watch(input: false)
          watcher = w
          lock.unlock()

          guard let batch = stage.next(w, timeout: 0.5) else {
            // closed by a control or a new stage, or it fell behind: watch again, from a fresh meta
            lock.lock()
            if watcher === w {
              stage.unwatch(w)
              watcher = nil
            }
            lock.unlock()
            _ = wake.wait(timeout: .now() + 0.2)
            continue
          }
          let drawn = batch.filter { $0.first.map(DioramaPeer.rendering.contains) ?? false }
          if drawn.isEmpty { continue }
          let sent = DispatchSemaphore(value: 0)
          connection.send(content: DioramaPeer.frame(drawn), completion: .contentProcessed { _ in sent.signal() })
          sent.wait()
        }
      }
    }
  }

  // MARK: the iPhone

  /// The viewer's end: a browser for Macs showing the drive, a connection to
  /// one of them, and that connection made again whenever it drops, until
  /// `stop()`.
  public final class DioramaWatch: @unchecked Sendable {
    public enum State: Sendable, Equatable {
      case stopped
      /// No Mac found yet, or the last one went away.
      case searching
      case connecting(String)
      case watching(String)
      /// Browsing cannot work: Local Network access is off, most likely.
      case unavailable(String)
    }

    /// Called on the watch's queue when the state or the Macs found change.
    public var onState: (@Sendable (State, [String]) -> Void)?
    /// Called on the watch's queue with each batch of messages; call `done`
    /// once the page has them, and only then is more read.
    public var onMessages: (@Sendable (_ messages: [Data], _ done: @escaping @Sendable () -> Void) -> Void)?

    private let queue = DispatchQueue(label: "jetlink-diorama-watch")
    private let hello: String
    private var browser: NWBrowser?
    private var found: [String: NWEndpoint] = [:]
    private var connection: NWConnection?
    private var connectedTo: String?
    private var preferred: String?
    private var control = "ready"
    private var state = State.stopped
    private var retry: DispatchWorkItem?
    /// For tests: dial this instead of browsing.
    private var direct: NWEndpoint?

    /// `name`: how this viewer appears in the Mac's list.
    public init(name: String) { hello = "hello:" + name }

    /// Browses, and connects to `preferring` once found, or to the first Mac found without one.
    public func start(preferring name: String?) {
      queue.async { [self] in
        preferred = name
        guard browser == nil, direct == nil else { return }
        let browser = NWBrowser(for: .bonjour(type: DioramaPeer.serviceType, domain: nil), using: DioramaPeer.parameters())
        browser.browseResultsChangedHandler = { [weak self] results, _ in self?.results(results) }
        browser.stateUpdateHandler = { [weak self] state in self?.browserState(state) }
        self.browser = browser
        set(.searching)
        browser.start(queue: queue)
      }
    }

    /// Dials `host:port` without browsing, as the tests do.
    func start(host: String, port: UInt16) {
      queue.async { [self] in
        direct = .hostPort(host: NWEndpoint.Host(host), port: NWEndpoint.Port(rawValue: port)!)
        found = ["test": direct!]
        set(.searching)
        connectIfIdle()
      }
    }

    public func stop() {
      queue.async { [self] in
        retry?.cancel()
        retry = nil
        browser?.cancel()
        browser = nil
        connection?.cancel()
        connection = nil
        connectedTo = nil
        direct = nil
        found = [:]
        set(.stopped)
      }
    }

    /// Watches this Mac instead, from now on.
    public func choose(_ name: String) {
      queue.async { [self] in
        preferred = name
        guard connectedTo != name else { return }
        connection?.cancel()
        connection = nil
        connectedTo = nil
        connectIfIdle()
      }
    }

    /// What the page posted (`ready`): sent now, and again on each new connection.
    public func send(_ line: String) {
      queue.async { [self] in
        control = line
        connection?.send(content: DioramaPeer.frame([Data(line.utf8)]), completion: .idempotent)
      }
    }

    private func set(_ state: State) {
      self.state = state
      onState?(state, found.keys.sorted())
    }

    private func browserState(_ state: NWBrowser.State) {
      switch state {
      case .failed(let error), .waiting(let error):
        if case .dns(let code) = error, code == -65570 {  // kDNSServiceErr_PolicyDenied
          set(.unavailable("Jetlink needs Local Network access to find your Mac. Turn it on in Settings > Privacy & Security > Local Network."))
        } else {
          set(.unavailable("Cannot look for Macs nearby: \(error)"))
        }
        if case .failed = state {
          // start over in a moment, as a new browser
          browser?.cancel()
          browser = nil
          queue.asyncAfter(deadline: .now() + 3) { [weak self] in
            guard let self, self.state != .stopped else { return }
            self.start(preferring: self.preferred)
          }
        }
      case .ready:
        if case .unavailable = self.state { set(.searching) }
      default: break
      }
    }

    private func results(_ results: Set<NWBrowser.Result>) {
      var found: [String: NWEndpoint] = [:]
      for result in results {
        if case .service(let name, _, _, _) = result.endpoint { found[name] = result.endpoint }
      }
      self.found = found
      onState?(state, found.keys.sorted())
      // a Mac that went away before its connection came up never fails it: let it go
      if let connection, let name = connectedTo, found[name] == nil, case .connecting = state {
        dropped(connection)
        return
      }
      connectIfIdle()
    }

    private func connectIfIdle() {
      guard connection == nil, state != .stopped else { return }
      let name = preferred.flatMap { found[$0] != nil ? $0 : nil } ?? found.keys.sorted().first
      guard let name, let endpoint = found[name] else {
        if case .unavailable = state {} else { set(.searching) }
        return
      }
      let connection = NWConnection(to: endpoint, using: DioramaPeer.parameters())
      self.connection = connection
      connectedTo = name
      set(.connecting(name))
      connection.stateUpdateHandler = { [weak self, weak connection] state in
        guard let self, let connection, self.connection === connection else { return }
        switch state {
        case .ready:
          connection.send(content: DioramaPeer.frame([Data(self.hello.utf8), Data(self.control.utf8)]), completion: .idempotent)
          self.set(.watching(name))
        case .waiting, .failed:
          // the Mac went away or its radio is busy: look again in a moment
          self.dropped(connection)
        default: break
        }
      }
      connection.start(queue: queue)
      receive(connection, reader: DioramaPeer.Reader(limit: DioramaPeer.maxMessage))
      // nor does one resolving a service that is gone: give it a few seconds
      queue.asyncAfter(deadline: .now() + DioramaWatch.connectTimeout) { [weak self, weak connection] in
        guard let self, let connection, self.connection === connection, case .connecting = self.state else { return }
        self.dropped(connection)
      }
    }

    static let connectTimeout: TimeInterval = 5

    private func receive(_ connection: NWConnection, reader: DioramaPeer.Reader) {
      connection.receive(minimumIncompleteLength: 1, maximumLength: 256 << 10) { [weak self] data, _, done, error in
        guard let self, self.connection === connection else { return }
        var reader = reader
        var messages: [Data] = []
        if let data, !data.isEmpty {
          do {
            messages = try reader.take(data)
          } catch {
            self.dropped(connection)
            return
          }
        }
        if done || error != nil {
          if !messages.isEmpty { self.onMessages?(messages) {} }
          self.dropped(connection)
          return
        }
        if messages.isEmpty || self.onMessages == nil {
          self.receive(connection, reader: reader)
        } else {
          self.onMessages?(messages, self.more(connection, reader))
        }
      }
    }

    /// Reads on once the page has the last batch.
    private func more(_ connection: NWConnection, _ reader: DioramaPeer.Reader) -> @Sendable () -> Void {
      let queue = self.queue
      return { [weak self] in
        queue.async { [weak self] in self?.receive(connection, reader: reader) }
      }
    }

    private func dropped(_ connection: NWConnection) {
      guard self.connection === connection else { return }
      connection.cancel()
      self.connection = nil
      connectedTo = nil
      set(.searching)
      retry?.cancel()
      let retry = DispatchWorkItem { [weak self] in
        self?.retry = nil
        self?.connectIfIdle()
      }
      self.retry = retry
      queue.asyncAfter(deadline: .now() + 1, execute: retry)
    }
  }
#endif
