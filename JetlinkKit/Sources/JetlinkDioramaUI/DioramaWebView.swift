#if canImport(WebKit) && canImport(SwiftUI)
  import Foundation
  import JetlinkDiorama
  import JetlinkServer
  import SwiftUI
  import WebKit

  /// Where the page's messages come from: the stage on this device's own
  /// server (`StageFeed`), or a Mac nearby (`DioramaRemote`).
  @MainActor
  public protocol DioramaFeed: AnyObject {
    /// From the meta on, the frames the models see as well with `input`;
    /// `deliver` returns once the page has a batch, and no more come before
    /// it does. A second start replaces the first.
    func start(input: Bool, deliver: @escaping @MainActor @Sendable ([Data]) async -> Void)
    func stop()
  }

  /// The stage on this device's server, a watcher of it while the page is up.
  @MainActor
  public final class StageFeed: DioramaFeed {
    public let stage: DioramaStage
    private var relay: Task<Void, Never>?

    public init(stage: DioramaStage) { self.stage = stage }

    public func start(input: Bool, deliver: @escaping @MainActor @Sendable ([Data]) async -> Void) {
      stop()
      let stage = stage
      relay = Task.detached(priority: .utility) {
        while !Task.isCancelled {
          // a watcher that fell behind is dropped: watch again, from a fresh meta the page starts over from
          let watcher = stage.watch(input: input)
          while !Task.isCancelled, let batch = stage.next(watcher, timeout: 0.5) {
            if !batch.isEmpty { await deliver(batch) }
          }
          stage.unwatch(watcher)
          if !Task.isCancelled { try? await Task.sleep(for: .milliseconds(200)) }
        }
      }
    }

    public func stop() {
      relay?.cancel()
      relay = nil
    }
  }

  #if os(macOS)
    typealias PlatformViewRepresentable = NSViewRepresentable
  #else
    typealias PlatformViewRepresentable = UIViewRepresentable
  #endif

  /// The page in a web view: its files from JetlinkDiorama's bundle through a
  /// scheme of its own (a file URL may not fetch), and the feed's messages
  /// pushed in once it says it is ready. The same on a Mac and an iPhone.
  public struct DioramaWebView: PlatformViewRepresentable {
    public static let scheme = "jetlink-diorama"
    let feed: any DioramaFeed
    /// The page's Model input toggle: off for a feed that never carries the frames.
    let modelInput: Bool

    public init(feed: any DioramaFeed, modelInput: Bool = true) {
      self.feed = feed
      self.modelInput = modelInput
    }

    public func makeCoordinator() -> Coordinator { Coordinator(feed: feed) }

    #if os(macOS)
      public func makeNSView(context: Context) -> WKWebView { make(context) }
      public func updateNSView(_ view: WKWebView, context: Context) { update(view, context) }
      public static func dismantleNSView(_ view: WKWebView, coordinator: Coordinator) { dismantle(view, coordinator) }
    #else
      public func makeUIView(context: Context) -> WKWebView { make(context) }
      public func updateUIView(_ view: WKWebView, context: Context) { update(view, context) }
      public static func dismantleUIView(_ view: WKWebView, coordinator: Coordinator) { dismantle(view, coordinator) }
    #endif

    private func make(_ context: Context) -> WKWebView {
      let configuration = WKWebViewConfiguration()
      configuration.setURLSchemeHandler(PageFiles(), forURLScheme: DioramaWebView.scheme)
      configuration.userContentController.add(context.coordinator, name: "diorama")
      // the page's own errors, into the server's log beside the stage's
      configuration.userContentController.addUserScript(
        WKUserScript(
          source: """
            addEventListener('error', (e) => { try { webkit.messageHandlers.diorama.postMessage('error: ' + e.message + ' at ' + e.lineno); } catch (_) {} });
            """, injectionTime: .atDocumentStart, forMainFrameOnly: true))
      let view = WKWebView(frame: .zero, configuration: configuration)
      #if os(iOS)
        // the page fills the view and draws its own background
        view.scrollView.isScrollEnabled = false
        view.scrollView.contentInsetAdjustmentBehavior = .never
        view.isOpaque = false
        view.backgroundColor = .clear
      #endif
      view.navigationDelegate = context.coordinator
      context.coordinator.webView = view
      let query = modelInput ? "bridge=1" : "bridge=1&noinput=1"
      view.load(URLRequest(url: URL(string: "\(DioramaWebView.scheme)://page/diorama.html?\(query)")!))
      return view
    }

    private func update(_ view: WKWebView, _ context: Context) {
      if context.coordinator.feed !== feed {
        // the server restarted, or the source changed: a new feed
        context.coordinator.stop()
        context.coordinator.feed = feed
        view.reload()
      }
    }

    private static func dismantle(_ view: WKWebView, _ coordinator: Coordinator) {
      coordinator.stop()
      view.configuration.userContentController.removeScriptMessageHandler(forName: "diorama")
    }

    /// Starts the feed when the page asks, and hands the page its batches.
    @MainActor
    public final class Coordinator: NSObject, WKScriptMessageHandler, WKNavigationDelegate {
      var feed: any DioramaFeed
      weak var webView: WKWebView?
      /// The page's Model input toggle: the frames the models see as well.
      private var input = false

      init(feed: any DioramaFeed) { self.feed = feed }

      public func userContentController(_ controller: WKUserContentController, didReceive message: WKScriptMessage) {
        guard let body = message.body as? String else { return }
        if body == "ready" || body == "input:on" || body == "input:off" {
          // the page asks once it is up, and again from the start when the toggle changes
          input = body == "input:on"
          feed.start(input: input) { [weak self] batch in await self?.push(batch) }
        } else if body.hasPrefix("error: ") {
          Log.write(.warning, "diorama", "the page: \(body.dropFirst(7))")
        }
      }

      /// WebKit ended the page's process (out of memory, or it crashed): a white
      /// view until something reloads it, so this does.
      public func webViewWebContentProcessDidTerminate(_ webView: WKWebView) {
        Log.write(.warning, "diorama", "the page's web process ended; reloading it")
        stop()
        webView.reload()
      }

      func stop() { feed.stop() }

      /// Returns once the page has run the batch, so a slow page holds the feed back.
      private func push(_ batch: [Data]) async {
        guard let webView else { return }
        let script = "Diorama.pushAll([" + batch.map { "'\($0.base64EncodedString())'" }.joined(separator: ",") + "])"
        await withCheckedContinuation { (done: CheckedContinuation<Void, Never>) in
          webView.evaluateJavaScript(script) { _, _ in done.resume() }
        }
      }
    }

    /// The page's files, from JetlinkDiorama's bundle.
    final class PageFiles: NSObject, WKURLSchemeHandler {
      func webView(_ webView: WKWebView, start task: any WKURLSchemeTask) {
        guard let url = task.request.url, let file = DioramaPage.file(url.lastPathComponent) else {
          task.didFailWithError(URLError(.fileDoesNotExist))
          return
        }
        let mime = String(file.type.split(separator: ";")[0])
        task.didReceive(
          URLResponse(url: url, mimeType: mime, expectedContentLength: file.data.count, textEncodingName: file.type.contains("utf-8") ? "utf-8" : nil))
        task.didReceive(file.data)
        task.didFinish()
      }

      func webView(_ webView: WKWebView, stop task: any WKURLSchemeTask) {}
    }
  }
#endif
