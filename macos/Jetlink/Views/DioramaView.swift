import JetlinkDiorama
import SwiftUI
import WebKit

/// Diorama: the drive as a live 3-D miniature, drawn by the same page the
/// Jetson serves, in a web view fed straight from the server's stage. The
/// stage's networks run only while this view is open.
struct DioramaView: View {
  @Environment(ServerStore.self) private var server
  @Environment(AppSettings.self) private var settings

  var body: some View {
    if let stage = server.diorama {
      DioramaWebView(stage: stage)
    } else {
      ContentUnavailableView {
        Label("Diorama is off", systemImage: "cube.transparent")
      } description: {
        if server.runState != .serving {
          Text("Start the server to see the drive.")
        } else {
          Text(
            "Diorama needs its networks in \(settings.dioramaFolder.path(percentEncoded: false)). Export them on a computer with PyTorch with `python -m diorama export --out <folder>`, choose the folder in Settings, and restart the server."
          )
        }
      } actions: {
        if server.runState == .serving {
          SettingsLink { Text("Open Settings") }
        }
      }
    }
  }
}

/// The page in a web view: its files from the app's bundle through a scheme
/// of its own (a file URL may not fetch), and the stage's messages pushed in
/// once it says it is ready.
struct DioramaWebView: NSViewRepresentable {
  static let scheme = "jetlink-diorama"
  let stage: DioramaStage

  func makeCoordinator() -> Coordinator { Coordinator(stage: stage) }

  func makeNSView(context: Context) -> WKWebView {
    let configuration = WKWebViewConfiguration()
    configuration.setURLSchemeHandler(PageFiles(), forURLScheme: DioramaWebView.scheme)
    configuration.userContentController.add(context.coordinator, name: "diorama")
    let view = WKWebView(frame: .zero, configuration: configuration)
    context.coordinator.webView = view
    view.load(URLRequest(url: URL(string: "\(DioramaWebView.scheme)://page/diorama.html?bridge=1")!))
    return view
  }

  func updateNSView(_ view: WKWebView, context: Context) {
    if context.coordinator.stage !== stage {
      // the server restarted: a new stage
      context.coordinator.stop()
      context.coordinator.stage = stage
      view.reload()
    }
  }

  static func dismantleNSView(_ view: WKWebView, coordinator: Coordinator) {
    coordinator.stop()
    view.configuration.userContentController.removeScriptMessageHandler(forName: "diorama")
  }

  /// Watches the stage while the page is up, and hands it the messages in batches.
  @MainActor
  final class Coordinator: NSObject, WKScriptMessageHandler {
    var stage: DioramaStage
    weak var webView: WKWebView?
    private var watcher: DioramaStage.Watcher?
    private var relay: Task<Void, Never>?

    init(stage: DioramaStage) { self.stage = stage }

    func userContentController(_ controller: WKUserContentController, didReceive message: WKScriptMessage) {
      if message.body as? String == "ready" { start() }
    }

    /// From the stage's meta on: a page that reloads asks again.
    func start() {
      stop()
      let stage = stage
      let watcher = stage.watch()
      self.watcher = watcher
      relay = Task.detached(priority: .utility) { [weak self] in
        while !Task.isCancelled, let batch = stage.next(watcher, timeout: 0.5) {
          if batch.isEmpty { continue }
          let script = "Diorama.pushAll([" + batch.map { "'\($0.base64EncodedString())'" }.joined(separator: ",") + "])"
          await MainActor.run { self?.webView?.evaluateJavaScript(script, completionHandler: nil) }
        }
      }
    }

    func stop() {
      relay?.cancel()
      relay = nil
      if let watcher { stage.unwatch(watcher) }
      watcher = nil
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
