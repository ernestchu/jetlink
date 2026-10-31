import JetlinkDiorama
import JetlinkRegistry
import JetlinkServer
import SwiftUI
import WebKit

/// Diorama: the drive as a live 3-D miniature, drawn by the same page the
/// Jetson serves, in a web view fed straight from the server's stage. The
/// stage's models run only while this view is open.
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
        switch server.dioramaDownload {
        case .downloading(let fraction):
          ProgressView(value: fraction) {
            Text("Downloading Diorama's models…")
          } currentValueLabel: {
            Text(fraction.formatted(.percent.precision(.fractionLength(0))))
          }
          .frame(maxWidth: 320)
        case .failed(let message):
          Text("The models did not download: \(message)")
        default:
          if hasModels {
            Text(server.runState == .serving ? "Restart the server to load the models." : "Start the server to see the drive.")
          } else {
            Text(
              "Diorama draws the drive with two models, about 720 MB, that run on the GPU only while this view is open. Download them into \(settings.dioramaFolder.path(percentEncoded: false)), or choose a folder you exported them to in Settings."
            )
          }
          if hasModels, let problem = server.dioramaProblem {
            Text(problem).font(.callout).foregroundStyle(.secondary).textSelection(.enabled)
          }
        }
      } actions: {
        switch server.dioramaDownload {
        case .downloading:
          Button("Cancel") { server.cancelDioramaDownload() }
        case .checking:
          ProgressView().controlSize(.small)
        case .failed:
          Button("Try Again") { server.checkDiorama() }
        default:
          if !hasModels {
            Button("Download Models…") { server.checkDiorama() }.buttonStyle(.borderedProminent)
          }
          SettingsLink { Text("Open Settings") }
        }
      }
      .sheet(isPresented: confirming) {
        if case .confirm(let bytes) = server.dioramaDownload {
          DioramaLicenseSheet(bytes: bytes, folder: settings.dioramaFolder) { agreed in
            if agreed { server.downloadDiorama() } else { server.cancelDioramaDownload() }
          }
        }
      }
    }
  }

  /// diorama.json is written last, so its being there means the graphs are too.
  private var hasModels: Bool {
    FileManager.default.fileExists(atPath: settings.dioramaFolder.appending(path: DioramaModels.manifest).path(percentEncoded: false))
  }

  private var confirming: Binding<Bool> {
    Binding {
      if case .confirm = server.dioramaDownload { true } else { false }
    } set: { shown in
      if !shown { server.cancelDioramaDownload() }
    }
  }
}

/// Before 720 MB comes down: what it is, where it goes, and whose terms it comes under.
struct DioramaLicenseSheet: View {
  let bytes: Int64
  let folder: URL
  let answer: (Bool) -> Void

  var body: some View {
    VStack(alignment: .leading, spacing: 14) {
      Text("Download Diorama's models?").font(.headline)
      Text(
        "\(bytes.formatted(.byteCount(style: .file))) from huggingface.co/\(DioramaModels.repo), into \(folder.path(percentEncoded: false))."
      )
      .fixedSize(horizontal: false, vertical: true)
      VStack(alignment: .leading, spacing: 8) {
        Text("**Segmentation:** Mask2Former by Meta. Its code is under the MIT License; its weights were trained on Mapillary Vistas, whose terms allow non-commercial use only.")
          .fixedSize(horizontal: false, vertical: true)
        HStack(spacing: 14) {
          Link("Mask2Former", destination: URL(string: "https://huggingface.co/facebook/mask2former-swin-large-mapillary-vistas-panoptic")!)
          Link("Mapillary Vistas terms", destination: URL(string: "https://www.mapillary.com/dataset/vistas")!)
        }
        Text("**Headings:** DD3D by Toyota Research Institute, code and weights under the MIT License.")
          .fixedSize(horizontal: false, vertical: true)
        Link("DD3D", destination: URL(string: "https://github.com/TRI-ML/dd3d")!)
      }
      .font(.callout)
      HStack {
        Spacer()
        Button("Cancel", role: .cancel) { answer(false) }.keyboardShortcut(.cancelAction)
        Button("Agree and Download") { answer(true) }.keyboardShortcut(.defaultAction)
      }
    }
    .padding(20)
    .frame(width: 460)
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
    // the page's own errors, into the server's log beside the stage's
    configuration.userContentController.addUserScript(
      WKUserScript(
        source: """
          addEventListener('error', (e) => { try { webkit.messageHandlers.diorama.postMessage('error: ' + e.message + ' at ' + e.lineno); } catch (_) {} });
          """, injectionTime: .atDocumentStart, forMainFrameOnly: true))
    let view = WKWebView(frame: .zero, configuration: configuration)
    view.navigationDelegate = context.coordinator
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
  final class Coordinator: NSObject, WKScriptMessageHandler, WKNavigationDelegate {
    var stage: DioramaStage
    weak var webView: WKWebView?
    private var watcher: DioramaStage.Watcher?
    private var relay: Task<Void, Never>?
    /// The page's Model input toggle: the frames the models see as well.
    private var input = false

    init(stage: DioramaStage) { self.stage = stage }

    func userContentController(_ controller: WKUserContentController, didReceive message: WKScriptMessage) {
      guard let body = message.body as? String else { return }
      if body == "ready" {
        start()
      } else if body == "input:on" || body == "input:off" {
        input = body == "input:on"
        if let watcher { stage.showInput(watcher, input) }
      } else if body.hasPrefix("error: ") {
        Log.write(.warning, "diorama", "the page: \(body.dropFirst(7))")
      }
    }

    /// WebKit ended the page's process (out of memory, or it crashed): a white
    /// view until something reloads it, so this does.
    func webViewWebContentProcessDidTerminate(_ webView: WKWebView) {
      Log.write(.warning, "diorama", "the page's web process ended; reloading it")
      stop()
      webView.reload()
    }

    /// From the stage's meta on: a page that reloads asks again.
    func start() {
      stop()
      let stage = stage
      let watcher = stage.watch(input: input)
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
