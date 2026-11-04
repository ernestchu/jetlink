import JetlinkDiorama
import JetlinkDioramaUI
import JetlinkRegistry
import JetlinkServer
import SwiftUI

/// Diorama: the drive as a live 3-D miniature, drawn natively in Metal as the
/// Jetson's page draws it, fed straight from the server's stage. The
/// stage's models run only while this view is open, or an iPhone watches.
struct DioramaView: View {
  @Environment(ServerStore.self) private var server
  @Environment(AppSettings.self) private var settings

  var body: some View {
    content
      .toolbar {
        if settings.dioramaShare {
          ToolbarItem { DioramaViewersButton() }
        }
      }
  }

  @ViewBuilder
  private var content: some View {
    if let feed = server.dioramaFeed {
      DioramaSceneView(feed: feed)
    } else {
      ContentUnavailableView {
        Label("Diorama is off", systemImage: "cube.transparent")
      } description: {
        switch server.dioramaDownloads.state {
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
              "Diorama draws the drive with a segmentation model, about 90 MB, that runs on the GPU only while this view is open. Download it into \(settings.dioramaFolder.path(percentEncoded: false)), or choose a folder you exported it to in Settings."
            )
          }
          if hasModels, let problem = server.dioramaProblem {
            Text(problem).font(.callout).foregroundStyle(.secondary).textSelection(.enabled)
          }
        }
      } actions: {
        switch server.dioramaDownloads.state {
        case .downloading:
          Button("Cancel") { server.dioramaDownloads.cancel() }
        case .checking:
          ProgressView().controlSize(.small)
        case .failed:
          Button("Try Again") { server.dioramaDownloads.check() }
        default:
          if !hasModels {
            Button("Download Models…") { server.dioramaDownloads.check() }.buttonStyle(.borderedProminent)
          }
          SettingsLink { Text("Open Settings") }
        }
      }
      .sheet(isPresented: server.dioramaDownloads.confirming) {
        if let bytes = server.dioramaDownloads.bytesToConfirm {
          DioramaLicenseSheet(bytes: bytes, destination: settings.dioramaFolder.path(percentEncoded: false)) { agreed in
            if agreed { server.dioramaDownloads.download(into: settings.dioramaFolder) } else { server.dioramaDownloads.cancel() }
          }
        }
      }
    }
  }

  private var hasModels: Bool { DioramaDownloads.present(in: settings.dioramaFolder) }
}

/// The iPhones watching the drive, in the toolbar while Settings shows it on them.
struct DioramaViewersButton: View {
  @Environment(ServerStore.self) private var server
  @State private var shown = false

  var body: some View {
    Button {
      shown.toggle()
    } label: {
      Label(title, systemImage: server.dioramaViewers.isEmpty ? "iphone" : "iphone.radiowaves.left.and.right")
        .labelStyle(.titleAndIcon)
    }
    .help("iPhones watching the drive")
    .popover(isPresented: $shown, arrowEdge: .bottom) {
      VStack(alignment: .leading, spacing: 10) {
        if let problem = server.dioramaShareProblem {
          Text("iPhones cannot find this Mac: \(problem)").fixedSize(horizontal: false, vertical: true)
        } else if server.dioramaViewers.isEmpty {
          Text("No iPhone is watching. Open Jetlink's Diorama tab on an iPhone nearby and choose Mac.")
            .fixedSize(horizontal: false, vertical: true)
        } else {
          ForEach(server.dioramaViewers) { viewer in
            LabeledContent(viewer.name) {
              Text(viewer.address).foregroundStyle(.secondary)
            }
          }
          Button("Disconnect All") { server.disconnectDioramaViewers() }
        }
        SettingsLink { Text("Diorama Settings…") }
      }
      .padding(14)
      .frame(width: 320)
    }
  }

  private var title: String {
    switch server.dioramaViewers.count {
    case 0: "No iPhones"
    case 1: "1 iPhone"
    case let n: "\(n) iPhones"
    }
  }
}
