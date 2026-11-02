import JetlinkDiorama
import JetlinkDioramaUI
import SwiftUI

/// Diorama: the drive as a live 3-D miniature, drawn natively in Metal as the
/// Mac draws it, and as the Jetson's page does. Either a Mac nearby runs the comma and the models
/// and this only watches, peer to peer, or this iPhone's own server runs the
/// models, which Settings has to turn on first.
struct DioramaScreen: View {
  @Environment(AppModel.self) private var app
  @Environment(\.scenePhase) private var scenePhase
  @State private var shown = false

  var body: some View {
    @Bindable var settings = app.settings
    NavigationStack {
      Group {
        switch settings.dioramaSource {
        case .mac: DioramaFromMac()
        case .phone: DioramaOnPhone()
        }
      }
      .navigationTitle("Diorama")
      .navigationBarTitleDisplayMode(.inline)
      .toolbar {
        ToolbarItem(placement: .principal) {
          Picker("Source", selection: $settings.dioramaSource) {
            Text("Mac").tag(DioramaSource.mac)
            Text("This \(ThisDevice.name)").tag(DioramaSource.phone)
          }
          .pickerStyle(.segmented)
          .fixedSize()
        }
        if settings.dioramaSource == .mac, app.dioramaRemote.macs.count > 1 {
          ToolbarItem(placement: .topBarTrailing) { MacMenu() }
        }
      }
    }
    .onAppear {
      shown = true
      update()
    }
    .onDisappear {
      shown = false
      update()
    }
    .onChange(of: scenePhase) { update() }
    .onChange(of: settings.dioramaSource) { update() }
  }

  /// Looks for Macs only while the tab is in front and watching one, and
  /// keeps the screen on meanwhile. Coming back from the background looks again.
  private func update() {
    let active = shown && scenePhase == .active
    app.dioramaShown = active
    if active && app.settings.dioramaSource == .mac {
      app.dioramaRemote.activate()
    } else {
      app.dioramaRemote.deactivate()
    }
  }
}

/// A Mac's drive: the scene always, which says it waits for the comma until
/// messages come, and above it which Mac this is, or that none is found.
private struct DioramaFromMac: View {
  @Environment(AppModel.self) private var app

  var body: some View {
    // a Mac never sends the frames the models see, so there is no Model input
    DioramaSceneView(feed: app.dioramaRemote, modelInput: false)
      .ignoresSafeArea(edges: .horizontal)
      .overlay(alignment: .top) {
        if let note {
          Label(note.text, systemImage: note.symbol)
            .font(.footnote)
            .padding(.horizontal, 14)
            .padding(.vertical, 10)
            .glassEffect(.regular, in: .capsule)
            .padding(.horizontal, 16)
            .padding(.top, 12)
            .multilineTextAlignment(.leading)
        }
      }
  }

  private var note: (text: String, symbol: String)? {
    switch app.dioramaRemote.state {
    case .stopped, .searching:
      ("Looking for a Mac with Diorama shown on iPhones nearby. Turn it on in Jetlink on the Mac: Settings > General > Diorama.", "magnifyingglass")
    case .connecting(let name): ("Connecting to \(name)…", "desktopcomputer")
    case .watching: nil
    case .unavailable(let why): (why, "exclamationmark.triangle")
    }
  }
}

/// Which Mac to watch, when more than one is found.
private struct MacMenu: View {
  @Environment(AppModel.self) private var app

  var body: some View {
    Menu {
      ForEach(app.dioramaRemote.macs, id: \.self) { name in
        Button {
          app.dioramaRemote.preferred = name
          app.settings.dioramaMac = name
        } label: {
          if case .watching(name) = app.dioramaRemote.state {
            Label(name, systemImage: "checkmark")
          } else {
            Text(name)
          }
        }
      }
    } label: {
      Label("Mac", systemImage: "desktopcomputer")
    }
  }
}

/// The models on this phone's own server: off until turned on, then
/// downloaded once, then the scene while the comma is on the cable.
struct DioramaOnPhone: View {
  @Environment(AppModel.self) private var app

  var body: some View {
    if !app.settings.dioramaOnPhone {
      ContentUnavailableView {
        Label("Diorama on This \(ThisDevice.name) Is Off", systemImage: "cube.transparent")
      } description: {
        Text(DioramaOnPhone.cost)
      } actions: {
        Button("Turn On") { app.settings.dioramaOnPhone = true }
          .buttonStyle(.borderedProminent)
      }
    } else if let feed = app.server.dioramaFeed {
      DioramaSceneView(feed: feed)
        .ignoresSafeArea(edges: .horizontal)
    } else {
      setup
    }
  }

  /// What it costs, said once here and once in Settings.
  static let cost =
    "Diorama runs a segmentation model, a 90 MB download, on this \(ThisDevice.name)'s GPU beside the driving model, while this tab is open. It warms the phone and drains the battery faster, and the driving model always comes first: Diorama pauses whenever a frame runs slow, and takes at most half the GPU's time. To keep the phone cool, run Jetlink on a Mac and watch it here instead."

  private var downloads: DioramaDownloads { app.server.dioramaDownloads }
  private var hasModels: Bool { DioramaDownloads.present(in: PhoneSettings.dioramaFolder) }

  @ViewBuilder
  private var setup: some View {
    @Bindable var downloads = downloads
    ContentUnavailableView {
      Label("Diorama", systemImage: "cube.transparent")
    } description: {
      switch downloads.state {
      case .downloading(let fraction):
        ProgressView(value: fraction) {
          Text("Downloading Diorama's model…")
        } currentValueLabel: {
          Text(fraction.formatted(.percent.precision(.fractionLength(0))))
        }
        .padding(.top, 8)
      case .failed(let message):
        Text("The model did not download: \(message)")
      default:
        if !hasModels {
          Text("Diorama needs its segmentation model, about 90 MB. Download it over Wi-Fi.")
        } else if let problem = app.server.dioramaProblem {
          Text(problem)
        } else {
          Text("Jetlink's server is not running.")
        }
      }
    } actions: {
      switch downloads.state {
      case .downloading:
        Button("Cancel") { downloads.cancel() }
      case .checking:
        ProgressView()
      case .failed:
        Button("Try Again") { downloads.check() }
      default:
        if !hasModels {
          Button("Download Model…") { downloads.check() }.buttonStyle(.borderedProminent)
        }
      }
    }
    .sheet(isPresented: downloads.confirming) {
      if let bytes = downloads.bytesToConfirm {
        DioramaLicenseSheet(bytes: bytes, destination: "Jetlink on this \(ThisDevice.name)") { agreed in
          if agreed { app.server.downloadDiorama() } else { downloads.cancel() }
        }
      }
    }
  }
}
