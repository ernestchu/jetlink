#if canImport(SwiftUI)
  import Foundation
  import JetlinkRegistry
  import Observation
  import SwiftUI
  import Synchronization

  /// Fetching Diorama's models into a folder, for a view: the repo is asked
  /// what there is, the user agrees to the size and the licence, and then it
  /// comes down. The Mac's and the iPhone's alike.
  @MainActor
  @Observable
  public final class DioramaDownloads {
    public enum State: Equatable, Sendable {
      case idle
      /// Asking the repo what there is to fetch.
      case checking
      /// The user agrees to the licenses and the size first.
      case confirm(bytes: Int64)
      case downloading(Double)
      case failed(String)
    }

    public private(set) var state: State = .idle
    /// Called once the models are in the folder, for the server to pick them up.
    @ObservationIgnored public var finished: (@MainActor () -> Void)?

    @ObservationIgnored private var plan: DioramaModels.Plan?
    @ObservationIgnored private var task: Task<Void, Never>?
    @ObservationIgnored private var stopFlag = StopFlag()

    public init() {}

    /// diorama.json is written last, so its being there means the models are too.
    public static func present(in folder: URL) -> Bool {
      FileManager.default.fileExists(atPath: folder.appending(path: DioramaModels.manifest).path(percentEncoded: false))
    }

    /// Looks up the models' files and sizes, then waits for `download(into:)`.
    public func check() {
      guard task == nil else { return }
      state = .checking
      task = Task { [weak self] in
        do {
          // a mirror with Hugging Face's layout, for testing: -dioramaModelsHost http://...
          let host = UserDefaults.standard.string(forKey: "dioramaModelsHost") ?? DioramaModels.defaultHost
          let plan = try await DioramaModels.plan(host: host, coreml: true)
          self?.plan = plan
          self?.state = .confirm(bytes: plan.bytes)
        } catch {
          self?.state = .failed((error as? RegistryError)?.message ?? "\(error)")
        }
        self?.task = nil
      }
    }

    /// Into `folder`, then `finished`.
    public func download(into folder: URL) {
      guard task == nil, let plan else { return }
      let stop = StopFlag()
      stopFlag = stop
      state = .downloading(0)
      task = Task { [weak self] in
        do {
          try await DioramaModels.fetch(
            plan, into: folder,
            progress: { fraction in
              // each hop is a task of its own, so they may land out of order: the bar only grows
              Task { @MainActor in if case .downloading(let shown) = self?.state, fraction > shown { self?.state = .downloading(fraction) } }
            },
            shouldStop: { stop.isSet })
          self?.state = .idle
          self?.plan = nil
          self?.finished?()
        } catch let error as RegistryError where error.kind == .cancelled {
          self?.state = .idle
        } catch {
          self?.state = .failed((error as? RegistryError)?.message ?? "\(error)")
        }
        self?.task = nil
      }
    }

    public func cancel() {
      stopFlag.set()
      if case .confirm = state { state = .idle }
    }

    /// Bound to the licence sheet: shown while the user is asked.
    public var confirming: Binding<Bool> {
      Binding {
        if case .confirm = self.state { true } else { false }
      } set: { shown in
        if !shown { self.cancel() }
      }
    }

    public var bytesToConfirm: Int64? {
      if case .confirm(let bytes) = state { bytes } else { nil }
    }

    /// Asked between a download's chunks, set from the main actor.
    private final class StopFlag: Sendable {
      private let value = Atomic<Bool>(false)
      var isSet: Bool { value.load(ordering: .relaxed) }
      func set() { value.store(true, ordering: .relaxed) }
    }
  }

  /// Before about 90 MB comes down: what it is, where it goes, and whose terms it comes under.
  public struct DioramaLicenseSheet: View {
    let bytes: Int64
    /// Where it goes, as the user knows the place.
    let destination: String
    let answer: (Bool) -> Void

    public init(bytes: Int64, destination: String, answer: @escaping (Bool) -> Void) {
      self.bytes = bytes
      self.destination = destination
      self.answer = answer
    }

    public var body: some View {
      VStack(alignment: .leading, spacing: 14) {
        Text("Download Diorama's models?").font(.headline)
        Text("\(bytes.formatted(.byteCount(style: .file))) from huggingface.co/\(DioramaModels.repo), into \(destination).")
          .fixedSize(horizontal: false, vertical: true)
        VStack(alignment: .leading, spacing: 8) {
          Text(
            "**Segmentation:** Mask2Former by Meta. Its code is under the MIT License; its weights are under CC BY-NC 4.0, and were trained on Mapillary Vistas, whose terms allow non-commercial use only."
          )
          .fixedSize(horizontal: false, vertical: true)
          HStack(spacing: 14) {
            Link("Mask2Former", destination: URL(string: "https://github.com/facebookresearch/Mask2Former/blob/main/MODEL_ZOO.md")!)
            Link("Mapillary Vistas terms", destination: URL(string: "https://www.mapillary.com/dataset/vistas")!)
          }
        }
        .font(.callout)
        HStack {
          Spacer()
          Button("Cancel", role: .cancel) { answer(false) }.keyboardShortcut(.cancelAction)
          Button("Agree and Download") { answer(true) }.keyboardShortcut(.defaultAction)
            #if os(iOS)
              .buttonStyle(.borderedProminent)
            #endif
        }
      }
      .padding(20)
      #if os(macOS)
        .frame(width: 460)
      #else
        .presentationDetents([.medium, .large])
      #endif
    }
  }

#endif
