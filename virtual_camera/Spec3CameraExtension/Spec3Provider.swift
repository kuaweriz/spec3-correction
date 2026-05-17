import CoreMedia
import CoreMediaIO
import CoreVideo
import Foundation
import IOKit.audio
import os.log

private let frameRate: Int32 = 30
private let streamWidth: Int32 = 1280
private let streamHeight: Int32 = 720
private let frameMagic: UInt32 = 0x53335033
private let frameVersion: UInt32 = 1
private let frameHeaderSize: Int = 64
private let frameFormatBGRA: UInt32 = 1

final class Spec3FrameReader {
    private let frameURL: URL
    private var lastSequence: UInt64 = 0

    init() {
        let support = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Application Support/spec3 correction", isDirectory: true)
        self.frameURL = support.appendingPathComponent("virtual_camera_frame.bgra")
    }

    func copyLatestFrame(into pixelBuffer: CVPixelBuffer) -> Bool {
        guard let data = try? Data(contentsOf: frameURL, options: [.mappedIfSafe]) else {
            return false
        }
        guard data.count >= frameHeaderSize else {
            return false
        }

        let magic = data.u32(at: 0)
        let version = data.u32(at: 4)
        let headerSize = Int(data.u32(at: 8))
        let width = Int(data.u32(at: 12))
        let height = Int(data.u32(at: 16))
        let stride = Int(data.u32(at: 20))
        let format = data.u32(at: 24)
        let sequence = data.u64(at: 32)

        guard magic == frameMagic,
              version == frameVersion,
              headerSize >= frameHeaderSize,
              format == frameFormatBGRA,
              width == Int(streamWidth),
              height == Int(streamHeight),
              stride >= width * 4,
              sequence > 0,
              sequence % 2 == 0 else {
            return false
        }

        let requiredSize = headerSize + stride * height
        guard data.count >= requiredSize else {
            return false
        }

        CVPixelBufferLockBaseAddress(pixelBuffer, [])
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, []) }

        guard let dstBase = CVPixelBufferGetBaseAddress(pixelBuffer) else {
            return false
        }
        let dstStride = CVPixelBufferGetBytesPerRow(pixelBuffer)
        data.withUnsafeBytes { rawBuffer in
            guard let srcBase = rawBuffer.baseAddress?.advanced(by: headerSize) else {
                return
            }
            for row in 0..<height {
                memcpy(
                    dstBase.advanced(by: row * dstStride),
                    srcBase.advanced(by: row * stride),
                    min(dstStride, stride)
                )
            }
        }
        lastSequence = sequence
        return true
    }
}

private extension Data {
    func u32(at offset: Int) -> UInt32 {
        guard offset + 4 <= count else { return 0 }
        return self[offset..<offset + 4].withUnsafeBytes { UInt32(littleEndian: $0.load(as: UInt32.self)) }
    }

    func u64(at offset: Int) -> UInt64 {
        guard offset + 8 <= count else { return 0 }
        return self[offset..<offset + 8].withUnsafeBytes { UInt64(littleEndian: $0.load(as: UInt64.self)) }
    }
}

final class Spec3DeviceSource: NSObject, CMIOExtensionDeviceSource {
    private(set) var device: CMIOExtensionDevice!

    private var streamSource: Spec3StreamSource!
    private var streamingCounter: UInt32 = 0
    private var timer: DispatchSourceTimer?
    private let timerQueue = DispatchQueue(
        label: "spec3.virtual.camera.timer",
        qos: .userInteractive,
        autoreleaseFrequency: .workItem
    )
    private var videoDescription: CMFormatDescription!
    private var bufferPool: CVPixelBufferPool!
    private let frameReader = Spec3FrameReader()
    private var frameIndex: UInt64 = 0

    init(localizedName: String) {
        super.init()
        let deviceID = UUID(uuidString: "6F15A4AE-7FBA-483B-9D2E-4774F3E71133")!
        device = CMIOExtensionDevice(localizedName: localizedName, deviceID: deviceID, legacyDeviceID: nil, source: self)

        let dims = CMVideoDimensions(width: streamWidth, height: streamHeight)
        CMVideoFormatDescriptionCreate(
            allocator: kCFAllocatorDefault,
            codecType: kCVPixelFormatType_32BGRA,
            width: dims.width,
            height: dims.height,
            extensions: nil,
            formatDescriptionOut: &videoDescription
        )

        let pixelBufferAttributes: NSDictionary = [
            kCVPixelBufferWidthKey: dims.width,
            kCVPixelBufferHeightKey: dims.height,
            kCVPixelBufferPixelFormatTypeKey: videoDescription.mediaSubType,
            kCVPixelBufferIOSurfacePropertiesKey: [:] as NSDictionary
        ]
        CVPixelBufferPoolCreate(kCFAllocatorDefault, nil, pixelBufferAttributes, &bufferPool)

        let videoStreamFormat = CMIOExtensionStreamFormat(
            formatDescription: videoDescription,
            maxFrameDuration: CMTime(value: 1, timescale: frameRate),
            minFrameDuration: CMTime(value: 1, timescale: frameRate),
            validFrameDurations: nil
        )
        let streamID = UUID(uuidString: "3F496D9A-8875-4EC1-9F4B-7381B6F7C1E0")!
        streamSource = Spec3StreamSource(
            localizedName: "spec3 correction Video",
            streamID: streamID,
            streamFormat: videoStreamFormat,
            device: device
        )
        do {
            try device.addStream(streamSource.stream)
        } catch {
            fatalError("Failed to add stream: \(error.localizedDescription)")
        }
    }

    var availableProperties: Set<CMIOExtensionProperty> {
        [.deviceTransportType, .deviceModel]
    }

    func deviceProperties(forProperties properties: Set<CMIOExtensionProperty>) throws -> CMIOExtensionDeviceProperties {
        let deviceProperties = CMIOExtensionDeviceProperties(dictionary: [:])
        if properties.contains(.deviceTransportType) {
            deviceProperties.transportType = kIOAudioDeviceTransportTypeVirtual
        }
        if properties.contains(.deviceModel) {
            deviceProperties.model = "spec3 correction virtual camera"
        }
        return deviceProperties
    }

    func setDeviceProperties(_ deviceProperties: CMIOExtensionDeviceProperties) throws {}

    func startStreaming() {
        guard bufferPool != nil else {
            return
        }
        streamingCounter += 1
        if streamingCounter > 1 {
            return
        }

        timer = DispatchSource.makeTimerSource(flags: .strict, queue: timerQueue)
        timer?.schedule(deadline: .now(), repeating: 1.0 / Double(frameRate), leeway: .milliseconds(2))
        timer?.setEventHandler { [weak self] in
            self?.sendFrame()
        }
        timer?.resume()
    }

    func stopStreaming() {
        if streamingCounter > 1 {
            streamingCounter -= 1
            return
        }
        streamingCounter = 0
        timer?.cancel()
        timer = nil
    }

    private func sendFrame() {
        var pixelBuffer: CVPixelBuffer?
        let err = CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault, bufferPool, &pixelBuffer)
        guard err == 0, let pixelBuffer else {
            os_log(.error, "spec3 virtual camera: out of pixel buffers %{public}d", err)
            return
        }

        if !frameReader.copyLatestFrame(into: pixelBuffer) {
            drawFallback(into: pixelBuffer)
        }

        var sampleBuffer: CMSampleBuffer!
        var timingInfo = CMSampleTimingInfo()
        timingInfo.presentationTimeStamp = CMClockGetTime(CMClockGetHostTimeClock())
        let sampleErr = CMSampleBufferCreateForImageBuffer(
            allocator: kCFAllocatorDefault,
            imageBuffer: pixelBuffer,
            dataReady: true,
            makeDataReadyCallback: nil,
            refcon: nil,
            formatDescription: videoDescription,
            sampleTiming: &timingInfo,
            sampleBufferOut: &sampleBuffer
        )
        guard sampleErr == 0 else {
            os_log(.error, "spec3 virtual camera: sample buffer failed %{public}d", sampleErr)
            return
        }
        streamSource.stream.send(
            sampleBuffer,
            discontinuity: [],
            hostTimeInNanoseconds: UInt64(timingInfo.presentationTimeStamp.seconds * Double(NSEC_PER_SEC))
        )
    }

    private func drawFallback(into pixelBuffer: CVPixelBuffer) {
        frameIndex += 1
        CVPixelBufferLockBaseAddress(pixelBuffer, [])
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, []) }
        guard let base = CVPixelBufferGetBaseAddress(pixelBuffer) else {
            return
        }
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        let rowBytes = CVPixelBufferGetBytesPerRow(pixelBuffer)
        memset(base, 18, rowBytes * height)
        let stripe = Int(frameIndex % UInt64(max(1, height)))
        for y in max(0, stripe - 3)..<min(height, stripe + 3) {
            let row = base.advanced(by: y * rowBytes)
            for x in 0..<width {
                let pixel = row.advanced(by: x * 4)
                pixel.storeBytes(of: UInt8(92), as: UInt8.self)
                pixel.advanced(by: 1).storeBytes(of: UInt8(132), as: UInt8.self)
                pixel.advanced(by: 2).storeBytes(of: UInt8(190), as: UInt8.self)
                pixel.advanced(by: 3).storeBytes(of: UInt8(255), as: UInt8.self)
            }
        }
    }
}

final class Spec3StreamSource: NSObject, CMIOExtensionStreamSource {
    private(set) var stream: CMIOExtensionStream!

    let device: CMIOExtensionDevice
    private let streamFormat: CMIOExtensionStreamFormat

    init(localizedName: String, streamID: UUID, streamFormat: CMIOExtensionStreamFormat, device: CMIOExtensionDevice) {
        self.device = device
        self.streamFormat = streamFormat
        super.init()
        stream = CMIOExtensionStream(localizedName: localizedName, streamID: streamID, direction: .source, clockType: .hostTime, source: self)
    }

    var formats: [CMIOExtensionStreamFormat] { [streamFormat] }

    var activeFormatIndex: Int = 0

    var availableProperties: Set<CMIOExtensionProperty> {
        [.streamActiveFormatIndex, .streamFrameDuration]
    }

    func streamProperties(forProperties properties: Set<CMIOExtensionProperty>) throws -> CMIOExtensionStreamProperties {
        let streamProperties = CMIOExtensionStreamProperties(dictionary: [:])
        if properties.contains(.streamActiveFormatIndex) {
            streamProperties.activeFormatIndex = 0
        }
        if properties.contains(.streamFrameDuration) {
            streamProperties.frameDuration = CMTime(value: 1, timescale: frameRate)
        }
        return streamProperties
    }

    func setStreamProperties(_ streamProperties: CMIOExtensionStreamProperties) throws {
        if let activeFormatIndex = streamProperties.activeFormatIndex {
            self.activeFormatIndex = activeFormatIndex
        }
    }

    func authorizedToStartStream(for client: CMIOExtensionClient) -> Bool {
        true
    }

    func startStream() throws {
        guard let deviceSource = device.source as? Spec3DeviceSource else {
            fatalError("Unexpected source type \(String(describing: device.source))")
        }
        deviceSource.startStreaming()
    }

    func stopStream() throws {
        guard let deviceSource = device.source as? Spec3DeviceSource else {
            fatalError("Unexpected source type \(String(describing: device.source))")
        }
        deviceSource.stopStreaming()
    }
}

final class Spec3ProviderSource: NSObject, CMIOExtensionProviderSource {
    private(set) var provider: CMIOExtensionProvider!
    private var deviceSource: Spec3DeviceSource!

    init(clientQueue: DispatchQueue?) {
        super.init()
        provider = CMIOExtensionProvider(source: self, clientQueue: clientQueue)
        deviceSource = Spec3DeviceSource(localizedName: "spec3 correction Camera")
        do {
            try provider.addDevice(deviceSource.device)
        } catch {
            fatalError("Failed to add device: \(error.localizedDescription)")
        }
    }

    func connect(to client: CMIOExtensionClient) throws {}
    func disconnect(from client: CMIOExtensionClient) {}

    var availableProperties: Set<CMIOExtensionProperty> {
        [.providerManufacturer]
    }

    func providerProperties(forProperties properties: Set<CMIOExtensionProperty>) throws -> CMIOExtensionProviderProperties {
        let providerProperties = CMIOExtensionProviderProperties(dictionary: [:])
        if properties.contains(.providerManufacturer) {
            providerProperties.manufacturer = "spec3 correction"
        }
        return providerProperties
    }

    func setProviderProperties(_ providerProperties: CMIOExtensionProviderProperties) throws {}
}
