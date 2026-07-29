import Foundation
import Metal

public class MetalComputeEngine {
    private var device: MTLDevice?

    public init() {
        self.device = MTLCreateSystemDefaultDevice()
    }

    public func isNeuralEngineAvailable() -> Bool {
        return self.device != nil
    }

    public func quantizeFP16toINT8(buffer: [Float32]) -> [Int8] {
        return buffer.map { Int8(clamping: Int(round($0 * 127.0))) }
    }
}
