import Foundation
import Metal

struct Config {
    var groups = 875
    var iterations = 3
    var dumpArtifacts = false
    var artifactDirectory = "artifacts/latest"
}

func parseConfig(arguments: [String]) -> Config {
    var config = Config()
    var positionals: [String] = []
    var index = 0

    while index < arguments.count {
        let argument = arguments[index]
        switch argument {
        case "--dump-runtime-artifacts":
            config.dumpArtifacts = true
        case "--artifact-dir":
            if index + 1 >= arguments.count {
                fputs("error: --artifact-dir requires a path\n", stderr)
                exit(2)
            }
            index += 1
            config.artifactDirectory = arguments[index]
        default:
            positionals.append(argument)
        }
        index += 1
    }

    if let groups = positionals.first, let parsed = Int(groups) {
        config.groups = parsed
    }

    if positionals.count > 1, let iterations = Int(positionals[1]) {
        config.iterations = iterations
    }

    return config
}

let config = parseConfig(arguments: Array(CommandLine.arguments.dropFirst()))

let barrierSource = """
#include <metal_stdlib>
using namespace metal;

struct Uniforms {
    uint groups;
};

inline float2 cmul(float2 a, float2 b) {
    return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

kernel void repro(
    device float2* out [[buffer(0)]],
    constant Uniforms& u [[buffer(1)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]
) {
    if(tid >= 25) return;

    threadgroup float2 sdata[125];
    float2 r[5];
    uint group = tg.x;

    for (uint i = 0; i < 5; ++i) {
        uint idx = i * 25 + tid;
        float a = float((group * 131u + idx * 17u) & 1023u) * 0.001f;
        float b = float((group * 197u + idx * 29u + 7u) & 1023u) * 0.001f;
        r[i] = float2(a, b);
    }

    for (uint stage = 0; stage < 8; ++stage) {
        for (uint i = 0; i < 5; ++i) {
            uint idx = i * 25 + tid;
            uint writeIndex = (idx * 37u + 7u + stage * 11u) % 125u;
            sdata[writeIndex] = r[i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint i = 0; i < 5; ++i) {
            uint idx = i * 25 + tid;
            uint readIndex = (idx * 73u + 19u + stage * 17u) % 125u;
            float angle = float((idx + stage * 13u) % 125u) * 0.0502654824f;
            float2 tw = float2(cos(angle), sin(angle));
            float2 v = sdata[readIndex];
            r[i] = cmul(v + r[i], tw) + float2(float(stage) * 0.01f, float(i) * 0.005f);
        }

        for (uint i = 0; i < 5; ++i) {
            uint idx = i * 25 + tid;
            uint writeIndex = (idx * 51u + 3u + stage * 23u) % 125u;
            sdata[writeIndex] = r[i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint i = 0; i < 5; ++i) {
            uint idx = i * 25 + tid;
            uint readIndex = (idx * 99u + 41u + stage * 5u) % 125u;
            float angle = float((group + idx * 3u + stage * 7u) % 125u) * 0.0502654824f;
            float2 tw = float2(cos(angle), -sin(angle));
            float2 v = sdata[readIndex];
            r[i] = cmul(v, tw) + r[i] * 0.5f + float2(float(group) * 0.001f, float(stage) * 0.002f);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    uint base = group * 125u;
    for (uint i = 0; i < 5; ++i) {
        out[base + i * 25u + tid] = r[i];
    }
}
"""

let fenceSource = barrierSource.replacingOccurrences(
    of: "threadgroup_barrier(mem_flags::mem_threadgroup);",
    with: """
    atomic_thread_fence(mem_flags::mem_device | mem_flags::mem_threadgroup | mem_flags::mem_texture, memory_order_seq_cst, thread_scope_device);
            threadgroup_barrier(mem_flags::mem_threadgroup);
    """
)

struct Variant {
    let name: String
    let slug: String
    let source: String
    let threadLimited: Bool
}

struct Result {
    let variant: Variant
    let mismatchCount: Int
    let firstMismatch: Int?
    let expected: SIMD2<Float>?
    let actual: SIMD2<Float>?
}

struct Uniforms {
    var groups: UInt32
}

struct CompiledVariant {
    let variant: Variant
    let descriptor: MTLComputePipelineDescriptor
    let pipeline: MTLComputePipelineState
}

func makeVariants() -> [Variant] {
    [
        Variant(name: "fence baseline", slug: "fence-baseline", source: fenceSource, threadLimited: false),
        Variant(name: "barrier baseline", slug: "barrier-baseline", source: barrierSource, threadLimited: false),
        Variant(name: "fence thread_limited", slug: "fence-thread-limited", source: fenceSource, threadLimited: true),
        Variant(name: "barrier thread_limited", slug: "barrier-thread-limited", source: barrierSource, threadLimited: true),
    ]
}

func makePipelineDescriptor(function: MTLFunction, variant: Variant) -> MTLComputePipelineDescriptor {
    let descriptor = MTLComputePipelineDescriptor()
    descriptor.label = variant.name
    descriptor.computeFunction = function
    descriptor.maxTotalThreadsPerThreadgroup = variant.threadLimited ? 32 : 33
    return descriptor
}

func compileVariant(_ variant: Variant, device: MTLDevice) throws -> CompiledVariant {
    let library = try device.makeLibrary(source: variant.source, options: nil)
    guard let function = library.makeFunction(name: "repro") else {
        throw NSError(domain: "repro", code: 1, userInfo: [NSLocalizedDescriptionKey: "Missing repro function for \(variant.name)"])
    }

    let descriptor = makePipelineDescriptor(function: function, variant: variant)
    let pipeline = try device.makeComputePipelineState(descriptor: descriptor, options: [], reflection: nil)
    return CompiledVariant(variant: variant, descriptor: descriptor, pipeline: pipeline)
}

func run(compiledVariant: CompiledVariant, groups: Int, device: MTLDevice) throws -> [SIMD2<Float>] {
    guard let queue = device.makeCommandQueue() else {
        throw NSError(domain: "repro", code: 2, userInfo: [NSLocalizedDescriptionKey: "Unable to create command queue"])
    }

    let count = groups * 125
    guard let output = device.makeBuffer(length: count * MemoryLayout<SIMD2<Float>>.stride, options: .storageModeShared) else {
        throw NSError(domain: "repro", code: 3, userInfo: [NSLocalizedDescriptionKey: "Unable to allocate output buffer"])
    }

    var uniforms = Uniforms(groups: UInt32(groups))
    guard let uniformBuffer = device.makeBuffer(bytes: &uniforms, length: MemoryLayout<Uniforms>.stride, options: .storageModeShared) else {
        throw NSError(domain: "repro", code: 4, userInfo: [NSLocalizedDescriptionKey: "Unable to allocate uniform buffer"])
    }

    guard let commandBuffer = queue.makeCommandBuffer(),
          let encoder = commandBuffer.makeComputeCommandEncoder()
    else {
        throw NSError(domain: "repro", code: 5, userInfo: [NSLocalizedDescriptionKey: "Unable to create command encoder"])
    }

    encoder.setComputePipelineState(compiledVariant.pipeline)
    encoder.setBuffer(output, offset: 0, index: 0)
    encoder.setBuffer(uniformBuffer, offset: 0, index: 1)
    encoder.dispatchThreadgroups(
        MTLSize(width: groups, height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: 25, height: 1, depth: 1)
    )
    encoder.endEncoding()
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()

    if let error = commandBuffer.error {
        throw error
    }

    let ptr = output.contents().bindMemory(to: SIMD2<Float>.self, capacity: count)
    return (0..<count).map { ptr[$0] }
}

func compare(reference: [SIMD2<Float>], candidate: [SIMD2<Float>], variant: Variant) -> Result {
    let tolerance: Float = 1e-3
    for index in 0..<reference.count {
        let a = reference[index]
        let b = candidate[index]
        if abs(a.x - b.x) > tolerance || abs(a.y - b.y) > tolerance {
            let mismatches = zip(reference, candidate).filter {
                abs($0.x - $1.x) > tolerance || abs($0.y - $1.y) > tolerance
            }.count
            return Result(variant: variant, mismatchCount: mismatches, firstMismatch: index, expected: a, actual: b)
        }
    }
    return Result(variant: variant, mismatchCount: 0, firstMismatch: nil, expected: nil, actual: nil)
}

func format(_ value: SIMD2<Float>) -> String {
    String(format: "(%.6f, %.6f)", value.x, value.y)
}

func writeText(_ text: String, to url: URL) throws {
    try text.write(to: url, atomically: true, encoding: .utf8)
}

func makeBinaryArchive(device: MTLDevice, descriptor: MTLComputePipelineDescriptor, url: URL) throws {
    let archiveDescriptor = MTLBinaryArchiveDescriptor()
    archiveDescriptor.url = nil

    let archive = try device.makeBinaryArchive(descriptor: archiveDescriptor)
    try archive.addComputePipelineFunctions(descriptor: descriptor)
    try archive.serialize(to: url)
}

func emitArtifacts(compiledVariants: [CompiledVariant], device: MTLDevice, rootPath: String) throws {
    let fileManager = FileManager.default
    let directory = URL(fileURLWithPath: rootPath, isDirectory: true)
    try fileManager.createDirectory(at: directory, withIntermediateDirectories: true)

    var manifestLines = [
        "device: \(device.name)",
        "artifact_dir: \(directory.path)",
        "",
    ]

    for compiledVariant in compiledVariants {
        let variantDirectory = directory.appendingPathComponent(compiledVariant.variant.slug, isDirectory: true)
        try fileManager.createDirectory(at: variantDirectory, withIntermediateDirectories: true)

        let sourceURL = variantDirectory.appendingPathComponent("\(compiledVariant.variant.slug).metal")
        let archiveURL = variantDirectory.appendingPathComponent("\(compiledVariant.variant.slug).binarchive")

        try writeText(compiledVariant.variant.source, to: sourceURL)
        try makeBinaryArchive(device: device, descriptor: compiledVariant.descriptor, url: archiveURL)

        let metadata = """
        variant: \(compiledVariant.variant.name)
        slug: \(compiledVariant.variant.slug)
        thread_limited: \(compiledVariant.variant.threadLimited)
        pipeline.maxTotalThreadsPerThreadgroup: \(compiledVariant.pipeline.maxTotalThreadsPerThreadgroup)
        pipeline.threadExecutionWidth: \(compiledVariant.pipeline.threadExecutionWidth)
        pipeline.staticThreadgroupMemoryLength: \(compiledVariant.pipeline.staticThreadgroupMemoryLength)
        binary_archive: \(archiveURL.lastPathComponent)
        """
        try writeText(metadata + "\n", to: variantDirectory.appendingPathComponent("metadata.txt"))
        manifestLines.append("\(compiledVariant.variant.slug): \(variantDirectory.path)")
    }

    try writeText(manifestLines.joined(separator: "\n"), to: directory.appendingPathComponent("manifest.txt"))
}

guard let device = MTLCreateSystemDefaultDevice() else {
    fputs("No Metal device found.\n", stderr)
    exit(1)
}

let variants = makeVariants()

do {
    let compiledVariants = try variants.map { try compileVariant($0, device: device) }

    if config.dumpArtifacts {
        try emitArtifacts(compiledVariants: compiledVariants, device: device, rootPath: config.artifactDirectory)
    }

    let reference = try run(compiledVariant: compiledVariants[0], groups: config.groups, device: device)
    var observedFailure = false

    print("Metal barrier repro")
    print("device: \(device.name)")
    print("groups: \(config.groups)")
    print("iterations: \(config.iterations)")
    print("reference: fence baseline")
    if config.dumpArtifacts {
        print("artifacts: \(config.artifactDirectory)")
    }
    print("")

    for compiledVariant in compiledVariants {
        var worst = Result(variant: compiledVariant.variant, mismatchCount: 0, firstMismatch: nil, expected: nil, actual: nil)

        for _ in 0..<config.iterations {
            let output = try run(compiledVariant: compiledVariant, groups: config.groups, device: device)
            let result = compare(reference: reference, candidate: output, variant: compiledVariant.variant)
            if result.mismatchCount > worst.mismatchCount {
                worst = result
            }
        }

        if worst.mismatchCount == 0 {
            print("\(compiledVariant.variant.name): PASS")
        } else {
            observedFailure = true
            print("\(compiledVariant.variant.name): FAIL mismatches=\(worst.mismatchCount) firstIndex=\(worst.firstMismatch!) expected=\(format(worst.expected!)) actual=\(format(worst.actual!))")
        }
    }

    print("")
    if observedFailure {
        print("overall: FAIL")
        exit(1)
    } else {
        print("overall: PASS")
        exit(0)
    }
} catch {
    fputs("error: \(error)\n", stderr)
    exit(1)
}
