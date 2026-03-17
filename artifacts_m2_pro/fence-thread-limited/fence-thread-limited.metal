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
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

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
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

        for (uint i = 0; i < 5; ++i) {
            uint idx = i * 25 + tid;
            uint readIndex = (idx * 99u + 41u + stage * 5u) % 125u;
            float angle = float((group + idx * 3u + stage * 7u) % 125u) * 0.0502654824f;
            float2 tw = float2(cos(angle), -sin(angle));
            float2 v = sdata[readIndex];
            r[i] = cmul(v, tw) + r[i] * 0.5f + float2(float(group) * 0.001f, float(stage) * 0.002f);
        }

        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    }

    uint base = group * 125u;
    for (uint i = 0; i < 5; ++i) {
        out[base + i * 25u + tid] = r[i];
    }
}