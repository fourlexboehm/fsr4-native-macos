"""Generate an experimental Winograd SIMD CNB + temporal POST kernel.

The INT8 result stays in threadgroup memory. The imported temporal resolve and
the existing SIMD quantization are preserved; unknown source shapes fail closed.
"""
from __future__ import annotations
import re


def fused_post_source(post: str, simd: str, header: str) -> str:
    boundary = simd.index('// Stride-2 2x2 downsample')
    neural = simd[:boundary] + '\n} // namespace fsr411_simd\n'
    neural = neural.replace('#include "fsr411_model.h"', header, 1)
    signature = 'threadgroup typename Tile<WINO>::type *tile, uint2 group, uint tid, ushort sg, ushort lane) {'
    if neural.count(signature) != 1:
        raise ValueError('SIMD block signature changed')
    neural = neural.replace(signature, signature.replace(') {', ', threadgroup uchar *localOutput) {'))
    store = '*(device ushort *)(scratch + fsr411::tensor_address(p.output, x, y, ch)) = pack2(v0, v1);'
    if neural.count(store) != 1:
        raise ValueError('SIMD CNB output store changed')
    neural = neural.replace(store, '*(threadgroup ushort *)(localOutput + ((y - uint(y0)) * 16u + x - uint(x0)) * 16u + ch) = pack2(v0, v1);')

    match = re.search(r'kernel void (\w+)\((.*?)\)\n\{', post, re.S)
    if not match:
        raise ValueError('POST signature not found')
    args = match[2]
    coordinate = re.search(r'^        uint2 (_\d+) = .*gl_WorkGroupID.xy \* uint2\(16u\)\);$', post, re.M)
    if not coordinate:
        raise ValueError('POST pixel mapping changed')
    pixel = coordinate[1]
    body = post[match.end()-1:]
    body = body.replace(coordinate[0], f'        uint2 {pixel} = fusedPixel;', 1)
    cache = re.compile(r'        uint fsr412CacheWord = .*?;\n((?:        uint _\d+ = ScratchBuffer\._m0\[fsr412CacheWord \+ [0-3]u\];\n){4})')
    found = cache.search(body)
    if not found:
        raise ValueError('POST compact cache reads changed')
    names = re.findall(r'uint (_\d+) =', found[1])
    body = body[:found.start()] + ''.join(f'        uint {name} = fusedValue[{i}];\n' for i, name in enumerate(names)) + body[found.end():]
    # Only the once-per-frame exposure store depends on the global thread ID.
    body = body.replace('gl_GlobalInvocationID.xy', 'fusedPixel')
    body = re.sub(r'^        uint _\d+ = gl_LocalInvocationID.x >> 2u;\n', '', body, flags=re.M)
    if any(name in body for name in ('gl_LocalInvocationID', 'gl_WorkGroupID', 'gl_GlobalInvocationID')):
        raise ValueError('unhandled POST thread-coordinate use')
    helper_args = re.sub(r'\s*\[\[.*?\]\]', '', args)
    helper_args = re.sub(r', uint3 gl_(?:GlobalInvocationID|WorkGroupID|LocalInvocationID)', '', helper_args)
    helper = post[:match.start()] + '__attribute__((always_inline)) inline void fsr412_resolve_local(' + helper_args + ', uint2 fusedPixel, uint4 fusedValue)\n' + body
    # Same texture bindings as ordinary POST. Buffers 2..4 are neural parameters
    # and weights; buffer 1 remains the frame constants.
    kernel_args = re.sub(r', uint3 gl_(?:GlobalInvocationID|WorkGroupID|LocalInvocationID) \[\[.*?\]\]', '', args)
    call_args = ['ScratchBuffer', 'MLSR_Optimized_Constants']
    call_args += re.findall(r'\b(\w+) \[\[texture\(\d+\)\]\]', args)
    return '#define FSR411_SPECIALIZE 1\n' + neural + '\n' + helper + '\n' + f'''
kernel void fsr412_simd_fused_post({kernel_args},
    constant fsr411::PassParams &raw [[buffer(2)]],
    device const uchar *weights [[buffer(3)]], device const half *wh [[buffer(4)]],
    uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {{
    FSR411_LOCAL_PARAMS;
    threadgroup char window[18 * 10 * 16];
    threadgroup uint4 result[16 * 8];
    // No global tensor writes: every threadgroup owns its private CNB result.
    fsr411_simd::block<16, 1, 4, 4, false, true>(
        (device uchar *)&ScratchBuffer, weights, wh, p, window, group, tid, sg, lane,
        (threadgroup uchar *)result);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint2 pixel = group * uint2(16, 8) + uint2(tid % 16, tid / 16);
    if (pixel.x < p.output.width && pixel.y < p.output.height)
        fsr412_resolve_local({', '.join(call_args)}, pixel, result[tid]);
}}
'''
