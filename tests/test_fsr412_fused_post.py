import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from fsr412.fused_post import fused_post_source


class FusedPostSourceTests(unittest.TestCase):
    simd = '''#include "fsr411_model.h"
namespace fsr411_simd {
void block(threadgroup typename Tile<WINO>::type *tile, uint2 group, uint tid, ushort sg, ushort lane) {
*(device ushort *)(scratch + fsr411::tensor_address(p.output, x, y, ch)) = pack2(v0, v1);
}
// Stride-2 2x2 downsample
'''
    post = '''kernel void original(const device type_RWByteAddressBuffer& ScratchBuffer [[buffer(0)]], constant type_MLSR_Optimized_Constants& MLSR_Optimized_Constants [[buffer(1)]], texture2d<float> r_input_color [[texture(0)]], uint3 gl_GlobalInvocationID [[thread_position_in_grid]], uint3 gl_WorkGroupID [[threadgroup_position_in_grid]], uint3 gl_LocalInvocationID [[thread_position_in_threadgroup]])
{
        uint _1231 = gl_LocalInvocationID.x >> 2u;
        uint2 _1244 = uint2(0) + (gl_WorkGroupID.xy * uint2(16u));
        uint fsr412CacheWord = (_1244.y * 15360u + _1244.x * 16u) >> 2u;
        uint _2540 = ScratchBuffer._m0[fsr412CacheWord + 0u];
        uint _2583 = ScratchBuffer._m0[fsr412CacheWord + 1u];
        uint _2626 = ScratchBuffer._m0[fsr412CacheWord + 2u];
        uint _2669 = ScratchBuffer._m0[fsr412CacheWord + 3u];
}
'''

    def test_local_handoff_and_uniform_barrier(self):
        text = fused_post_source(self.post, self.simd, '// model ABI')
        self.assertIn('fsr411_simd::block<16, 1, 4, 4, false, true>', text)
        self.assertIn('threadgroup uint4 result[16 * 8]', text)
        self.assertIn('uint _2540 = fusedValue[0]', text)
        self.assertNotIn('ScratchBuffer._m0[fsr412CacheWord', text)
        self.assertNotIn('*(device ushort *)(scratch +', text)
        self.assertLess(text.index('threadgroup_barrier(mem_flags::mem_threadgroup);'),
                        text.index('if (pixel.x < p.output.width'))

    def test_reject_unknown_store(self):
        with self.assertRaisesRegex(ValueError, 'output store changed'):
            fused_post_source(self.post, self.simd.replace('pack2(v0, v1)', 'other()'), '')

    def test_reject_unknown_mapping(self):
        with self.assertRaisesRegex(ValueError, 'pixel mapping changed'):
            fused_post_source(self.post.replace('uint2(16u)', 'uint2(32u)'), self.simd, '')

    def test_reject_unknown_cache(self):
        with self.assertRaisesRegex(ValueError, 'cache reads changed'):
            fused_post_source(self.post.replace('fsr412CacheWord + 3u', 'fsr412CacheWord + 4u'), self.simd, '')


if __name__ == '__main__':
    unittest.main()
