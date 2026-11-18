#include "theft4_frame_interpolation.h"
#include "theft4_native_metal.h"
#include "theft4_perf_stats.h"
#include "theft4_retail_mode.h"
#include <algorithm>
#import <MetalFX/MetalFX.h>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <mutex>

// FSR3 Lite adapts the structure of AMD FidelityFX FSR3 frame interpolation
// (optical flow -> game/optical motion selection -> midpoint motion field ->
// disocclusion-aware warp -> inpaint). FidelityFX SDK: MIT License,
// Copyright (c) 2023 Advanced Micro Devices, Inc. This is an independent,
// much smaller Metal implementation, not AMD's shader code.

namespace {
struct Camera {
  std::array<float,16> vp{}, proj{};
  bool valid=false;
  float dt=0;
  uint64_t serial=0;
};
std::mutex camera_mutex;
Camera latest_camera;
std::atomic<uint64_t> camera_frames{0}, camera_valid_frames{0};
std::atomic<int> status{0};

bool Invert(const float* m, float* out) {
  float inv[16];
  inv[0]=m[5]*m[10]*m[15]-m[5]*m[11]*m[14]-m[9]*m[6]*m[15]+m[9]*m[7]*m[14]+m[13]*m[6]*m[11]-m[13]*m[7]*m[10];
  inv[4]=-m[4]*m[10]*m[15]+m[4]*m[11]*m[14]+m[8]*m[6]*m[15]-m[8]*m[7]*m[14]-m[12]*m[6]*m[11]+m[12]*m[7]*m[10];
  inv[8]=m[4]*m[9]*m[15]-m[4]*m[11]*m[13]-m[8]*m[5]*m[15]+m[8]*m[7]*m[13]+m[12]*m[5]*m[11]-m[12]*m[7]*m[9];
  inv[12]=-m[4]*m[9]*m[14]+m[4]*m[10]*m[13]+m[8]*m[5]*m[14]-m[8]*m[6]*m[13]-m[12]*m[5]*m[10]+m[12]*m[6]*m[9];
  inv[1]=-m[1]*m[10]*m[15]+m[1]*m[11]*m[14]+m[9]*m[2]*m[15]-m[9]*m[3]*m[14]-m[13]*m[2]*m[11]+m[13]*m[3]*m[10];
  inv[5]=m[0]*m[10]*m[15]-m[0]*m[11]*m[14]-m[8]*m[2]*m[15]+m[8]*m[3]*m[14]+m[12]*m[2]*m[11]-m[12]*m[3]*m[10];
  inv[9]=-m[0]*m[9]*m[15]+m[0]*m[11]*m[13]+m[8]*m[1]*m[15]-m[8]*m[3]*m[13]-m[12]*m[1]*m[11]+m[12]*m[3]*m[9];
  inv[13]=m[0]*m[9]*m[14]-m[0]*m[10]*m[13]-m[8]*m[1]*m[14]+m[8]*m[2]*m[13]+m[12]*m[1]*m[10]-m[12]*m[2]*m[9];
  inv[2]=m[1]*m[6]*m[15]-m[1]*m[7]*m[14]-m[5]*m[2]*m[15]+m[5]*m[3]*m[14]+m[13]*m[2]*m[7]-m[13]*m[3]*m[6];
  inv[6]=-m[0]*m[6]*m[15]+m[0]*m[7]*m[14]+m[4]*m[2]*m[15]-m[4]*m[3]*m[14]-m[12]*m[2]*m[7]+m[12]*m[3]*m[6];
  inv[10]=m[0]*m[5]*m[15]-m[0]*m[7]*m[13]-m[4]*m[1]*m[15]+m[4]*m[3]*m[13]+m[12]*m[1]*m[7]-m[12]*m[3]*m[5];
  inv[14]=-m[0]*m[5]*m[14]+m[0]*m[6]*m[13]+m[4]*m[1]*m[14]-m[4]*m[2]*m[13]-m[12]*m[1]*m[6]+m[12]*m[2]*m[5];
  inv[3]=-m[1]*m[6]*m[11]+m[1]*m[7]*m[10]+m[5]*m[2]*m[11]-m[5]*m[3]*m[10]-m[9]*m[2]*m[7]+m[9]*m[3]*m[6];
  inv[7]=m[0]*m[6]*m[11]-m[0]*m[7]*m[10]-m[4]*m[2]*m[11]+m[4]*m[3]*m[10]+m[8]*m[2]*m[7]-m[8]*m[3]*m[6];
  inv[11]=-m[0]*m[5]*m[11]+m[0]*m[7]*m[9]+m[4]*m[1]*m[11]-m[4]*m[3]*m[9]-m[8]*m[1]*m[7]+m[8]*m[3]*m[5];
  inv[15]=m[0]*m[5]*m[10]-m[0]*m[6]*m[9]-m[4]*m[1]*m[10]+m[4]*m[2]*m[9]+m[8]*m[1]*m[6]-m[8]*m[2]*m[5];
  const float det=m[0]*inv[0]+m[1]*inv[4]+m[2]*inv[8]+m[3]*inv[12];
  if(!std::isfinite(det)||std::fabs(det)<1e-20f)return false;
  for(int i=0;i<16;++i)out[i]=inv[i]/det;
  return true;
}

// Must match FIConstants in the shader source below.
struct MotionConstants {
  float inv_vp[16];
  float prev_vp[16];
  float size[2];
  float sign;
  float camera_valid;
};
// Must match FlowConstants in the shader source below.
struct FlowConstants {
  uint32_t width, height;          // output (full) size
  uint32_t flow_w, flow_h;         // blocks of 8x8 output pixels
  uint32_t half_w, half_h;         // luma pyramid level 1
  uint32_t camera_valid;           // camera motion texture holds this frame's motion
  uint32_t depth_valid;            // depth texture holds this frame's depth
  uint32_t depth_reversed;
  uint32_t have_prev_flow;
  float pad0, pad1;
};

const char* kShaderSource = R"MSL(
#include <metal_stdlib>
using namespace metal;
struct FIV { float4 p [[position]]; };
vertex FIV theft4_fi_vs(uint id [[vertex_id]]) {
  float2 uv = float2((id << 1) & 2, id & 2);
  FIV o; o.p = float4(uv * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0); return o;
}
fragment float4 theft4_fi_copy(FIV in [[stage_in]], texture2d<float> src [[texture(0)]]) {
  uint2 c = min(uint2(in.p.xy), uint2(src.get_width() - 1, src.get_height() - 1));
  return src.read(c);
}
// Copy for the generated FSR3 Lite frame: pixels neither real frame could
// explain (alpha 0) pull colour from the push-pull inpaint levels.
fragment float4 theft4_fi_copy_inpaint(FIV in [[stage_in]], texture2d<float> src [[texture(0)]],
                                       texture2d<float> l1 [[texture(1)]], texture2d<float> l2 [[texture(2)]]) {
  constexpr sampler s(filter::linear, address::clamp_to_edge);
  uint2 c = min(uint2(in.p.xy), uint2(src.get_width() - 1, src.get_height() - 1));
  float4 v = src.read(c);
  if (v.a > 0.5) return float4(v.rgb, 1.0);
  float2 uv = (float2(c) + 0.5) / float2(src.get_width(), src.get_height());
  float4 a = l1.sample(s, uv);
  if (a.a > 1e-3) return float4(a.rgb / a.a, 1.0);
  float4 b = l2.sample(s, uv);
  return float4(b.a > 1e-3 ? b.rgb / b.a : v.rgb, 1.0);
}
struct FIConstants { float4x4 inv_vp; float4x4 prev_vp; float2 size; float sign; float camera_valid; };
struct FIMotionOut { float2 motion [[color(0)]]; float depth [[depth(any)]]; };
static FIMotionOut motion_from_depth(float z, float2 uv, constant FIConstants& k) {
  FIMotionOut o; o.depth = z; o.motion = float2(0.0);
  if (k.camera_valid < 0.5) return o;
  // Row-vector guest matrices are bound unchanged, so clip = M^T * v here.
  float4 w = k.inv_vp * float4(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, z, 1.0);
  if (abs(w.w) > 1e-7) {
    w /= w.w;
    float4 pc = k.prev_vp * w;
    if (pc.w > 1e-6) {
      float2 pn = pc.xy / pc.w;
      float2 puv = float2(pn.x * 0.5 + 0.5, 0.5 - pn.y * 0.5);
      o.motion = (puv - uv) * k.sign;
    }
  }
  return o;
}
fragment FIMotionOut theft4_fi_motion(FIV in [[stage_in]], depth2d<float> d [[texture(0)]],
                                      constant FIConstants& k [[buffer(0)]]) {
  float2 uv = in.p.xy / k.size;
  uint2 dc = min(uint2(uv * float2(d.get_width(), d.get_height())),
                 uint2(d.get_width() - 1, d.get_height() - 1));
  return motion_from_depth(d.read(dc), uv, k);
}
fragment FIMotionOut theft4_fi_motion_ms(FIV in [[stage_in]], depth2d_ms<float> d [[texture(0)]],
                                         constant FIConstants& k [[buffer(0)]]) {
  float2 uv = in.p.xy / k.size;
  uint2 dc = min(uint2(uv * float2(d.get_width(), d.get_height())),
                 uint2(d.get_width() - 1, d.get_height() - 1));
  return motion_from_depth(d.read(dc, 0), uv, k);
}

// ---------------- FSR3 Lite (compute) ----------------
struct FlowConstants {
  uint width, height, flow_w, flow_h, half_w, half_h;
  uint camera_valid, depth_valid, depth_reversed, have_prev_flow;
  float pad0, pad1;
};
static float luma3(float3 c) { return dot(saturate(c), float3(0.299, 0.587, 0.114)); }
// Level-1 luma (half resolution, 2x2 box) of the colour history image.
kernel void fsr3l_luma(texture2d<float, access::read> color [[texture(0)]],
                       texture2d<float, access::write> luma [[texture(1)]],
                       uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= luma.get_width() || gid.y >= luma.get_height()) return;
  uint2 m = uint2(color.get_width() - 1, color.get_height() - 1);
  uint2 b = gid * 2;
  float l = luma3(color.read(min(b, m)).rgb) + luma3(color.read(min(b + uint2(1, 0), m)).rgb) +
            luma3(color.read(min(b + uint2(0, 1), m)).rgb) + luma3(color.read(min(b + uint2(1, 1), m)).rgb);
  luma.write(float4(l * 0.25), gid);
}
// SAD of a 4x4 half-res patch (one 8x8 output block) between current and
// previous luma, previous displaced by mv (half-res pixels).
static float block_sad(texture2d<float, access::sample> cur, texture2d<float, access::sample> prev,
                       float2 origin, float2 mv, float2 inv) {
  constexpr sampler s(filter::linear, address::clamp_to_edge);
  float sad = 0.0;
  for (int y = 0; y < 4; ++y)
    for (int x = 0; x < 4; ++x) {
      float2 p = origin + float2(x, y) + 0.5;
      sad += abs(cur.sample(s, p * inv).r - prev.sample(s, (p + mv) * inv).r);
    }
  return sad;
}
// Block-matching optical flow, one thread per 8x8 output block. Candidates:
// zero, game (camera) motion, last frame's flow here and at 4 neighbours;
// then a +-2 half-pixel refinement around the best. Output: motion from the
// current to the previous frame in OUTPUT pixels (xy), SAD per pixel (z),
// 1 in w.
kernel void fsr3l_flow(texture2d<float, access::sample> cur [[texture(0)]],
                       texture2d<float, access::sample> prev [[texture(1)]],
                       texture2d<float, access::sample> camera_motion [[texture(2)]],
                       texture2d<float, access::read> prev_flow [[texture(3)]],
                       texture2d<float, access::write> flow [[texture(4)]],
                       constant FlowConstants& k [[buffer(0)]],
                       uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= k.flow_w || gid.y >= k.flow_h) return;
  constexpr sampler s(filter::nearest, address::clamp_to_edge);
  float2 inv = 1.0 / float2(k.half_w, k.half_h);
  float2 origin = float2(gid) * 4.0;  // half-res top-left of this block
  float2 cands[7];
  int n = 0;
  cands[n++] = float2(0.0);
  if (k.camera_valid != 0) {
    float2 uv = (float2(gid) * 8.0 + 4.0) / float2(k.width, k.height);
    float2 m = camera_motion.sample(s, uv).xy * float2(k.width, k.height);
    cands[n++] = m * 0.5;
  }
  if (k.have_prev_flow != 0) {
    int2 g = int2(gid), mx = int2(k.flow_w - 1, k.flow_h - 1);
    cands[n++] = prev_flow.read(uint2(g)).xy * 0.5;
    cands[n++] = prev_flow.read(uint2(clamp(g + int2(-1, 0), int2(0), mx))).xy * 0.5;
    cands[n++] = prev_flow.read(uint2(clamp(g + int2(1, 0), int2(0), mx))).xy * 0.5;
    cands[n++] = prev_flow.read(uint2(clamp(g + int2(0, -1), int2(0), mx))).xy * 0.5;
    cands[n++] = prev_flow.read(uint2(clamp(g + int2(0, 1), int2(0), mx))).xy * 0.5;
  }
  float2 best = float2(0.0);
  float best_sad = 1e9;
  for (int i = 0; i < n; ++i) {
    // Small bias toward zero/camera motion keeps flat areas stable.
    float sad = block_sad(cur, prev, origin, cands[i], inv) + (i == 0 ? 0.0 : 0.02 * float(i > 1));
    if (sad < best_sad) { best_sad = sad; best = cands[i]; }
  }
  float2 center = best;
  for (int y = -2; y <= 2; ++y)
    for (int x = -2; x <= 2; ++x) {
      if (x == 0 && y == 0) continue;
      float2 mv = center + float2(x, y);
      float sad = block_sad(cur, prev, origin, mv, inv);
      if (sad < best_sad) { best_sad = sad; best = mv; }
    }
  flow.write(float4(best * 2.0, best_sad / 16.0, 1.0), gid);
}
// Clear the midpoint key buffer.
kernel void fsr3l_clear(device atomic_uint* keys [[buffer(0)]], constant FlowConstants& k [[buffer(1)]],
                        uint gid [[thread_position_in_grid]]) {
  if (gid < k.flow_w * k.flow_h) atomic_store_explicit(&keys[gid], 0xFFFFFFFFu, memory_order_relaxed);
}
// Scatter each block's motion to where it sits halfway between the two
// frames. The nearest surface (depth) wins via atomic_min on a key holding a
// 16-bit closeness and the source block index (FSR3's motion-field
// reconstruction idea, at block resolution).
kernel void fsr3l_scatter(texture2d<float, access::read> flow [[texture(0)]],
                          depth2d<float, access::sample> depth [[texture(1)]],
                          device atomic_uint* keys [[buffer(0)]],
                          constant FlowConstants& k [[buffer(1)]],
                          uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= k.flow_w || gid.y >= k.flow_h) return;
  constexpr sampler s(filter::nearest, address::clamp_to_edge);
  float2 m = flow.read(gid).xy;
  float2 center = float2(gid) * 8.0 + 4.0;
  float2 mid = center + 0.5 * m;
  float closeness = 0.5;
  if (k.depth_valid != 0) {
    float z = depth.sample(s, center / float2(k.width, k.height));
    closeness = k.depth_reversed != 0 ? 1.0 - z : z;  // smaller = nearer
  }
  uint key = (uint(saturate(closeness) * 65535.0) << 16) | (gid.y * k.flow_w + gid.x);
  int2 dst = int2(floor(mid / 8.0));
  if (dst.x < 0 || dst.y < 0 || dst.x >= int(k.flow_w) || dst.y >= int(k.flow_h)) return;
  atomic_fetch_min_explicit(&keys[uint(dst.y) * k.flow_w + uint(dst.x)], key, memory_order_relaxed);
}
// Midpoint motion per block: the winner's motion, or (hole) the average of
// valid neighbours within 2 blocks; w = 1 direct, 0.5 filled, 0 unknown.
kernel void fsr3l_resolve(texture2d<float, access::read> flow [[texture(0)]],
                          device atomic_uint* keys [[buffer(0)]],
                          texture2d<float, access::write> mid [[texture(1)]],
                          constant FlowConstants& k [[buffer(1)]],
                          uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= k.flow_w || gid.y >= k.flow_h) return;
  uint key = atomic_load_explicit(&keys[gid.y * k.flow_w + gid.x], memory_order_relaxed);
  if (key != 0xFFFFFFFFu) {
    uint src = key & 0xFFFFu;
    mid.write(float4(flow.read(uint2(src % k.flow_w, src / k.flow_w)).xy, 0.0, 1.0), gid);
    return;
  }
  float2 sum = float2(0.0); float count = 0.0;
  for (int y = -2; y <= 2; ++y)
    for (int x = -2; x <= 2; ++x) {
      int2 p = int2(gid) + int2(x, y);
      if (p.x < 0 || p.y < 0 || p.x >= int(k.flow_w) || p.y >= int(k.flow_h)) continue;
      uint kk = atomic_load_explicit(&keys[uint(p.y) * k.flow_w + uint(p.x)], memory_order_relaxed);
      if (kk == 0xFFFFFFFFu) continue;
      uint src = kk & 0xFFFFu;
      sum += flow.read(uint2(src % k.flow_w, src / k.flow_w)).xy; count += 1.0;
    }
  if (count > 0.0) mid.write(float4(sum / count, 0.0, 0.5), gid);
  else mid.write(float4(flow.read(gid).xy, 0.0, 0.0), gid);
}
// Per-pixel interpolation at t = 0.5. The previous frame is sampled at
// x + m/2 and the current at x - m/2 (m = midpoint motion, current->previous).
// Matching colours: average of the two motion-compensated samples. Otherwise
// the side whose own flow agrees with m is visible (disocclusion: current;
// occlusion: previous). Unexplained pixels get alpha 0 for the inpaint.
kernel void fsr3l_interpolate(texture2d<float, access::sample> cur [[texture(0)]],
                              texture2d<float, access::sample> prev [[texture(1)]],
                              texture2d<float, access::sample> mid [[texture(2)]],
                              texture2d<float, access::sample> flow [[texture(3)]],
                              texture2d<float, access::write> out [[texture(4)]],
                              constant FlowConstants& k [[buffer(0)]],
                              uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= k.width || gid.y >= k.height) return;
  constexpr sampler lin(filter::linear, address::clamp_to_edge);
  constexpr sampler near(filter::nearest, address::clamp_to_edge);
  float2 size = float2(k.width, k.height);
  float2 x = float2(gid) + 0.5;
  float4 mv = mid.sample(lin, x / size);  // linear between block centres
  float2 m = mv.xy;
  float2 pc = x - 0.5 * m, pp = x + 0.5 * m;
  float4 c = cur.sample(lin, pc / size);
  float4 p = prev.sample(lin, pp / size);
  float d = abs(luma3(c.rgb) - luma3(p.rgb));
  float2 fc = flow.sample(near, pc / size).xy;  // current frame's own motion there
  bool cur_ok = length(fc - m) < max(2.0, 0.25 * length(m));
  bool inside_c = all(pc >= 0.0) && all(pc < size);
  bool inside_p = all(pp >= 0.0) && all(pp < size);
  float4 o;
  if (d < 0.06 && inside_c && inside_p) {
    o = float4(0.5 * (c.rgb + p.rgb), 1.0);
  } else if (cur_ok && inside_c) {
    o = float4(c.rgb, 1.0);
  } else if (inside_p && mv.w > 0.25) {
    o = float4(p.rgb, 1.0);
  } else {
    o = float4(inside_c ? c.rgb : p.rgb, 0.0);
  }
  out.write(o, gid);
}
// Push step of the inpaint: premultiplied average of valid pixels.
kernel void fsr3l_push(texture2d<float, access::read> src [[texture(0)]],
                       texture2d<float, access::write> dst [[texture(1)]],
                       constant uint& factor [[buffer(0)]],
                       constant uint& premultiplied [[buffer(1)]],
                       uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
  uint2 m = uint2(src.get_width() - 1, src.get_height() - 1);
  float4 acc = float4(0.0);
  for (uint y = 0; y < factor; ++y)
    for (uint x = 0; x < factor; ++x) {
      float4 v = src.read(min(gid * factor + uint2(x, y), m));
      acc += premultiplied != 0 ? v : float4(v.rgb * v.a, v.a);
    }
  dst.write(acc / float(factor * factor), gid);
}
// Game motion for MetalFX when the camera is unavailable: optical flow,
// converted to the interpolator's normalized (uv) units.
kernel void fsr3l_flow_to_motion(texture2d<float, access::sample> flow [[texture(0)]],
                                 texture2d<float, access::write> motion [[texture(1)]],
                                 constant FlowConstants& k [[buffer(0)]],
                                 uint2 gid [[thread_position_in_grid]]) {
  if (gid.x >= k.width || gid.y >= k.height) return;
  constexpr sampler lin(filter::linear, address::clamp_to_edge);
  float2 size = float2(k.width, k.height);
  float2 m = flow.sample(lin, (float2(gid) + 0.5) / size).xy;
  motion.write(float4(m / size, 0.0, 0.0), gid);
}
)MSL";

enum class Path { None, MetalFX, Lite };

struct State {
  id<MTLDevice> device=nil;
  uint32_t width=0,height=0;
  int requested_mode=0;          // mode these resources were built for
  Path path=Path::None;
  int fallback_reason=0;         // why MetalFX fell back to FSR3 Lite
  bool shaders_failed=false;
  id<MTLFXFrameInterpolator> interpolator=nil;
  id<MTLLibrary> library=nil;
  id<MTLRenderPipelineState> copy_rgba16=nil, copy_drawable=nil, copy_inpaint=nil, motion=nil, motion_ms=nil;
  id<MTLComputePipelineState> luma=nil, flow=nil, clear=nil, scatter=nil, resolve=nil, interp=nil, push=nil, to_motion=nil;
  MTLPixelFormat drawable_format=MTLPixelFormatInvalid;
  id<MTLDepthStencilState> depth_always=nil;
  std::array<id<MTLTexture>,2> colors{};
  std::array<id<MTLTexture>,2> lumas{};
  std::array<id<MTLTexture>,2> flows{};
  id<MTLTexture> interpolated=nil, motion_texture=nil, depth_texture=nil, mid=nil, inpaint1=nil, inpaint2=nil;
  id<MTLBuffer> keys=nil;
  uint32_t flow_w=0, flow_h=0, half_w=0, half_h=0;
  uint32_t index=0;
  bool have_previous=false, have_prev_flow=false;
  bool motion_ready=false;       // depth (and camera motion) written for this frame
  bool camera_ready=false;
  bool depth_reversed=false;
  Camera previous_camera;
  bool have_previous_camera=false;
  uint64_t last_commit_ns=0;
  double interval_s=1.0/30.0;
  uint64_t sequence=0;
  uint64_t trim_seen=0;
  int last_reason=THEFT4_FG_OFF;
};
State state;
std::mutex state_mutex;  // render worker in practice; guards lazy creation
std::atomic<bool> presented_frame{false};

// Next-launch default comes from the launcher; the governor and the benchmark
// change it at runtime through theft4_opt_set_fg_mode.
int RequestedMode() {
  static std::once_flag once;
  std::call_once(once,[] {
    NSUserDefaults* d=NSUserDefaults.standardUserDefaults;
    int mode=0;
    if([d objectForKey:@THEFT4_FRAME_GENERATION_MODE_KEY])mode=int([d integerForKey:@THEFT4_FRAME_GENERATION_MODE_KEY]);
    else if([d boolForKey:@THEFT4_FRAME_INTERPOLATION_KEY])mode=1;  // build-5 switch
    theft4_opt_set_fg_mode(mode);
  });
  return theft4_opt_fg_mode();
}

void Report(int reason) {
  if(reason!=state.last_reason) {
    state.last_reason=reason;
    std::fprintf(stderr,"theft4-fg: mode=%d path=%s reason=%d (%s)\n",state.requested_mode,
                 state.path==Path::MetalFX?"metalfx":state.path==Path::Lite?"fsr3-lite":"none",
                 reason,theft4_fg_reason_text(reason));
  }
  status.store(reason);
  theft4_render_info_set_fg(reason,reason==THEFT4_FG_GENERATING?
      (state.path==Path::MetalFX?1:state.path==Path::Lite?2:0):0);
}

id<MTLRenderPipelineState> FIPipeline(id<MTLLibrary> library, NSString* fragment,
                                   MTLPixelFormat color, MTLPixelFormat depth) {
  MTLRenderPipelineDescriptor* d=[MTLRenderPipelineDescriptor new];
  d.vertexFunction=[library newFunctionWithName:@"theft4_fi_vs"];
  d.fragmentFunction=[library newFunctionWithName:fragment];
  if(!d.vertexFunction||!d.fragmentFunction)return nil;
  d.colorAttachments[0].pixelFormat=color;
  d.depthAttachmentPixelFormat=depth;
  NSError* error=nil;
  auto pipeline=[library.device newRenderPipelineStateWithDescriptor:d error:&error];
  if(!pipeline)std::fprintf(stderr,"theft4-fg: pipeline %s failed: %s\n",fragment.UTF8String,
                           error.localizedDescription.UTF8String?:"");
  return pipeline;
}
id<MTLComputePipelineState> Kernel(id<MTLLibrary> library, NSString* name) {
  id<MTLFunction> f=[library newFunctionWithName:name];
  if(!f)return nil;
  NSError* error=nil;
  auto pipeline=[library.device newComputePipelineStateWithFunction:f error:&error];
  if(!pipeline)std::fprintf(stderr,"theft4-fg: kernel %s failed: %s\n",name.UTF8String,
                           error.localizedDescription.UTF8String?:"");
  return pipeline;
}

id<MTLTexture> MakeTexture(id<MTLDevice> device, MTLPixelFormat format, uint32_t w, uint32_t h,
                           MTLTextureUsage usage) {
  MTLTextureDescriptor* t=[MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format
      width:std::max(1u,w) height:std::max(1u,h) mipmapped:NO];
  t.storageMode=MTLStorageModePrivate;t.usage=usage;
  return [device newTextureWithDescriptor:t];
}

bool BuildShaders(id<MTLDevice> device) {
  if(state.library)return true;
  NSError* error=nil;
  id<MTLLibrary> library=[device newLibraryWithSource:@(kShaderSource) options:nil error:&error];
  if(!library) {
    std::fprintf(stderr,"theft4-fg: shader library failed: %s\n",error.localizedDescription.UTF8String?:"");
    return false;
  }
  state.copy_rgba16=FIPipeline(library,@"theft4_fi_copy",MTLPixelFormatRGBA16Float,MTLPixelFormatInvalid);
  state.motion=FIPipeline(library,@"theft4_fi_motion",MTLPixelFormatRG16Float,MTLPixelFormatDepth32Float);
  state.motion_ms=FIPipeline(library,@"theft4_fi_motion_ms",MTLPixelFormatRG16Float,MTLPixelFormatDepth32Float);
  state.luma=Kernel(library,@"fsr3l_luma");state.flow=Kernel(library,@"fsr3l_flow");
  state.clear=Kernel(library,@"fsr3l_clear");state.scatter=Kernel(library,@"fsr3l_scatter");
  state.resolve=Kernel(library,@"fsr3l_resolve");state.interp=Kernel(library,@"fsr3l_interpolate");
  state.push=Kernel(library,@"fsr3l_push");state.to_motion=Kernel(library,@"fsr3l_flow_to_motion");
  MTLDepthStencilDescriptor* ds=[MTLDepthStencilDescriptor new];
  ds.depthCompareFunction=MTLCompareFunctionAlways;ds.depthWriteEnabled=YES;
  state.depth_always=[device newDepthStencilStateWithDescriptor:ds];
  const bool ok=state.copy_rgba16&&state.motion&&state.motion_ms&&state.luma&&state.flow&&state.clear&&
      state.scatter&&state.resolve&&state.interp&&state.push&&state.to_motion&&state.depth_always;
  if(ok)state.library=library;
  return ok;
}

void Release() {
  state.interpolator=nil;
  for(auto& t:state.colors)t=nil;
  for(auto& t:state.lumas)t=nil;
  for(auto& t:state.flows)t=nil;
  state.interpolated=state.motion_texture=state.depth_texture=state.mid=state.inpaint1=state.inpaint2=nil;
  state.keys=nil;state.have_previous=state.have_prev_flow=state.motion_ready=false;
  state.path=Path::None;state.width=state.height=0;
}

// Resources for both paths: history, flow, motion/depth. The interpolator
// itself only for MetalFX.
bool Create(id<MTLDevice> device,uint32_t w,uint32_t h,int mode) {
  Release();
  state.device=device;state.width=w;state.height=h;state.requested_mode=mode;state.fallback_reason=0;
  if(!BuildShaders(device)){state.shaders_failed=true;Report(THEFT4_FG_SHADER_FAILED);return false;}
  Path path=mode==2?Path::Lite:Path::MetalFX;
  MTLTextureUsage color_usage=MTLTextureUsageRenderTarget|MTLTextureUsageShaderRead;
  MTLTextureUsage motion_usage=MTLTextureUsageRenderTarget|MTLTextureUsageShaderRead;
  MTLTextureUsage depth_usage=MTLTextureUsageRenderTarget|MTLTextureUsageShaderRead;
  MTLTextureUsage out_usage=MTLTextureUsageShaderRead|MTLTextureUsageShaderWrite|MTLTextureUsageRenderTarget;
  if(path==Path::MetalFX) {
    if(![MTLFXFrameInterpolatorDescriptor supportsDevice:device]) {
      state.fallback_reason=THEFT4_FG_UNSUPPORTED_DEVICE;path=Path::Lite;
      std::fprintf(stderr,"theft4-fg: MTLFXFrameInterpolatorDescriptor supportsDevice: NO -> FSR3 Lite\n");
    } else {
      MTLFXFrameInterpolatorDescriptor* d=[MTLFXFrameInterpolatorDescriptor new];
      d.colorTextureFormat=MTLPixelFormatRGBA16Float;
      d.outputTextureFormat=MTLPixelFormatRGBA16Float;
      d.depthTextureFormat=MTLPixelFormatDepth32Float;
      d.motionTextureFormat=MTLPixelFormatRG16Float;
      d.inputWidth=w;d.inputHeight=h;d.outputWidth=w;d.outputHeight=h;
      state.interpolator=[d newFrameInterpolatorWithDevice:device];
      if(!state.interpolator) {
        state.fallback_reason=THEFT4_FG_DESCRIPTOR_FAILED;path=Path::Lite;
        std::fprintf(stderr,"theft4-fg: newFrameInterpolatorWithDevice failed (%ux%u) -> FSR3 Lite\n",w,h);
      } else {
        color_usage|=state.interpolator.colorTextureUsage;
        motion_usage|=state.interpolator.motionTextureUsage;
        depth_usage|=state.interpolator.depthTextureUsage;
        out_usage|=state.interpolator.outputTextureUsage;
      }
    }
  }
  state.path=path;
  for(auto& c:state.colors)c=MakeTexture(device,MTLPixelFormatRGBA16Float,w,h,color_usage);
  state.interpolated=MakeTexture(device,MTLPixelFormatRGBA16Float,w,h,out_usage);
  state.motion_texture=MakeTexture(device,MTLPixelFormatRG16Float,w,h,motion_usage|MTLTextureUsageShaderWrite);
  state.depth_texture=MakeTexture(device,MTLPixelFormatDepth32Float,w,h,depth_usage);
  // FSR3 Lite / optical flow at reduced resolution: luma at 1/2, flow at 1/8.
  state.half_w=(w+1)/2;state.half_h=(h+1)/2;state.flow_w=(w+7)/8;state.flow_h=(h+7)/8;
  for(auto& l:state.lumas)l=MakeTexture(device,MTLPixelFormatR16Float,state.half_w,state.half_h,
      MTLTextureUsageShaderRead|MTLTextureUsageShaderWrite);
  for(auto& f:state.flows)f=MakeTexture(device,MTLPixelFormatRGBA16Float,state.flow_w,state.flow_h,
      MTLTextureUsageShaderRead|MTLTextureUsageShaderWrite);
  state.mid=MakeTexture(device,MTLPixelFormatRGBA16Float,state.flow_w,state.flow_h,
      MTLTextureUsageShaderRead|MTLTextureUsageShaderWrite);
  state.inpaint1=MakeTexture(device,MTLPixelFormatRGBA16Float,(w+3)/4,(h+3)/4,MTLTextureUsageShaderRead|MTLTextureUsageShaderWrite);
  state.inpaint2=MakeTexture(device,MTLPixelFormatRGBA16Float,(w+15)/16,(h+15)/16,MTLTextureUsageShaderRead|MTLTextureUsageShaderWrite);
  state.keys=[device newBufferWithLength:std::max<NSUInteger>(4,NSUInteger(state.flow_w)*state.flow_h*4)
      options:MTLResourceStorageModePrivate];
  bool ok=state.interpolated&&state.motion_texture&&state.depth_texture&&state.mid&&state.inpaint1&&
      state.inpaint2&&state.keys&&state.flow_w*state.flow_h<=65535;
  for(auto& c:state.colors)ok=ok&&c;
  for(auto& l:state.lumas)ok=ok&&l;
  for(auto& f:state.flows)ok=ok&&f;
  if(!ok){Release();state.width=w;state.height=h;state.requested_mode=mode;Report(THEFT4_FG_TEXTURE_FAILED);return false;}
  std::fprintf(stderr,"theft4-fg: %s ready %ux%u flow %ux%u%s%s\n",path==Path::MetalFX?"MetalFX interpolator":"FSR3 Lite",
               w,h,state.flow_w,state.flow_h,state.fallback_reason?" fallback-from-metalfx reason=":"",
               state.fallback_reason?theft4_fg_reason_text(state.fallback_reason):"");
  Report(THEFT4_FG_NO_PREV_FRAME);
  return true;
}

// Camera parameters for the MetalFX interpolator, from the guest projection
// matrix (row-major D3D style: [0]=x scale, [5]=y scale, [10]/[14] depth).
void ConfigureCamera(const Camera& c) {
  const float* p=c.proj.data();
  float fov=60.f,aspect=float(state.width)/float(std::max(1u,state.height)),near_plane=0.1f,far_plane=5000.f;
  bool reversed=false;
  if(std::isfinite(p[5])&&std::fabs(p[5])>1e-4f) {
    fov=2.f*std::atan(1.f/std::fabs(p[5]))*180.f/float(M_PI);
    if(std::fabs(p[0])>1e-4f)aspect=std::fabs(p[5]/p[0]);
  }
  const float a=p[10],b=p[14];
  if(std::isfinite(a)&&std::isfinite(b)&&std::fabs(a)>1e-7f) {
    const float t0=std::fabs(b/a), t1=std::fabs(a-1.f)>1e-7f?std::fabs(b/(a-1.f)):0.f;
    if(t0>0&&t1>0) {
      reversed=t0>t1;  // z=0 farther than z=1 -> reversed depth
      near_plane=std::min(t0,t1);far_plane=std::max(t0,t1);
    }
  }
  state.depth_reversed=reversed;
  if(auto fi=state.interpolator) {
    fi.fieldOfView=fov;fi.aspectRatio=aspect;fi.nearPlane=near_plane;fi.farPlane=far_plane;
    fi.depthReversed=reversed;
  }
}

void CopyPass(id<MTLCommandBuffer> cb,id<MTLRenderPipelineState> pipeline,id<MTLTexture> src,id<MTLTexture> dst,
              id<MTLTexture> extra1=nil,id<MTLTexture> extra2=nil) {
  MTLRenderPassDescriptor* rp=[MTLRenderPassDescriptor renderPassDescriptor];
  rp.colorAttachments[0].texture=dst;rp.colorAttachments[0].loadAction=MTLLoadActionDontCare;
  rp.colorAttachments[0].storeAction=MTLStoreActionStore;
  auto e=[cb renderCommandEncoderWithDescriptor:rp];
  [e setRenderPipelineState:pipeline];[e setFragmentTexture:src atIndex:0];
  if(extra1)[e setFragmentTexture:extra1 atIndex:1];
  if(extra2)[e setFragmentTexture:extra2 atIndex:2];
  [e drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];[e endEncoding];
}
void Dispatch(id<MTLComputeCommandEncoder> e,id<MTLComputePipelineState> p,uint32_t w,uint32_t h) {
  [e setComputePipelineState:p];
  const NSUInteger tw=std::min<NSUInteger>(8,p.threadExecutionWidth),th=std::max<NSUInteger>(1,std::min<NSUInteger>(8,p.maxTotalThreadsPerThreadgroup/tw));
  [e dispatchThreadgroups:MTLSizeMake((w+tw-1)/tw,(h+th-1)/th,1) threadsPerThreadgroup:MTLSizeMake(tw,th,1)];
}

// Optical flow for this frame (both paths). Needs current + previous colour.
void EncodeFlow(id<MTLComputeCommandEncoder> e,id<MTLTexture> current,id<MTLTexture> previous_luma,
                id<MTLTexture> current_luma,id<MTLTexture> out_flow,id<MTLTexture> prev_flow,
                const FlowConstants& k) {
  [e setTexture:current atIndex:0];[e setTexture:current_luma atIndex:1];
  Dispatch(e,state.luma,state.half_w,state.half_h);
  [e setTexture:current_luma atIndex:0];[e setTexture:previous_luma atIndex:1];
  [e setTexture:state.motion_texture atIndex:2];[e setTexture:prev_flow atIndex:3];[e setTexture:out_flow atIndex:4];
  [e setBytes:&k length:sizeof k atIndex:0];
  Dispatch(e,state.flow,state.flow_w,state.flow_h);
}
}  // namespace

extern "C" void theft4_fi_note_camera(const float* vp,const float* proj,int valid,float dt) {
  camera_frames.fetch_add(1,std::memory_order_relaxed);
  const bool ok=valid&&vp&&proj;
  if(ok)camera_valid_frames.fetch_add(1,std::memory_order_relaxed);
  if(!theft4_opt_fg_mode())return;  // nothing to keep while frame generation is off
  std::lock_guard lock(camera_mutex);
  latest_camera.valid=ok;
  if(vp)std::memcpy(latest_camera.vp.data(),vp,64);
  if(proj)std::memcpy(latest_camera.proj.data(),proj,64);
  latest_camera.dt=dt;++latest_camera.serial;
}
extern "C" uint64_t theft4_camera_valid_frames(void){return camera_valid_frames.load(std::memory_order_relaxed);}
extern "C" uint64_t theft4_camera_frames(void){return camera_frames.load(std::memory_order_relaxed);}
extern "C" int theft4_fi_status(void) {
  if(!theft4_opt_fg_mode())return 0;
  return status.load();
}

namespace theft4::metal::fi {
void MarkPresentedFrame(bool presented){presented_frame.store(presented,std::memory_order_relaxed);}
bool PresentedFrameMarked(){return presented_frame.load(std::memory_order_relaxed);}
bool Wanted() { return RequestedMode()!=0; }

bool Prepare(id<MTLDevice> device,uint32_t w,uint32_t h) {
  const int mode=RequestedMode();
  if(!mode||!device||!w||!h)return false;
  std::lock_guard lock(state_mutex);
  const uint64_t trim=theft4_opt_trim_generation();
  if(trim!=state.trim_seen) {
    // Memory warning: drop history/flow; rebuilt below on demand.
    state.trim_seen=trim;
    if(state.path!=Path::None)std::fprintf(stderr,"theft4-fg: released history on memory trim\n");
    Release();
  }
  if(state.shaders_failed)return false;
  if(state.device==device&&state.width==w&&state.height==h&&state.requested_mode==mode&&state.path!=Path::None)return true;
  if(state.device==device&&state.width==w&&state.height==h&&state.requested_mode==mode&&state.path==Path::None)
    return false;  // creation already failed for this exact configuration
  return Create(device,w,h,mode);
}

void NoteNoDepth(int reason) {
  std::lock_guard lock(state_mutex);
  state.motion_ready=false;state.camera_ready=false;
  if(state.path!=Path::None&&state.last_reason!=THEFT4_FG_GENERATING)Report(reason);
}

void EncodeMotion(Frame& frame,id<MTLTexture> scene_depth) {
  std::lock_guard lock(state_mutex);
  state.motion_ready=false;state.camera_ready=false;
  if(state.path==Path::None||!scene_depth)return;
  const bool multisample=scene_depth.textureType==MTLTextureType2DMultisample;
  const MTLPixelFormat f=scene_depth.pixelFormat;
  if((f!=MTLPixelFormatDepth32Float&&f!=MTLPixelFormatDepth32Float_Stencil8&&f!=MTLPixelFormatDepth16Unorm)||
     !(scene_depth.usage&MTLTextureUsageShaderRead)||
     (scene_depth.textureType!=MTLTextureType2D&&!multisample)) {
    static int logged=0;
    if(logged++<4)std::fprintf(stderr,"theft4-fg: scene depth not sampleable format=%lu usage=%lu type=%lu\n",
        (unsigned long)f,(unsigned long)scene_depth.usage,(unsigned long)scene_depth.textureType);
    Report(THEFT4_FG_DEPTH_FORMAT);return;
  }
  Camera camera;{std::lock_guard c(camera_mutex);camera=latest_camera;}
  MotionConstants k{};
  bool camera_ok=camera.valid&&Invert(camera.vp.data(),k.inv_vp);
  const Camera& prev=state.have_previous_camera?state.previous_camera:camera;
  std::memcpy(k.prev_vp,prev.vp.data(),64);
  k.size[0]=float(state.width);k.size[1]=float(state.height);
  static const bool invert=[NSUserDefaults.standardUserDefaults boolForKey:@"Theft4FIMotionInvert"];
  k.sign=invert?-1.f:1.f;
  k.camera_valid=camera_ok?1.f:0.f;
  id<MTLRenderPipelineState> pipeline=multisample?state.motion_ms:state.motion;
  std::string error;
  const bool encoded=frame.EncodeExternal([&](id<MTLCommandBuffer> cb) {
    MTLRenderPassDescriptor* rp=[MTLRenderPassDescriptor renderPassDescriptor];
    rp.colorAttachments[0].texture=state.motion_texture;
    rp.colorAttachments[0].loadAction=MTLLoadActionDontCare;rp.colorAttachments[0].storeAction=MTLStoreActionStore;
    rp.depthAttachment.texture=state.depth_texture;
    rp.depthAttachment.loadAction=MTLLoadActionDontCare;rp.depthAttachment.storeAction=MTLStoreActionStore;
    auto e=[cb renderCommandEncoderWithDescriptor:rp];
    if(!e)return;
    if(!theft4_retail_mode())e.label=@"Theft4 FG motion+depth";
    [e setRenderPipelineState:pipeline];[e setDepthStencilState:state.depth_always];
    [e setFragmentTexture:scene_depth atIndex:0];[e setFragmentBytes:&k length:sizeof k atIndex:0];
    [e drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];[e endEncoding];
  },error);
  if(!encoded){Report(THEFT4_FG_ENCODE_FAILED);return;}
  state.motion_ready=true;state.camera_ready=camera_ok;
  if(camera_ok) {
    ConfigureCamera(camera);
    state.previous_camera=camera;state.have_previous_camera=true;
  }
}

bool Present(id<MTLCommandQueue> queue,CAMetalLayer* layer,id<MTLTexture> color,uint64_t commit_ns,std::string& error) {
  std::lock_guard lock(state_mutex);
  if(state.path==Path::None||!queue||!layer||!color){error="Frame generation unavailable";return false;}
  if(!state.copy_drawable||state.drawable_format!=layer.pixelFormat) {
    state.copy_drawable=FIPipeline(state.library,@"theft4_fi_copy",layer.pixelFormat,MTLPixelFormatInvalid);
    state.copy_inpaint=FIPipeline(state.library,@"theft4_fi_copy_inpaint",layer.pixelFormat,MTLPixelFormatInvalid);
    state.drawable_format=layer.pixelFormat;
    if(!state.copy_drawable||!state.copy_inpaint){Report(THEFT4_FG_SHADER_FAILED);error="Frame generation drawable pipeline failed";return false;}
  }
  if(state.last_commit_ns&&commit_ns>state.last_commit_ns) {
    const double interval=double(commit_ns-state.last_commit_ns)/1e9;
    if(interval<0.25)state.interval_s=state.interval_s*0.8+interval*0.2;
    else state.have_previous=false;  // long gap (menu, pause): do not interpolate across it
  }
  state.last_commit_ns=commit_ns;
  const double half=std::clamp(state.interval_s*0.5,1.0/120.0,1.0/15.0);
  const uint32_t index=state.index;
  id<MTLTexture> current=state.colors[index];
  id<MTLTexture> previous=state.colors[index^1u];
  id<MTLTexture> current_luma=state.lumas[index], previous_luma=state.lumas[index^1u];
  id<MTLTexture> current_flow=state.flows[index], previous_flow=state.flows[index^1u];
  const bool can=state.have_previous;
  const bool perf=theft4_perf_enabled();
  id<MTLCommandBuffer> cb=[queue commandBuffer];
  if(!cb){Report(THEFT4_FG_ENCODE_FAILED);error="Frame generation command buffer failed";return false;}
  if(!theft4_retail_mode())cb.label=@"Theft4 frame generation";
  CopyPass(cb,state.copy_rgba16,color,current);
  FlowConstants k{};
  k.width=state.width;k.height=state.height;k.flow_w=state.flow_w;k.flow_h=state.flow_h;
  k.half_w=state.half_w;k.half_h=state.half_h;k.camera_valid=state.camera_ready;
  k.depth_valid=state.motion_ready;k.depth_reversed=state.depth_reversed;k.have_prev_flow=state.have_prev_flow;
  bool generate=false;
  int reason=THEFT4_FG_GENERATING;
  if(!can) reason=THEFT4_FG_NO_PREV_FRAME;
  else if(state.path==Path::MetalFX&&!state.motion_ready) reason=state.last_reason<0?state.last_reason:THEFT4_FG_NO_DEPTH;
  // Always refresh luma (so the next frame has a previous); flow only when used.
  id<MTLComputeCommandEncoder> ce=[cb computeCommandEncoder];
  if(!ce){[cb commit];Report(THEFT4_FG_ENCODE_FAILED);error="Frame generation encoder failed";return false;}
  if(reason==THEFT4_FG_GENERATING) {
    const bool need_flow=state.path==Path::Lite||!state.camera_ready;
    if(need_flow) {
      EncodeFlow(ce,current,previous_luma,current_luma,current_flow,previous_flow,k);
      state.have_prev_flow=true;
    } else {
      [ce setTexture:current atIndex:0];[ce setTexture:current_luma atIndex:1];
      Dispatch(ce,state.luma,state.half_w,state.half_h);
      state.have_prev_flow=false;
    }
    if(state.path==Path::Lite) {
      [ce setBuffer:state.keys offset:0 atIndex:0];[ce setBytes:&k length:sizeof k atIndex:1];
      [ce setComputePipelineState:state.clear];
      const NSUInteger n=NSUInteger(state.flow_w)*state.flow_h;
      [ce dispatchThreadgroups:MTLSizeMake((n+63)/64,1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];
      [ce setTexture:current_flow atIndex:0];[ce setTexture:state.depth_texture atIndex:1];
      Dispatch(ce,state.scatter,state.flow_w,state.flow_h);
      [ce setTexture:current_flow atIndex:0];[ce setTexture:state.mid atIndex:1];
      Dispatch(ce,state.resolve,state.flow_w,state.flow_h);
      [ce setTexture:current atIndex:0];[ce setTexture:previous atIndex:1];[ce setTexture:state.mid atIndex:2];
      [ce setTexture:current_flow atIndex:3];[ce setTexture:state.interpolated atIndex:4];
      [ce setBytes:&k length:sizeof k atIndex:0];
      Dispatch(ce,state.interp,state.width,state.height);
      uint32_t factor=4,premultiplied=0;
      [ce setTexture:state.interpolated atIndex:0];[ce setTexture:state.inpaint1 atIndex:1];
      [ce setBytes:&factor length:4 atIndex:0];[ce setBytes:&premultiplied length:4 atIndex:1];
      Dispatch(ce,state.push,uint32_t(state.inpaint1.width),uint32_t(state.inpaint1.height));
      factor=4;premultiplied=1;
      [ce setTexture:state.inpaint1 atIndex:0];[ce setTexture:state.inpaint2 atIndex:1];
      [ce setBytes:&factor length:4 atIndex:0];[ce setBytes:&premultiplied length:4 atIndex:1];
      Dispatch(ce,state.push,uint32_t(state.inpaint2.width),uint32_t(state.inpaint2.height));
    } else if(!state.camera_ready) {
      // MetalFX without camera matrices: optical flow becomes its motion.
      [ce setTexture:current_flow atIndex:0];[ce setTexture:state.motion_texture atIndex:1];
      [ce setBytes:&k length:sizeof k atIndex:0];
      Dispatch(ce,state.to_motion,state.width,state.height);
    }
    generate=true;
  } else {
    [ce setTexture:current atIndex:0];[ce setTexture:current_luma atIndex:1];
    Dispatch(ce,state.luma,state.half_w,state.half_h);
    state.have_prev_flow=false;
  }
  [ce endEncoding];
  if(generate&&state.path==Path::MetalFX) {
    auto fi=state.interpolator;
    fi.colorTexture=current;fi.prevColorTexture=previous;
    fi.depthTexture=state.depth_texture;fi.motionTexture=state.motion_texture;
    fi.outputTexture=state.interpolated;
    fi.motionVectorScaleX=float(state.width);fi.motionVectorScaleY=float(state.height);
    fi.jitterOffsetX=0;fi.jitterOffsetY=0;  // the game renders without TAA jitter
    fi.deltaTime=float(state.interval_s);
    [fi encodeToCommandBuffer:cb];
  }
  if(generate) {
    id<CAMetalDrawable> generated=[layer nextDrawable];
    if(generated) {
      if(state.path==Path::Lite)CopyPass(cb,state.copy_inpaint,state.interpolated,generated.texture,state.inpaint1,state.inpaint2);
      else CopyPass(cb,state.copy_drawable,state.interpolated,generated.texture);
      const uint64_t sequence=++state.sequence;
      id<MTLCommandBuffer> owner=cb;
      // Counted when actually shown (presentedTime > 0), not when queued.
      [generated addPresentedHandler:^(id<MTLDrawable> shown) {
        if(shown.presentedTime<=0)return;
        theft4_perf_count_generated();
        if(!perf)return;
        theft4_perf_event p{};p.kind=THEFT4_PERF_EVENT_PRESENTED;p.sequence=uint64_t(shown.drawableID);
        p.gpu_end_ns=uint64_t(shown.presentedTime*1e9);p.value=1;
        theft4_perf_record(&p);
        theft4_perf_event g{};g.kind=THEFT4_PERF_EVENT_INTERPOLATED;g.sequence=sequence;g.commit_ns=commit_ns;
        g.gpu_start_ns=uint64_t(owner.GPUStartTime*1e9);g.gpu_end_ns=uint64_t(owner.GPUEndTime*1e9);
        g.value=p.gpu_end_ns>commit_ns?p.gpu_end_ns-commit_ns:0;
        theft4_perf_record(&g);
      }];
      [cb presentDrawable:generated afterMinimumDuration:half];
    } else {
      generate=false;reason=THEFT4_FG_NO_DRAWABLE;
    }
  }
  if(perf) {
    [cb addCompletedHandler:^(id<MTLCommandBuffer> done) {
      if(done.GPUEndTime>done.GPUStartTime)theft4_perf_add_gpu_busy(uint64_t((done.GPUEndTime-done.GPUStartTime)*1e9));
    }];
  }
  [cb commit];
  // The real frame follows in its own command buffer and display slot.
  id<MTLCommandBuffer> real=[queue commandBuffer];
  id<CAMetalDrawable> drawable=real?[layer nextDrawable]:nil;
  if(drawable) {
    CopyPass(real,state.copy_drawable,current,drawable.texture);
    if(perf) [drawable addPresentedHandler:^(id<MTLDrawable> shown) {
      theft4_perf_event p{};p.kind=THEFT4_PERF_EVENT_PRESENTED;p.sequence=uint64_t(shown.drawableID);
      p.gpu_end_ns=shown.presentedTime>0?uint64_t(shown.presentedTime*1e9):0;p.value=0;
      theft4_perf_record(&p);
    }];
    if(generate)[real presentDrawable:drawable afterMinimumDuration:half];
    else [real presentDrawable:drawable];
  }
  if(real)[real commit];
  state.have_previous=true;state.index^=1u;state.motion_ready=false;state.camera_ready=false;
  Report(generate?THEFT4_FG_GENERATING:reason);
  return true;
}
}  // namespace theft4::metal::fi
