#pragma once
// Theft4-MetalFX: frame generation. Launcher: Graphics -> FRAME GENERATION
// (Off / MetalFX / FSR3 Lite). Runtime-switchable (theft4_opt_set_fg_mode).
//
// Two real interpolators share one input path:
//  * MetalFX: Apple MTLFXFrameInterpolator (iOS 26 SDK).
//  * FSR3 Lite: a compact Metal-compute adaptation of the ideas in AMD
//    FidelityFX FSR3 frame interpolation (MIT): block-matching optical flow on
//    a half-resolution luma pair, game (camera) motion as a predictor, a
//    depth-keyed atomic scatter of motion to the midpoint frame, motion
//    hole fill, per-pixel bidirectional warp with disocclusion selection and a
//    two-level push-pull inpaint for pixels neither frame can explain.
//    Used when chosen, and automatically when MetalFX is unsupported or its
//    interpolator cannot be created.
//
// Inputs produced by this renderer every real frame while FG is on:
//  * color: the frame's final presented image (rendered offscreen),
//  * depth: the scene depth of the pass with the most game draws (single or
//    multi-sample, sampled in place; no copies of the game's texture),
//  * camera motion: reprojection of that depth with the current/previous
//    view-projection (EnvironmentalDataV1). When the camera is unavailable the
//    optical flow supplies the motion (both paths).
// Presentation: generated frame, then the real frame, each held for at least
// half of the measured real-frame interval. No duplication, no plain blending.
// Off costs nothing per frame beyond one relaxed atomic load.
#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>
#include <cstdint>
#include <string>

#define THEFT4_FRAME_INTERPOLATION_KEY "Theft4MetalFXFrameInterpolation"  // legacy bool (build 5)
#define THEFT4_FRAME_GENERATION_MODE_KEY "Theft4FrameGenerationMode"     // 0 off, 1 MetalFX, 2 FSR3 Lite

namespace theft4::metal {
class Frame;
namespace fi {
// Cheap per-frame check: a frame generation mode is selected.
bool Wanted();
// Lazily (re)creates the state for the selected mode at this output size.
// True when some interpolator (MetalFX or FSR3 Lite fallback) can run.
bool Prepare(id<MTLDevice> device, uint32_t output_width, uint32_t output_height);
// Backend marks the submission it is about to make as a presented frame that
// frame generation will present (render worker only).
void MarkPresentedFrame(bool presented);
bool PresentedFrameMarked();
// Called by the frame adapter right after the main scene pass. Encodes the
// depth + camera-motion pass into this frame's command buffer.
void EncodeMotion(Frame& frame, id<MTLTexture> scene_depth);
// The adapter found no usable scene depth pass this frame.
void NoteNoDepth(int reason);
// Called by the backend after the real frame's command buffer was committed.
// Copies color, generates and presents generated + real frames.
// Returns false (and presents nothing) when the backend must fall back.
bool Present(id<MTLCommandQueue> queue, CAMetalLayer* layer, id<MTLTexture> color,
             uint64_t commit_ns, std::string& error);
}  // namespace fi
}  // namespace theft4::metal

extern "C" {
// From the title frontend, before each presented frame is submitted.
void theft4_fi_note_camera(const float* view_projection16, const float* projection16,
                           int valid, float time_step_seconds);
// Legacy status for the perf summary: 0 off, 1 generating, <0 THEFT4_FG_* reason.
int theft4_fi_status(void);
}
