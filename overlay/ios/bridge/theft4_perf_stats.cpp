#include "theft4_perf_stats.h"

#include <atomic>
#include <os/lock.h>
#include <time.h>

namespace {
constexpr uint64_t kCapacity = 32768;  // ~9 min at 60 events/s, bounded RAM
theft4_perf_event ring[kCapacity];
uint64_t head = 0;  // protected by lock
os_unfair_lock lock = OS_UNFAIR_LOCK_INIT;
std::atomic<bool> enabled{false};
std::atomic<int> enable_refs{0};
std::atomic<uint64_t> gpu_busy{0};
std::atomic<int> fg_mode{0}, enhance{0}, frame_cap{30};
std::atomic<uint64_t> trim_generation{0};
os_unfair_lock info_lock = OS_UNFAIR_LOCK_INIT;
theft4_render_info info{};
std::atomic<uint64_t> timeouts{0};
thread_local uint64_t submit_begin_ns = 0;
std::atomic<uint64_t> rendered_total{0};
std::atomic<uint64_t> generated_total{0};
}  // namespace

uint64_t theft4_perf_now_ns(void) { return clock_gettime_nsec_np(CLOCK_UPTIME_RAW); }

uint64_t theft4_perf_start(void) {
  os_unfair_lock_lock(&lock);
  const uint64_t cursor = head;
  os_unfair_lock_unlock(&lock);
  if (enable_refs.fetch_add(1, std::memory_order_acq_rel) == 0)
    timeouts.store(0, std::memory_order_relaxed);
  enabled.store(true, std::memory_order_release);
  return cursor;
}
void theft4_perf_stop(void) {
  int refs = enable_refs.load(std::memory_order_acquire);
  while (refs > 0 && !enable_refs.compare_exchange_weak(refs, refs - 1, std::memory_order_acq_rel)) {}
  if (refs <= 1) enabled.store(false, std::memory_order_release);
}
uint64_t theft4_perf_cursor(void) {
  os_unfair_lock_lock(&lock);
  const uint64_t cursor = head;
  os_unfair_lock_unlock(&lock);
  return cursor;
}
bool theft4_perf_enabled(void) { return enabled.load(std::memory_order_relaxed); }

void theft4_perf_record(const theft4_perf_event* event) {
  if (!event || !enabled.load(std::memory_order_relaxed)) return;
  if (event->kind == THEFT4_PERF_EVENT_SLOT_TIMEOUT) timeouts.fetch_add(1, std::memory_order_relaxed);
  os_unfair_lock_lock(&lock);
  ring[head % kCapacity] = *event;
  ++head;
  os_unfair_lock_unlock(&lock);
}

void theft4_perf_mark_submit_begin(void) {
  submit_begin_ns = enabled.load(std::memory_order_relaxed) ? theft4_perf_now_ns() : 0;
}
uint64_t theft4_perf_take_submit_begin(void) {
  const uint64_t value = submit_begin_ns;
  submit_begin_ns = 0;
  return value;
}

uint32_t theft4_perf_read(uint64_t* cursor, theft4_perf_event* out, uint32_t capacity, uint64_t* lost) {
  if (!cursor || !out || !capacity) return 0;
  if (lost) *lost = 0;
  os_unfair_lock_lock(&lock);
  if (head > *cursor + kCapacity) {
    if (lost) *lost = head - kCapacity - *cursor;
    *cursor = head - kCapacity;
  }
  uint32_t count = 0;
  while (*cursor < head && count < capacity) out[count++] = ring[(*cursor)++ % kCapacity];
  os_unfair_lock_unlock(&lock);
  return count;
}
uint64_t theft4_perf_slot_timeouts(void) { return timeouts.load(std::memory_order_relaxed); }

void theft4_perf_count_rendered(void) { rendered_total.fetch_add(1, std::memory_order_relaxed); }
void theft4_perf_count_generated(void) { generated_total.fetch_add(1, std::memory_order_relaxed); }
uint64_t theft4_perf_rendered_total(void) { return rendered_total.load(std::memory_order_relaxed); }
uint64_t theft4_perf_generated_total(void) { return generated_total.load(std::memory_order_relaxed); }

uint64_t theft4_perf_gpu_busy_ns(void) { return gpu_busy.load(std::memory_order_relaxed); }
void theft4_perf_add_gpu_busy(uint64_t ns) { gpu_busy.fetch_add(ns, std::memory_order_relaxed); }

void theft4_opt_set_fg_mode(int mode) { fg_mode.store(mode < 0 || mode > 2 ? 0 : mode, std::memory_order_relaxed); }
int theft4_opt_fg_mode(void) { return fg_mode.load(std::memory_order_relaxed); }
void theft4_opt_set_enhance(int on) { enhance.store(on ? 1 : 0, std::memory_order_relaxed); }
int theft4_opt_enhance(void) { return enhance.load(std::memory_order_relaxed); }
void theft4_opt_set_frame_cap(int fps) { frame_cap.store(fps < 0 ? 0 : fps > 120 ? 120 : fps, std::memory_order_relaxed); }
int theft4_opt_frame_cap(void) { return frame_cap.load(std::memory_order_relaxed); }
void theft4_opt_request_trim(void) { trim_generation.fetch_add(1, std::memory_order_relaxed); }
uint64_t theft4_opt_trim_generation(void) { return trim_generation.load(std::memory_order_relaxed); }

void theft4_render_info_get(theft4_render_info* out) {
  if (!out) return;
  os_unfair_lock_lock(&info_lock); *out = info; os_unfair_lock_unlock(&info_lock);
}
void theft4_render_info_set_present(uint32_t sw, uint32_t sh, uint32_t dw, uint32_t dh, uint32_t source_format) {
  os_unfair_lock_lock(&info_lock);
  info.present_source_w = sw; info.present_source_h = sh; info.output_w = dw; info.output_h = dh;
  info.source_format = source_format;
  os_unfair_lock_unlock(&info_lock);
}
void theft4_render_info_set_scaler(uint32_t scaler, uint32_t iw, uint32_t ih, uint32_t ow, uint32_t oh) {
  os_unfair_lock_lock(&info_lock);
  info.scaler = scaler; info.scaler_in_w = iw; info.scaler_in_h = ih; info.scaler_out_w = ow; info.scaler_out_h = oh;
  os_unfair_lock_unlock(&info_lock);
}
void theft4_render_info_set_drawable_format(uint32_t format) {
  os_unfair_lock_lock(&info_lock); info.drawable_format = format; os_unfair_lock_unlock(&info_lock);
}
void theft4_render_info_set_fg(int reason, int active_mode) {
  os_unfair_lock_lock(&info_lock); info.fg_reason = reason; info.fg_active_mode = active_mode; os_unfair_lock_unlock(&info_lock);
}
const char* theft4_fg_reason_text(int reason) {
  switch (reason) {
    case THEFT4_FG_OFF: return "off";
    case THEFT4_FG_GENERATING: return "generating";
    case THEFT4_FG_UNSUPPORTED_DEVICE: return "unsupported device";
    case THEFT4_FG_DESCRIPTOR_FAILED: return "descriptor failed";
    case THEFT4_FG_SHADER_FAILED: return "shader failed";
    case THEFT4_FG_NO_PREV_FRAME: return "no prev frame";
    case THEFT4_FG_NO_DEPTH: return "no depth";
    case THEFT4_FG_DEPTH_FORMAT: return "depth format";
    case THEFT4_FG_NO_CAMERA: return "no camera";
    case THEFT4_FG_NO_DRAWABLE: return "no drawable";
    case THEFT4_FG_TEXTURE_FAILED: return "texture failed";
    case THEFT4_FG_ENCODE_FAILED: return "encode failed";
    case THEFT4_FG_THERMAL_PAUSED: return "thermal pause";
    case THEFT4_FG_NOT_IN_GAME: return "not in game";
    default: return "unknown";
  }
}
