#include "theft4_motion_blur.h"

#include <atomic>
#include <limits>

#include <rex/logging.h>
#include <rex/runtime.h>
#include "gta4_init.h"
#include <CoreFoundation/CoreFoundation.h>
#include <bit>
#include <cmath>

namespace {
namespace blur = theft4::motion_blur;
std::atomic<bool> enabled{true};
std::atomic<bool> trace{false};
std::atomic<uint32_t> events{0};

bool Readable(uint8_t* base, uint32_t address, uint32_t size) {
    if (!base || !address || !size) return false;
    const uint64_t last = uint64_t{address} + size - 1;
    if (last > std::numeric_limits<uint32_t>::max()) return false;
    auto* kernel = REX_KERNEL_STATE();
    auto* memory = kernel ? kernel->memory() : nullptr;
    auto* heap = memory ? memory->LookupHeap(address) : nullptr;
    if (!heap || heap != memory->LookupHeap(static_cast<uint32_t>(last))) return false;
    const auto access = heap->QueryRangeAccess(address, static_cast<uint32_t>(last));
    using rex::memory::PageAccess;
    return access == PageAccess::kReadWrite || access == PageAccess::kReadOnly ||
           access == PageAccess::kExecuteReadWrite || access == PageAccess::kExecuteReadOnly;
}
}  // namespace

// Theft4-MetalFX: city density. The iOS target does not link the desktop
// population hooks, so rescale GTA IV's population multiplier globals from
// this per-frame boundary. A value is treated as a fresh game request only
// when it differs from what we last wrote; zero and negatives are preserved.
namespace citydensity {
constexpr uint32_t kRandomCarDensity = 0x82AA1888;
constexpr uint32_t kParkedCarDensity = 0x82AA188C;
constexpr uint32_t kPedDensity = 0x82AA2940;
constexpr uint32_t kScenarioPedCurrent = 0x82AA2944;
constexpr uint32_t kScenarioPedNext = 0x82AA2948;
const double kLegacyScales[] = {0.35, 0.6, 0.8, 1.0};  // old index keys

// Percent key (5..100) written by the launcher; falls back to the older
// index key, then to Medium (60%). Values outside 1..100 are ignored.
double ReadScale(CFStringRef percent_key, CFStringRef legacy_key) {
    Boolean ok = false;
    const CFIndex pct = CFPreferencesGetAppIntegerValue(percent_key, kCFPreferencesCurrentApplication, &ok);
    if (ok && pct >= 1 && pct <= 100) return double(pct) / 100.0;
    ok = false;
    const CFIndex v = CFPreferencesGetAppIntegerValue(legacy_key, kCFPreferencesCurrentApplication, &ok);
    if (ok && v >= 0 && v <= 3) return kLegacyScales[v];
    return 0.6;
}

// Each slot remembers the game's own request (the value it last stored) and
// what we wrote, so the scale can change at runtime (governor / benchmark)
// and stock (100%) restores the game's value exactly.
struct Slot { uint32_t address; uint32_t written = 0; float requested = 0; bool have = false; };
Slot slots[] = {{kRandomCarDensity}, {kParkedCarDensity}, {kPedDensity},
                {kScenarioPedCurrent}, {kScenarioPedNext}};
std::atomic<int> state{0};  // 0 unknown, 1 ready, -1 unreadable
// Runtime scales in per-mille; generation bumps on every change.
std::atomic<int> traffic_pm{-1}, ped_pm{-1};
std::atomic<uint32_t> generation{1};
uint32_t applied_generation = 0;  // game thread only
double traffic_scale = 1.0, ped_scale = 1.0;

void Apply(uint8_t* base) {
    int st = state.load(std::memory_order_relaxed);
    if (st < 0) return;
    if (st == 0) {
        if (traffic_pm.load(std::memory_order_relaxed) < 0) {
            traffic_pm.store(int(ReadScale(CFSTR("Theft4TrafficDensityPct"), CFSTR("Theft4TrafficDensity")) * 1000 + 0.5));
            ped_pm.store(int(ReadScale(CFSTR("Theft4PedDensityPct"), CFSTR("Theft4PedDensity")) * 1000 + 0.5));
        }
        const bool ok = Readable(base, kRandomCarDensity, 8) && Readable(base, kPedDensity, 12);
        state.store(ok ? 1 : -1, std::memory_order_relaxed);
        if (!ok) return;
    }
    const uint32_t gen = generation.load(std::memory_order_acquire);
    const bool rescale = gen != applied_generation;
    if (rescale) {
        applied_generation = gen;
        traffic_scale = traffic_pm.load(std::memory_order_relaxed) / 1000.0;
        ped_scale = ped_pm.load(std::memory_order_relaxed) / 1000.0;
    }
    // Stock everywhere and nothing of ours left in memory: nothing to do.
    if (traffic_scale == 1.0 && ped_scale == 1.0 && !rescale) {
        bool any = false;
        for (const Slot& s : slots) any = any || s.have;
        if (!any) return;
    }
    for (int i = 0; i < 5; ++i) {
        Slot& s = slots[i];
        const double scale = i < 2 ? traffic_scale : ped_scale;
        const uint32_t raw = REX_LOAD_U32(s.address);
        if (s.have && raw == s.written) {
            if (!rescale) continue;
            // Our value is still in place: rescale the remembered request.
        } else {
            const float requested = std::bit_cast<float>(raw);
            if (!std::isfinite(requested) || requested <= 0.0f) { s.have = false; continue; }
            s.requested = requested;
        }
        const float scaled = static_cast<float>(s.requested * scale);
        const uint32_t out = std::bit_cast<uint32_t>(scaled);
        if (out != raw) REX_STORE_U32(s.address, out);
        s.written = out;
        s.have = scale != 1.0;  // at 100% the game's own value is back in place
    }
}
}  // namespace citydensity

// Runtime density (percent of the game's own request, 1..100). Takes effect
// at the next population update; existing cars/peds are not deleted, the
// spawners simply create fewer (or more) from then on.
extern "C" void theft4_city_density_set(int traffic_percent, int ped_percent) {
    auto clamp = [](int v) { return v < 1 ? 1 : v > 100 ? 100 : v; };
    citydensity::traffic_pm.store(clamp(traffic_percent) * 10, std::memory_order_relaxed);
    citydensity::ped_pm.store(clamp(ped_percent) * 10, std::memory_order_relaxed);
    citydensity::generation.fetch_add(1, std::memory_order_release);
}

// Read-only snapshot for Long Performance Capture: configured scales, state
// (0 not yet applied, 1 active, -1 unreadable) and the last value written in
// each guest multiplier (random cars, parked cars, peds, scenario cur/next).
extern "C" int theft4_city_density_snapshot(double* traffic, double* peds, float* values5) {
    const int tp = citydensity::traffic_pm.load(std::memory_order_relaxed);
    const int pp = citydensity::ped_pm.load(std::memory_order_relaxed);
    if (traffic) *traffic = tp >= 0 ? tp / 1000.0 : citydensity::traffic_scale;
    if (peds) *peds = pp >= 0 ? pp / 1000.0 : citydensity::ped_scale;
    const int st = citydensity::state.load(std::memory_order_relaxed);
    if (values5) {
        for (int i = 0; i < 5; ++i)
            values5[i] = citydensity::slots[i].have ? std::bit_cast<float>(citydensity::slots[i].written) : -1.0f;
    }
    return st;
}

void theft4::motion_blur::Configure(bool value, bool diagnostics) noexcept {
    enabled.store(value, std::memory_order_relaxed);
    trace.store(diagnostics, std::memory_order_relaxed);
    events.store(0, std::memory_order_relaxed);
}

// The iOS target does not link desktop gta4_presentation_hooks.cpp. Intercept
// this same pass-selection boundary, not a shader binding: the original helper
// installs the selected variant's own shader, constants and sampler layout.
extern "C" void sub_822CF300(PPCContext& ctx, uint8_t* base) {
    citydensity::Apply(base);
    const bool use_blur = enabled.load(std::memory_order_relaxed);
    const bool observe = trace.load(std::memory_order_relaxed);
    const uint32_t requested = ctx.r6.u32;
    if (ctx.lr != blur::kCompositeCaller || ctx.r4.u32 != 0 ||
        (use_blur && !observe)) {
        __imp__sub_822CF300(ctx, base);
        return;
    }
    const uint32_t postfx = ctx.r3.u32;
    bool valid = false;
    if (Readable(base, postfx, blur::kTechniqueOffset + sizeof(uint32_t))) {
        const uint32_t effect = REX_LOAD_U32(postfx + 108);
        valid = ctx.r5.u32 != 0 &&
                ctx.r5.u32 == REX_LOAD_U32(postfx + blur::kTechniqueOffset) &&
                Readable(base, effect, 28);
    }
    const uint32_t selected =
        blur::SelectPass(requested, use_blur, static_cast<uint32_t>(ctx.lr), ctx.r4.u32, valid);
    // Opt-in, bounded evidence only; no per-frame retail logging.
    if (observe && events.fetch_add(1, std::memory_order_relaxed) < 32) {
        REXLOG_INFO("Theft4 motion-blur: enabled={} requested={} selected={} valid={}",
                    use_blur, requested, selected, valid);
    }
    if (selected != requested) ctx.r6.u64 = selected;
    __imp__sub_822CF300(ctx, base);
}
