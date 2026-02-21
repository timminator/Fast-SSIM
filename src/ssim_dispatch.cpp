#pragma warning(disable:4996)
#include "ssim.h"
#include <stdlib.h>
#include "thread_pool.h"

#ifdef _MSC_VER
    #include <intrin.h>
#endif

// --- ThreadPool Implementation ---
unsigned int GetHardwareThreadCount() {
    unsigned int threads = std::thread::hardware_concurrency();
    return (threads == 0) ? 4 : threads;
}

ThreadPool::ThreadPool(size_t threads) : stop(false) {
    for(size_t i = 0; i<threads; ++i)
        workers.emplace_back([this] {
            for(;;) {
                std::function<void()> task;
                {
                    std::unique_lock<std::mutex> lock(this->queue_mutex);
                    this->condition.wait(lock, [this]{ return this->stop || !this->tasks.empty(); });
                    if(this->stop && this->tasks.empty()) return;
                    task = std::move(this->tasks.front());
                    this->tasks.pop();
                }
                task();
            }
        });
}

ThreadPool::~ThreadPool() {
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        stop = true;
    }
    condition.notify_all();
    for(std::thread &worker: workers) worker.join();
}

ThreadPool* global_pool = nullptr;
std::once_flag pool_init_flag;

void ensure_pool_initialized() {
    std::call_once(pool_init_flag, [](){
        global_pool = new ThreadPool(GetHardwareThreadCount());
    });
}

// --- CPU Feature Detection ---
#ifdef _MSC_VER

// Helper for OS support check (XCR0)
static unsigned long long safe_xgetbv() {
    return _xgetbv(_XCR_XFEATURE_ENABLED_MASK);
}

extern "C" int CheckCpuSupport() {
    if (getenv("SSIM_FORCE_SSE") != nullptr) return 2;

    int regs[4] = {0};

    // 1. Check CPU basics (AVX)
    __cpuid(regs, 1);
    bool has_avx = (regs[2] & (1 << 28)) != 0;
    bool has_osxsave = (regs[2] & (1 << 27)) != 0;
    bool has_fma = (regs[2] & (1 << 12)) != 0;
    if (!has_osxsave || !has_avx) return 1;

    // 2. Check OS YMM state support (XCR0)
    if ((safe_xgetbv() & 6) != 6) return 1; 

    // 3. Check AVX2
    __cpuidex(regs, 7, 0);
    bool has_avx2 = (regs[1] & (1 << 5)) != 0;

    if (!has_avx2) return 2; // Missing AVX2
    if (!has_fma)  return 3; // Missing FMA

    return 0;
}
#else
extern "C" int CheckCpuSupport() {
    if (getenv("SSIM_FORCE_SSE") != nullptr) return 2;

    if (!__builtin_cpu_supports("avx"))  return 1;
    if (!__builtin_cpu_supports("avx2")) return 2;
    if (!__builtin_cpu_supports("fma"))  return 3;

    return 0;
}
#endif

// --- Forward Declarations from namespaces ---
namespace avx2_impl {
    double MSE_Byte_Impl(Byte* pX, Byte* pY, int s, int w, int h);
    double MSE_Float_Impl(float* pX, float* pY, int s, int w, int h);
    float PSNR_Byte_Impl(Byte* pX, Byte* pY, int s, int w, int h, int m);
    float PSNR_Float_Impl(float* pX, float* pY, int s, int w, int h, double m);
    float SSIM_Byte_Impl(Byte* pX, Byte* pY, int wb, int w, int h, int ws, int m);
    float SSIM_Float_Impl(float* pX, float* pY, int s, int w, int h, int ws, double m);
    float SSIM_Byte_Slow_Impl(Byte* pX, Byte* pY, int wb, int w, int h, int ws);
}

namespace sse_impl {
    double MSE_Byte_Impl(Byte* pX, Byte* pY, int s, int w, int h);
    double MSE_Float_Impl(float* pX, float* pY, int s, int w, int h);
    float PSNR_Byte_Impl(Byte* pX, Byte* pY, int s, int w, int h, int m);
    float PSNR_Float_Impl(float* pX, float* pY, int s, int w, int h, double m);
    float SSIM_Byte_Impl(Byte* pX, Byte* pY, int wb, int w, int h, int ws, int m);
    float SSIM_Float_Impl(float* pX, float* pY, int s, int w, int h, int ws, double m);
    float SSIM_Byte_Slow_Impl(Byte* pX, Byte* pY, int wb, int w, int h, int ws);
}

// --- Function Pointers ---
static double (*g_MSE_Byte)(Byte*, Byte*, int, int, int) = nullptr;
static double (*g_MSE_Float)(float*, float*, int, int, int) = nullptr;
static float (*g_PSNR_Byte)(Byte*, Byte*, int, int, int, int) = nullptr;
static float (*g_PSNR_Float)(float*, float*, int, int, int, double) = nullptr;
static float (*g_SSIM_Byte)(Byte*, Byte*, int, int, int, int, int) = nullptr;
static float (*g_SSIM_Float)(float*, float*, int, int, int, int, double) = nullptr;
static float (*g_SSIM_Byte_Slow)(Byte*, Byte*, int, int, int, int) = nullptr;

// --- Dispatch Initialization ---
static bool InitDispatch() {
    if (CheckCpuSupport() == 0) {
        // CPU supports AVX2 + FMA
        g_MSE_Byte = avx2_impl::MSE_Byte_Impl;
        g_MSE_Float = avx2_impl::MSE_Float_Impl;
        g_PSNR_Byte = avx2_impl::PSNR_Byte_Impl;
        g_PSNR_Float = avx2_impl::PSNR_Float_Impl;
        g_SSIM_Byte = avx2_impl::SSIM_Byte_Impl;
        g_SSIM_Float = avx2_impl::SSIM_Float_Impl;
        g_SSIM_Byte_Slow = avx2_impl::SSIM_Byte_Slow_Impl;
    } else {
        // Fallback to SSE
        g_MSE_Byte = sse_impl::MSE_Byte_Impl;
        g_MSE_Float = sse_impl::MSE_Float_Impl;
        g_PSNR_Byte = sse_impl::PSNR_Byte_Impl;
        g_PSNR_Float = sse_impl::PSNR_Float_Impl;
        g_SSIM_Byte = sse_impl::SSIM_Byte_Impl;
        g_SSIM_Float = sse_impl::SSIM_Float_Impl;
        g_SSIM_Byte_Slow = sse_impl::SSIM_Byte_Slow_Impl;
    }
    return true;
}

// Run dispatcher once globally before main()
static bool g_dispatch_initialized = InitDispatch();

// --- Public API Wrappers ---
double MSE_Byte(Byte* pX, Byte* pY, int s, int w, int h) { return g_MSE_Byte(pX, pY, s, w, h); }
double MSE_Float(float* pX, float* pY, int s, int w, int h) { return g_MSE_Float(pX, pY, s, w, h); }
float PSNR_Byte(Byte* pX, Byte* pY, int s, int w, int h, int m) { return g_PSNR_Byte(pX, pY, s, w, h, m); }
float PSNR_Float(float* pX, float* pY, int s, int w, int h, double m) { return g_PSNR_Float(pX, pY, s, w, h, m); }
float SSIM_Byte(Byte* pX, Byte* pY, int wb, int w, int h, int ws, int m) { return g_SSIM_Byte(pX, pY, wb, w, h, ws, m); }
float SSIM_Float(float* pX, float* pY, int s, int w, int h, int ws, double m) { return g_SSIM_Float(pX, pY, s, w, h, ws, m); }
float SSIM_Byte_Slow(Byte* pX, Byte* pY, int wb, int w, int h, int ws) { return g_SSIM_Byte_Slow(pX, pY, wb, w, h, ws); }
