#include "ssim.h"
#include "math.h"
#include "stdio.h"
#include "emmintrin.h"
#include "tmmintrin.h"
#include "string.h"

// OpenMP Support
#if defined(_OPENMP)
    #include <omp.h>
#endif

// AVX2 Support
#if defined(__AVX2__)
    #include <immintrin.h>
#endif

// CPU Feature Detection
#ifdef _MSC_VER
    #include <intrin.h>
    
    // Helper for OS support check (XCR0)
    static unsigned long long safe_xgetbv() {
        return _xgetbv(_XCR_XFEATURE_ENABLED_MASK);
    }

    extern "C" __declspec(dllexport) int CheckCpuSupport() {
        int regs[4] = {0};

        // 1. Check CPU basics (AVX)
        __cpuid(regs, 1);
        bool has_avx = (regs[2] & (1 << 28)) != 0;
        bool has_osxsave = (regs[2] & (1 << 27)) != 0;
        bool has_fma = (regs[2] & (1 << 12)) != 0;

        if (!has_osxsave || !has_avx) return 1;

        // 2. Check OS YMM state support (XCR0)
        unsigned long long xcr0 = safe_xgetbv();
        if ((xcr0 & 6) != 6) return 1; 

        // 3. Check AVX2
        __cpuidex(regs, 7, 0);
        bool has_avx2 = (regs[1] & (1 << 5)) != 0;

        if (!has_avx2) return 2; // Missing AVX2
        if (!has_fma)  return 3; // Missing FMA

        return 0;
    }

#else
    extern "C" __attribute__((visibility("default"))) int CheckCpuSupport() {
        if (!__builtin_cpu_supports("avx"))  return 1;
        if (!__builtin_cpu_supports("avx2")) return 2;
        if (!__builtin_cpu_supports("fma"))  return 3;
        
        return 0;
    }

#endif

#define MAX(a, b) ((a)<(b)?(b):(a))
#define MIN(a, b) ((a)<(b)?(a):(b))
#define ABS(a) ((a)<0?-(a):(a))
#define SQR(x) ((x)*(x))
#define AlignBytes(x) (((x)+31)/32*32) 

// --- SSE Helpers ---
static inline __m128 GetF32x4(__m128i u16)
{
    __m128i zero=_mm_setzero_si128();
    __m128i x8_0 = _mm_unpacklo_epi8(u16, zero);
    return _mm_cvtepi32_ps(_mm_unpacklo_epi16(x8_0, zero));
}

static inline __m128 LoadF32x4(Byte pData[8])
{
    return GetF32x4(_mm_loadl_epi64((__m128i*)pData));
}

// --- AVX2 Helpers ---
#if defined(__AVX2__)
static inline __m256 GetF32x8(__m128i u8_16) 
{
    __m256i u16 = _mm256_cvtepu8_epi32(u8_16); 
    return _mm256_cvtepi32_ps(u16);            
}

static inline __m256 LoadF32x8(Byte* pData)
{
    return GetF32x8(_mm_loadl_epi64((__m128i*)pData)); 
}
#endif

// --- MSE Implementations ---

template <typename Data, typename T>
static double MSE_Data_t(Data* pDataX, Data* pDataY, int widthBytes, int width, int height)
{
    double sum=0;
    int cn=widthBytes/width, width3=cn*width, width4=width3/4*4;
    
    #pragma omp parallel for reduction(+:sum)
    for(int y=0; y<height; y++)
    {
        Data *pX=pDataX+y*widthBytes;
        Data *pY=pDataY+y*widthBytes;
        T s_local = 0;
        for(int x=0; x<width4; x+=4)
        {
            T diff0=pY[x+0]-pX[x+0];
            T diff1=pY[x+1]-pX[x+1];
            T diff2=pY[x+2]-pX[x+2];
            T diff3=pY[x+3]-pX[x+3];
            s_local += diff0*diff0 + diff1*diff1 + diff2*diff2 + diff3*diff3;
        }
        for(int x=width4; x<width3; x++)
        {
            T diff=pY[x]-pX[x];
            s_local += diff*diff;
        }
        sum += s_local;
    }
    return sum/(height*width3);
}

static double MSE_Data_Byte(Byte* pDataX, Byte* pDataY, int widthBytes, int width, int height)
{
    double sum=0;
    int width3=widthBytes/width*width;
    
#if defined(__AVX2__)
    int width32 = width3 / 32 * 32;
    #pragma omp parallel for reduction(+:sum)
    for(int y=0; y<height; y++)
    {
        Byte *pX=pDataX+y*widthBytes;
        Byte *pY=pDataY+y*widthBytes;
        int x=0;
        __m256i s_acc = _mm256_setzero_si256(); 
        
        for(; x<width32; x+=32)
        {
            __m256i x_vec = _mm256_loadu_si256((__m256i*)(pX+x));
            __m256i y_vec = _mm256_loadu_si256((__m256i*)(pY+x));
            
            __m256i x_lo = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(x_vec, 0));
            __m256i x_hi = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(x_vec, 1));
            __m256i y_lo = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(y_vec, 0));
            __m256i y_hi = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(y_vec, 1));

            __m256i diff_lo = _mm256_sub_epi16(x_lo, y_lo);
            __m256i diff_hi = _mm256_sub_epi16(x_hi, y_hi);

            s_acc = _mm256_add_epi32(s_acc, _mm256_madd_epi16(diff_lo, diff_lo));
            s_acc = _mm256_add_epi32(s_acc, _mm256_madd_epi16(diff_hi, diff_hi));
        }

        int s0 = 0;
        int* temp = (int*)&s_acc;
        for(int k=0; k<8; k++) s0 += temp[k];

        for(; x<width3; x++)
        {
            int d=pY[x]-pX[x];
            s0 += d*d;
        }
        sum += s0;
    }
#else
    int width8=width3/8*8;
    #pragma omp parallel for reduction(+:sum)
    for(int y=0; y<height; y++)
    {
        Byte *pX=pDataX+y*widthBytes;
        Byte *pY=pDataY+y*widthBytes;
        int x=0;
        __m128i zero=_mm_setzero_si128();
        __m128i s4 = _mm_setzero_si128();
        int s0=0;
        for(x=0; x<width8; x+=8)
        {
            __m128i x8 = _mm_unpacklo_epi8(_mm_loadl_epi64((__m128i*)(pX+x)), zero);
            __m128i y8 = _mm_unpacklo_epi8(_mm_loadl_epi64((__m128i*)(pY+x)), zero);
            __m128i d8 = _mm_sub_epi16(x8, y8);
            __m128i z8 = _mm_mullo_epi16(d8, d8);
            __m128i z8_0 = _mm_unpacklo_epi16(z8, zero);
            __m128i z8_1 = _mm_unpackhi_epi16(z8, zero);
            s4 = _mm_add_epi32(s4, _mm_add_epi32(z8_0, z8_1));
        }
        int* pS=(int*)&s4;
        s0=pS[0]+pS[1]+pS[2]+pS[3];
        for(; x<width3; x++)
        {
            int d=pY[x]-pX[x];
            s0 += d*d;
        }
        sum += s0;
    }
#endif
    return sum/(height*width3);
}

double MSE_Byte(Byte* pDataX, Byte* pDataY, int step, int width, int height)
{
    return MSE_Data_Byte(pDataX, pDataY, step, width, height);
}
double MSE_Float(float* pDataX, float* pDataY, int step, int width, int height)
{
    return MSE_Data_t<float, float>(pDataX, pDataY, step, width, height);
}

template <typename Data, typename T>
static float PSNR_Data_t(Data* pDataX, Data* pDataY, int widthBytes, int width, int height, double maxVal)
{
    double mse = MSE_Data_t<Data, T>(pDataX, pDataY, widthBytes, width, height);
    mse = MAX(mse, 1e-10);
    return (float)(10*log10(maxVal*maxVal/mse));
}
static float PSNR_Data_Byte(Byte* pDataX, Byte* pDataY, int step, int width, int height, int maxVal)
{
    double mse=MSE_Data_Byte(pDataX, pDataY, step, width, height);
    mse = MAX(mse, 1e-10);
    return (float)(10*log10(maxVal*maxVal/mse));
}
float PSNR_Float(float* pDataX, float* pDataY, int step, int width, int height, double maxVal)
{
    return PSNR_Data_t<float, float>(pDataX, pDataY, step, width, height, maxVal);
}
float PSNR_Byte(Byte* pDataX, Byte* pDataY, int step, int width, int height, int maxVal)
{
    return PSNR_Data_Byte(pDataX, pDataY, step, width, height, maxVal);
}

// --- SSIM Processing Class ---
template<typename Data>
struct WinSum3F
{
    float *pSumX, *pSumY;
    float *pSumXX, *pSumXY, *pSumYY;
    int win_size;
    int widthBytes, width, height;
    Data *pDataX, *pDataY;
    int bufLen, cn;
    float maxVal;
    
    WinSum3F(Data* pDataX, Data* pDataY, int widthBytes, int width, int height, int win_size, double maxVal)
    {
        this->win_size=win_size;
        this->width=width;
        this->height=height;
        this->cn=widthBytes/width;
        this->pDataX=pDataX;
        this->pDataY=pDataY;
        this->widthBytes=widthBytes;
        this->maxVal=(float)maxVal;

        this->bufLen=AlignBytes((width+win_size)*cn+64); 
        this->pSumX=new float[bufLen*5];
        this->pSumY=this->pSumX+bufLen;
        this->pSumXX=this->pSumX+bufLen*2;
        this->pSumXY=this->pSumX+bufLen*3;
        this->pSumYY=this->pSumX+bufLen*4;
        
        memset(this->pSumX, 0, bufLen*5*sizeof(float));
    }
    
    ~WinSum3F()
    {
        if(this->pSumX) delete[] this->pSumX;
    }

    void ProcessStrip(int start_y, int end_y, double &outSum, int &outCount)
    {
        InitWindow(start_y); 
        if (cn == 3) DoFilter3_Loop(start_y, end_y, outSum, outCount);
        else         DoFilter1_Loop(start_y, end_y, outSum, outCount);
    }

private:
    inline void InitWindow(int y)
    {
        for(int i=0; i<bufLen*5; i++) pSumX[i]=0;

        int half = win_size/2;
        int row_start = y - half;
        int row_end = y + half;
        
        for (int r = row_start; r <= row_end; r++)
        {
            if (r >= 0 && r < height) 
            {
                Data *pX = pDataX + r * widthBytes;
                Data *pY = pDataY + r * widthBytes;
                for (int x = 0; x < widthBytes; x++)
                {
                    float valX = (float)pX[x];
                    float valY = (float)pY[x];
                    pSumX[x] += valX;
                    pSumY[x] += valY;
                    pSumXX[x] += valX * valX;
                    pSumXY[x] += valX * valY;
                    pSumYY[x] += valY * valY;
                }
            }
        }
    }

    void DoFilter1_Loop(int start_y, int end_y, double &sum, int &count)
    {
        float k1=0.01f, k2=0.03f, c1=SQR(k1*maxVal), c2=SQR(k2*maxVal);
        int half_size=win_size/2;
        float invA=(float)(1.0/(win_size*win_size));
        float conv_norm=(float)(win_size*win_size)/(win_size*win_size-1);

        for(int y=start_y; y<end_y; y++)
        {
            float s=0;
            float sumX=0, sumY=0, sumXX=0, sumXY=0, sumYY=0;
            
            for(int i=0; i<win_size; i++)
            {
                sumX+=pSumX[i]; sumY+=pSumY[i];
                sumXX+=pSumXX[i]; sumXY+=pSumXY[i]; sumYY+=pSumYY[i];
            }

            for(int x=half_size; x<width-half_size; x++)
            {
                float meanX=sumX*invA, meanY=sumY*invA;
                float meanXX=meanX*meanX, meanXY=meanX*meanY, meanYY=meanY*meanY;
                float sigmaXX=conv_norm*(sumXX*invA-meanXX);
                float sigmaXY=conv_norm*(sumXY*invA-meanXY);
                float sigmaYY=conv_norm*(sumYY*invA-meanYY);
                float u1=2*meanXY+c1, u2=meanXX+meanYY+c1;
                float v1=2*sigmaXY+c2, v2=sigmaXX+sigmaYY+c2;
                sum += (u1*v1)/(u2*v2);
                count++;

                int i=(x+half_size+1), j=(x-half_size);
                sumX += pSumX[i]-pSumX[j];
                sumY += pSumY[i]-pSumY[j];
                sumXX += pSumXX[i]-pSumXX[j];
                sumXY += pSumXY[i]-pSumXY[j];
                sumYY += pSumYY[i]-pSumYY[j];
            }
            
            if(sizeof(Data)==1) UpdateNextRow_Byte(y+1);
            else UpdateNextRow(y+1);
        }
    }

    void DoFilter3_Loop(int start_y, int end_y, double &sum, int &count)
    {
        float k1=0.01f, k2=0.03f, c1=SQR(k1*maxVal), c2=SQR(k2*maxVal);
        int half_size=win_size/2;
        float invA=(float)(1.0/(win_size*win_size));
        float conv_norm=(float)(win_size*win_size)/(win_size*win_size-1);

#if defined(__AVX2__)
        for(int y=start_y; y<end_y; y++)
        {
            __m256 s8 = _mm256_setzero_ps();
            __m256 invA8=_mm256_set1_ps(invA), conv_norm8=_mm256_set1_ps(conv_norm);
            __m256 c1_8 = _mm256_set1_ps(c1), c2_8 = _mm256_set1_ps(c2);
            __m256 two8 = _mm256_set1_ps(2.0f);

            __m256 sumX8 = _mm256_loadu_ps(pSumX);
            __m256 sumY8 = _mm256_loadu_ps(pSumY);
            __m256 sumXX8 = _mm256_loadu_ps(pSumXX); 
            __m256 sumXY8 = _mm256_loadu_ps(pSumXY);
            __m256 sumYY8 = _mm256_loadu_ps(pSumYY);

            for(int i=3; i<win_size*3; i+=3)
            {
                sumX8 = _mm256_add_ps(_mm256_loadu_ps(pSumX+i), sumX8);
                sumY8 = _mm256_add_ps(_mm256_loadu_ps(pSumY+i), sumY8);
                sumXX8 = _mm256_add_ps(_mm256_loadu_ps(pSumXX+i), sumXX8);
                sumXY8 = _mm256_add_ps(_mm256_loadu_ps(pSumXY+i), sumXY8);
                sumYY8 = _mm256_add_ps(_mm256_loadu_ps(pSumYY+i), sumYY8);
            }

            int x;
            int x_limit = width - half_size - 1; 
            for(x=half_size; x<x_limit; x+=2)
            {
                __m256 meanX8 = _mm256_mul_ps(sumX8, invA8);
                __m256 meanY8 = _mm256_mul_ps(sumY8, invA8);
                __m256 meanXX8 = _mm256_mul_ps(meanX8, meanX8);
                __m256 meanXY8 = _mm256_mul_ps(meanX8, meanY8);
                __m256 meanYY8 = _mm256_mul_ps(meanY8, meanY8);

#ifdef __FMA__
                __m256 T_XX = _mm256_mul_ps(sumXX8, invA8);
                __m256 T_XY = _mm256_mul_ps(sumXY8, invA8);
                __m256 T_YY = _mm256_mul_ps(sumYY8, invA8);
                __m256 sigmaXX8 = _mm256_mul_ps(_mm256_sub_ps(T_XX, meanXX8), conv_norm8);
                __m256 sigmaXY8 = _mm256_mul_ps(_mm256_sub_ps(T_XY, meanXY8), conv_norm8);
                __m256 sigmaYY8 = _mm256_mul_ps(_mm256_sub_ps(T_YY, meanYY8), conv_norm8);
                __m256 u1_8 = _mm256_fmadd_ps(meanXY8, two8, c1_8);
                __m256 v1_8 = _mm256_fmadd_ps(sigmaXY8, two8, c2_8);
#else
                __m256 sigmaXX8 = _mm256_mul_ps(_mm256_sub_ps(_mm256_mul_ps(sumXX8, invA8), meanXX8), conv_norm8);
                __m256 sigmaXY8 = _mm256_mul_ps(_mm256_sub_ps(_mm256_mul_ps(sumXY8, invA8), meanXY8), conv_norm8);
                __m256 sigmaYY8 = _mm256_mul_ps(_mm256_sub_ps(_mm256_mul_ps(sumYY8, invA8), meanYY8), conv_norm8);
                __m256 u1_8 = _mm256_add_ps(_mm256_add_ps(meanXY8, meanXY8), c1_8);
                __m256 v1_8 = _mm256_add_ps(_mm256_add_ps(sigmaXY8, sigmaXY8), c2_8);
#endif
                __m256 u2_8 = _mm256_add_ps(_mm256_add_ps(meanXX8, meanYY8), c1_8);
                __m256 v2_8 = _mm256_add_ps(_mm256_add_ps(sigmaXX8, sigmaYY8), c2_8);
                __m256 z8 = _mm256_div_ps(_mm256_mul_ps(u1_8, v1_8), _mm256_mul_ps(u2_8, v2_8));

                s8 = _mm256_add_ps(s8, z8);
                count += 6;

                int i=3*(x+half_size+1);
                int j=3*(x-half_size);
                
                __m256 add_cols = _mm256_add_ps(_mm256_loadu_ps(pSumX + i), _mm256_loadu_ps(pSumX + i + 3));
                __m256 sub_cols = _mm256_add_ps(_mm256_loadu_ps(pSumX + j), _mm256_loadu_ps(pSumX + j + 3));
                sumX8 = _mm256_add_ps(sumX8, _mm256_sub_ps(add_cols, sub_cols));

                add_cols = _mm256_add_ps(_mm256_loadu_ps(pSumY + i), _mm256_loadu_ps(pSumY + i + 3));
                sub_cols = _mm256_add_ps(_mm256_loadu_ps(pSumY + j), _mm256_loadu_ps(pSumY + j + 3));
                sumY8 = _mm256_add_ps(sumY8, _mm256_sub_ps(add_cols, sub_cols));

                add_cols = _mm256_add_ps(_mm256_loadu_ps(pSumXX + i), _mm256_loadu_ps(pSumXX + i + 3));
                sub_cols = _mm256_add_ps(_mm256_loadu_ps(pSumXX + j), _mm256_loadu_ps(pSumXX + j + 3));
                sumXX8 = _mm256_add_ps(sumXX8, _mm256_sub_ps(add_cols, sub_cols));

                add_cols = _mm256_add_ps(_mm256_loadu_ps(pSumXY + i), _mm256_loadu_ps(pSumXY + i + 3));
                sub_cols = _mm256_add_ps(_mm256_loadu_ps(pSumXY + j), _mm256_loadu_ps(pSumXY + j + 3));
                sumXY8 = _mm256_add_ps(sumXY8, _mm256_sub_ps(add_cols, sub_cols));

                add_cols = _mm256_add_ps(_mm256_loadu_ps(pSumYY + i), _mm256_loadu_ps(pSumYY + i + 3));
                sub_cols = _mm256_add_ps(_mm256_loadu_ps(pSumYY + j), _mm256_loadu_ps(pSumYY + j + 3));
                sumYY8 = _mm256_add_ps(sumYY8, _mm256_sub_ps(add_cols, sub_cols));
            }
            
            float* pS = (float*)&s8;
            sum += pS[0]+pS[1]+pS[2] + pS[3]+pS[4]+pS[5];

            for(; x<width-half_size; x++)
            {
                 float *pSumX_f = (float*)&sumX8; float *pSumY_f = (float*)&sumY8;
                 float *pSumXX_f = (float*)&sumXX8; float *pSumXY_f = (float*)&sumXY8; float *pSumYY_f = (float*)&sumYY8;
                 
                 for(int k=0; k<3; k++) {
                     float mx = pSumX_f[k]*invA, my = pSumY_f[k]*invA;
                     float mxx = mx*mx, mxy = mx*my, myy = my*my;
                     float sxx = conv_norm * (pSumXX_f[k]*invA - mxx);
                     float sxy = conv_norm * (pSumXY_f[k]*invA - mxy);
                     float syy = conv_norm * (pSumYY_f[k]*invA - myy);
                     float u1 = 2*mxy + c1, v1 = 2*sxy + c2;
                     float u2 = mxx+myy+c1, v2 = sxx+syy+c2;
                     sum += (u1*v1)/(u2*v2);
                     count++;
                 }
            }
            
            if(sizeof(Data)==1) UpdateNextRow_Byte(y+1);
            else UpdateNextRow(y+1);
        }

#else
        // --- Legacy SSE Path ---
        for(int y=start_y; y<end_y; y++)
        {
            __m128 s4 = _mm_set_ps1(0);
            __m128 invA4=_mm_set_ps1(invA), conv_norm4=_mm_set_ps1(conv_norm);
            __m128 sumX4 = _mm_loadu_ps(pSumX), sumY4 = _mm_loadu_ps(pSumY);
            __m128 sumXX4 = _mm_loadu_ps(pSumXX), sumXY4 = _mm_loadu_ps(pSumXY), sumYY4 = _mm_loadu_ps(pSumYY);
            
            for(int i=3; i<win_size*3; i+=3)
            {
                sumX4 = _mm_add_ps(_mm_loadu_ps(pSumX+i), sumX4);
                sumY4 = _mm_add_ps(_mm_loadu_ps(pSumY+i), sumY4);
                sumXX4 = _mm_add_ps(_mm_loadu_ps(pSumXX+i), sumXX4);
                sumXY4 = _mm_add_ps(_mm_loadu_ps(pSumXY+i), sumXY4);
                sumYY4 = _mm_add_ps(_mm_loadu_ps(pSumYY+i), sumYY4);
            }
            
            for(int x=half_size; x<width-half_size; x++)
            {
                __m128 meanX4 = _mm_mul_ps(sumX4, invA4);
                __m128 meanY4 = _mm_mul_ps(sumY4, invA4);
                __m128 meanXX4 = _mm_mul_ps(meanX4, meanX4);
                __m128 meanXY4 = _mm_mul_ps(meanX4, meanY4);
                __m128 meanYY4 = _mm_mul_ps(meanY4, meanY4);
                __m128 sigmaXX4 = _mm_mul_ps(_mm_sub_ps(_mm_mul_ps(sumXX4, invA4), meanXX4), conv_norm4);
                __m128 sigmaXY4 = _mm_mul_ps(_mm_sub_ps(_mm_mul_ps(sumXY4, invA4), meanXY4), conv_norm4);
                __m128 sigmaYY4 = _mm_mul_ps(_mm_sub_ps(_mm_mul_ps(sumYY4, invA4), meanYY4), conv_norm4);

                __m128 c1_4 = _mm_set_ps1(c1), c2_4 = _mm_set_ps1(c2);
                __m128 u1_4 = _mm_add_ps(_mm_add_ps(meanXY4, meanXY4), c1_4);
                __m128 u2_4 = _mm_add_ps(_mm_add_ps(meanXX4, meanYY4), c1_4);
                __m128 v1_4 = _mm_add_ps(_mm_add_ps(sigmaXY4, sigmaXY4), c2_4);
                __m128 v2_4 = _mm_add_ps(_mm_add_ps(sigmaXX4, sigmaYY4), c2_4);
                __m128 z1_4 = _mm_mul_ps(u1_4, v1_4), z2_4 = _mm_mul_ps(u2_4, v2_4);
                __m128 z4 = _mm_div_ps(z1_4, z2_4);

                s4 = _mm_add_ps(s4, z4);
                count += 3;

                int i=3*(x+half_size+1), j=3*(x-half_size);
                sumX4 = _mm_add_ps(_mm_sub_ps(_mm_loadu_ps(pSumX+i), _mm_loadu_ps(pSumX+j)), sumX4);
                sumY4 = _mm_add_ps(_mm_sub_ps(_mm_loadu_ps(pSumY+i), _mm_loadu_ps(pSumY+j)), sumY4);
                sumXX4 = _mm_add_ps(_mm_sub_ps(_mm_loadu_ps(pSumXX+i), _mm_loadu_ps(pSumXX+j)), sumXX4);
                sumXY4 = _mm_add_ps(_mm_sub_ps(_mm_loadu_ps(pSumXY+i), _mm_loadu_ps(pSumXY+j)), sumXY4);
                sumYY4 = _mm_add_ps(_mm_sub_ps(_mm_loadu_ps(pSumYY+i), _mm_loadu_ps(pSumYY+j)), sumYY4);
            }
            float *pS=(float*)&s4;
            sum += pS[0]+pS[1]+pS[2];
            if(sizeof(Data)==1)
                UpdateNextRow_Byte(y+1);
            else
                UpdateNextRow(y+1);
        }
#endif
    }

    inline void UpdateNextRow(int y)
    {
        int half_win=win_size/2;
        if(y+half_win<height)
        {
            Data *pX0=pDataX+(y-1-half_win)*widthBytes, *pX1=pDataX+(y+half_win)*widthBytes;
            Data *pY0=pDataY+(y-1-half_win)*widthBytes, *pY1=pDataY+(y+half_win)*widthBytes;
            int x=0;
#if defined(__AVX2__)
            int width8 = widthBytes/8*8;
            for(; x<width8; x+=8)
            {
                 __m256 x0_8 = _mm256_loadu_ps((const float*)(pX0+x));
                 __m256 x1_8 = _mm256_loadu_ps((const float*)(pX1+x));
                 __m256 y0_8 = _mm256_loadu_ps((const float*)(pY0+x));
                 __m256 y1_8 = _mm256_loadu_ps((const float*)(pY1+x));
                 
                 _mm256_storeu_ps(pSumX+x, _mm256_add_ps(_mm256_loadu_ps(pSumX+x), _mm256_sub_ps(x1_8, x0_8)));
                 _mm256_storeu_ps(pSumY+x, _mm256_add_ps(_mm256_loadu_ps(pSumY+x), _mm256_sub_ps(y1_8, y0_8)));
#ifdef __FMA__
                 __m256 xx_diff = _mm256_fmsub_ps(x1_8, x1_8, _mm256_mul_ps(x0_8, x0_8));
                 __m256 xy_diff = _mm256_fmsub_ps(x1_8, y1_8, _mm256_mul_ps(x0_8, y0_8));
                 __m256 yy_diff = _mm256_fmsub_ps(y1_8, y1_8, _mm256_mul_ps(y0_8, y0_8));
#else
                 __m256 xx_diff = _mm256_sub_ps(_mm256_mul_ps(x1_8, x1_8), _mm256_mul_ps(x0_8, x0_8));
                 __m256 xy_diff = _mm256_sub_ps(_mm256_mul_ps(x1_8, y1_8), _mm256_mul_ps(x0_8, y0_8));
                 __m256 yy_diff = _mm256_sub_ps(_mm256_mul_ps(y1_8, y1_8), _mm256_mul_ps(y0_8, y0_8));
#endif
                 _mm256_storeu_ps(pSumXX+x, _mm256_add_ps(_mm256_loadu_ps(pSumXX+x), xx_diff));
                 _mm256_storeu_ps(pSumXY+x, _mm256_add_ps(_mm256_loadu_ps(pSumXY+x), xy_diff));
                 _mm256_storeu_ps(pSumYY+x, _mm256_add_ps(_mm256_loadu_ps(pSumYY+x), yy_diff));
            }
#else
            int width4=widthBytes/4*4;
            for(x=0; x<width4; x+=4)
            {
                __m128 x0_4 = _mm_set_ps(pX0[x+3], pX0[x+2], pX0[x+1], pX0[x+0]);
                __m128 x1_4 = _mm_set_ps(pX1[x+3], pX1[x+2], pX1[x+1], pX1[x+0]);
                __m128 y0_4 = _mm_set_ps(pY0[x+3], pY0[x+2], pY0[x+1], pY0[x+0]);
                __m128 y1_4 = _mm_set_ps(pY1[x+3], pY1[x+2], pY1[x+1], pY1[x+0]);
                _mm_storeu_ps(pSumX+x, _mm_add_ps(_mm_loadu_ps(pSumX+x), _mm_sub_ps(x1_4, x0_4)));
                _mm_storeu_ps(pSumY+x, _mm_add_ps(_mm_loadu_ps(pSumY+x), _mm_sub_ps(y1_4, y0_4)));
                _mm_storeu_ps(pSumXX+x, _mm_add_ps(_mm_loadu_ps(pSumXX+x), _mm_sub_ps(_mm_mul_ps(x1_4, x1_4), _mm_mul_ps(x0_4, x0_4))));
                _mm_storeu_ps(pSumXY+x, _mm_add_ps(_mm_loadu_ps(pSumXY+x), _mm_sub_ps(_mm_mul_ps(x1_4, y1_4), _mm_mul_ps(x0_4, y0_4))));
                _mm_storeu_ps(pSumYY+x, _mm_add_ps(_mm_loadu_ps(pSumYY+x), _mm_sub_ps(_mm_mul_ps(y1_4, y1_4), _mm_mul_ps(y0_4, y0_4))));
            }
#endif
            for(; x<widthBytes; x++)
            {
                float x1=pX1[x], x0=pX0[x], y0=pY0[x], y1=pY1[x];
                pSumX[x] += x1-x0;
                pSumY[x] += y1-y0;
                pSumXX[x]+= x1*x1-x0*x0;
                pSumXY[x]+= x1*y1-x0*y0;
                pSumYY[x]+= y1*y1-y0*y0;
            }
        }
    }
    
    inline void UpdateNextRow_Byte(int y)
    {
        int half_win=win_size/2;
        if(y+half_win<height)
        {
            Data *pX0=pDataX+(y-1-half_win)*widthBytes, *pX1=pDataX+(y+half_win)*widthBytes;
            Data *pY0=pDataY+(y-1-half_win)*widthBytes, *pY1=pDataY+(y+half_win)*widthBytes;
            int x=0;
#if defined(__AVX2__)
            int width8=(widthBytes)/8*8;
            for(x=0; x<width8; x+=8)
            {
                __m256 x0_8 = LoadF32x8((Byte*)(pX0+x));
                __m256 x1_8 = LoadF32x8((Byte*)(pX1+x));
                __m256 y0_8 = LoadF32x8((Byte*)(pY0+x));
                __m256 y1_8 = LoadF32x8((Byte*)(pY1+x));
                
                _mm256_storeu_ps(pSumX+x, _mm256_add_ps(_mm256_loadu_ps(pSumX+x), _mm256_sub_ps(x1_8, x0_8)));
                _mm256_storeu_ps(pSumY+x, _mm256_add_ps(_mm256_loadu_ps(pSumY+x), _mm256_sub_ps(y1_8, y0_8)));

#ifdef __FMA__
                __m256 xx_diff = _mm256_fmsub_ps(x1_8, x1_8, _mm256_mul_ps(x0_8, x0_8));
                __m256 xy_diff = _mm256_fmsub_ps(x1_8, y1_8, _mm256_mul_ps(x0_8, y0_8));
                __m256 yy_diff = _mm256_fmsub_ps(y1_8, y1_8, _mm256_mul_ps(y0_8, y0_8));
#else
                __m256 xx_diff = _mm256_sub_ps(_mm256_mul_ps(x1_8, x1_8), _mm256_mul_ps(x0_8, x0_8));
                __m256 xy_diff = _mm256_sub_ps(_mm256_mul_ps(x1_8, y1_8), _mm256_mul_ps(x0_8, y0_8));
                __m256 yy_diff = _mm256_sub_ps(_mm256_mul_ps(y1_8, y1_8), _mm256_mul_ps(y0_8, y0_8));
#endif
                _mm256_storeu_ps(pSumXX+x, _mm256_add_ps(_mm256_loadu_ps(pSumXX+x), xx_diff));
                _mm256_storeu_ps(pSumXY+x, _mm256_add_ps(_mm256_loadu_ps(pSumXY+x), xy_diff));
                _mm256_storeu_ps(pSumYY+x, _mm256_add_ps(_mm256_loadu_ps(pSumYY+x), yy_diff));
            }
#else
            int width4=(widthBytes-8)/4*4;
            for(x=0; x<width4; x+=4)
            {
                __m128 x0_4 = LoadF32x4((Byte*)(pX0+x));
                __m128 x1_4 = LoadF32x4((Byte*)(pX1+x));
                __m128 y0_4 = LoadF32x4((Byte*)(pY0+x));
                __m128 y1_4 = LoadF32x4((Byte*)(pY1+x));
                _mm_storeu_ps(pSumX+x, _mm_add_ps(_mm_loadu_ps(pSumX+x), _mm_sub_ps(x1_4, x0_4)));
                _mm_storeu_ps(pSumY+x, _mm_add_ps(_mm_loadu_ps(pSumY+x), _mm_sub_ps(y1_4, y0_4)));
                _mm_storeu_ps(pSumXX+x, _mm_add_ps(_mm_loadu_ps(pSumXX+x), _mm_sub_ps(_mm_mul_ps(x1_4, x1_4), _mm_mul_ps(x0_4, x0_4))));
                _mm_storeu_ps(pSumXY+x, _mm_add_ps(_mm_loadu_ps(pSumXY+x), _mm_sub_ps(_mm_mul_ps(x1_4, y1_4), _mm_mul_ps(x0_4, y0_4))));
                _mm_storeu_ps(pSumYY+x, _mm_add_ps(_mm_loadu_ps(pSumYY+x), _mm_sub_ps(_mm_mul_ps(y1_4, y1_4), _mm_mul_ps(y0_4, y0_4))));
            }
#endif
            for(; x<widthBytes; x++)
            {
                float x1=pX1[x], x0=pX0[x], y0=pY0[x], y1=pY1[x];
                pSumX[x] += x1-x0;
                pSumY[x] += y1-y0;
                pSumXX[x]+= x1*x1-x0*x0;
                pSumXY[x]+= x1*y1-x0*y0;
                pSumYY[x]+= y1*y1-y0*y0;
            }
        }
    }
};

// --- Main Entry Points ---
template<typename Data>
float SSIM_Generic(Data* pDataX, Data* pDataY, int widthBytes, int width, int height, int win_size, double maxVal)
{
    double totalSum = 0;
    int totalCount = 0;
    int half_size = win_size/2;

    #pragma omp parallel reduction(+:totalSum, totalCount)
    {
        WinSum3F<Data> worker(pDataX, pDataY, widthBytes, width, height, win_size, maxVal);
        
        int loop_start = half_size;
        int loop_end = height - half_size;
        int total_rows = loop_end - loop_start;
        
        #if defined(_OPENMP)
            int n_threads = omp_get_num_threads();
            int tid = omp_get_thread_num();
        #else
            int n_threads = 1;
            int tid = 0;
        #endif
        
        int rows_per_thread = total_rows / n_threads;
        int my_start = loop_start + tid * rows_per_thread;
        int my_end = (tid == n_threads - 1) ? loop_end : my_start + rows_per_thread;

        if (my_start < my_end) {
            double localSum = 0;
            int localCount = 0;
            worker.ProcessStrip(my_start, my_end, localSum, localCount);
            totalSum += localSum;
            totalCount += localCount;
        }
    }

    if (totalCount == 0) return 0.0f;
    return (float)(totalSum / totalCount);
}

float SSIM_Byte(Byte* pDataX, Byte* pDataY, int widthBytes, int width, int height, int win_size, int maxVal)
{
    return SSIM_Generic<Byte>(pDataX, pDataY, widthBytes, width, height, win_size, (double)maxVal);
}
float SSIM_Float(float* pDataX, float* pDataY, int step, int width, int height, int win_size, double maxVal)
{
    return SSIM_Generic<float>(pDataX, pDataY, step, width, height, win_size, maxVal);
}

template <typename T, typename Data>
static inline void calcMeanValue(Data* pDataX, Data* pDataY, int widthBytes, int width, int height, int cx, int cy, int idx, int win_size,
    OUT T& meanX, OUT T& meanY, OUT T& sigmaXX, OUT T& sigmaXY, OUT T& sigmaYY)
{
    int cn=widthBytes/width, half_size=win_size/2;
    meanX=meanY=sigmaXX=sigmaXY=sigmaYY=0;
    for(int y=cy-half_size; y<=cy+half_size; y++)
    {
        Data *pX=pDataX+y*widthBytes;
        Data *pY=pDataY+y*widthBytes;
        for(int x=(cx-half_size)*cn+idx; x<(cx+half_size+1)*cn; x+=cn)
        {
            T fx=pX[x], fy=pY[x];
            meanX += fx;
            meanY += fy;
            sigmaXX += fx*fx;
            sigmaXY += fx*fy;
            sigmaYY += fy*fy;
        }
    }
    T invA=1/(T)(win_size*win_size);
    meanX=meanX*invA;  meanY=meanY*invA;
    sigmaXX=sigmaXX*invA;  sigmaXY=sigmaXY*invA;  sigmaYY=sigmaYY*invA;
}

template <typename T>
float SSIM_Byte_Slow_t(Byte* pDataX, Byte* pDataY, int widthBytes, int width, int height, int win_size)
{
    int half_size=win_size/2;
    win_size = 2*half_size+1;
    int count=0;
    double sum=0;
    T k1=0.01f, k2=0.03f, max_val=255, c1=SQR(k1*max_val), c2=SQR(k2*max_val);
    T conv_norm=(T)win_size*win_size/(win_size*win_size-1);
    int cn=widthBytes/width;
    for(int y=half_size; y<height-half_size; y++)
    {
        for(int k=0; k<cn; k++)
        {
            for(int x=half_size; x<width-half_size; x++)
            {
                T meanX, meanY, meanXX, meanXY, meanYY, sigmaXX, sigmaXY, sigmaYY;
                calcMeanValue(pDataX, pDataY, widthBytes, width, height, x, y, k, win_size, meanX, meanY, sigmaXX, sigmaXY, sigmaYY);
                meanXX = meanX*meanX;
                meanXY = meanX*meanY;
                meanYY = meanY*meanY;
                sigmaXX = conv_norm*(sigmaXX-meanXX);
                sigmaXY = conv_norm*(sigmaXY-meanXY);
                sigmaYY = conv_norm*(sigmaYY-meanYY);

                T u1=2*meanXY+c1, v1=2*sigmaXY+c2;
                T u2=meanXX+meanYY+c1, v2=sigmaXX+sigmaYY+c2;
                T z=(u1*v1)/(u2*v2);
                sum += z;
                count++;
            }
        }
    }
    return float(sum/count);
}
float SSIM_Byte_Slow(Byte* pDataX, Byte* pDataY, int widthBytes, int width, int height, int win_size)
{
    return SSIM_Byte_Slow_t<float>(pDataX, pDataY, widthBytes, width, height, win_size);
}