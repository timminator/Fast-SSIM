#if defined(_MSC_VER) && defined(__AVX2__)
    #ifndef __FMA__
        #define __FMA__ 1
    #endif
#endif

#include "ssim_impl.hpp"
