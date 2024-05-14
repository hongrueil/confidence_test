#include <stdint.h>
#include <math.h>


//******************************************************************************
//
//! \ingroup dsplib_support_device
//!
//! \brief Real Q15 multiply with 32-bit result returned.
//
//******************************************************************************


#define __q7mpyl(A, B)  ((int16_t)(((int16_t)(A) * (int16_t)(B)) << 1))
#define __q15mpyl(A, B)  ((int32_t)(((int32_t)(A) * (int32_t)(B)) << 1)) // 這邊多 << 一次是為了抵銷 >> 16，這樣分子多乘的 2^15 再 << 1 之後剛好可以跟 >> 16抵銷成我們要的樣子
//#define __q15mpyl(A, B)  ((int32_t)(((int32_t)(A) * (int32_t)(B))))
#define __saturate(x, min, max) (((x)>(max))?(max):(((x)<(min))?(min):(x)))

typedef int8_t _q7;
typedef int16_t _q15;
typedef int32_t _iq31;
typedef _q15 fixed;

//#define _Q15(A)                 ((_q15)((((uint32_t)1 << 15) * __saturate(A,-1.0,32767.0/32768.0)))) & short(0XFF00)
#define _Q15(A)                 ((_q15)((((uint32_t)1 << 15) * __saturate(A * 0.6,-1.0,32767.0/32768.0)))) // A need to multiply a  scale factor
#define _Q15_i(A)                 ((_q15)((((uint32_t)1 << 15) * __saturate(A,-1.0,32767.0/32768.0)))) 
#define _Q7(A)                  ((_q7)((((uint16_t)1 << 7) * __saturate(A,-1.0,32767.0/32768.0))))
                                

static inline int64_t __saturated_add_iq64(int64_t x, int64_t y)
{
    return (int64_t)__saturate((int64_t)x + (int64_t)y, INT64_MIN, INT64_MAX);
}


static inline _iq31 __saturated_add_iq31(_iq31 x, _iq31 y)
{
    return (_iq31)__saturate((int64_t)x + (int64_t)y, INT32_MIN, INT32_MAX);
}

static inline _q15 __saturated_add_q15(_q15 x, _q15 y)
{
    return (_q15)__saturate((int32_t)x + (int32_t)y, INT16_MIN, INT16_MAX);
}