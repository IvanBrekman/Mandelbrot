#include <cstdio>
#include <ctime>

#include <immintrin.h>
#include <xmmintrin.h>
#include <pmmintrin.h>

struct ComplexRect {
    float xa = 0.f;
    float ya = 0.f;
    float xb = 0.f;
    float yb = 0.f;
};

const int IMG_X         = 900;
const int IMG_Y         = 600;
const int R2_MAX        = 100;
const int ITERATIONS    = 256;

void mand_no_sse(ComplexRect rect) {
    for (int iy = 0; iy < IMG_Y; iy++) {
        for (int ix = 0; ix < IMG_X; ix++) {
            float x0 = ix * (rect.xb - rect.xa) / IMG_X + rect.xa;
            float y0 = iy * (rect.yb - rect.ya) / IMG_Y + rect.ya;

            int n = 0;
            float x = x0, y = y0;

            for (; n < ITERATIONS; n++) {
                float x2 = x * x;
                float y2 = y * y;
                float xy = x * y;

                float r2 = x2 + y2;

                if (r2 >= R2_MAX) break;

                x = x2 - y2 + x0;
                y = xy + xy + y0;
            }
        }
    }
}

void mand_with_sse(ComplexRect rect) {
    float w_sh  = (rect.xb - rect.xa) / IMG_X;
    float h_sh  = (rect.yb - rect.ya) / IMG_Y;

    __m256 _r2_max_ar = _mm256_set1_ps(R2_MAX);
    __m256 _xa_ps     = _mm256_set1_ps(rect.xa);
    __m256 _w_sh_ps   = _mm256_set1_ps(w_sh);
    __m256 _one_ps    = _mm256_set1_ps(1);

    __m256 _3210_w_ps = _mm256_set_ps(0, 0, 0, 0, 3, 2, 1, 0);
           _3210_w_ps = _mm256_mul_ps(_3210_w_ps, _w_sh_ps);

    for (int iy = 0; iy < IMG_Y; iy++) {
        for (int ix = 0; ix < IMG_X; ix += 4) {
            __m256 _x0 = _mm256_set1_ps(ix * w_sh);
                   _x0 = _mm256_add_ps(_x0, _3210_w_ps);
                   _x0 = _mm256_add_ps(_x0, _xa_ps);
            __m256 _y0 = _mm256_set1_ps(iy * h_sh + rect.ya);

            __m256 _x  = _mm256_movehdup_ps(_x0);
            __m256 _y  = _mm256_movehdup_ps(_y0);

            __m256i _n  = _mm256_set1_epi32(0);
            for (int iter = 0; iter < ITERATIONS; iter++) {
                __m256 _x2 = _mm256_mul_ps(_x, _x);
                __m256 _y2 = _mm256_mul_ps(_y, _y);
                __m256 _xy = _mm256_mul_ps(_x, _y);

                __m256 _r2 = _mm256_add_ps(_x2, _y2);

                __m256 _cmp = _mm256_cmp_ps(_r2, _r2_max_ar, _CMP_LE_OS);

                int mask = _mm256_movemask_ps(_cmp);
                if (!mask) break;

                _n = _mm256_add_epi32(_n, _mm256_cvtps_epi32(_mm256_and_ps(_cmp, _one_ps)));

                _x = _mm256_sub_ps(_x2, _y2); _x = _mm256_add_ps(_x, _x0);
                _y = _mm256_add_ps(_xy, _xy); _y = _mm256_add_ps(_y, _y0);
            }
        }
    }
}

void ded_mand(ComplexRect rect) {
    const float ROI_X = -1.325f,
                ROI_Y = 0;
    
    const int    nMax  = 256;
    const float  dx    = 1/800.f, dy = 1/800.f;
    const __m128 r2Max = _mm_set_ps1 (100.f);
    const __m128 _255  = _mm_set_ps1 (255.f);
    const __m128 _3210 = _mm_set_ps  (3.f, 2.f, 1.f, 0.f);
    
    const __m128 nmax  = _mm_set_ps1 (nMax);

    float xC = 0.f, yC = 0.f, scale = 1.f;

    for (int iy = 0; iy < 600; iy++) 
        {

        float x0 = ( (          - 400.f) * dx + ROI_X + xC ) * scale,
              y0 = ( ((float)iy - 300.f) * dy + ROI_Y + yC ) * scale;

        for (int ix = 0; ix < 800; ix += 4, x0 += dx*4)
            {
            __m128 X0 = _mm_add_ps (_mm_set_ps1 (x0), _mm_mul_ps (_3210, _mm_set_ps1 (dx)));
            __m128 Y0 =             _mm_set_ps1 (y0);

            __m128 X = X0, Y = Y0;
            
            __m128i N = _mm_setzero_si128();

            for (int n = 0; n < nMax; n++)
                {
                __m128 x2 = _mm_mul_ps (X, X),
                        y2 = _mm_mul_ps (Y, Y);
                        
                __m128 r2 = _mm_add_ps (x2, y2);

                __m128 cmp = _mm_cmple_ps (r2, r2Max);
                int mask   = _mm_movemask_ps (cmp);
                if (!mask) break;

                N = _mm_sub_epi32 (N, _mm_castps_si128 (cmp));

                __m128 xy = _mm_mul_ps (X, Y);

                X = _mm_add_ps (_mm_sub_ps (x2, y2), X0);
                Y = _mm_add_ps (_mm_add_ps (xy, xy), Y0);
                }

            __m128 I = _mm_mul_ps (_mm_sqrt_ps (_mm_sqrt_ps (_mm_div_ps (_mm_cvtepi32_ps (N), nmax))), _255);
            }
        }
}

int calculate_fps(void (tested_func) (ComplexRect rect), ComplexRect* rect, int repeats=1) {
    int start_time = clock();

    tested_func(*rect);

    return clock() - start_time;
}

int main(void) {
    ComplexRect rect = { -2, -1, 1, 1 };

    int fps1 = calculate_fps(mand_no_sse,   &rect);
    int fps2 = calculate_fps(mand_with_sse, &rect);
    int fps3 = calculate_fps(ded_mand,      &rect);

    printf("FPS no sse/avx: %d\n",   fps1);
    printf("FPS with   avx: %d\n",   fps2);
    printf("FPS ded    sse: %d\n\n", fps3);

    return 0;
}
