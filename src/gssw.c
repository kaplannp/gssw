/* The MIT License

   Copyright (c) 2012-2015 Boston College, 2014-2018 Wellcome Sanger Institute

   Permission is hereby granted, free of charge, to any person obtaining
   a copy of this software and associated documentation files (the
   "Software"), to deal in the Software without restriction, including
   without limitation the rights to use, copy, modify, merge, publish,
   distribute, sublicense, and/or sell copies of the Software, and to
   permit persons to whom the Software is furnished to do so, subject to
   the following conditions:

   The above copyright notice and this permission notice shall be
   included in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
   EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
   MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
   NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
   BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
   ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
   CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
   SOFTWARE.
*/

/* Contact: Erik Garrison <erik.garrison@gmail.com> */

/*
 *  Created by Mengyao Zhao on 6/22/10.
 *  Generalized to operate on graphs by Erik Garrison and renamed gssw.c
 */
#define SIMDE_ENABLE_NATIVE_ALIASES
#include "simde/x86/sse2.h"
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <inttypes.h>
#include <assert.h>
#include "gssw.h"

/* Internal-only struct for returning alignment end positions */
typedef struct {
    uint16_t score;
    int32_t ref;     //0-based position
    int32_t read;    //alignment ending position on read, 0-based
} gssw_alignment_end;

#ifdef __GNUC__
#define LIKELY(x) __builtin_expect((x),1)
#define UNLIKELY(x) __builtin_expect((x),0)
#else
#define LIKELY(x) (x)
#define UNLIKELY(x) (x)
#endif

/* Convert the coordinate in the scoring matrix into the coordinate in one line of the band. */
#define set_u(u, w, i, j) { int x=(i)-(w); x=x>0?x:0; (u)=(j)-x+1; }

/* Convert the coordinate in the direction matrix into the coordinate in one line of the band. */
#define set_d(u, w, i, j, p) { int x=(i)-(w); x=x>0?x:0; x=(j)-x; (u)=x*3+p; }

/*! @function
  @abstract  Round an integer to the next closest power-2 integer.
  @param  x  integer to be rounded (in place)
  @discussion x will be modified.
 */
#define kroundup32(x) (--(x), (x)|=(x)>>1, (x)|=(x)>>2, (x)|=(x)>>4, (x)|=(x)>>8, (x)|=(x)>>16, ++(x))


/* Generate query profile rearrange query sequence & calculate the weight of match/mismatch. */
__m128i* gssw_qP_byte (const int8_t* read_num,
                       const int8_t* mat,
                       const int32_t readLen,
                       const int32_t n,    /* the edge length of the squre matrix mat */
                       uint8_t bias,
                       int8_t start_full_length_bonus,
                       int8_t end_full_length_bonus) {

    int32_t segLen = (readLen + 15) / 16; /* Split the 128 bit register into 16 pieces.
                                     Each piece is 8 bit. Split the read into 16 segments.
                                     Calculate 16 segments in parallel.
                                     This holds the number of segments needed to fit the read.
                                   */
    __m128i* vProfile = (__m128i*)malloc(n * segLen * sizeof(__m128i));
    int8_t* t = (int8_t*)vProfile; // This points to each byte in the profile vector, one at a time
    // nt tracks the nucleotide we're computing the profile for. We do each possible character.
    // i tracks which swizzled register we're working on
    // j tracks the character in the read we're working on
    // segNum counts which of the 16 read segments we're working on, each of which gets its own byte in each swizzled register
    int32_t nt, i, j, segNum;

    /* Generate query profile rearrange query sequence & calculate the weight of match/mismatch */
    for (nt = 0; LIKELY(nt < n); nt ++) {

        // special logic for first vector to add bonus for full length left alignment
        if (segLen > 0) {
            j = 0;
            // add bonus to first position in first register (corresponds to first position in read)
            // also account for the start potentially being the end
            *t = j>= readLen ? bias : mat[nt * n + read_num[j]] + bias +
                start_full_length_bonus + (j == readLen - 1 ? end_full_length_bonus : 0);
            t++;
            j += segLen;
            // use normal score for the rest of the vector
            // except for the last base which also gets a bonus
            for (segNum = 1; LIKELY(segNum < 16) ; segNum ++) {
                *t = j>= readLen ? bias : mat[nt * n + read_num[j]] + bias + (j == readLen - 1 ? end_full_length_bonus : 0);
                t++;
                j += segLen;
            }
        }

        for (i = 1; i < segLen; i ++) {
            j = i;
            for (segNum = 0; LIKELY(segNum < 16) ; segNum ++) {
                // Now handle all the vectors after the first
                *t = j>= readLen ? bias : mat[nt * n + read_num[j]] + bias + (j == readLen - 1 ? end_full_length_bonus : 0);
                t++;
                j += segLen;
            }
        }
    }
    return vProfile;
}

/**
 * Look up the value in a profile matrix for the given base code observed at the given read index.
 * Useful for non-swizzled access to the the swizzled profile.
 */
uint8_t profile_get_byte(__m128i* vProfile, int32_t readLen, int32_t read_position, int32_t observed_base) {
    // Profile is stored by observed base (most significant), then by position in the segment, then by segment in the read (lwast significant).

    // How long is a segment? We have 16.
    int32_t segLen = (readLen + 15) / 16;
    // What segment are we in of the 16?
    int32_t segment = read_position / segLen;
    // And where are we in that segment?
    int32_t pos_in_segment = read_position % segLen;

    // Look at the profile as a byte array
    uint8_t* profile_bytes = (uint8_t*) vProfile;

    return profile_bytes[observed_base * (segLen * 16) + pos_in_segment * 16 + segment];
}

/**
 * Swizzle a vector of bytes into a "striped" vector, organized first by
 * position in segment and then by segment of 16. Size must be a multiple of 16.
 */
void swizzle_byte(uint8_t* to_swizzle, int32_t size) {
    if (size == 0) {
        // Nothing to do!
        return;
    }

    uint8_t* scratch = (uint8_t*) malloc(size * sizeof(uint8_t));
    if(scratch == NULL) {
        fprintf(stderr, "error:[gssw] Could not allocate swizzle buffer.\n");
        exit(1);
    }
    // Copy the data out of the way
    memcpy(scratch, to_swizzle, size);

    // How long is a segment? We have 16.
    int32_t segLen = (size + 15) / 16;

    // We'll walk this through the destination array.
    int32_t cursor = 0;

    int32_t pos_in_segment;
    for(pos_in_segment = 0; pos_in_segment < segLen; pos_in_segment++) {
        // For each position in a segment

        int32_t segNum;
        for (segNum = 0; segNum < 16; segNum++) {
            // For each segment

            // Grab the byte
            to_swizzle[cursor] = scratch[segNum * segLen + pos_in_segment];
            // Write the next byte at the next position
            cursor++;
        }

    }
   free(scratch);
}

/**
 * Unswizzle a swizzled vector of bytes into a normal start-to-end vector of bytes.
 * Size must be a multiple of 16.
 */
void unswizzle_byte(uint8_t* to_unswizzle, int32_t size) {
    if (size == 0) {
        // Nothing to do!
        return;
    }

    uint8_t* scratch = (uint8_t*) malloc(size * sizeof(uint8_t));
    if(scratch == NULL) {
        fprintf(stderr, "error:[gssw] Could not allocate unswizzle buffer.\n");
        exit(1);
    }
    // Copy the data out of the way
    memcpy(scratch, to_unswizzle, size);

    // How long is a segment? We have 16.
    int32_t segLen = (size + 15) / 16;

    int32_t i;
    for (i = 0; i < size; i++) {
        // Swizzled vector is arranged first by position in segment, then by segment (of 16)
        // So go to the right position in the segment, and then to the right segment, and get the value
        // And save it to the right place in the unswizzled vector.
        to_unswizzle[i] = scratch[(i % segLen) * 16 + (i / segLen)];
    }
   free(scratch);
}

/**
 * Saturation arithmetic subtraction. (like the "subs" SSE2 intrinsics)
 * Compute a - b, returning 0 if it would be negative.
 */
uint8_t subs_byte(uint8_t a, uint8_t b) {
    if (b > a) {
        return 0;
    }
    return a - b;
}

/**
 * Saturation arithmetic addition. (like the "addss" SSE2 intrinsics)
 * Compute a + b, returning max
 */
uint8_t adds_byte(uint8_t a, uint8_t b) {
    uint16_t sum = (uint16_t) a + (uint16_t) b;
    if (sum > 255) {
        return 255;
    }
    return sum;
}

/**
 * We need a max for bytes.
 */
uint8_t max_byte(uint8_t a, uint8_t b) {
    if (a > b) {
        return a;
    }
    return b;
}


/* To determine the maximum values within each vector, rather than between vectors. */

#define m128i_max16(m, vm) \
    (vm) = _mm_max_epu8((vm), _mm_srli_si128((vm), 8)); \
    (vm) = _mm_max_epu8((vm), _mm_srli_si128((vm), 4)); \
    (vm) = _mm_max_epu8((vm), _mm_srli_si128((vm), 2)); \
    (vm) = _mm_max_epu8((vm), _mm_srli_si128((vm), 1)); \
    (m) = _mm_extract_epi16((vm), 0)

// See https://stackoverflow.com/q/33824300 for this unsigned comparison macro
// for the missing unsigned comparison instruction _mm_cmpgt_epu8
#define m128i_cmpgt(v0, v1) \
         _mm_cmpgt_epi8(_mm_xor_si128(v0, _mm_set1_epi8(-128)), \
                        _mm_xor_si128(v1, _mm_set1_epi8(-128)))

/* Striped Smith-Waterman
   Record the highest score of each reference position.
   Return the alignment score and ending position of the best alignment, 2nd best alignment, etc.
   Gap begin and gap extension are different.
   wight_match > 0, all other weights < 0.
   The returned positions are 0-based.
 */
gssw_alignment_end* gssw_sw_sse2_byte (const int8_t* ref,
                                       int8_t ref_dir,    // 0: forward ref; 1: reverse ref
                                       int32_t refLen,
                                       int32_t readLen,
                                       const uint8_t weight_gapO, /* will be used as - */
                                       const uint8_t weight_gapE, /* will be used as - */
                                       __m128i* vProfile,
                                       uint8_t terminate,    /* the best alignment score: used to terminate
                                                               the matrix calculation when locating the
                                                               alignment beginning point. If this score
                                                               is set to 0, it will not be used */
                                       uint8_t bias,  /* Shift 0 point to a positive value. */
                                       int32_t maskLen,
                                       gssw_align* alignment, /* to save seed */
                                       const gssw_seed* seed) {     /* to seed the alignment */

    uint8_t max = 0;                             /* the max alignment score */
    int32_t end_read = readLen - 1;
    int32_t end_ref = -1; /* 0_based best alignment ending point; Initialized as isn't aligned -1. */
    int32_t segLen = (readLen + 15) / 16; /* number of segment */

    /* Initialize buffers used in alignment */
    __m128i* pvHStore;
    __m128i* pvHLoad;
    __m128i* pvHmax;
    __m128i* pvE;
    // Extra arrays used during lazy F loop
    __m128i* pvEStore;
    __m128i* pvFStore;
    /* Note use of aligned memory.  Return value of 0 means success for posix_memalign. */
    if (!(!posix_memalign((void**)&pvHStore,     sizeof(__m128i), segLen*sizeof(__m128i)) &&
          !posix_memalign((void**)&pvHLoad,      sizeof(__m128i), segLen*sizeof(__m128i)) &&
          !posix_memalign((void**)&pvHmax,       sizeof(__m128i), segLen*sizeof(__m128i)) &&
          !posix_memalign((void**)&pvE,          sizeof(__m128i), segLen*sizeof(__m128i)) &&
          !posix_memalign((void**)&pvEStore,     sizeof(__m128i), segLen*sizeof(__m128i)) &&
          !posix_memalign((void**)&pvFStore,     sizeof(__m128i), segLen*sizeof(__m128i)) &&
          !posix_memalign((void**)&alignment->seed.pvE,      sizeof(__m128i), segLen*sizeof(__m128i)) &&
          !posix_memalign((void**)&alignment->seed.pvHStore, sizeof(__m128i), segLen*sizeof(__m128i)))) {
        fprintf(stderr, "error:[gssw] Could not allocate memory required for alignment buffers.\n");
        exit(1);
    }

    /* Workaround: zero memory ourselves because we don't have an aligned calloc */
    memset(pvHStore,                 0, segLen*sizeof(__m128i));
    memset(pvHLoad,                  0, segLen*sizeof(__m128i));
    memset(pvHmax,                   0, segLen*sizeof(__m128i));
    memset(pvE,                      0, segLen*sizeof(__m128i));
    memset(pvEStore,                 0, segLen*sizeof(__m128i));
    memset(pvFStore,                 0, segLen*sizeof(__m128i));
    memset(alignment->seed.pvE,      0, segLen*sizeof(__m128i));
    memset(alignment->seed.pvHStore, 0, segLen*sizeof(__m128i));

    /* if we are running a seeded alignment, copy over the seeds */
    if (seed) {
        memcpy(pvE, seed->pvE, segLen*sizeof(__m128i));
        memcpy(pvHStore, seed->pvHStore, segLen*sizeof(__m128i));
    }

    /* Define 16 byte 0 vector. */
    __m128i vZero = _mm_set1_epi32(0);

    /* Used for iteration */
    int32_t i, j;

    /* 16 byte insertion begin vector */
    __m128i vGapO = _mm_set1_epi8(weight_gapO);

    /* 16 byte insertion extension vector */
    __m128i vGapE = _mm_set1_epi8(weight_gapE);

    /* 16 byte bias vector */
    __m128i vBias = _mm_set1_epi8(bias);

    __m128i vMaxScore = vZero; /* Trace the highest score of the whole SW matrix. */
    __m128i vMaxMark = vZero; /* Trace the highest score till the previous column. */
    __m128i vTemp;
    int32_t begin = 0, end = refLen, step = 1;

    /* outer loop to process the reference sequence */
    if (ref_dir == 1) {
        begin = refLen - 1;
        end = -1;
        step = -1;
    }
    for (i = begin; LIKELY(i != end); i += step) {
        // For each column

        int32_t cmp;
        __m128i e = vZero, vF = vZero, vMaxColumn = vZero; /* Initialize F value to 0.
                               Any errors to vH values will be corrected in the Lazy_F loop.
                             */
        //max16(maxColumn[i], vMaxColumn);
        //fprintf(stderr, "middle[%d]: %d\n", i, maxColumn[i]);

        // Load the last column's last H value in each segment
        //__m128i vH = pvHStore[segLen - 1];
        __m128i vH = _mm_load_si128 (pvHStore + (segLen - 1));
        // Shift it over (TODO: why??? We only shift this initial read and not later reads.)
        vH = _mm_slli_si128 (vH, 1); /* Shift the 128-bit value in vH left by 1 byte. */
        // Find the profile entries for matching this column's ref base against each read base.
        __m128i* vP = vProfile + ref[i] * segLen; /* Right part of the vProfile */

        /* Swap the 2 H buffers. */
        __m128i* pv = pvHLoad;
        pvHLoad = pvHStore;
        pvHStore = pv;

        /* inner loop to process the query sequence */
        for (j = 0; LIKELY(j < segLen); ++j) {
            // For each vector of cursor positions within this column
            // at position j in each segment

            // Add the profile scores for matching against this ref base
            vH = _mm_adds_epu8(vH, _mm_load_si128(vP + j));
            // And subtract out the profile's bias (so profile scores can be <0)
            vH = _mm_subs_epu8(vH, vBias); /* vH will be always > 0 because of saturation arithmetic */
            //    max16(maxColumn[i], vH);
            //    fprintf(stderr, "H[%d]: %d\n", i, maxColumn[i]);
            /*
            int8_t* t;
            int32_t ti;
            fprintf(stdout, "%d\n", i);
            for (t = (int8_t*)&vH, ti = 0; ti < 16; ++ti) fprintf(stdout, "%d\t", *t++);
            fprintf(stdout, "\n");
            */

            // So now vH has the scores we would get if we did all matches/mismatches from the previous column.
            // Next we are going to replace entries if we have a better score from a gap matrix.

            /* Get max from vH, vE and vF. */
            e = _mm_load_si128(pvE + j);
            //_mm_store_si128(vE + j, e);

            // So e holds the *current* column's read gap open/extend scores,
            // which we computed on the *previous* column's pass.
            // vF stores the current column and *current* cursor position's ref
            // gap open/extend scores, which we computed on the *previous*
            // cursor position.

            vH = _mm_max_epu8(vH, e);
            vH = _mm_max_epu8(vH, vF);
            vMaxColumn = _mm_max_epu8(vMaxColumn, vH);

            // So now vH has the correct (modulo wrong F values) H matrix entries.

            // max16(maxColumn[i], vMaxColumn);
            //fprintf(stdout, "middle[%d]: %d\n", i, maxColumn[i]);
            //fprintf(stdout, "i=%d, j=%d\t", i, j);
            //for (t = (int8_t*)&vMaxColumn, ti = 0; ti < 16; ++ti) fprintf(stdout, "%d\t", *t++);
            //fprintf(stdout, "\n");

            /* Save vH values. */
            _mm_store_si128(pvHStore + j, vH);

            /* Save the vE and vF values they derived from */
            _mm_store_si128(pvEStore + j, e);
            _mm_store_si128(pvFStore + j, vF);

            // Now we need to compute the E values for the *next* column, based
            // on our non-F-loop-processed H values

            /* Update vE value. */
            vH = _mm_subs_epu8(vH, vGapO); /* saturation arithmetic, result >= 0 */
            e = _mm_subs_epu8(e, vGapE);
            e = _mm_max_epu8(e, vH);

            // And we compute the F values for the next cursor position.

            /* Compute new vF value, giving F matrix values at next cursor position */
            vF = _mm_subs_epu8(vF, vGapE);
            vF = _mm_max_epu8(vF, vH); // We already charged a gap open against vH

            /* Save the E values we computed for the next column */
            _mm_store_si128(pvE + j, e);

            /* Load the next vH. */
            vH = _mm_load_si128(pvHLoad + j);
        }


        /* reset pointers to the start of the saved data */
        j = 0;
        vH = _mm_load_si128 (pvHStore + j);

        /*
         * Wrap vF around from the end of each segment to the start of the next.
         */
        vF = _mm_slli_si128 (vF, 1);

        // So now we're looking at the F value for every first position, after a
        // full pass. So the first F is guaranteed to be right, and other Fs
        // will be right if nothing had to propagate down more than 16 bases.

        // We're also looking at the H values that should be derived from those
        // F values.

        // Now we need to work out if we actually want to update anything. We
        // need to do an F loop if we would modify H, or if we would improve
        // over the old F.

        // If we beat the stored H
        vTemp = m128i_cmpgt (vF, vH);
        cmp = _mm_movemask_epi8 (vTemp);
        // Or we beat the stored F
        vTemp = _mm_load_si128 (pvFStore + j);
        vTemp = m128i_cmpgt (vF, vTemp);
        cmp |= _mm_movemask_epi8 (vTemp);
        while (cmp != 0x0000)
        {
            // Then we do the update

            // Update this stripe of the H matrix
            vH = _mm_max_epu8 (vH, vF);
            vMaxColumn = _mm_max_epu8(vMaxColumn, vH);
            _mm_store_si128 (pvHStore + j, vH);

            // Update the E matrix for the next column
            // Since we may have changed the H matrix
            // This is to allow a gap-to-gap transition in the alignment
            e = _mm_load_si128(pvE + j);
            // The H matrix can only get better, so the gap open scores can only
            // get better, so the E matrix can only get better too.
            vTemp = _mm_subs_epu8(vH, vGapO);
            e = _mm_max_epu8(e, vTemp);
            _mm_store_si128(pvE + j, e);
            // TODO: Instead of doing this, would it be smarter to just compute
            // the E matrix for each column when we're doing its H matrix? Or
            // would the extra buffer slow us down more than the extra compute?


            // Save the stripe of the F matrix
            // Only add in better F scores. Sometimes during this loop we'll
            // recompute worse ones.
            vTemp = _mm_load_si128 (pvFStore + j);
            vTemp = _mm_max_epu8 (vTemp, vF);
            _mm_store_si128(pvFStore + j, vTemp);

            // Then think about extending
            vF = _mm_subs_epu8 (vF, vGapE);
            // We never need to think about gap opens because nothing that came
            // from a gap open can ever change, because you won't close and then
            // immediately open a gap.

            j++;
            if (j >= segLen)
            {
                // Wrap around to the next segment again
                j = 0;
                vF = _mm_slli_si128 (vF, 1);
            }

            // Again compute if H or F needs updating based on this new set of F
            // values.
            vH = _mm_load_si128 (pvHStore + j);

            // See if we beat the stored H
            vTemp = m128i_cmpgt (vF, vH);
            cmp = _mm_movemask_epi8 (vTemp);
            // Or if we beat the stored F
            vTemp = _mm_load_si128 (pvFStore + j);
            vTemp = m128i_cmpgt (vF, vTemp);
            cmp |= _mm_movemask_epi8 (vTemp);
        }

        vMaxScore = _mm_max_epu8(vMaxScore, vMaxColumn);
        vTemp = _mm_cmpeq_epi8(vMaxMark, vMaxScore);
        cmp = _mm_movemask_epi8(vTemp);
        if (cmp != 0xffff) {
            uint8_t temp;
            vMaxMark = vMaxScore;
            m128i_max16(temp, vMaxScore);
            vMaxScore = vMaxMark;

            if (LIKELY(temp > max)) {
                max = temp;
                if (max + bias >= 255) break;    //overflow
                end_ref = i;

                /* Store the column with the highest alignment score in order to trace the alignment ending position on read. */
                for (j = 0; LIKELY(j < segLen); ++j) pvHmax[j] = pvHStore[j];

            }
        }

        /* Record the max score of current column. */
        //max16(maxColumn[i], vMaxColumn);
        //fprintf(stderr, "maxColumn[%d]: %d\n", i, maxColumn[i]);
        //if (maxColumn[i] == terminate) break;

    }

    //fprintf(stderr, "%p %p %p %p %p %p\n", *pmH, mH, pvHmax, pvE, pvHLoad, pvHStore);
    // save the last vH
    memcpy(alignment->seed.pvE,      pvE,      segLen*sizeof(__m128i));
    memcpy(alignment->seed.pvHStore, pvHStore, segLen*sizeof(__m128i));

    /* Trace the alignment ending position on read. */
    uint8_t *t = (uint8_t*)pvHmax;
    int32_t column_len = segLen * 16;
    for (i = 0; LIKELY(i < column_len); ++i, ++t) {
        int32_t temp;
        if (*t == max) {
            temp = i / 16 + i % 16 * segLen;
            if (temp < end_read) end_read = temp;
        }
    }

    //fprintf(stderr, "%p %p %p %p %p %p\n", *pmH, mH, pvHmax, pvE, pvHLoad, pvHStore);

    free(pvE);
    free(pvHmax);
    free(pvHLoad);
    free(pvHStore);
    free(pvEStore);
    free(pvFStore);

    /* Find the most possible 2nd best alignment. */
    gssw_alignment_end* bests = (gssw_alignment_end*) calloc(2, sizeof(gssw_alignment_end));
    bests[0].score = max + bias >= 255 ? 255 : max;
    bests[0].ref = end_ref;
    bests[0].read = end_read;


    return bests;
}

/* Simplified gssw_init: byte-only, no score_size parameter */
gssw_profile* gssw_init (const int8_t* read, const int32_t readLen, const int8_t* mat, const int32_t n,
                         int8_t start_full_length_bonus, int8_t end_full_length_bonus) {
    gssw_profile* p = (gssw_profile*)calloc(1, sizeof(struct gssw_profile));
    p->profile_byte = 0;
    p->bias = 0;
    int32_t bias = 0, i;
    for (i = 0; i < n*n; i++) if (mat[i] < bias) bias = mat[i];
    bias = abs(bias);
    p->bias = bias;
    p->profile_byte = gssw_qP_byte(read, mat, readLen, n, bias, start_full_length_bonus, end_full_length_bonus);
    p->read = read;
    p->mat = mat;
    p->readLen = readLen;
    p->n = n;
    return p;
}

void gssw_init_destroy (gssw_profile* p) {
    free(p->profile_byte);
    free(p);
}

/* Simplified gssw_fill: byte-only SSE2, no fallback */
gssw_align* gssw_fill (const gssw_profile* prof,
                       const int8_t* ref,
                       const int32_t refLen,
                       const uint8_t weight_gapO,
                       const uint8_t weight_gapE,
                       const int32_t maskLen,
                       gssw_seed* seed) {
    gssw_alignment_end* bests = 0;
    int32_t readLen = prof->readLen;
    gssw_align* alignment = gssw_align_create();
    if (maskLen < 15) {
        fprintf(stderr, "When maskLen < 15, the function ssw_align doesn't return 2nd best alignment information.\n");
    }
    bests = gssw_sw_sse2_byte(ref, 0, refLen, readLen, weight_gapO, weight_gapE,
                              prof->profile_byte, -1, prof->bias, maskLen, alignment, seed);
    if (bests[0].score == 255) {
        fprintf(stderr, "Warning: score overflow (255) in byte mode\n");
    }
    alignment->score1 = bests[0].score;
    alignment->ref_end1 = bests[0].ref;
    alignment->read_end1 = bests[0].read;
    if (maskLen >= 15) {
        alignment->score2 = bests[1].score;
        alignment->ref_end2 = bests[1].ref;
    } else {
        alignment->score2 = 0;
        alignment->ref_end2 = -1;
    }
    free(bests);
    return alignment;
}

gssw_align* gssw_align_create (void) {
    gssw_align* a = (gssw_align*)calloc(1, sizeof(gssw_align));
    a->seed.pvHStore = NULL;
    a->seed.pvE = NULL;
    a->ref_begin1 = -1;
    a->read_begin1 = -1;
    return a;
}

void gssw_align_destroy (gssw_align* a) {
    gssw_align_clear_matrix_and_seed(a);
    free(a);
}

void gssw_align_clear_matrix_and_seed (gssw_align* a) {
    free(a->seed.pvHStore);
    a->seed.pvHStore = NULL;
    free(a->seed.pvE);
    a->seed.pvE = NULL;
}

void gssw_seed_destroy(gssw_seed* s) {
    free(s->pvE);
    s->pvE = NULL;
    free(s->pvHStore);
    s->pvE = NULL;
    free(s);
}

//TODO: why is score_matrix even an argument here?
gssw_node* gssw_node_create(void* data,
                            const uint64_t id,
                            const char* seq,
                            const int8_t* nt_table,
                            const int8_t* score_matrix) {
    gssw_node* n = calloc(1, sizeof(gssw_node));
    int32_t len = strlen(seq);
    n->id = id;
    n->len = len;
    n->seq = (char*)malloc(len+1);
    strncpy(n->seq, seq, len); n->seq[len] = 0;
    n->data = data;
    n->num = gssw_create_num(seq, len, nt_table);
    n->count_prev = 0; // are these be set == 0 by calloc?
    n->count_next = 0;
    n->alignment = NULL;
    return n;
}

// for reuse of graph through multiple alignments
void gssw_node_clear_alignment(gssw_node* n) {
    gssw_align_destroy(n->alignment);
    n->alignment = NULL;
}

void gssw_profile_destroy(gssw_profile* prof) {
    free(prof->profile_byte);
    free(prof);
}

void gssw_node_destroy(gssw_node* n) {
    free(n->seq);
    free(n->num);
    free(n->prev);
    free(n->next);
    if (n->alignment) {
        gssw_align_destroy(n->alignment);
    }
    free(n);
}

//void node_clear_alignment(node* n) {
//    align_clear_matrix_and_seed(n->alignment);
//}

void gssw_node_add_prev(gssw_node* n, gssw_node* m) {
    ++n->count_prev;
    n->prev = (gssw_node**)realloc(n->prev, n->count_prev*sizeof(gssw_node*));
    n->prev[n->count_prev -1] = m;
}

void gssw_node_add_next(gssw_node* n, gssw_node* m) {
    ++n->count_next;
    n->next = (gssw_node**)realloc(n->next, n->count_next*sizeof(gssw_node*));
    n->next[n->count_next -1] = m;
}

void gssw_nodes_add_edge(gssw_node* n, gssw_node* m) {
    //fprintf(stderr, "connecting %u -> %u\n", n->id, m->id);
    // check that there isn't already an edge
    uint32_t k;
    // check to see if there is an edge from n -> m, and exit if so
    for (k=0; k<n->count_next; ++k) {
        if (n->next[k] == m) {
            return;
        }
    }
    gssw_node_add_next(n, m);
    gssw_node_add_prev(m, n);
}

void gssw_node_del_prev(gssw_node* n, gssw_node* m) {
    gssw_node** x = (gssw_node**)malloc(n->count_prev*sizeof(gssw_node*));
    int i = 0;
    gssw_node** np = n->prev;
    for ( ; i < n->count_prev; ++i, ++np) {
        if (*np != m) {
            x[i] = *np;
        }
    }
    free(n->prev);
    n->prev = x;
    --n->count_prev;
}

void gssw_node_del_next(gssw_node* n, gssw_node* m) {
    gssw_node** x = (gssw_node**)malloc(n->count_next*sizeof(gssw_node*));
    int i = 0;
    gssw_node** nn = n->next;
    for ( ; i < n->count_next; ++i, ++nn) {
        if (*nn != m) {
            x[i] = *nn;
        }
    }
    free(n->next);
    n->next = x;
    --n->count_next;
}

void gssw_nodes_del_edge(gssw_node* n, gssw_node* m) {
    gssw_node_del_next(n, m);
    gssw_node_del_prev(m, n);
}

void gssw_node_replace_prev(gssw_node* n, gssw_node* m, gssw_node* p) {
    int i = 0;
    gssw_node** np = n->prev;
    for ( ; i < n->count_prev; ++i, ++np) {
        if (*np == m) {
            *np = p;
        }
    }
}

void gssw_node_replace_next(gssw_node* n, gssw_node* m, gssw_node* p) {
    int i = 0;
    gssw_node** nn = n->next;
    for ( ; i < n->count_next; ++i, ++nn) {
        if (*nn == m) {
            *nn = p;
        }
    }
}

gssw_seed* gssw_create_seed_byte(int32_t readLen, gssw_node** prev, int32_t count) {
    int32_t j = 0, k = 0;
    for (k = 0; k < count; ++k) {
        if (!prev[k]->alignment) {
            fprintf(stderr, "error:[gssw] cannot align because node predecessors cannot provide seed\n");
            fprintf(stderr, "failing is node %llu\n", prev[k]->id);
            exit(1);
        }
    }
    __m128i vZero = _mm_set1_epi32(0);
    int32_t segLen = (readLen + 15) / 16;
    gssw_seed* seed = (gssw_seed*)calloc(1, sizeof(gssw_seed));
    if (!(!posix_memalign((void**)&seed->pvE,      sizeof(__m128i), segLen*sizeof(__m128i)) &&
          !posix_memalign((void**)&seed->pvHStore, sizeof(__m128i), segLen*sizeof(__m128i)))) {
        fprintf(stderr, "error:[gssw] Could not allocate memory for alignment seed\n"); exit(1);
        exit(1);
    }
    memset(seed->pvE,      0, segLen*sizeof(__m128i));
    memset(seed->pvHStore, 0, segLen*sizeof(__m128i));
    // take the max of all inputs
    __m128i pvE = vZero, pvH = vZero, ovE = vZero, ovH = vZero;
    for (j = 0; j < segLen; ++j) {
        pvE = vZero; pvH = vZero;
        for (k = 0; k < count; ++k) {
            ovE = _mm_load_si128(prev[k]->alignment->seed.pvE + j);
            ovH = _mm_load_si128(prev[k]->alignment->seed.pvHStore + j);
            pvE = _mm_max_epu8(pvE, ovE);
            pvH = _mm_max_epu8(pvH, ovH);
        }
        _mm_store_si128(seed->pvHStore + j, pvH);
        _mm_store_si128(seed->pvE + j, pvE);
    }
    return seed;
}

/* Simplified gssw_graph_fill_internal: byte-only, no word retry */
gssw_graph*
gssw_graph_fill_internal (gssw_graph* graph,
                          const char* read_seq,
                          const int8_t* nt_table,
                          const int8_t* score_matrix,
                          const uint8_t weight_gapO,
                          const uint8_t weight_gapE,
                          const int8_t start_full_length_bonus,
                          const int8_t end_full_length_bonus,
                          const int32_t maskLen) {
    int32_t read_length = strlen(read_seq);
    int8_t* read_num = gssw_create_num(read_seq, read_length, nt_table);
    gssw_profile* prof = gssw_init(read_num, read_length, score_matrix, 5,
                                    start_full_length_bonus, end_full_length_bonus);
    gssw_seed* seed = NULL;
    uint16_t max_score = 0;
    uint32_t i;
    gssw_node** npp = &graph->nodes[0];
    // seed the head nodes
    for (i = 0; i < graph->size; ++i, ++npp) {
        gssw_node* n = *npp;
        if (!n->count_prev) {
            seed = gssw_create_seed_byte(prof->readLen, n->prev, n->count_prev);
            gssw_node_fill(n, prof, weight_gapO, weight_gapE, maskLen, seed);
            gssw_seed_destroy(seed); seed = NULL;
            if (!graph->max_node || n->alignment->score1 > max_score) {
                graph->max_node = n;
                max_score = n->alignment->score1;
            }
        }
    }
    npp = &graph->nodes[0];
    // fill non-head nodes in topological order
    for (i = 0; i < graph->size; ++i, ++npp) {
        gssw_node* n = *npp;
        if (n->count_prev) {
            seed = gssw_create_seed_byte(prof->readLen, n->prev, n->count_prev);
            gssw_node_fill(n, prof, weight_gapO, weight_gapE, maskLen, seed);
            gssw_seed_destroy(seed); seed = NULL;
            if (!graph->max_node || n->alignment->score1 > max_score) {
                graph->max_node = n;
                max_score = n->alignment->score1;
            }
        }
    }
    free(read_num);
    gssw_profile_destroy(prof);
    return graph;
}

gssw_graph*
gssw_graph_fill (gssw_graph* graph,
                 const char* read_seq,
                 const int8_t* nt_table,
                 const int8_t* score_matrix,
                 const uint8_t weight_gapO,
                 const uint8_t weight_gapE,
                 const int8_t start_full_length_bonus,
                 const int8_t end_full_length_bonus,
                 const int32_t maskLen) {
    return gssw_graph_fill_internal(graph, read_seq, nt_table, score_matrix,
                                    weight_gapO, weight_gapE, start_full_length_bonus,
                                    end_full_length_bonus, maskLen);
}

gssw_graph*
gssw_graph_fill_pinned (gssw_graph* graph,
                        const char* read_seq,
                        const int8_t* nt_table,
                        const int8_t* score_matrix,
                        const uint8_t weight_gapO,
                        const uint8_t weight_gapE,
                        const int8_t start_full_length_bonus,
                        const int8_t end_full_length_bonus,
                        const int32_t maskLen) {
    return gssw_graph_fill_internal(graph, read_seq, nt_table, score_matrix,
                                    weight_gapO, weight_gapE, start_full_length_bonus,
                                    end_full_length_bonus, maskLen);
}

/* Simplified gssw_node_fill: byte-only SSE2, no fallback */
gssw_node*
gssw_node_fill (gssw_node* node,
                const gssw_profile* prof,
                const uint8_t weight_gapO,
                const uint8_t weight_gapE,
                const int32_t maskLen,
                const gssw_seed* seed) {
    gssw_alignment_end* bests = NULL;
    int32_t readLen = prof->readLen;
    gssw_align* alignment = node->alignment;
    if (alignment) {
        gssw_align_destroy(alignment);
    }
    node->alignment = alignment = gssw_align_create();
    bests = gssw_sw_sse2_byte((const int8_t*)node->num, 0, node->len, readLen,
                               weight_gapO, weight_gapE, prof->profile_byte, -1,
                               prof->bias, maskLen, alignment, seed);
    alignment->score1 = bests[0].score;
    alignment->ref_end1 = bests[0].ref;
    alignment->read_end1 = bests[0].read;
    if (maskLen >= 15) {
        alignment->score2 = bests[1].score;
        alignment->ref_end2 = bests[1].ref;
    } else {
        alignment->score2 = 0;
        alignment->ref_end2 = -1;
    }
    free(bests);
    return node;
}

gssw_graph* gssw_graph_create(uint32_t size) {
    gssw_graph* g = calloc(1, sizeof(gssw_graph));
    g->nodes = malloc(size*sizeof(gssw_node*));
    if (!g || !g->nodes) { fprintf(stderr, "error:[gssw] Could not allocate memory for graph of %u nodes.\n", size); exit(1); }
    return g;
}

void gssw_graph_clear_alignment(gssw_graph* g) {
    g->max_node = NULL;
    uint32_t i;
    for (i = 0; i < g->size; ++i) {
        gssw_node_clear_alignment(g->nodes[i]);
    }
}

void gssw_graph_destroy(gssw_graph* g) {
    uint32_t i;
    for (i = 0; i < g->size; ++i) {
        gssw_node_destroy(g->nodes[i]);
    }
    g->max_node = NULL;
    free(g->nodes);
    g->nodes = NULL;
    free(g);
}

uint32_t gssw_graph_add_node(gssw_graph* graph, gssw_node* node) {
    if (UNLIKELY(graph->size % 1024 == 0)) {
        size_t old_size = graph->size * sizeof(void*);
        size_t increment = 1024 * sizeof(void*);
        if (UNLIKELY(!(graph->nodes = realloc((void*)graph->nodes, old_size + increment)))) {
            fprintf(stderr, "error:[gssw] could not allocate memory for graph\n"); exit(1);
        }
    }
    ++graph->size;
    graph->nodes[graph->size-1] = node;
    return graph->size;
}

int8_t* gssw_create_num(const char* seq,
                        const int32_t len,
                        const int8_t* nt_table) {
    int32_t m;
    int8_t* num = (int8_t*)malloc(len);
    for (m = 0; m < len; ++m) num[m] = nt_table[(int)seq[m]];
    return num;
}


int8_t* gssw_create_score_matrix(int32_t match, int32_t mismatch) {
    // initialize scoring matrix for genome sequences
    //  A  C  G  T    N (or other ambiguous code)
    //  2 -2 -2 -2     0    A
    // -2  2 -2 -2     0    C
    // -2 -2  2 -2     0    G
    // -2 -2 -2  2     0    T
    //    0  0  0  0  0    N (or other ambiguous code)
    int32_t l, k, m;
    int8_t* mat = (int8_t*)calloc(25, sizeof(int8_t));
    for (l = k = 0; l < 4; ++l) {
        for (m = 0; m < 4; ++m) mat[k++] = l == m ? match : - mismatch;    /* weight_match : -weight_mismatch */
        mat[k++] = 0; // ambiguous base: no penalty
    }
    for (m = 0; m < 5; ++m) mat[k++] = 0;
    return mat;
}

int8_t* gssw_create_nt_table(void) {
    int8_t* ret_nt_table = calloc(128, sizeof(int8_t));
    int8_t nt_table[128] = {
        4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
        4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
        4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
        4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
        4, 0, 4, 1,  4, 4, 4, 2,  4, 4, 4, 4,  4, 4, 4, 4,
        4, 4, 4, 4,  3, 0, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,
        4, 0, 4, 1,  4, 4, 4, 2,  4, 4, 4, 4,  4, 4, 4, 4,
        4, 4, 4, 4,  3, 0, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4
    };
    memcpy(ret_nt_table, nt_table, 128*sizeof(int8_t));
    return ret_nt_table;
}

/* Rounds a double to nearest int8_t. */
int8_t gssw_round8_t(double x) {
    int8_t int_x = (int8_t) x;
    if (x >= 0.0) {
        if (x - int_x >= 0.5) {
            return int_x + (int8_t) 1;
        }
        else {
            return int_x;
        }
    }
    else {
        if (int_x - x >= 0.5) {
            return int_x - (int8_t) 1;
        }
        else {
            return int_x;
        }
    }
}

/* Simple (slow) algorithm for finding greatest common factor of the scores, not performance critical. */
int8_t gssw_score_gcf(const int8_t* score_matrix, int32_t alphabet_size) {
    int8_t* score_matrix_copy = (int8_t*) malloc(sizeof(int8_t) * alphabet_size * alphabet_size);
    int32_t i;
    for (i = 0; i < alphabet_size * alphabet_size; ++i) {
        score_matrix_copy[i] = score_matrix[i];
    }
    int8_t gcf = 1;
    int8_t factor = 2;
    int8_t min_score = 127;
    for (i = 0; i < alphabet_size * alphabet_size; ++i) {
        if (abs(score_matrix[i]) < min_score) {
            min_score = (int8_t) abs(score_matrix[i]);
        }
    }
    while (factor <= min_score / 2) {
        int8_t common_factor = 1;
        for (i = 0; i < alphabet_size * alphabet_size; ++i) {
            if (score_matrix_copy[i] % factor != 0) {
                common_factor = 0;
                break;
            }
        }
        if (common_factor) {
            gcf *= factor;
            for (i = 0; i < alphabet_size * alphabet_size; ++i) {
                score_matrix_copy[i] /= factor;
            }
            min_score /= factor;
        }
        else {
            factor++;
        }
    }
    free(score_matrix_copy);
    return gcf;
}

int8_t gssw_verify_valid_log_odds_score_matrix(const int8_t* score_matrix, const double* char_freqs,
                                               uint32_t alphabet_size) {
    int32_t i, j;
    int8_t contains_positive_score = 0.0;
    for (i = 0; i < alphabet_size * alphabet_size; i++) {
        if (score_matrix[i] > 0) {
            contains_positive_score = 1;
            break;
        }
    }
    if (!contains_positive_score) {
        return 0;
    }

    double expected_score = 0.0;
    for (i = 0; i < alphabet_size; i++) {
        for (j = 0; j < alphabet_size; j++) {
            expected_score += char_freqs[i] * char_freqs[j] * score_matrix[i * alphabet_size + j];
        }
    }
    return (int8_t) (expected_score < 0.0);
}

/* Returns the total probability in the distribution of aligned characters with a given logarithm base */
double gssw_alignment_partition_func(double lam, const int8_t* score_matrix, const double* char_freqs,
                                     uint32_t alphabet_size) {
    int32_t i, j;
    double partition = 0.0;
    for (i = 0; i < alphabet_size; i++) {
        for (j = 0; j < alphabet_size; j++) {
            partition += char_freqs[i] * char_freqs[j] * exp(lam * score_matrix[i * alphabet_size + j]);
        }
    }

    if (isnan(partition)) {
        fprintf(stderr, "error:[gssw] overflow error in log-odds base recovery subroutine.\n");
        exit(EXIT_FAILURE);
    }

    return partition;
}

/* Numerical routine to compute the base of the logarithm that translates alignment scores to log-odds */
double gssw_recover_log_base(const int8_t* score_matrix, const double* char_freqs, uint32_t alphabet_size, double tol) {

    if (!gssw_verify_valid_log_odds_score_matrix(score_matrix, char_freqs, alphabet_size)) {
        fprintf(stderr, "error:[gssw] score matrix does not correspond to log-odds of any distribution, cannot adjust for base quality.\n");
        exit(EXIT_FAILURE);
    }

    // searching for a positive value (because it's a base of a logarithm)
    double lower_bound;
    double upper_bound;

    // arbitrary starting point greater than zero
    double lam = 1.0;
    // search for a window containing lambda where total probability is 1
    double partition = gssw_alignment_partition_func(lam, score_matrix, char_freqs, alphabet_size);
    if (partition < 1.0) {
        lower_bound = lam;
        while (partition <= 1.0) {
            lower_bound = lam;
            lam *= 2.0;
            partition = gssw_alignment_partition_func(lam, score_matrix, char_freqs, alphabet_size);
        }
        upper_bound = lam;
    }
    else {
        upper_bound = lam;
        while (partition >= 1.0) {
            upper_bound = lam;
            lam /= 2.0;
            partition = gssw_alignment_partition_func(lam, score_matrix, char_freqs, alphabet_size);
        }
        lower_bound = lam;
    }

    // bisect to find a log base where total probability is 1
    while (upper_bound / lower_bound - 1.0 > tol) {
        lam = 0.5 * (lower_bound + upper_bound);
        if (gssw_alignment_partition_func(lam, score_matrix, char_freqs, alphabet_size) < 1.0) {
            lower_bound = lam;
        }
        else {
            upper_bound = lam;
        }
    }

    return 0.5 * (lower_bound + upper_bound);
}

double gssw_dna_recover_log_base(int8_t match, int8_t mismatch, double gc_content, double tol) {
    double gc_freq = gc_content / 2.0;
    double at_freq = 0.5 - gc_freq;
    double* nt_freqs = (double*) malloc(sizeof(double) * 4);
    nt_freqs[0] = at_freq; nt_freqs[1] = gc_freq; nt_freqs[2] = gc_freq; nt_freqs[3] = at_freq;
    int8_t* score_matrix = (int8_t*) malloc(sizeof(int8_t) * 16);
    int32_t i, j;
    for (i = 0; i < 4; i++) {
        for (j = 0; j < 4; j++) {
            score_matrix[i * 4 + j] = (i == j) ? match : -mismatch;
        }
    }
    double log_base = gssw_recover_log_base(score_matrix, nt_freqs, 4, 1e-12);
    free(nt_freqs);
    free(score_matrix);
    return log_base;
}

/* Returns a 3-dimensional matrix of quality-adjusted scores indexed by (qual score) x (ref base) x (query base). */
int8_t* gssw_adjusted_qual_matrix(uint8_t max_qual, const int8_t* score_matrix, const double* char_freqs,
                                  uint32_t alphabet_size, double tol){

    int32_t i, j, k, q;
    // recover base of logarithm used in log odds scores
    double lam;
    // factoring out GCF can avoid numerical problems without affecting correctness
    int8_t gcf = gssw_score_gcf(score_matrix, alphabet_size);
    int8_t* score_matrix_scaled = (int8_t*) malloc(sizeof(int8_t) * alphabet_size * alphabet_size);
    for (i = 0; i < alphabet_size * alphabet_size; i++) {
        score_matrix_scaled[i] = score_matrix[i] / gcf;
    }
    lam = gssw_recover_log_base(score_matrix_scaled, char_freqs, alphabet_size, tol) / gcf;
    free(score_matrix_scaled);

    // recover the emission probabilities of the align state of the HMM
    int32_t mat_size = alphabet_size * alphabet_size;
    double* align_prob = (double*) malloc(sizeof(double) * mat_size);

    for (i = 0; i < alphabet_size; i++) {
        for (j = 0; j < alphabet_size; j++) {
            align_prob[i * alphabet_size + j] = exp(lam * score_matrix[i * alphabet_size + j])
                                                      * char_freqs[i] * char_freqs[j];
        }
    }

    // compute the sum of the emission probabilities under a base error
    double* align_complement_prob = (double*) malloc(sizeof(double) * mat_size);
    for (i = 0; i < alphabet_size; i++) {
        for (j = 0; j < alphabet_size; j++) {
            align_complement_prob[i * alphabet_size + j] = 0.0;
            for (k = 0; k < alphabet_size; k++) {
                if (k != j) {
                    align_complement_prob[i * alphabet_size + j] += align_prob[i * alphabet_size + k];
                }
            }
        }
    }

    // quality score of random guessing
    int8_t lowest_meaningful_qual = gssw_round8_t(-10.0 * log10(1.0 - 1.0 / alphabet_size));

    // compute the adjusted alignment scores for each quality level
    int8_t* adj_qual_mat = (int8_t*) calloc(mat_size * (max_qual + (int8_t) 1), sizeof(int8_t));
    double score, err;
    for (q = lowest_meaningful_qual; q <= max_qual; q++) {
        err = pow(10.0, -q / 10.0);
        for (i = 0; i < alphabet_size; i++) {
            for (j = 0; j < alphabet_size; j++) {
                score = log(((1.0 - err) * align_prob[i * alphabet_size + j] + (err / (alphabet_size - 1.0)) * align_complement_prob[i * alphabet_size + j])
                            / (char_freqs[i] * ((1.0 - err) * char_freqs[j] + (err / (alphabet_size - 1.0)) * (1.0 - char_freqs[j]))));
                score /= lam;
                adj_qual_mat[q * mat_size + i * alphabet_size + j] = gssw_round8_t(score);
            }
        }
    }

    free(align_complement_prob);
    free(align_prob);

    return adj_qual_mat;
}

/* Returns a 3-dimensional matrix of quality-adjusted scores indexed by (qual score) x (ref base) x (query base)
 * that have been scaled up to (at most) a max score to accentuate differences, also adjusts value of gap penalties. */
int8_t* gssw_scaled_adjusted_qual_matrix(int8_t max_score, uint8_t max_qual, int8_t* gap_open_out, int8_t* gap_extend_out,
                                         const int8_t* score_matrix, const double* char_freqs, uint32_t alphabet_size,
                                         double tol) {

    int8_t gap_extend = *gap_extend_out;
    int8_t gap_open = *gap_open_out;

    // find largest integer multiplier that keeps all scores under maximum
    uint8_t multiplier = (uint8_t) abs(max_score);
    if (abs(max_score / gap_open) < multiplier) {
        multiplier = (uint8_t) max_score / gap_open;
    }
    if (abs(max_score / gap_extend) < multiplier) {
        multiplier = (uint8_t) max_score / gap_extend;
    }
    int32_t i;
    for (i = 0; i < alphabet_size * alphabet_size; i++) {
        if (abs(max_score / score_matrix[i]) < multiplier) {
            multiplier = (uint8_t) abs(max_score / score_matrix[i]);
        }
    }

    if (multiplier == 0) {
        fprintf(stderr, "error:[gssw] max scaled score is smaller than baseline score.\n");
        exit(EXIT_FAILURE);
    }

    // scale scores by multiplier
    int8_t* scaled_score_mat = (int8_t*) malloc(sizeof(int8_t) * alphabet_size * alphabet_size);

    for (i = 0; i < alphabet_size * alphabet_size; i++) {
        scaled_score_mat[i] = multiplier * score_matrix[i];
    }

    // compute adjusted score matrices
    int8_t* scaled_adj_qual_mat = gssw_adjusted_qual_matrix(max_qual, scaled_score_mat, char_freqs, alphabet_size, tol);

    free(scaled_score_mat);

    *gap_open_out = multiplier * gap_open;
    *gap_extend_out = multiplier * gap_extend;

    return scaled_adj_qual_mat;
}

int8_t* gssw_add_ambiguous_char_to_adjusted_matrix(int8_t* adj_mat, uint8_t max_qual, uint32_t alphabet_size) {
    int32_t mat_size = alphabet_size * alphabet_size;
    int32_t aug_alph_size = alphabet_size + 1;
    int32_t aug_mat_size = aug_alph_size * aug_alph_size;

    int8_t* aug_adj_mat = (int8_t*) malloc(sizeof(int8_t) * aug_mat_size * (max_qual + 1));

    int32_t q, i, j;
    for (q = 0; q <= max_qual; q++) {
        for (i = 0; i < aug_alph_size; i++) {
            for (j = 0; j < aug_alph_size; j++) {
                if (i == alphabet_size || j == alphabet_size) {
                    aug_adj_mat[q * aug_mat_size + i * aug_alph_size + j] = 0;
                }
                else {
                    aug_adj_mat[q * aug_mat_size + i * aug_alph_size + j] = adj_mat[q * mat_size + i * alphabet_size + j];
                }
            }
        }
    }

    return aug_adj_mat;
}

// automatically adds 0-scoring N to the final row and column
int8_t* gssw_dna_scaled_adjusted_qual_matrix(int8_t max_score, uint8_t max_qual, int8_t* gap_open_out,
                                             int8_t* gap_extend_out, int8_t match_score, int8_t mismatch_score,
                                             double gc_content, double tol) {

    double gc_freq = gc_content / 2.0;
    double at_freq = 0.5 - gc_freq;
    double* nt_freqs = (double*) malloc(sizeof(double) * 4);
    nt_freqs[0] = at_freq; nt_freqs[1] = gc_freq; nt_freqs[2] = gc_freq; nt_freqs[3] = at_freq;

    int32_t i, j;
    int8_t* score_matrix = (int8_t*) malloc(sizeof(int8_t) * 16);
    for (i = 0; i < 4; ++i) {
        for (j = 0; j < 4; ++j) {
            score_matrix[i * 4 + j] = (i == j) ? match_score : -mismatch_score;
        }
    }


    int8_t* adj_mat_init = gssw_scaled_adjusted_qual_matrix(max_score, max_qual, gap_open_out,
                                                            gap_extend_out, score_matrix,
                                                            nt_freqs, 4, tol);

    int8_t* adj_mat = gssw_add_ambiguous_char_to_adjusted_matrix(adj_mat_init, max_qual, 4);

    free(nt_freqs);
    free(score_matrix);
    free(adj_mat_init);

    return adj_mat;
}
