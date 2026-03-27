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

// Hardcoded scoring: match=1, mismatch=-4, n=4, bias=4, bonus=5/5
gssw_profile* gssw_init(const int8_t* read, int32_t readLen) {
    static const int8_t mat[16] = {
         1, -4, -4, -4,
        -4,  1, -4, -4,
        -4, -4,  1, -4,
        -4, -4, -4,  1
    };
    const int32_t n = 4;
    const uint8_t bias = 4;
    const int8_t start_bonus = 5;
    const int8_t end_bonus = 5;

    gssw_profile* p = (gssw_profile*)calloc(
        1, sizeof(struct gssw_profile));
    p->bias = bias;
    p->profile_byte = gssw_qP_byte(
        read, mat, readLen, n, bias, start_bonus, end_bonus);
    p->read = read;
    p->readLen = readLen;
    return p;
}

void gssw_init_destroy (gssw_profile* p) {
    free(p->profile_byte);
    free(p);
}

gssw_align* gssw_align_create (void) {
    gssw_align* a = (gssw_align*)calloc(1, sizeof(gssw_align));
    a->seed.pvHStore = NULL;
    a->seed.pvE = NULL;
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

void gssw_profile_destroy(gssw_profile* prof) {
    free(prof->profile_byte);
    free(prof);
}

void gssw_node_destroy(gssw_node* n) {
    free(n->num);
    free(n->prev);
    free(n->next);
    if (n->alignment) {
        gssw_align_destroy(n->alignment);
    }
    free(n);
}

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

gssw_graph* gssw_graph_create(uint32_t size) {
    gssw_graph* g = calloc(1, sizeof(gssw_graph));
    g->nodes = malloc(size*sizeof(gssw_node*));
    if (!g || !g->nodes) { fprintf(stderr, "error:[gssw] Could not allocate memory for graph of %u nodes.\n", size); exit(1); }
    return g;
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

// SoA graph fill: push-based seed propagation, returns best score
// Hardcoded: gapO=6, gapE=1, maskLen=15
uint16_t gssw_soa_graph_fill(gssw_soa_graph* graph,
                              const int8_t* read_num,
                              int32_t read_length) {
    const uint8_t weight_gapO = 6;
    const uint8_t weight_gapE = 1;
    const int32_t maskLen = 15;

    gssw_profile* prof = gssw_init(read_num, read_length);

    int32_t segLen = (read_length + 15) / 16;
    uint32_t N = graph->num_nodes;

    // Allocate per-node accumulated seeds (push-based)
    // Each node needs segLen __m128i for H and segLen for E
    __m128i** seed_H = (__m128i**)calloc(N, sizeof(__m128i*));
    __m128i** seed_E = (__m128i**)calloc(N, sizeof(__m128i*));
    for (uint32_t i = 0; i < N; i++) {
        seed_H[i] = (__m128i*)calloc(segLen, sizeof(__m128i));
        seed_E[i] = (__m128i*)calloc(segLen, sizeof(__m128i));
    }

    uint16_t max_score = 0;
    gssw_align* alignment = gssw_align_create();

    for (uint32_t i = 0; i < N; i++) {
        gssw_node_desc* nd = &graph->nodes[i];

        // Build input seed from accumulated H/E for this node
        gssw_seed seed;
        seed.pvHStore = seed_H[i];
        seed.pvE = seed_E[i];

        // Clear alignment's output seed from prior iteration
        gssw_align_clear_matrix_and_seed(alignment);

        // Run the SSE2 kernel on this node's sequence
        const int8_t* node_seq = graph->seqs + nd->seq_off;
        gssw_alignment_end* bests = gssw_sw_sse2_byte(
            node_seq, 0, nd->seq_len, read_length,
            weight_gapO, weight_gapE, prof->profile_byte, -1,
            prof->bias, maskLen, alignment, &seed);

        uint16_t score = bests[0].score;
        if (score > max_score) max_score = score;
        free(bests);

        // Push output seed to children (element-wise SIMD max)
        for (int16_t c = 0; c < nd->next_len; c++) {
            int16_t child = graph->nexts[nd->next_off + c];
            for (int32_t j = 0; j < segLen; j++) {
                seed_H[child][j] = _mm_max_epu8(
                    seed_H[child][j],
                    alignment->seed.pvHStore[j]);
                seed_E[child][j] = _mm_max_epu8(
                    seed_E[child][j],
                    alignment->seed.pvE[j]);
            }
        }
    }

    // Cleanup
    for (uint32_t i = 0; i < N; i++) {
        free(seed_H[i]);
        free(seed_E[i]);
    }
    free(seed_H);
    free(seed_E);
    gssw_align_destroy(alignment);
    gssw_profile_destroy(prof);

    return max_score;
}

void gssw_soa_graph_destroy(gssw_soa_graph* g) {
    if (!g) return;
    free(g->nodes);
    free(g->nexts);
    free(g->seqs);
    free(g);
}


