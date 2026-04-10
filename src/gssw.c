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

// Horizontal max of 16 uint8 lanes in a __m128i
#define m128i_max16(m, vm) \
    (vm) = _mm_max_epu8((vm), _mm_srli_si128((vm), 8)); \
    (vm) = _mm_max_epu8((vm), _mm_srli_si128((vm), 4)); \
    (vm) = _mm_max_epu8((vm), _mm_srli_si128((vm), 2)); \
    (vm) = _mm_max_epu8((vm), _mm_srli_si128((vm), 1)); \
    (m) = _mm_extract_epi16((vm), 0)

// Unsigned 8-bit compare-greater-than (missing from SSE2)
#define m128i_cmpgt(v0, v1) \
         _mm_cmpgt_epi8(_mm_xor_si128(v0, _mm_set1_epi8(-128)), \
                        _mm_xor_si128(v1, _mm_set1_epi8(-128)))


void gssw_profile_destroy(gssw_profile* prof) {
    free(prof->profile_byte);
    free(prof);
}

void gssw_soa_graph_destroy(gssw_soa_graph* g) {
    if (!g) return;
    free(g->nodes);
    free(g->nexts);
    free(g->seqs);
    free(g);
}


// ===== Allocation-free accelerator kernel =====

// Process one reference column. No allocation.
// Reads H from hPing, writes H to hPong. E and F are in-place.
static void gssw_sw_column(
    const __m128i* prof, const __m128i* hPing,
    __m128i* hPong, __m128i* pvE, __m128i* pvF,
    int8_t refBase, __m128i* vMaxColumn)
{
    const int32_t segLen = GSSW_SEG_LEN;
    const __m128i vZero = _mm_setzero_si128();
    const __m128i vGapO = _mm_set1_epi8(6);
    const __m128i vGapE = _mm_set1_epi8(1);
    const __m128i vBias = _mm_set1_epi8(4);
    const __m128i* vP = prof + refBase * segLen;

    // Load last segment's H, shift left 1 byte for striped dep
    __m128i vH = _mm_load_si128(hPing + (segLen - 1));
    vH = _mm_slli_si128(vH, 1);

    __m128i e, vF = vZero;
    *vMaxColumn = vZero;
    memset(pvF, 0, segLen * sizeof(__m128i));

    // Main inner loop
    int32_t j;
    for (j = 0; j < segLen; ++j) {
        vH = _mm_adds_epu8(vH, _mm_load_si128(vP + j));
        vH = _mm_subs_epu8(vH, vBias);

        e = _mm_load_si128(pvE + j);
        vH = _mm_max_epu8(vH, e);
        vH = _mm_max_epu8(vH, vF);
        *vMaxColumn = _mm_max_epu8(*vMaxColumn, vH);

        _mm_store_si128(hPong + j, vH);
        _mm_store_si128(pvF + j, vF);

        // Update E for next column
        vH = _mm_subs_epu8(vH, vGapO);
        e = _mm_subs_epu8(e, vGapE);
        e = _mm_max_epu8(e, vH);
        _mm_store_si128(pvE + j, e);

        // Update F for next segment
        vF = _mm_subs_epu8(vF, vGapE);
        vF = _mm_max_epu8(vF, vH);

        vH = _mm_load_si128(hPing + j);
    }

    // Lazy-F loop: fix cross-lane F propagation
    j = 0;
    vH = _mm_load_si128(hPong + j);
    vF = _mm_slli_si128(vF, 1);

    __m128i vTemp = m128i_cmpgt(vF, vH);
    int32_t cmp = _mm_movemask_epi8(vTemp);
    vTemp = _mm_load_si128(pvF + j);
    vTemp = m128i_cmpgt(vF, vTemp);
    cmp |= _mm_movemask_epi8(vTemp);

    while (cmp != 0x0000) {
        vH = _mm_max_epu8(vH, vF);
        *vMaxColumn = _mm_max_epu8(*vMaxColumn, vH);
        _mm_store_si128(hPong + j, vH);

        // Update E (gap-to-gap transition)
        e = _mm_load_si128(pvE + j);
        vTemp = _mm_subs_epu8(vH, vGapO);
        e = _mm_max_epu8(e, vTemp);
        _mm_store_si128(pvE + j, e);

        // Update stored F
        vTemp = _mm_load_si128(pvF + j);
        vTemp = _mm_max_epu8(vTemp, vF);
        _mm_store_si128(pvF + j, vTemp);

        vF = _mm_subs_epu8(vF, vGapE);

        j++;
        if (j >= segLen) {
            j = 0;
            vF = _mm_slli_si128(vF, 1);
        }

        vH = _mm_load_si128(hPong + j);
        vTemp = m128i_cmpgt(vF, vH);
        cmp = _mm_movemask_epi8(vTemp);
        vTemp = _mm_load_si128(pvF + j);
        vTemp = m128i_cmpgt(vF, vTemp);
        cmp |= _mm_movemask_epi8(vTemp);
    }
}

// Allocation-free graph alignment kernel.
uint16_t gssw_kernel(uint8_t* SPM) {
    const int32_t segLen = GSSW_SEG_LEN;

    __m128i* prof  = (__m128i*)(SPM + GSSW_PROF_OFF);
    __m128i* hPing = (__m128i*)(SPM + GSSW_HPING_OFF);
    __m128i* hPong = (__m128i*)(SPM + GSSW_HPONG_OFF);
    __m128i* pvE   = (__m128i*)(SPM + GSSW_E_OFF);
    __m128i* pvF   = (__m128i*)(SPM + GSSW_F_OFF);
    __m128i* best  = (__m128i*)(SPM + GSSW_BEST_OFF);

    // Graph region pointers
    gssw_spm_graph_meta* meta = (gssw_spm_graph_meta*)(SPM + GSSW_GRAPH_OFF);
    uint32_t numNodes = meta->num_nodes;
    gssw_spm_node_desc* nodeDescs = (gssw_spm_node_desc*)(
        SPM + GSSW_GRAPH_OFF + sizeof(gssw_spm_graph_meta));
    int16_t* childIds = (int16_t*)(
        (uint8_t*)nodeDescs + numNodes * sizeof(gssw_spm_node_desc));
    uint64_t nexts_bytes = ((uint64_t)meta->total_nexts * sizeof(int16_t) + 15) & ~15ULL;
    const int8_t* graphSeq = (const int8_t*)childIds + nexts_bytes;

    memset(best, 0, segLen * sizeof(__m128i));
    uint8_t overallMax = 0;

    for (uint32_t n = 0; n < numNodes; n++) {
        gssw_spm_node_desc* nd = &nodeDescs[n];

        // Load seed: H into hPing, E into pvE
        memcpy(hPing, nd->hSeed, segLen * sizeof(__m128i));
        memcpy(pvE, nd->eSeed, segLen * sizeof(__m128i));

        // Process each reference column
        const int8_t* seq = graphSeq + nd->seq_off;
        for (int32_t col = 0; col < nd->seq_len; col++) {
            __m128i vMaxColumn = _mm_setzero_si128();
            gssw_sw_column(prof, hPing, hPong, pvE, pvF,
                           seq[col], &vMaxColumn);

            // Update bestAlignment if this column is new best
            uint8_t colMax;
            m128i_max16(colMax, vMaxColumn);
            if (colMax > overallMax) {
                overallMax = colMax;
                memcpy(best, hPong, segLen * sizeof(__m128i));
            }

            // Swap H ping/pong
            __m128i* tmp = hPing;
            hPing = hPong;
            hPong = tmp;
        }

        // Push seed to children
        for (int16_t c = 0; c < nd->next_len; c++) {
            int16_t child = childIds[nd->next_off + c];
            gssw_spm_node_desc* cd = &nodeDescs[child];
            for (int32_t j = 0; j < segLen; j++) {
                cd->hSeed[j] = _mm_max_epu8(cd->hSeed[j], hPing[j]);
                cd->eSeed[j] = _mm_max_epu8(cd->eSeed[j], pvE[j]);
            }
        }
    }

    // Extract best score from bestAlignment
    __m128i vMax = _mm_setzero_si128();
    for (int32_t j = 0; j < segLen; j++)
        vMax = _mm_max_epu8(vMax, best[j]);
    uint8_t maxScore;
    m128i_max16(maxScore, vMax);
    return (uint16_t)maxScore;
}

// Pack an existing SoA graph + profile into SPM layout.
void gssw_spm_pack(uint8_t* SPM,
                    const gssw_soa_graph* graph,
                    const gssw_profile* prof)
{
    const int32_t segLen = GSSW_SEG_LEN;

    memcpy(SPM + GSSW_PROF_OFF, prof->profile_byte,
           4 * segLen * sizeof(__m128i));
    memset(SPM + GSSW_HPING_OFF, 0, (GSSW_GRAPH_OFF - GSSW_HPING_OFF));

    gssw_spm_graph_meta* meta = (gssw_spm_graph_meta*)(SPM + GSSW_GRAPH_OFF);
    meta->num_nodes = graph->num_nodes;
    meta->total_nexts = graph->total_nexts;
    meta->total_seq = graph->total_seq;
    meta->_pad = 0;

    gssw_spm_node_desc* nodeDescs = (gssw_spm_node_desc*)(
        SPM + GSSW_GRAPH_OFF + sizeof(gssw_spm_graph_meta));
    for (uint32_t i = 0; i < graph->num_nodes; i++) {
        const gssw_node_desc* src = &graph->nodes[i];
        gssw_spm_node_desc* dst = &nodeDescs[i];
        dst->seq_off  = src->seq_off;
        dst->seq_len  = src->seq_len;
        dst->next_off = src->next_off;
        dst->next_len = src->next_len;
        memset(dst->_pad, 0, sizeof(dst->_pad));
        memset(dst->hSeed, 0, segLen * sizeof(__m128i));
        memset(dst->eSeed, 0, segLen * sizeof(__m128i));
    }

    int16_t* childIds = (int16_t*)(
        (uint8_t*)nodeDescs + graph->num_nodes * sizeof(gssw_spm_node_desc));
    memcpy(childIds, graph->nexts, graph->total_nexts * sizeof(int16_t));
    uint64_t nexts_bytes = (uint64_t)graph->total_nexts * sizeof(int16_t);
    uint64_t nexts_padded = (nexts_bytes + 15) & ~15ULL;
    if (nexts_padded > nexts_bytes)
        memset((uint8_t*)childIds + nexts_bytes, 0, nexts_padded - nexts_bytes);

    int8_t* seqDst = (int8_t*)childIds + nexts_padded;
    memcpy(seqDst, graph->seqs, graph->total_seq);
}
