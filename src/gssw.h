/*
 *  gssw.h
 *
 *  Created by Erik Garrison, based on SSW by Mengyao Zhao.
 *
 */

#ifndef SSW_H
#define SSW_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdbool.h>
#define SIMDE_ENABLE_NATIVE_ALIASES
#include "simde/x86/sse4.1.h"

// A profile is a precomputed, striped score matrix for a read.
// 4 nucleotides * segLen vectors, each 16 bytes.
struct gssw_profile{
    __m128i* profile_byte;
    const int8_t* read;
    int32_t readLen;
    uint8_t bias;
};
typedef struct gssw_profile gssw_profile;

// Descriptor for one node in the SoA graph
typedef struct {
    int16_t seq_off;   // offset into seqs array
    int16_t seq_len;   // length of this node's numeric sequence
    int16_t next_off;  // offset into nexts array
    int16_t next_len;  // number of children
} gssw_node_desc;

// Struct-of-arrays graph (nodes indexed by topological order)
typedef struct {
    uint32_t num_nodes;
    gssw_node_desc* nodes;   // array[num_nodes]
    int16_t* nexts;          // flattened child node indices
    int8_t* seqs;            // flattened numeric sequences
    uint32_t total_nexts;    // total length of nexts array
    uint32_t total_seq;      // total length of seqs array
} gssw_soa_graph;


#ifdef __cplusplus
extern "C" {
#endif

void gssw_profile_destroy(gssw_profile* prof);
void gssw_soa_graph_destroy(gssw_soa_graph* g);

// ===== Allocation-free accelerator kernel =====

#define GSSW_READ_LEN  148
#define GSSW_SEG_LEN   10     // ceil(148/16)

// SPM fixed-region offsets (in bytes). Each vector is 16 bytes.
#define GSSW_PROF_OFF   0                 // 40 vectors
#define GSSW_HPING_OFF  (40 * 16)         //  640
#define GSSW_HPONG_OFF  (50 * 16)         //  800
#define GSSW_E_OFF      (60 * 16)         //  960
#define GSSW_F_OFF      (70 * 16)         // 1120
#define GSSW_BEST_OFF   (80 * 16)         // 1280
#define GSSW_GRAPH_OFF  (90 * 16)         // 1440

// Graph metadata header (at GRAPH_OFF)
typedef struct {
    uint32_t num_nodes;
    uint32_t total_nexts;
    uint32_t total_seq;
    uint32_t _pad;            // pad to 16 bytes
} gssw_spm_graph_meta;

// Per-node descriptor with inline seeds (336 bytes, 16-byte aligned)
typedef struct {
    int16_t seq_off;
    int16_t seq_len;
    int16_t next_off;
    int16_t next_len;
    int8_t  _pad[8];          // align hSeed to 16-byte boundary
    __m128i hSeed[GSSW_SEG_LEN];  // 160 bytes
    __m128i eSeed[GSSW_SEG_LEN];  // 160 bytes
} gssw_spm_node_desc;

// Compute total SPM size needed for a given graph
static inline uint64_t gssw_spm_size(
    uint32_t num_nodes, uint32_t total_nexts, uint32_t total_seq)
{
    uint64_t sz = GSSW_GRAPH_OFF;
    sz += sizeof(gssw_spm_graph_meta);
    sz += (uint64_t)num_nodes * sizeof(gssw_spm_node_desc);
    uint64_t nexts_bytes = (uint64_t)total_nexts * sizeof(int16_t);
    nexts_bytes = (nexts_bytes + 15) & ~15ULL;
    sz += nexts_bytes;
    sz += total_seq;
    return sz;
}

// Pack an existing SoA graph + profile into SPM layout.
void gssw_spm_pack(uint8_t* SPM,
                    const gssw_soa_graph* graph,
                    const gssw_profile* prof);

// Allocation-free graph alignment kernel.
// Returns the best alignment score.
uint16_t gssw_kernel(uint8_t* SPM);

#ifdef __cplusplus
}
#endif

#undef SIMDE_ENABLE_NATIVE_ALIASES
#endif  // SSW_H
