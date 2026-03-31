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

/*!	@typedef	structure of the query profile	*/
struct gssw_profile;
typedef struct gssw_profile gssw_profile;

// This file makes extensive use of SSE intrinsics, which operate on 128-bit
// values. These values are stored in the 128-bit special registers XMM0 through
// XMM7, as assigned by the compiler. In C, these values have type "__m128i",
// with the "i" meaning integer (because we do integer math on them). We have
// code to use these in alignments, either to store 16 single-byte values or 8
// double-byte values, depending on how large we're allowing the scores to get.

// We use j for indexes in the read (or "query"), and i for indexes in the
// reference (or "database"). Conceptually, the database runs rightwards along
// the top of the DP matrix, and the read runs downwards along the left.

// Note that this usage of i and j is BACKWARDS from the paper!

// Within each node, we perform a standard-ish Smith-Waterman alignment with
// affine gaps. This involves three matrices. H[i, j] gives the score of the
// very best alignment using read characters through j and reference characters
// through i, no matter what it ends in (match, mismatch, or gap). We also have
// E[i, j] which gives the score of the best such alignment ending in a gap in
// the read, and F[i, j], which gives the score of the best such alignment
// ending in a gap in the reference.

// During alignment, we work on one column (i.e. reference character) of all the
// DP matrices at a time. This column is *not* divided top to bottom into SSE
// vectors. Instead, the column is broken up across SSE vectors in a "striped"
// format, as follows:

// Take the read, and break it into 16 (or 8 for word-scored alignment) equal-
// length pieces, padding if necessary. Line those pieces up left to right. Then
// run through *that* matrix in rows. Each row is an SSE 128-bit chunk, and is
// processed all at once.

// Another way to think of the layout is to have 16 or 8 cursors moving down the
// column in parallel. Each cursor has easy access to the contents of the cell
// above it, except for the first one (which is fixed with some fix-up logic
// called the "lazy F loop").

// Each node has a notion of a "seed", which carries the E information (about
// gaps in the read that can continue through the node boundary) and the H
// information (about the best alignments overall before we entered this node's
// matrix). F information is not carried over, because gaps in the reference
// can't span multiple reference nodes.
typedef struct {
    // Stores the E values (best gap in read scores) for the *next* column to be
    // generated, in the matrix to be filled. They are known in advance.
    __m128i* pvE;
    // Stores the H values (overall best scores) from the previous column, before the matrix to be filled.
    __m128i* pvHStore;
} gssw_seed;


/*!	@typedef	structure of the alignment result
	@field	score1	the best alignment score
*/
typedef struct {
	uint16_t score1;
    gssw_seed seed;
} gssw_align;


// A profile is a sort of exploded score matrix. We fill a whole read by
// reference matrix with the score you would get by matching a particular
// character in the read with a particular character in the reference. There's
// no data dependence between any entries, so we can just fill this in.
struct gssw_profile{
    // We keep one version, stored striped, for byte-sized alignment
	__m128i* profile_byte;	// 0: none
	const int8_t* read;
	int32_t readLen;
	uint8_t bias;
};

//struct node;
//typedef struct node s_node;
typedef struct _gssw_node gssw_node;
typedef struct _gssw_node {
    uint64_t id;
    int8_t* num; // numerical encoding of sequence
    int32_t len; // length of sequence
    gssw_node** prev;
    int32_t count_prev;
    gssw_node** next;
    int32_t count_next;
    gssw_align* alignment;
} _gssw_node;

typedef struct {
    uint32_t size;
    gssw_node* max_node;
    gssw_node** nodes;
} gssw_graph;

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
#endif // __cplusplus

/*!	@function	Create the query profile using the query sequence.
	All scoring parameters are hardcoded: match=1, mismatch=-4, bias=4,
	start_full_length_bonus=5, end_full_length_bonus=5.
	@param	read	pointer to numeric read (0=A,1=C,2=G,3=T)
	@param	readLen	length of the query sequence
	@return	pointer to the query profile structure
*/
gssw_profile* gssw_init(const int8_t* read, int32_t readLen);

void gssw_init_destroy(gssw_profile* p);

gssw_align* gssw_align_create(void);
void gssw_align_destroy(gssw_align* a);
void gssw_align_clear_matrix_and_seed(gssw_align* a);

void gssw_profile_destroy(gssw_profile* prof);
void gssw_seed_destroy(gssw_seed* seed);

void gssw_node_destroy(gssw_node* n);
void gssw_node_add_prev(gssw_node* n, gssw_node* m);
void gssw_node_add_next(gssw_node* n, gssw_node* m);
void gssw_nodes_add_edge(gssw_node* n, gssw_node* m);

gssw_graph* gssw_graph_create(uint32_t size);
void gssw_graph_destroy(gssw_graph* graph);

// SoA graph alignment (push-based seed propagation, returns best score)
// All scoring params hardcoded: gapO=6, gapE=1, maskLen=15
// Profile must be pre-built via gssw_init()
uint16_t gssw_soa_graph_fill(gssw_soa_graph* graph,
                              gssw_profile* prof);

void gssw_soa_graph_destroy(gssw_soa_graph* g);

// Print cumulative timing breakdown (profile, kernel, overhead)
void gssw_print_timers(void);


#ifdef __cplusplus
}
#endif	// __cplusplus

#undef SIMDE_ENABLE_NATIVE_ALIASES
#endif	// SSW_H
