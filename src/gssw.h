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
	const int8_t* mat;
	int32_t readLen;
	int32_t n;
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
	@param	read	pointer to the query sequence; the query sequence needs to be numbers
	@param	readLen	length of the query sequence
	@param	mat	pointer to the substitution matrix; mat needs to be corresponding to the read sequence
	@param	n	the square root of the number of elements in mat (mat has n*n elements)
	@param	start_full_length_bonus	a bonus for aligning the full length of an alignment at the start of a query
	@param	end_full_length_bonus	a bonus for aligning the full length of an alignment at the end of a query
	@return	pointer to the query profile structure
	@note	example for parameter read and mat:
			If the query sequence is: ACGTATC, the sequence that read points to can be: 1234142
			Then if the penalty for match is 2 and for mismatch is -2, the substitution matrix of parameter mat will be:
			//A  C  G  T
			  2 -2 -2 -2 //A
			 -2  2 -2 -2 //C
			 -2 -2  2 -2 //G
			 -2 -2 -2  2 //T
			mat is the pointer to the array {2, -2, -2, -2, -2, 2, -2, -2, -2, -2, 2, -2, -2, -2, -2, 2}
*/
gssw_profile* gssw_init (const int8_t* read, const int32_t readLen, const int8_t* mat, const int32_t n,
                         int8_t start_full_length_bonus, int8_t end_full_length_bonus);

/*!	@function	Release the memory allocated by function ssw_init.
	@param	p	pointer to the query profile structure
*/
void gssw_init_destroy (gssw_profile* p);

gssw_align* gssw_align_create(void);


// @function	ssw alignment.
/*!	@function	Do Striped Smith-Waterman alignment.
	@param	prof	pointer to the query profile structure
	@param	ref	pointer to the target sequence; the target sequence needs to be numbers and corresponding to the mat parameter of
				function ssw_init
	@param	refLen	length of the target sequence
	@param	weight_gapO	the absolute value of gap open penalty
	@param	weight_gapE	the absolute value of gap extension penalty
	@param	flag	bitwise FLAG; (from high to low) bit 5: when setted as 1, function ssw_align will return the best alignment
					beginning position; bit 6: when setted as 1, if (ref_end1 - ref_begin1 < filterd && read_end1 - read_begin1
					< filterd), (whatever bit 5 is setted) the function will return the best alignment beginning position and
					cigar; bit 7: when setted as 1, if the best alignment score >= filters, (whatever bit 5 is setted) the function
 					will return the best alignment beginning position and cigar; bit 8: when setted as 1, (whatever bit 5, 6 or 7 is
					setted) the function will always return the best alignment beginning position and cigar. When flag == 0, only
					the optimal and sub-optimal scores and the optimal alignment ending position will be returned.
	@param	filters	score filter: when bit 7 of flag is setted as 1 and bit 8 is setted as 0, filters will be used (Please check the
					decription of the flag parameter for detailed usage.)
	@param	filterd	distance filter: when bit 6 of flag is setted as 1 and bit 8 is setted as 0, filterd will be used (Please check
					the decription of the flag parameter for detailed usage.)
	@param	maskLen	The distance between the optimal and suboptimal alignment ending position >= maskLen. We suggest to use
					readLen/2, if you don't have special concerns. Note: maskLen has to be >= 15, otherwise this function will NOT
					return the suboptimal alignment information. Detailed description of maskLen: After locating the optimal
					alignment ending position, the suboptimal alignment score can be heuristically found by checking the second
					largest score in the array that contains the maximal score of each column of the SW matrix. In order to avoid
					picking the scores that belong to the alignments sharing the partial best alignment, SSW C library masks the
					reference loci nearby (mask length = maskLen) the best alignment ending position and locates the second largest
					score from the unmasked elements.
	@return	pointer to the alignment result structure
	@note	Whatever the parameter flag is setted, this function will at least return the optimal and sub-optimal alignment score,
			and the optimal alignment ending positions on target and query sequences. If both bit 6 and 7 of the flag are setted
			while bit 8 is not, the function will return cigar only when both criteria are fulfilled. All returned positions are
			0-based coordinate.
*/
gssw_align* gssw_fill (const gssw_profile* prof,
                       const int8_t* ref,
                       const int32_t refLen,
                       const uint8_t weight_gapO,
                       const uint8_t weight_gapE,
                       const int32_t maskLen,
                       gssw_seed* seed);


/*!	@function	Release the memory allocated by function ssw_align.
	@param	a	pointer to the alignment result structure
*/
void gssw_align_destroy (gssw_align* a);

/*!	@function	Release the memory allocated for matrices and pvE in s_align.
	@param	a	pointer to the alignment result structure
*/
void gssw_align_clear_matrix_and_seed (gssw_align* a);

void gssw_profile_destroy(gssw_profile* prof);
void gssw_seed_destroy(gssw_seed* seed);
gssw_seed* gssw_create_seed_byte(int32_t readLen, gssw_node** prev, int32_t count);

void gssw_node_destroy(gssw_node* n);
void gssw_node_add_prev(gssw_node* n, gssw_node* m);
void gssw_node_add_next(gssw_node* n, gssw_node* m);
void gssw_nodes_add_edge(gssw_node* n, gssw_node* m);


gssw_node*
gssw_node_fill (gssw_node* node,
                const gssw_profile* prof,
                const uint8_t weight_gapO,
                const uint8_t weight_gapE,
                const int32_t maskLen,
                const gssw_seed* seed);

gssw_graph*
gssw_graph_fill (gssw_graph* graph,
                 const char* read_seq,
                 const int8_t* nt_table,
                 const int8_t* score_matrix,
                 const uint8_t weight_gapO,
                 const uint8_t weight_gapE,
                 const int8_t start_full_length_bonus,
                 const int8_t end_full_length_bonus,
                 const int32_t maskLen);

gssw_graph*
gssw_graph_fill_pinned (gssw_graph* graph,
                        const char* read_seq,
                        const int8_t* nt_table,
                        const int8_t* score_matrix,
                        const uint8_t weight_gapO,
                        const uint8_t weight_gapE,
                        const int8_t start_full_length_bonus,
                        const int8_t end_full_length_bonus,
                        const int32_t maskLen);

gssw_graph* gssw_graph_create(uint32_t size);
void gssw_graph_destroy(gssw_graph* graph);

// SoA graph alignment (push-based seed propagation, returns best score)
uint16_t gssw_soa_graph_fill(gssw_soa_graph* graph,
                              const char* read_seq,
                              const int8_t* nt_table,
                              const int8_t* score_matrix,
                              uint8_t weight_gapO,
                              uint8_t weight_gapE,
                              int8_t start_full_length_bonus,
                              int8_t end_full_length_bonus,
                              int32_t maskLen);

void gssw_soa_graph_destroy(gssw_soa_graph* g);

// some utility functions
int8_t* gssw_create_score_matrix(int32_t match, int32_t mismatch);
int8_t* gssw_create_nt_table(void);
int8_t* gssw_create_num(const char* seq,
                        const int32_t len,
                        const int8_t* nt_table);


#ifdef __cplusplus
}
#endif	// __cplusplus

#undef SIMDE_ENABLE_NATIVE_ALIASES
#endif	// SSW_H
