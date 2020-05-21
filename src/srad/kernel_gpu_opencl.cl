/**
 * Copyright (c)2008-2011 University of Virginia
 * Copyright (c)2020 Roy Spliet
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted without royalty fees or other restrictions,
 * provided that the following conditions are met:
 *
 * - Redistributions of source code must retain the above copyright notice,
 *   this list of conditions and the following disclaimer.
 * - Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 * - Neither the name of the University of Virginia, the Dept. of Computer
 *   Science, nor the names of its contributors may be used to endorse or
 *   promote products derived from this software without specific prior written
 *   permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE UNIVERSITY OF VIRGINIA OR THE SOFTWARE
 * AUTHORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
 * OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 * Modifications from upstream:
 * - Remove unused code,
 * - reduce_kernel: Clean up and deduplicate code,
 * - Add an NVIDIA-only reduction kernel that uses floating point atomics,
 * - Fix literals to be single-precision.
 */

//========================================================================================================================================================================================================200
//	DEFINE / INCLUDE
//========================================================================================================================================================================================================200

//======================================================================================================================================================150
//	MAIN FUNCTION HEADER
//======================================================================================================================================================150
#ifdef ECLIPSE
#define __kernel
#define __global
#define __local
#define CLK_LOCAL_MEM_FENCE 0
#define CLK_GLOBAL_MEM_FENCE 0
#endif

#include "src/srad/main.h"

//========================================================================================================================================================================================================200
//	Reduce KERNEL
//========================================================================================================================================================================================================200

__kernel void
reduce_kernel(	long d_Ne,													// number of elements in array
				long d_no,													// number of sums to reduce
				int d_mul,													// increment
				__global fp* d_sums,										// pointer to partial sums variable (DEVICE GLOBAL MEMORY)
				__global fp* d_sums2,
				int gridDim){

	// indexes
    int bx = get_group_id(0);												// get current horizontal block index
	int tx = get_local_id(0);												// get current horizontal thread index
	int ei = (bx*NUMBER_THREADS)+tx;										// unique thread id, more threads than actual elements !!!
	// int gridDim = (int)get_group_size(0)/(int)get_local_size(0);			// number of workgroups
	int nf = NUMBER_THREADS-(gridDim*NUMBER_THREADS-d_no);				// number of elements assigned to last block
	int df = NUMBER_THREADS;															// divisibility factor for the last block

	// statistical
	__local fp d_psum[NUMBER_THREADS];										// data for block calculations allocated by every block in its shared memory
	__local fp d_psum2[NUMBER_THREADS];

	// counters
	int i;

	// copy data to shared memory
	if(ei<d_no){															// do only for the number of elements, omit extra threads
		d_psum[tx] = d_sums[ei*d_mul];
		d_psum2[tx] = d_sums2[ei*d_mul];
	}

    // Lingjie Zhang modificated at Nov 1, 2015
	//	barrier(CLK_LOCAL_MEM_FENCE);
	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE); // Lukasz proposed, Ke modified 2015/12/12 22:31:00
    // end Lingjie Zhang modification

	if (nf != NUMBER_THREADS && bx == (gridDim - 1)) {
		// figure out divisibility
		df = 1 << (31 - clz(nf));
	}

	// reduction of sums if all blocks are full (rare case)
	// sum of every 2, 4, ..., NUMBER_THREADS elements
	for(i=2; i<=df; i <<= 1){
		// sum of elements
		if(((tx+1) & (i-1)) == 0){											// every ith
			d_psum[tx] = d_psum[tx] + d_psum[tx-i/2];
			d_psum2[tx] = d_psum2[tx] + d_psum2[tx-i/2];
		}
		// synchronization
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	// remainder / final summation by last thread
	if(tx==(df-1)){										//
		// compute the remainder and final summation by last busy thread
		if (nf != NUMBER_THREADS && bx == (gridDim - 1)) {
			for(i=(bx*NUMBER_THREADS)+df; i<(bx*NUMBER_THREADS)+nf; i++){						//
				d_psum[tx] = d_psum[tx] + d_sums[i];
				d_psum2[tx] = d_psum2[tx] + d_sums2[i];
			}
		}
		// final sumation by last thread in every block
		d_sums[bx*d_mul*NUMBER_THREADS] = d_psum[tx];
		d_sums2[bx*d_mul*NUMBER_THREADS] = d_psum2[tx];
	}
}


inline void
atomic_add_fp(volatile float __global *ptr, float val)
{
#ifdef NV_SM_20
	/* XXX: This is obviously not portable, but makes a 50x perf difference.
	 * someone should propose a fp32 atomic extension to OpenCL */
	float oldval;

	asm volatile ("atom.global.add.f32 %0, [%1], %2;" :
			"=f"(oldval) : "l"(ptr), "f"(val));
//#else
//	volatile int __global *iptr = (volatile int __global *) ptr;
//	union {
//		int i;
//		float f;
//	} oldval, newval;
//
//	do {
//		oldval.i = *iptr;
//		newval.f = oldval.f + val;
//	} while (atomic_cmpxchg(iptr,
//			oldval.i, newval.i) != oldval.i);
#endif
}

__kernel void
reduce_kernel_fpatom(	long d_Ne,													// number of elements in array
				long d_no,													// number of sums to reduce
				int d_mul,													// increment
				volatile __global fp* d_sums,										// pointer to partial sums variable (DEVICE GLOBAL MEMORY)
				volatile __global fp* d_sums2,
				int gridDim){

	// indexes											// get current horizontal thread index
	int ei = get_global_id(0) + 1;
	float sums_acc[2] = {0,0};

	//volatile __local fp d_psum[1];										// data for block calculations allocated by every block in its shared memory
	//volatile __local fp d_psum2[1];// unique thread id, more threads than actual elements !!!
	// int gridDim = (int)get_group_size(0)/(int)get_local_size(0);			// number of workgroups
	// counters
	for (unsigned int i = ei; i < d_no; i += get_global_size(0)) {
		sums_acc[0] += d_sums[i];
		sums_acc[1] += d_sums2[i];
	}

    // Lingjie Zhang modificated at Nov 1, 2015
	//	barrier(CLK_LOCAL_MEM_FENCE);
	//barrier(CLK_LOCAL_MEM_FENCE); // Lukasz proposed, Ke modified 2015/12/12 22:31:00

	// copy data to shared memory
	if (ei<d_no) {														// do only for the number of elements, omit extra threads
		atomic_add_fp(&d_sums[0], sums_acc[0]);
		atomic_add_fp(&d_sums2[0], sums_acc[1]);
	}
}

//========================================================================================================================================================================================================200
//	SRAD KERNEL
//========================================================================================================================================================================================================200

// BUG, IF STILL PRESENT, COULD BE SOMEWHERE IN THIS CODE, MEMORY ACCESS OUT OF BOUNDS

__kernel void 
srad_kernel(fp d_lambda, 
			int d_Nr, 
			int d_Nc, 
			long d_Ne, 
			__global int* d_iN, 
			__global int* d_iS, 
			__global int* d_jE, 
			__global int* d_jW, 
			__global fp* d_dN, 
			__global fp* d_dS, 
			__global fp* d_dE, 
			__global fp* d_dW, 
			fp d_q0sqr, 
			__global fp* d_c, 
			__global fp* d_I){

	// indexes
    int bx = get_group_id(0);												// get current horizontal block index
	int tx = get_local_id(0);												// get current horizontal thread index
	int ei = bx*NUMBER_THREADS+tx;											// more threads than actual elements !!!
	int row;																// column, x position
	int col;																// row, y position

	// variables
	fp d_Jc;
	fp d_dN_loc, d_dS_loc, d_dW_loc, d_dE_loc;
	fp d_c_loc;
	fp d_G2,d_L,d_num,d_den,d_qsqr;

	// figure out row/col location in new matrix
	row = (ei+1) % d_Nr - 1;													// (0-n) row
	col = (ei+1) / d_Nr + 1 - 1;												// (0-n) column
	if((ei+1) % d_Nr == 0){
		row = d_Nr - 1;
		col = col - 1;
	}

	if(ei<d_Ne){															// make sure that only threads matching jobs run
		
		// directional derivatives, ICOV, diffusion coefficent
		d_Jc = d_I[ei];														// get value of the current element
		
		// directional derivates (every element of IMAGE)(try to copy to shared memory or temp files)
		d_dN_loc = d_I[d_iN[row] + d_Nr*col] - d_Jc;						// north direction derivative
		d_dS_loc = d_I[d_iS[row] + d_Nr*col] - d_Jc;						// south direction derivative
		d_dW_loc = d_I[row + d_Nr*d_jW[col]] - d_Jc;						// west direction derivative
		d_dE_loc = d_I[row + d_Nr*d_jE[col]] - d_Jc;						// east direction derivative
	         
		// normalized discrete gradient mag squared (equ 52,53)
		d_G2 = (d_dN_loc*d_dN_loc + d_dS_loc*d_dS_loc + d_dW_loc*d_dW_loc + d_dE_loc*d_dE_loc) / (d_Jc*d_Jc);	// gradient (based on derivatives)
		
		// normalized discrete laplacian (equ 54)
		d_L = (d_dN_loc + d_dS_loc + d_dW_loc + d_dE_loc) / d_Jc;			// laplacian (based on derivatives)

		// ICOV (equ 31/35)
		d_num  = (0.5f*d_G2) - ((1.f/16.f)*(d_L*d_L)) ;						// num (based on gradient and laplacian)
		d_den  = 1.f + (0.25f*d_L);												// den (based on laplacian)
		d_qsqr = d_num/(d_den*d_den);										// qsqr (based on num and den)
	 
		// diffusion coefficent (equ 33) (every element of IMAGE)
		d_den = (d_qsqr-d_q0sqr) / (d_q0sqr * (1.f+d_q0sqr)) ;				// den (based on qsqr and q0sqr)
		d_c_loc = clamp(1.f / (1.f+d_den),0.f,1.f);	// diffusion coefficient (based on den)
								// Clamped to [0.f,1.f]

		// save data to global memory
		d_dN[ei] = d_dN_loc; 
		d_dS[ei] = d_dS_loc; 
		d_dW[ei] = d_dW_loc; 
		d_dE[ei] = d_dE_loc;
		d_c[ei] = d_c_loc;
			
	}
	
}

//========================================================================================================================================================================================================200
//	SRAD2 KERNEL
//========================================================================================================================================================================================================200

// BUG, IF STILL PRESENT, COULD BE SOMEWHERE IN THIS CODE, MEMORY ACCESS OUT OF BOUNDS

__kernel void 
srad2_kernel(	fp d_lambda, 
				int d_Nr, 
				int d_Nc, 
				long d_Ne, 
				__global int* d_iN, 
				__global int* d_iS, 
				__global int* d_jE, 
				__global int* d_jW,
				__global fp* d_dN, 
				__global fp* d_dS, 
				__global fp* d_dE, 
				__global fp* d_dW, 
				__global fp* d_c, 
				__global fp* d_I){

	// indexes
    int bx = get_group_id(0);												// get current horizontal block index
	int tx = get_local_id(0);												// get current horizontal thread index
	int ei = bx*NUMBER_THREADS+tx;											// more threads than actual elements !!!
	int row;																// column, x position
	int col;																// row, y position

	// variables
	fp d_cN,d_cS,d_cW,d_cE;
	fp d_D;

	// figure out row/col location in new matrix
	row = (ei+1) % d_Nr - 1;												// (0-n) row
	col = (ei+1) / d_Nr + 1 - 1;											// (0-n) column
	if((ei+1) % d_Nr == 0){
		row = d_Nr - 1;
		col = col - 1;
	}

	if(ei<d_Ne){															// make sure that only threads matching jobs run

		// diffusion coefficent
		d_cN = d_c[ei];														// north diffusion coefficient
		d_cS = d_c[d_iS[row] + d_Nr*col];										// south diffusion coefficient
		d_cW = d_c[ei];														// west diffusion coefficient
		d_cE = d_c[row + d_Nr * d_jE[col]];									// east diffusion coefficient

		// divergence (equ 58)
		d_D = d_cN*d_dN[ei] + d_cS*d_dS[ei] + d_cW*d_dW[ei] + d_cE*d_dE[ei];// divergence

		// image update (equ 61) (every element of IMAGE)
		d_I[ei] = d_I[ei] + 0.25f*d_lambda*d_D;								// updates image (based on input time step and divergence)

	}

}
