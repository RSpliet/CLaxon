/* SPDX-License-Identifier: NCSA
 ***************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************
 * Modifications from upstream:
 * - Use M_PI_F macro instead of M_PI. Avoids emitting double-precision logic
 *   on NVIDIA platforms,
 * - Use native_sin()/native_cor() instead of sin()/cos(). slightly lower
 *   precision, but tests hardware at native speed.
 * - Remove unused code.
 */

#ifdef ECLIPSE
#define __kernel
#define __global
#define __local
#define __private
#define CLK_LOCAL_MEM_FENCE 0
typedef struct { float x; float y; } float2;
#endif

// Block index
// Thread index
#define tx  get_global_id(0)

// Possible values are 2, 4, 8 and 16
#define R 2

inline float2 cmpMul( float2 a, float2 b ) { return (float2)( a.x*b.x-a.y*b.y, a.x*b.y+a.y*b.x ); }

#ifndef M_PI_F
#define M_PI_F 3.141592653589793238462643f
#endif
  
inline void GPU_FFT2(__private float2 *v1, __private float2 *v2 ) { 
  float2 v0 = *v1;
  *v1 = v0 + *v2; 
  *v2 = v0 - *v2;
}

inline void global_GPU_FFT2(__private float2* v){
  GPU_FFT2(v, v+1);
}
  
     
int GPU_expand(int idxL, int N1, int N2 ){ 
  return (idxL/N1)*N1*N2 + (idxL%N1); 
}      

void GPU_FftIteration(int j, int Ns, __global float2* data0, __global float2* data1, int N) { 
  __private float2 v[R];
  int idxS = j;       
  float angle = -2.f*M_PI_F*(j%Ns)/(Ns*2);
  
  for( int r=0; r<R; r++ ) { 
    v[r] = data0[idxS+r*N/R];
    v[r] = cmpMul(v[r],((float2)(native_cos((float) r*angle), native_sin((float) r*angle))));
  }

  global_GPU_FFT2( v );

  int idxD = GPU_expand(j,Ns,R); 

  for( int r=0; r<R; r++ ){
    data1[idxD+r*Ns] = v[r];
  } 	

}      

__kernel void GPU_FFT_Global(int Ns, __global float2* data0, __global float2* data1, int N) { 
  /* Pick the right window */
  data0+=get_global_id(1)*N;
  data1+=get_global_id(1)*N;	 
  GPU_FftIteration( tx, Ns, data0, data1, N);  
}      

