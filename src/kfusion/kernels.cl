/* SPDX-License-Identifier: MIT

 Copyright (c) 2011-2013 Gerhard Reitmayr, TU Graz

 Copyright (c) 2014 University of Edinburgh, Imperial College, University of Manchester.
 Developed in the PAMELA project, EPSRC Programme Grant EP/K008730/1

 This code is licensed under the MIT License.

 *
 * Modifications from upstream:
 * - Remove unused kernels.
 * - HalfSampleRobustImage: unroll inner loop by 2
 */

/************** TYPES ***************/

#ifdef ECLIPSE
#define __kernel
#define __global
#define __local
#define __private
#define CLK_LOCAL_MEM_FENCE 0
typedef struct { float x; float y; } float2;
typedef struct { float x; float y; float z; } float3;
typedef struct { float x; float y; float z; float w; float3 xyz; } float4;
typedef struct { int x; int y; } int2;
typedef struct { int x; int y; int z; } int3;
typedef struct { unsigned int x; unsigned int y; } uint2;
typedef struct { unsigned int x; unsigned int y; unsigned int z; } uint3;
typedef struct { short x; short y; } short2;
typedef struct { unsigned char x; unsigned char y; unsigned char z; } uchar3;
typedef struct { unsigned char x; unsigned char y; unsigned char z; unsigned char w; } uchar4;
#endif

#define INVALID -2.f

typedef struct sTrackData {
	int result;
	float error;
	float J[6];
} TrackData;

typedef struct sMatrix4 {
	float4 data[4];
} Matrix4;

/************** FUNCTIONS ***************/

inline float3 Mat4TimeFloat3(Matrix4 M, float3 v) {
	return (float3)(
			dot((float3)(M.data[0].x, M.data[0].y, M.data[0].z), v)
					+ M.data[0].w,
			dot((float3)(M.data[1].x, M.data[1].y, M.data[1].z), v)
					+ M.data[1].w,
			dot((float3)(M.data[2].x, M.data[2].y, M.data[2].z), v)
					+ M.data[2].w);
}

inline float3 myrotate(const Matrix4 M, const float3 v) {
	return (float3)(dot((float3)(M.data[0].x, M.data[0].y, M.data[0].z), v),
			dot((float3)(M.data[1].x, M.data[1].y, M.data[1].z), v),
			dot((float3)(M.data[2].x, M.data[2].y, M.data[2].z), v));
}

/************** KFUSION KERNELS ***************/
// inVertex iterate
__kernel void trackKernel (
		__global TrackData * output,
		const uint2 outputSize,
		__global const float * inVertex,// float3
		const uint2 inVertexSize,
		__global const float * inNormal,// float3
		const uint2 inNormalSize,
		__global const float * refVertex,// float3
		const uint2 refVertexSize,
		__global const float * refNormal,// float3
		const uint2 refNormalSize,
		const Matrix4 Ttrack,
		const Matrix4 view,
		const float dist_threshold,
		const float normal_threshold
) {

	const uint2 pixel = (uint2)(get_global_id(0),get_global_id(1));

	if(pixel.x >= inVertexSize.x || pixel.y >= inVertexSize.y ) {return;}

	float3 inNormalPixel = vload3(pixel.x + inNormalSize.x * pixel.y,inNormal);

	if(inNormalPixel.x == INVALID ) {
		output[pixel.x + outputSize.x * pixel.y].result = -1;
		return;
	}

	float3 inVertexPixel = vload3(pixel.x + inVertexSize.x * pixel.y,inVertex);
	const float3 projectedVertex = Mat4TimeFloat3 (Ttrack , inVertexPixel);
	const float3 projectedPos = Mat4TimeFloat3 ( view , projectedVertex);
	const float2 projPixel = (float2) ( projectedPos.x / projectedPos.z + 0.5f, projectedPos.y / projectedPos.z + 0.5f);

	if(projPixel.x < 0.f || projPixel.x > refVertexSize.x-1.f || projPixel.y < 0.f || projPixel.y > refVertexSize.y-1.f ) {
		output[pixel.x + outputSize.x * pixel.y].result = -2;
		return;
	}

	const uint2 refPixel = (uint2) (projPixel.x, projPixel.y);
	const float3 referenceNormal = vload3(refPixel.x + refNormalSize.x * refPixel.y,refNormal);

	if(referenceNormal.x == INVALID) {
		output[pixel.x + outputSize.x * pixel.y].result = -3;
		return;
	}

	const float3 diff = vload3(refPixel.x + refVertexSize.x * refPixel.y,refVertex) - projectedVertex;
	const float3 projectedNormal = myrotate(Ttrack, inNormalPixel);

	if(length(diff) > dist_threshold ) {
		output[pixel.x + outputSize.x * pixel.y].result = -4;
		return;
	}

	if(dot(projectedNormal, referenceNormal) < normal_threshold) {
		output[pixel.x + outputSize.x * pixel.y] .result = -5;
		return;
	}

	output[pixel.x + outputSize.x * pixel.y].result = 1;
	output[pixel.x + outputSize.x * pixel.y].error = dot(referenceNormal, diff);

	vstore3(referenceNormal,0,(output[pixel.x + outputSize.x * pixel.y].J));
	vstore3(cross(projectedVertex, referenceNormal),1,(output[pixel.x + outputSize.x * pixel.y].J));

}

__kernel void depth2vertexKernel( __global float * vertex, // float3
		const uint2 vertexSize ,
		const __global float * depth,
		const uint2 depthSize ,
		const Matrix4 invK ) {

	uint2 pixel = (uint2) (get_global_id(0),get_global_id(1));
	float3 vert = (float3)(get_global_id(0),get_global_id(1),1.0f);

	if(pixel.x >= depthSize.x || pixel.y >= depthSize.y ) {
		return;
	}

	float3 res = (float3) (0);

	if(depth[pixel.x + depthSize.x * pixel.y] > 0) {
		res = depth[pixel.x + depthSize.x * pixel.y] * (myrotate(invK, (float3)(pixel.x, pixel.y, 1.f)));
	}

	vstore3(res, pixel.x + vertexSize.x * pixel.y,vertex); 	// vertex[pixel] =

}

__kernel void vertex2normalKernel( __global float * normal,    // float3
		const uint2 normalSize,
		const __global float * vertex ,
		const uint2 vertexSize ) {  // float3

	uint2 pixel = (uint2) (get_global_id(0),get_global_id(1));

	if(pixel.x >= vertexSize.x || pixel.y >= vertexSize.y )
	return;

	uint2 vleft = (uint2)(max((int)(pixel.x)-1,0), pixel.y);
	uint2 vright = (uint2)(min(pixel.x+1,vertexSize.x-1), pixel.y);
	uint2 vup = (uint2)(pixel.x, max((int)(pixel.y)-1,0));
	uint2 vdown = (uint2)(pixel.x, min(pixel.y+1,vertexSize.y-1));

	const float3 left = vload3(vleft.x + vertexSize.x * vleft.y,vertex);
	const float3 right = vload3(vright.x + vertexSize.x * vright.y,vertex);
	const float3 up = vload3(vup.x + vertexSize.x * vup.y,vertex);
	const float3 down = vload3(vdown.x + vertexSize.x * vdown.y,vertex);

	if(left.z == 0 || right.z == 0|| up.z ==0 || down.z == 0) {
		vstore3((float3)(INVALID,INVALID,INVALID),pixel.x + normalSize.x * pixel.y,normal);
		return;
	}
	const float3 dxv = right - left;
	const float3 dyv = down - up;
	vstore3((float3) normalize(cross(dyv, dxv)), pixel.x + pixel.y * normalSize.x, normal );

}

__kernel void halfSampleRobustImageKernel(__global float * out,
		__global const float * in,
		const uint2 inSize,
		const float e_d,
		const int r) {

	uint2 pixel = (uint2) (get_global_id(0),get_global_id(1));
	uint2 outSize = inSize / 2;

	const uint2 centerPixel = 2 * pixel;

	float sum = 0.0f;
	float t = 0.0f;
	const float center = in[centerPixel.x + centerPixel.y * inSize.x];
	for(int i = -r + 1; i <= r; ++i) {
		#pragma unroll 2
		for(int j = -r + 1; j <= r; ++j) {
			int2 from = (int2)(clamp((int2)(centerPixel.x + j, centerPixel.y + i), (int2)(0), (int2)(inSize.x - 1, inSize.y - 1)));
			float current = in[from.x + from.y * inSize.x];
			if(fabs(current - center) < e_d) {
				sum += 1.0f;
				t += current;
			}
		}
	}
	out[pixel.x + pixel.y * outSize.x] = t / sum;

}

