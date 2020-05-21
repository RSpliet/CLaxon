/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2016 Roy Spliet, University of Cambridge
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/* Launch in 3D x*y*c */
__kernel void cl_max_pooling(float __global *in, float __global *out,
		int poolingFactor, int stride) {
	unsigned int x = get_global_id(0);
	unsigned int y = get_global_id(1);
	unsigned int c = get_global_id(2);
	unsigned int w = get_global_size(0);
	unsigned int s = w * get_global_size(1);
	unsigned int inWidth = w*stride+(poolingFactor - stride);
	unsigned int inSfc = inWidth *
			(get_global_size(1)*stride+(poolingFactor - stride));
	float maxSoFar = FLT_MIN;

	/* Find the maximum value in the local window. */
	for (int dy=0; dy<poolingFactor; dy++) {
		unsigned int yy = y*stride+dy;

		/* By default NVIDIA unrolls by 4. Unrolling by 3 instead
		 * gives 25% performance, abusing my knowledge of the
		 * poolingFactor */
		#pragma unroll 3
		for (int dx=0; dx<poolingFactor; dx++) {
			unsigned int xx = x*stride+dx;

			float value = in[c*inSfc + yy*inWidth + xx];
			maxSoFar = fmax(maxSoFar, value);
		}
	}

	out[c*s + y*w + x] = maxSoFar;
}
