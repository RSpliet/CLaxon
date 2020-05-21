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

void copy_kernels(const float __global *in, float __local *out,
		size_t elems, size_t off)
{
	event_t copy;

	async_work_group_copy(out, in + off, elems, copy);
	wait_group_events(1, &copy);
}

/* Launch in 3D x*y*c */
__kernel void cl_convolution(float __global *in, const float __global *kernels,
		float __global *out, unsigned int kernelSize,
		unsigned int inChannels, float __local *lkernels) {
	/* Integer saves ~6% over size_t by reducing emulated
	 * 64-bit int support. Careful not to overflow. */
	unsigned int x = get_global_id(0);
	unsigned int y = get_global_id(1);
	unsigned int c = get_global_id(2);
	unsigned int height = get_global_size(1);
	unsigned int width = get_global_size(0);
	unsigned int cLocal;
	unsigned int inWidth = width + kernelSize - 1;
	unsigned int inSurface = (height + kernelSize - 1) * inWidth;

	/* Apply convolution */
	float imageVal, kernelVal;
	float filterSum = 0;

	/* Copy kernels to local memory - brings a ~15% perf benefit */
	unsigned int ksize = kernelSize*kernelSize*inChannels;
	cLocal = get_local_id(2);
	unsigned int offChans = c - cLocal;
	copy_kernels(kernels, lkernels, ksize * get_local_size(2),
			ksize * offChans);

	cLocal *= ksize;

	/* Channels innermost brings ~2% perf benefit for limited parallelism
	 * but ~16% overhead with bigger datasets (more output chans) */
	for (unsigned int inCh = 0; inCh < inChannels; inCh++) {
		for (unsigned int ky = 0; ky < kernelSize; ky++) {
			for (unsigned int kx = 0; kx < kernelSize; kx++) {
				imageVal = in[inCh*inSurface+
					      (y+ky)*inWidth+
					      (x+kx)];
				kernelVal = lkernels[cLocal+
						     inCh*kernelSize*kernelSize +
						    ky*kernelSize +
						    kx];
				filterSum += imageVal * kernelVal;
			}
		}
	}
	out[c*height*width + y*width + x] = filterSum;
}
