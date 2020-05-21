/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2020 Roy Spliet, University of Cambridge
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

#ifdef ECLIPSE
#define __kernel
#define __global
#define __local
#define __private
#define __constant
#define CLK_LOCAL_MEM_FENCE 0
#endif

/* Launch in 1D, one work-item per output */
__kernel void
cl_relu(float __global *in, float __constant *biases, float __global *weights,
		unsigned int inSize, float __global *out)
{
	unsigned int x = get_global_id(0);
	unsigned int width = get_global_size(0);
	unsigned int line = 0;
	unsigned int i;

	float val = biases[x];

	for (i = 0; i < inSize; i++) {
		val += in[i] * weights[line + x];
		line += width;
	}

	out[x] = fmax(0.0f, val);
}
