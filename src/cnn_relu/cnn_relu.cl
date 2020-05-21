/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2016 Daniel Bates and Roy Spliet, University of Cambridge
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
__kernel void cl_relu(float __global *in, float __constant *biases,
		float __global *out, int shiftAmount) {
	unsigned int x = get_global_id(0);
	unsigned int y = get_global_id(1);
	unsigned int c = get_global_id(2);
	unsigned int height = get_global_size(1);
	unsigned int width = get_global_size(0);
	float val;

	val = (in[c*height*width + y*width + x] + biases[c]);
	out[c*height*width + y*width + x] = fmax(0.0f, val);
}
