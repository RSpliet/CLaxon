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
 *
 * This approach is based on Mark Harris et al.'s "Parallel Prefix Sum (Scan)
 * with CUDA", GPU Gems 3.
 */

/* Launch in 1D, as many threads as data elements, padded to next multiple of
 * block size. Block size must be POT */
__kernel void prefix_sum(unsigned int __global *in, unsigned int __local *in_l,
		unsigned int __global *out, unsigned int __global *incr) {
	size_t lx = get_local_id(0);
	size_t lthreads;
	size_t off;
	event_t copy;
	int i;
	unsigned int tmp;

	/* Copy data to local memory */
	copy = async_work_group_copy(in_l,
			&in[get_group_id(0) * get_local_size(0) * 2],
			get_local_size(0) * 2, 0);
	wait_group_events(1, &copy);

	/* Up-sweep */
	lthreads = get_local_size(0);
	for (i = 2; i <= get_local_size(0) * 2; i <<= 1, lthreads >>= 1) {
		if (lx < lthreads) {
			off = mad24((int)lx + 1, i, -1);
			in_l[off] += in_l[off - (i >> 1)];
		}

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	/* Write result to incr */
	if (lx == 0) {
		if (incr)
			incr[get_group_id(0)] = in_l[off];
		in_l[off] = 0;
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	/* Down-sweep */
	lthreads = 1;
	for (i = get_local_size(0) * 2; i >= 2; i >>= 1, lthreads <<= 1) {
		if (lx < lthreads) {
			off = mad24((int)lx + 1, i, -1);
			tmp = in_l[off];
			in_l[off] += in_l[off - (i >> 1)];
			in_l[off - (i >> 1)] = tmp;
		}

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	/* Copy result back to global memory */
	copy = async_work_group_copy(
			&out[get_group_id(0) * get_local_size(0) * 2],
			in_l, get_local_size(0) * 2, 0);
	wait_group_events(1, &copy);

	return;
}

/* Launch in 1D, as many threads as data elements, padded to next multiple of
 * block size. Skipped the first (2 * workgroup size) entries as incr[0] is
 * known to be 0 */
__kernel void prefix_sum_post(volatile unsigned int __global *data,
		unsigned int __global *incr) {
	data[mad24((int)get_local_size(0), 2, (int)get_global_id(0))] +=
			incr[(get_group_id(0) >> 1) + 1];

	return;
}
