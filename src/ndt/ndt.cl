/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2017 Roy Spliet, University of Cambridge
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

/* Data structured as "struct of arrays" for locality reasons.
 * in[dim][pt_idx]
 * q[dim][cell_idx]
 * C[y][x][cell_idx]
 */

inline int coord_to_bin(float x, float y, float z, float bins_dim)
{
	int bin;

	if (x < 0.f || x > bins_dim ||
	    y < 0.f || y > bins_dim ||
	    z < 0.f || z > bins_dim)
		return -1;

	bin = floor(x) + (floor(y) * bins_dim) +
			(floor(z) * bins_dim * bins_dim);

	return bin;
}

/* Launch in 1D */
__kernel void kernel_ins_cnt(float __global *data_x, unsigned int elems,
		float bins_dim, int __global *data_bin,
		volatile int __global *bin_size) {

	int bin;
	unsigned int n = get_global_id(0);

	bin = coord_to_bin(data_x[n], data_x[n + elems], data_x[n + (2*elems)],
			bins_dim);
	data_bin[n] = bin;

	if (bin < 0)
		return;

	atomic_inc(&bin_size[bin]);
}

/* Launch in 1D */
__kernel void kernel_reindex(float __global *in_x, unsigned int in_elems,
		float __global *out_x, unsigned int out_elems,
		int __global *data_bin, volatile int __global *bin_prefix,
		volatile __global int *bin_idx) {

	int bin, elem;
	size_t n = get_global_id(0);

	bin = data_bin[n];
	if (bin < 0)
		return;

	elem = bin_prefix[bin] + atomic_inc(&bin_idx[bin]);

	out_x[elem] = in_x[n];
	out_x[elem+out_elems] = in_x[n+in_elems];
	out_x[elem+(2*out_elems)] = in_x[n+(2*in_elems)];

	return;
}

/* Cofactor expansion over first row
 *
 * WARNING! We optimise assuming the following equalities that result from
 * covariant matrix construction - these are not generic methods!
 * in[1] == in[3]
 * in[2] == in[6]
 * in[5] == in[7]
 * Saves 36 instructions on GK107 and reduces register pressure to below 30.
 */
float
determinant_3x3(float in[9])
{
	return 	in[0] * (in[4] * in[8] - in[5] * in[5]) -
		in[1] * (in[1] * in[8] - in[5] * in[2]) +
		in[2] * (in[1] * in[5] - in[4] * in[2]);
}

/* Cramers rule for matrix inversion */
void
invert_3x3(float in[9], float out[9])
{
	float det = determinant_3x3(in);

	if (det == 0.0f) {
		/* Cannot be inverted */
		return;
	}

	out[0] =  (in[4] * in[8] - in[5] * in[5]) / det;
	out[1] = -(in[1] * in[8] - in[2] * in[5]) / det;
	out[2] =  (in[1] * in[5] - in[2] * in[4]) / det;
	out[3] = -(in[1] * in[8] - in[5] * in[2]) / det;
	out[4] =  (in[0] * in[8] - in[2] * in[2]) / det;
	out[5] = -(in[0] * in[5] - in[2] * in[1]) / det;
	out[6] =  (in[1] * in[5] - in[4] * in[2]) / det;
	out[7] = -(in[0] * in[5] - in[1] * in[2]) / det;
	out[8] =  (in[0] * in[4] - in[1] * in[1]) / det;

	return;
}

/* Launch in 2D, one thread per item
 * Transformation matrix is pre-calculated on the host
 * XXX: transform mat in const or local mem? */
__kernel void
ndt_vec_transform(float __global *in, unsigned int elems,
		__constant float transform_mat[12],
		float __global *out)
{
	unsigned int elem_idx =
		get_global_id(1) * get_global_size(0) +
		get_global_id(0);

	if (elem_idx >= elems)
		return;

	float in_tmp[3];
	float out_tmp[3];

	in_tmp[0] = in[elem_idx];
	in_tmp[1] = in[elems + elem_idx];
	in_tmp[2] = in[mad24(elems, (unsigned int) 2, elem_idx)];

	out_tmp[0] = in_tmp[0] * transform_mat[0] + transform_mat[3];
	out_tmp[0] += in_tmp[1] * transform_mat[1];
	out_tmp[0] += in_tmp[2] * transform_mat[2];

	out_tmp[1] = in_tmp[0] * transform_mat[4] + transform_mat[7];
	out_tmp[1] += in_tmp[1] * transform_mat[5];
	out_tmp[1] += in_tmp[2] * transform_mat[6];

	out_tmp[2] = in_tmp[0] * transform_mat[8] + transform_mat[11];
	out_tmp[2] += in_tmp[1] * transform_mat[9];
	out_tmp[2] += in_tmp[2] * transform_mat[10];

	out[elem_idx] = out_tmp[0];
	out[elems + elem_idx] = out_tmp[1];
	out[mad24(elems, (unsigned int) 2, elem_idx)] = out_tmp[2];
}

/* Launch in 1D - all cells */
__kernel void
ndt_cell_qC(float __global *in, unsigned int in_elems, int __global *bin_elems,
		int __global *bin_prefix, float __global *q, float __global *C)
{
	unsigned int cell_idx = get_global_id(0);
	unsigned int cells = get_global_size(0);
	unsigned int i;

	/* If fewer than 5 samples, ignore */
	if (bin_elems[cell_idx] < 5) {
		for(i = 0; i < 3; i++)
			q[(i * cells) + cell_idx] = 0.f;

		for(i = 0; i < 9; i++)
			C[(i * cells) + cell_idx] = 0.f;

		return;
	}

	/* mean vector q */
	/* Keep around in regs to avoid many memory accesses */
	float q_tmp[3];

	for (i = 0; i < 3; i++) {
		unsigned int j;

		q_tmp[i] = 0.f;

		unsigned int off = (i * in_elems) + bin_prefix[cell_idx];
		for (j = 0; j < bin_elems[cell_idx]; j++, off++) {
			q_tmp[i] += in[off];
		}

		q_tmp[i] /= j;
		q[(i * cells) + cell_idx] = q_tmp[i];
	}

	/* covariant matrix C */
	float c_tmp[9] = {0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f};
	float c_inv[9];

	{
		unsigned int off = bin_prefix[cell_idx];

		for (i = 0; i < bin_elems[cell_idx]; i++, off++) {
			float in_tmp[3];

			in_tmp[0] = in[off] - q_tmp[0];
			in_tmp[1] = in[in_elems + off] - q_tmp[1];
			in_tmp[2] = in[(2 * in_elems) + off] - q_tmp[2];

			c_tmp[0] += in_tmp[0] * in_tmp[0];
			c_tmp[1] += in_tmp[0] * in_tmp[1];
			c_tmp[2] += in_tmp[0] * in_tmp[2];

			c_tmp[3] += in_tmp[0] * in_tmp[1];
			c_tmp[4] += in_tmp[1] * in_tmp[1];
			c_tmp[5] += in_tmp[1] * in_tmp[2];

			c_tmp[6] += in_tmp[2] * in_tmp[0];
			c_tmp[7] += in_tmp[1] * in_tmp[2];
			c_tmp[8] += in_tmp[2] * in_tmp[2];
		}
	}

	for (i = 0; i < 9; i++)
		c_tmp[i] /= (bin_elems[cell_idx] - 1);

	invert_3x3(c_tmp, c_inv);

	#pragma unroll
	for (i = 0; i < 9; i++)
		C[(i * cells) + cell_idx] = c_inv[i];

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
#else
	volatile int __global *iptr = (volatile int __global *) ptr;
	union {
		int i;
		float f;
	} oldval, newval;

	do {
		oldval.i = *iptr;
		newval.f = oldval.f + val;
	} while (atomic_cmpxchg(iptr,
			oldval.i, newval.i) != oldval.i);
#endif
}

/* Launch in 1D - elems */
__kernel void
ndt_elem_q(float __global *in_x, unsigned int in_elems, int __global *cell,
		volatile int __global *bin_elems,
		const float bins_dim, volatile float __global *q)
{
	unsigned int elem_idx =
		get_global_id(1) * get_global_size(0) +
		get_global_id(0);

	if (elem_idx >= in_elems)
		return;

	unsigned int i;
	int cell_idx;
	unsigned int cells = bins_dim * bins_dim * bins_dim;

	/* mean vector q */
	float q_tmp[3];

	for (i = 0; i < 3; i++)
		q_tmp[i] = in_x[elem_idx + (i * in_elems)];

	cell_idx = coord_to_bin(q_tmp[0], q_tmp[1], q_tmp[2], bins_dim);
	cell[elem_idx] = cell_idx; /* Caching this saves ~5% on ndt_elem_C */
	if (cell_idx < 0)
		return;

	for (i = 0; i < 3; i++)
		atomic_add_fp(&q[cell_idx + (i* cells)], q_tmp[i]);

	atomic_inc(&bin_elems[cell_idx]);
}

/* Launch in 1D - elems */
__kernel void
ndt_elem_C(float __global *in_x, unsigned int in_elems, int __global *cell,
		volatile int __global *bin_elems,
		const float bins_dim, volatile float __global *q,
		volatile float __global *C)
{
	unsigned int elem_idx =
		get_global_id(1) * get_global_size(0) +
		get_global_id(0);

	if (elem_idx >= in_elems)
		return;

	unsigned int cells = bins_dim * bins_dim * bins_dim;
	int cell_idx;
	int cell_elems;
	int i;
	float q_tmp[3];

	cell_idx = cell[elem_idx];
	if (cell_idx < 0)
		return;

	cell_elems = bin_elems[cell_idx];
	if (cell_elems < 5)
		return;

	for (i = 0; i < 3; i++)
		q_tmp[i] = q[cell_idx + (i * cells)] / cell_elems;

	/* covariant matrix C */
	float in_tmp[3];

	for (i = 0; i < 3; i++)
		in_tmp[i] = in_x[(i * in_elems) + elem_idx] - q_tmp[i];

	atomic_add_fp(&C[cell_idx], in_tmp[0] * in_tmp[0]);
	atomic_add_fp(&C[cell_idx + cells], in_tmp[0] * in_tmp[1]);
	atomic_add_fp(&C[cell_idx + (2 * cells)], in_tmp[0] * in_tmp[2]);
	atomic_add_fp(&C[cell_idx + (3 * cells)], in_tmp[0] * in_tmp[1]);
	atomic_add_fp(&C[cell_idx + (4 * cells)], in_tmp[1] * in_tmp[1]);
	atomic_add_fp(&C[cell_idx + (5 * cells)], in_tmp[1] * in_tmp[2]);
	atomic_add_fp(&C[cell_idx + (6 * cells)], in_tmp[0] * in_tmp[2]);
	atomic_add_fp(&C[cell_idx + (7 * cells)], in_tmp[1] * in_tmp[2]);
	atomic_add_fp(&C[cell_idx + (8 * cells)], in_tmp[2] * in_tmp[2]);
}

/* Launch 1D - cells*/
__kernel void
ndt_elem_qC_post(float __global *in_x, unsigned int in_elems,
		int __global *bin_elems,
		const float bins_dim, float __global *q, float __global *C)
{
	unsigned int cell_idx = get_global_id(0);
	unsigned int cells = get_global_size(0);
	unsigned int i;

	int cell_elems = bin_elems[cell_idx];

	if (cell_elems < 5)
		return;

	for (i = 0; i < 3; i++)
		q[cell_idx + (i* cells)] = q[cell_idx + (i * in_elems)]
					     / cell_elems;

	float c_tmp[9];
	float c_inv[9];

	for (i = 0; i < 9; i++)
		c_tmp[i] = C[cell_idx + (i* cells)] / (bin_elems[cell_idx] - 1);

	invert_3x3(c_tmp, c_inv);

	for (i = 0; i < 9; i++)
		C[(i * cells) + cell_idx] = c_inv[i];
}
