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

inline int coord_to_bin(float x, float y, float z, float bins_dim)
{
	float b_x, b_y, b_z;

	b_x = floor(x * bins_dim);
	b_y = floor(y * bins_dim);
	b_z = floor(z * bins_dim);

	return b_x + (b_y * bins_dim) + (b_z * bins_dim * bins_dim);
}

/* Launch in 1D */
__kernel void kernel_ins_cnt(float __global *data_x, float bins_dim,
		int __global *data_bin, volatile int __global *bin_size) {

	int bin;
	unsigned int n = get_global_id(0);
	unsigned int elems = get_global_size(0);

	bin = coord_to_bin(data_x[n], data_x[n + elems], data_x[n + 2*elems],
			bins_dim);
	data_bin[n] = bin;
	atomic_inc(&bin_size[bin]);
}

/* Launch in 1D */
__kernel void kernel_reindex(float __global *in_x, float __global *out_x,
		int __global *data_bin, volatile int __global *bin_prefix,
		volatile __global int *bin_idx) {

	int bin, elem;
	size_t n = get_global_id(0);
	unsigned int elems = get_global_size(0);

	bin = data_bin[n];
	elem = bin_prefix[bin] + atomic_inc(&bin_idx[bin]);

	out_x[elem] = in_x[n];
	out_x[elem + elems] = in_x[n + elems];
	out_x[elem + (2*elems)] = in_x[n + (2*elems)];

	return;
}

inline float manhattan_dist_3d(float x, float y, float z, float that_x,
		float that_y, float that_z)
{
	return pown(that_x - x, 2) + pown(that_y - y, 2) + pown(that_z - z, 2);
}

/* Launch in 1D */
__kernel void kernel_nn(float __global *in_x, float bins_dim, float rsquare,
		int b, int __global *bin_elems, int __global *bin_prefix,
		int __global *nn)
{
	/* Truncating to 32-bits increases occupancy on AMD RX460
	 * Result is ~10% more perf */
	unsigned int n;
	unsigned int i;
	int b_x, b_y, b_z;
	unsigned int i_x, i_y, i_z;
	float x, y, z;
	unsigned int bin;
	int neighbour = -1;
	float neigh_dist = FLT_MAX;
	float dist;
	unsigned int elems = get_global_size(0);

	/* Find coords for my point */
	n = get_global_id(0);

	x = in_x[n];
	y = in_x[n + elems];
	z = in_x[n + (2*elems)];

	b_x = floor(x * bins_dim);
	b_y = floor(y * bins_dim);
	b_z = floor(z * bins_dim);

	/* Iterate bins.
	 * Note: a layed approach (like peeling an onion) only allows early
	 * exit if points are always in the centre of a bin. Instead of
	 * attempting heuristics to further reduce the search-space, brute-force
	 * all bins for now. */
	for (i_z = max(0, b_z - b); i_z <= min((int)bins_dim, b_z + b); i_z++) {
		for (i_y = max(0, b_y - b); i_y <= min((int)bins_dim , b_y + b);
		     i_y++) {
			for (i_x = max(0, b_x - b);
			     i_x <= min((int)bins_dim , b_x + b); i_x++) {

				bin = i_x + (i_y * bins_dim) +
					(i_z * bins_dim * bins_dim);

				for (i = bin_prefix[bin];
				     i < bin_prefix[bin] + bin_elems[bin];
				     i++) {

					if (i == n)
						continue;

					dist = manhattan_dist_3d(x, y, z,
						    in_x[i], in_x[i + elems],
						    in_x[i + (2*elems)]);

					if (dist <= rsquare &&
					    dist < neigh_dist) {
						neighbour = i;
						neigh_dist = dist;
					}
				}
			}
		}
	}

	nn[n] = neighbour;
}

/* Launch in 1D */
__kernel void kernel_nn_centoids(float __global *in_x, float bins_dim,
		float rsquare, int b, unsigned int __global *bin_elems,
		unsigned int __global *bin_prefix,
		float __global *cent_x)
{
	/* Truncating n to 32-bits increases occupancy on AMD RX460 */
	unsigned int n;
	unsigned int i;
	int b_x, b_y, b_z;
	unsigned int i_x, i_y, i_z;
	float x, y, z;
	unsigned int bin;
	float dist;
	float3 centoid = {0.f,0.f,0.f};
	unsigned int k = 0;
	unsigned int elems = get_global_size(0);

	/* Find coords for my point */
	n = get_global_id(0);

	x = in_x[n];
	y = in_x[n + elems];
	z = in_x[n + (2*elems)];

	b_x = floor(x * bins_dim);
	b_y = floor(y * bins_dim);
	b_z = floor(z * bins_dim);

	/* Iterate bins.
	 * Note: a layed approach (like peeling an onion) only allows early
	 * exit if points are always in the centre of a bin. Instead of
	 * attempting heuristics to further reduce the search-space, brute-force
	 * all bins for now. */
	for (i_z = max(0, b_z - b); i_z <= min((int)bins_dim, b_z + b); i_z++) {
		for (i_y = max(0, b_y - b); i_y <= min((int)bins_dim , b_y + b);
		     i_y++) {
			for (i_x = max(0, b_x - b);
			     i_x <= min((int)bins_dim , b_x + b); i_x++) {

				bin = i_x + (i_y * bins_dim) +
					(i_z * bins_dim * bins_dim);

				for (i = bin_prefix[bin];
				     i < bin_prefix[bin] + bin_elems[bin];
				     i++) {

					if (i == n)
						continue;

					dist = manhattan_dist_3d(x, y, z,
						    in_x[i], in_x[i + elems],
						    in_x[i + (2*elems)]);

					if (dist <= rsquare) {
						centoid.x += in_x[i];
						centoid.y += in_x[i + elems];
						centoid.z += in_x[i + (2*elems)];
						k++;
					}
				}
			}
		}
	}

	centoid /= k;
	cent_x[n] = centoid.x;
	cent_x[n + elems] = centoid.y;
	cent_x[n + (2*elems)] = centoid.z;
}
