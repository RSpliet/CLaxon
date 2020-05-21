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

#include <stdarg.h>
#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <unistd.h>
#include <inttypes.h>
#include <stdbool.h>
#include <math.h>

#include "lib/opencl.h"
#include "lib/csv.h"
#include "frnn/prefix_sum.h"

enum AXIS {
	X = 0,
	Y,
	Z
};

#define RADIUS 0.01f

const cl_float bins_dim = 100.0f; /* Bins per dimension */
const cl_float radius = RADIUS;    /* Radius for neighbour search */
const cl_float rsquare = RADIUS * RADIUS;

void
usage(char *prg)
{
	printf("%s - Fixed-radix near neighbours\n", prg);
	printf("Options:\n");
	printf("\t-?\t\t This help\n");
	printf("\t-i <file>\t Input file (default: "
			"data/frnn/frnn_stanbun_000.txt)\n");
	printf("\t-v\t\t Verbose: print neighbours\n");
	opencl_usage();
}

cl_mem
frnn_sort(cl_context ctx, cl_command_queue q, cl_program prg,
		size_t elems, cl_mem in, cl_mem *bin_elems,
		cl_mem *bin_prefix, cl_ulong *time_ns)
{
	cl_mem out;
	cl_mem in_bin, bin_idx;
	cl_kernel kernel_ins_cnt, kernel_reindex;
	cl_int error;
	size_t bins;
	const int zero = 0;
	cl_event time;
	cl_ulong t = 0ul;

	/* Determine grid point */
	kernel_ins_cnt = clCreateKernel(prg, "kernel_ins_cnt", &error);
	if (error != CL_SUCCESS) {
		fprintf(stderr, "Could not create kernel\n");
		return NULL;
	}

	in_bin = clCreateBuffer(ctx, CL_MEM_READ_WRITE, elems * sizeof(int),
			NULL, &error);
	if (error != CL_SUCCESS) {
		fprintf(stderr, "Could not create out buffer\n");
		return NULL;
	}

	bins = prefix_sum_elems_ceil(ctx, bins_dim * bins_dim * bins_dim, NULL);
	*bin_elems = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
			bins * sizeof(int), NULL, &error);
	if (error != CL_SUCCESS) {
		fprintf(stderr, "Could not create out buffer\n");
		return NULL;
	}
	clEnqueueFillBuffer(q, *bin_elems, &zero, sizeof(int), 0,
			bins * sizeof(int), 0, NULL, NULL);

	error =  clSetKernelArg(kernel_ins_cnt, 0, sizeof(cl_mem), &in);
	error |= clSetKernelArg(kernel_ins_cnt, 1, sizeof(cl_float), &bins_dim);
	error |= clSetKernelArg(kernel_ins_cnt, 2, sizeof(cl_mem), &in_bin);
	error |= clSetKernelArg(kernel_ins_cnt, 3, sizeof(cl_mem), bin_elems);
	if (error != CL_SUCCESS) {
		fprintf(stderr, "One of the arguments could not be set: %d.\n",
				error);
		return NULL;
	}

	const size_t dims[] = {elems};
	error = clEnqueueNDRangeKernel(q, kernel_ins_cnt, 1, NULL, dims, NULL,
			0, NULL, &time);
	if (error != CL_SUCCESS) {
		fprintf(stderr, "Could not enqueue kernel execution: %d\n",
				error);
		return NULL;
	}

	clFinish(q);
	if (time_ns) {
		t = opencl_exec_time(time);
		printf("Time determining bins: %lins\n", t);
		*time_ns += t;
		t = 0ul;
	}

	clReleaseEvent(time);

	/* Prefix sum to determine grid offsets */
	*bin_prefix = prefix_sum(ctx, q, *bin_elems, bins, &t);
	if (time_ns) {
		printf("Time prefix-sum: %lins\n", t);
		*time_ns += t;
	}

	/* Reorder elements into new buffers */
	kernel_reindex = clCreateKernel(prg, "kernel_reindex", &error);
	if (error != CL_SUCCESS) {
		fprintf(stderr, "Could not create kernel\n");
		return NULL;
	}

	bin_idx = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
			bins * sizeof(int), NULL, &error);
	if (error != CL_SUCCESS) {
		fprintf(stderr, "Could not create out buffer\n");
		return NULL;
	}
	clEnqueueFillBuffer(q, bin_idx, &zero, sizeof(int), 0,
			bins * sizeof(int), 0, NULL, NULL);

	out = clCreateBuffer(ctx, CL_MEM_READ_ONLY,
			3 * elems * sizeof(float), NULL, &error);
	if (error != CL_SUCCESS) {
		fprintf(stderr, "Could not create reordered data buffer\n");
		return NULL;
	}

	error =  clSetKernelArg(kernel_reindex, 0, sizeof(cl_mem), &in);
	error |= clSetKernelArg(kernel_reindex, 1, sizeof(cl_mem), &out);
	error |= clSetKernelArg(kernel_reindex, 2, sizeof(cl_mem), &in_bin);
	error |= clSetKernelArg(kernel_reindex, 3, sizeof(cl_mem), bin_prefix);
	error |= clSetKernelArg(kernel_reindex, 4, sizeof(cl_mem), &bin_idx);
	if (error != CL_SUCCESS) {
		printf("One of the arguments could not be set: %d.\n", error);
		return NULL;
	}
	error = clEnqueueNDRangeKernel(q, kernel_reindex, 1, NULL, dims, NULL,
			0, NULL, &time);
	if (error != CL_SUCCESS) {
		printf("Could not enqueue kernel execution: %d\n", error);
		return NULL;
	}

	clFinish(q);
	if (time_ns) {
		t = opencl_exec_time(time);
		printf("Time reindexing: %lins\n", t);
		*time_ns += t;
	}

	clReleaseEvent(time);

	/* Tear-down */
	clReleaseMemObject(bin_idx);
	clReleaseMemObject(in_bin);
	clReleaseKernel(kernel_ins_cnt);
	clReleaseKernel(kernel_reindex);

	return out;
}

cl_mem
frnn_nn(cl_context ctx, cl_command_queue q, cl_program prg,
		size_t elems, cl_mem in, cl_mem bin_elems,
		cl_mem bin_prefix, cl_ulong *time_ns)
{
	cl_mem nn;
	cl_kernel kernel_nn;
	cl_int error;
	cl_event time;
	cl_int b;
	cl_ulong t;

	/* Determine grid point */
	kernel_nn = clCreateKernel(prg, "kernel_nn", &error);
	if (error != CL_SUCCESS) {
		fprintf(stderr, "Could not create kernel\n");
		return NULL;
	}

	nn = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, elems * sizeof(int),
			NULL, &error);
	if (error != CL_SUCCESS) {
		fprintf(stderr, "Could not create out buffer\n");
		return NULL;
	}

	b = ceil(radius * bins_dim);

	error =  clSetKernelArg(kernel_nn, 0, sizeof(cl_mem), &in);
	error |= clSetKernelArg(kernel_nn, 1, sizeof(cl_float), &bins_dim);
	error |= clSetKernelArg(kernel_nn, 2, sizeof(cl_float), &rsquare);
	error |= clSetKernelArg(kernel_nn, 3, sizeof(cl_int), &b);
	error |= clSetKernelArg(kernel_nn, 4, sizeof(cl_mem), &bin_elems);
	error |= clSetKernelArg(kernel_nn, 5, sizeof(cl_mem), &bin_prefix);
	error |= clSetKernelArg(kernel_nn, 6, sizeof(cl_mem), &nn);
	if (error != CL_SUCCESS) {
		fprintf(stderr, "One of the arguments could not be set: %d.\n",
				error);
		return NULL;
	}

	const size_t dims[] = {elems};
	error = clEnqueueNDRangeKernel(q, kernel_nn, 1, NULL, dims, NULL,
			0, NULL, &time);
	if (error != CL_SUCCESS) {
		fprintf(stderr, "Could not enqueue kernel execution: %d\n",
				error);
		return NULL;
	}

	clFinish(q);
	if (time_ns) {
		t = opencl_exec_time(time);
		printf("Time determining nearest neighbour: %lins\n", t);
		*time_ns += t;
	}

	clReleaseEvent(time);

	/* Tear-down */
	clReleaseKernel(kernel_nn);

	return nn;
}

cl_mem
frnn_centoids(cl_context ctx, cl_command_queue q, cl_program prg,
		size_t elems, cl_mem in, cl_mem bin_elems,
		cl_mem bin_prefix, cl_ulong *time_ns)
{
	cl_mem out;
	cl_kernel kernel_nn;
	cl_int error;
	cl_event time;
	cl_int b;
	cl_ulong t;

	out = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY,
			3 * elems * sizeof(float), NULL, &error);
	if (error != CL_SUCCESS) {
		fprintf(stderr, "Could not create centoid buffer\n");
		return NULL;
	}

	/* Determine grid point */
	kernel_nn = clCreateKernel(prg, "kernel_nn_centoids", &error);
	if (error != CL_SUCCESS) {
		fprintf(stderr, "Could not create kernel\n");
		return NULL;
	}

	b = ceil(radius * bins_dim);

	error =  clSetKernelArg(kernel_nn, 0, sizeof(cl_mem), &in);
	error |= clSetKernelArg(kernel_nn, 1, sizeof(cl_float), &bins_dim);
	error |= clSetKernelArg(kernel_nn, 2, sizeof(cl_float), &rsquare);
	error |= clSetKernelArg(kernel_nn, 3, sizeof(cl_int), &b);
	error |= clSetKernelArg(kernel_nn, 4, sizeof(cl_mem), &bin_elems);
	error |= clSetKernelArg(kernel_nn, 5, sizeof(cl_mem), &bin_prefix);
	error |= clSetKernelArg(kernel_nn, 6, sizeof(cl_mem), &out);
	if (error != CL_SUCCESS) {
		fprintf(stderr, "One of the arguments could not be set: %d.\n",
				error);
		return NULL;
	}

	const size_t dims[] = {elems};
	error = clEnqueueNDRangeKernel(q, kernel_nn, 1, NULL, dims, NULL,
			0, NULL, &time);
	if (error != CL_SUCCESS) {
		fprintf(stderr, "Could not enqueue kernel execution: %d\n",
				error);
		return NULL;
	}

	clFinish(q);
	if (time_ns) {
		t = opencl_exec_time(time);
		printf("Time determining centoids: %lins\n", t);
		*time_ns += t;
	}

	clReleaseEvent(time);

	/* Tear-down */
	clReleaseKernel(kernel_nn);

	return out;
}

int main(int argc, char **argv)
{
	int c;
	int ret;
	char *file = "data/frnn/frnn_stanbun_000.txt";
	int64_t data_entries;
	float **data;

	cl_context ctx;
	cl_command_queue q;
	cl_program prg;
	cl_mem cldata, bin_elems, bin_prefix, nn, centoids;
	cl_mem cldata_ordered;
	cl_int error;
	cl_ulong time_ns = 0;
	bool verbose = false, verbose_centoids = false;

	int *result;
	float **data_ordered, **cents;
	int i;

	while ((c = getopt (argc, argv, "?i:vc"OPENCL_OPTS)) != -1)
	{
		switch (c) {
		case '?':
			usage(argv[0]);
			return 0;
		case 'i':
			file = strdup(optarg);
			break;
		case 'v':
			verbose = true;
			break;
		case 'c':
			verbose_centoids = true;
			break;
		default:
			ret = opencl_parse_option(c, optarg);
			if (ret != 0) {
				usage(argv[0]);
				return -1;
			}
		}
	}

	data_entries = csv_file_read_float_n(file, 3, &data);
	printf("Read %"PRIi64" entries\n", data_entries);

	if (data_entries > UINT32_MAX) {
		/* This limitation stems from the conversion of global id in
		 * frnn.cl from size_t to 32-bit int. Improves AMD performance
		 * by about 6% probably due to reduced register pressure.
		 */
		fprintf(stderr, "Data size (%"PRIu64") too large for"
				"benchmark\n", data_entries);
	}

	ctx = opencl_create_context();
	if (!ctx) {
		usage(argv[0]);
		return -1;
	}

	q = opencl_create_cmdqueue(ctx);
	if (!q) {
		usage(argv[0]);
		return -1;
	}

	const char *programs = {
		"src/frnn/frnn.cl"
	};
	prg = opencl_compile_program(ctx, 1, &programs);

	cldata = clCreateBuffer(ctx, CL_MEM_READ_ONLY,
			data_entries * 3 * sizeof(float), NULL, &error);
	if (error != CL_SUCCESS) {
		fprintf(stderr, "Could not create data buffer\n");
		return -1;
	}

	error =  clEnqueueWriteBuffer(q, cldata, CL_FALSE, 0,
			data_entries * 3 * sizeof(cl_float), data[X], 0,
			NULL, NULL);
	if (error != CL_SUCCESS) {
		fprintf(stderr, "Could not enqueue data write\n");
		return -1;
	}

	cldata_ordered = frnn_sort(ctx, q, prg, data_entries, cldata,
			&bin_elems, &bin_prefix, &time_ns);

	/* Now find nearest neighbours for each element.
	 *
	 * The current implementation only seeks the one closest neighbour
	 * within specified radius.
	 * A search for more (or all) neighbours is feasible, but requires
	 * either;
	 * - Linked lists. Requires a heterogeneous memory model and some
	 *     form of (poor-mans) malloc. The former can technically be
	 *     assumed in embedded systems. The latter requires significant
	 *     overprovisioning of the heap as compute cores cannot request
	 *     memory pages from the OS at runtime.
	 * - An n*n table. Of limited use as post-processing requires n^2
	 *     algorithms to iterate the table.
	 * kNN within radius can technically be done with static lists of k*n,
	 * but requires very awkward sorting algorithms that are likely to
	 * increase register and/or local memory usage of kernels to a point of
	 * diminishing returns.
	 *
	 * XXX: Let user study make an informed decision about benchmark
	 * requirements
	 */

	nn = frnn_nn(ctx, q, prg, data_entries, cldata_ordered, bin_elems,
			bin_prefix, &time_ns);


	if (verbose | verbose_centoids) {
		result = malloc(data_entries * sizeof(int));

		data_ordered = malloc(3 * sizeof(float *));
		for (i = 0; i < 3; i++) {
			data_ordered[i] = malloc(data_entries * sizeof(int));
			clEnqueueReadBuffer(q, cldata_ordered, CL_FALSE, 0,
					sizeof(int) * data_entries,
					data_ordered[i], 0, NULL, NULL);
		}
	}

	if (verbose) {
		printf("Neighbours: \n");
		clEnqueueReadBuffer(q, nn, CL_TRUE, 0,
				sizeof(int) * data_entries, result, 0, NULL,
				NULL);
		for(i = 0; i < data_entries; i++)
			printf("%i (%.3f, %.3f, %.3f): %i\n", i,
					data_ordered[X][i], data_ordered[Y][i],
					data_ordered[Z][i], result[i]);
	}

	centoids = frnn_centoids(ctx, q, prg, data_entries, cldata_ordered,
			bin_elems, bin_prefix, &time_ns);

	if (verbose_centoids) {
		cents = malloc(3 * data_entries * sizeof(float *));
		clEnqueueReadBuffer(q, centoids, CL_TRUE, 0,
				sizeof(float) * data_entries * 3,
				cents, 0, NULL, NULL);
		clFinish(q);

		printf("Centoids: \n");

		for(i = 0; i < data_entries; i++)
			printf("%i (%.3f, %.3f, %.3f): (%.3f, %.3f, %.3f)\n", i,
					data_ordered[X][i], data_ordered[Y][i],
					data_ordered[Z][i], cents[X][i],
					cents[Y][i], cents[Z][i]);
	}

	printf("\n");
	printf("Total execution time (excl data upload): %lins\n", time_ns);

	/* Tear down */
	clReleaseMemObject(cldata);
	clReleaseMemObject(cldata_ordered);
	clReleaseMemObject(centoids);
	clReleaseMemObject(bin_elems);
	clReleaseMemObject(bin_prefix);

	opencl_teardown(&ctx, &q, &prg);
	free(data[X]);
	free(data);

	return 0;
}
