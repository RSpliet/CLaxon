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

#include <stdarg.h>
#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <unistd.h>
#include <inttypes.h>

#include "lib/opencl.h"
#include "lib/csv.h"
#include "frnn/prefix_sum.h"

#define FILE_1 "data/ndt/room_scan1.txt"
#define FILE_2 "data/ndt/room_scan2.txt"

enum AXIS {
	X = 0,
	Y,
	Z
};

const cl_float bins_dim = 40.0f; /* Bins per dimension */

void
usage()
{
	printf("ndt - test program for OpenCL routines\n");
	printf("Options:\n");
	printf("\t-?\t\t This help\n");
	printf("\t-i <file>\t Base input file (default: "FILE_1")\n");
	printf("\t-b <file>\t Unregistered input file (default: "FILE_2")\n");
	opencl_usage();
}

void
calc_translation(float x, float y, float z, float rz, float target[12])
{
	target[3] = x;
	target[7] = y;
	target[11] = z;

	target[0] = 1.f;
	target[1] = 0.f;
	target[2] = 0.f;
	target[4] = 0.f;
	target[5] = 1.f;
	target[6] = 0.f;
	target[8] = 0.f;
	target[9] = 0.f;
	target[10] = 1.f;
}

unsigned int
sorted_entries(cl_command_queue q, cl_mem bin_elems, cl_mem bin_prefix)
{
	unsigned int last_bin;
	unsigned int prefix;
	unsigned int count;

	last_bin = (bins_dim * bins_dim * bins_dim) - 1;
	clEnqueueReadBuffer(q, bin_prefix, CL_FALSE, sizeof(cl_uint) * last_bin,
			sizeof(cl_uint), &prefix, 0, NULL, NULL);
	clEnqueueReadBuffer(q, bin_elems, CL_TRUE, sizeof(cl_uint) * last_bin,
				sizeof(cl_uint), &count, 0, NULL, NULL);

	return prefix + count;
}

cl_mem
ndt_sort(cl_context ctx, cl_command_queue q, cl_program prg,
		unsigned int elems, cl_mem in, unsigned int *sorted_elems,
		cl_mem *bin_elems, cl_mem *bin_prefix, cl_ulong *time_ns)
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
	error |= clSetKernelArg(kernel_ins_cnt, 1, sizeof(cl_uint), &elems);
	error |= clSetKernelArg(kernel_ins_cnt, 2, sizeof(cl_float), &bins_dim);
	error |= clSetKernelArg(kernel_ins_cnt, 3, sizeof(cl_mem), &in_bin);
	error |= clSetKernelArg(kernel_ins_cnt, 4, sizeof(cl_mem), bin_elems);
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

	*sorted_elems = sorted_entries(q, *bin_elems, *bin_prefix);

	/* Reorder elements into new buffers */
	kernel_reindex = clCreateKernel(prg, "kernel_reindex", &error);
	if (error != CL_SUCCESS) {
		fprintf(stderr, "Could not create kernel\n");
		return NULL;
	}

	bin_idx = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS,
			bins * sizeof(int), NULL, &error);
	if (error != CL_SUCCESS) {
		fprintf(stderr, "Could not create out buffer\n");
		return NULL;
	}
	clEnqueueFillBuffer(q, bin_idx, &zero, sizeof(int), 0,
			bins * sizeof(int), 0, NULL, NULL);

	out = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY,
			3 * *sorted_elems * sizeof(float), NULL, &error);
	if (error != CL_SUCCESS) {
		fprintf(stderr, "Could not create reordered data buffer\n");
		return NULL;
	}

	error =  clSetKernelArg(kernel_reindex, 0, sizeof(cl_mem), &in);
	error |= clSetKernelArg(kernel_reindex, 1, sizeof(cl_uint), &elems);
	error |= clSetKernelArg(kernel_reindex, 2, sizeof(cl_mem), &out);
	error |= clSetKernelArg(kernel_reindex, 3, sizeof(cl_uint), sorted_elems);
	error |= clSetKernelArg(kernel_reindex, 4, sizeof(cl_mem), &in_bin);
	error |= clSetKernelArg(kernel_reindex, 5, sizeof(cl_mem), bin_prefix);
	error |= clSetKernelArg(kernel_reindex, 6, sizeof(cl_mem), &bin_idx);
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

int
ndt_cell_qC(cl_context ctx, cl_command_queue q, cl_program prg,
		cl_mem data, unsigned int elems, cl_mem bin_elems,
		cl_mem bin_prefix, cl_ulong *time_ns)
{
	cl_kernel kernel;
	cl_int error;
	cl_mem out_q, out_C;
	cl_event time;
	cl_ulong time_diff = 0l;

	kernel = clCreateKernel(prg, "ndt_cell_qC", &error);
	if (error != CL_SUCCESS) {
		printf("Could not create kernel\n");
		return -1;
	}

	out_q = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
			elems * 3 * sizeof(float), NULL, &error); /* XXX */
	if (error != CL_SUCCESS) {
		printf("Could not create out buffer\n");
		return -1;
	}

	out_C = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
			elems * 9 * sizeof(float), NULL, &error); /* XXX */
	if (error != CL_SUCCESS) {
		printf("Could not create out buffer\n");
		return -1;
	}

	error =  clSetKernelArg(kernel, 0, sizeof(cl_mem), &data);
	error |= clSetKernelArg(kernel, 1, sizeof(cl_uint), &elems);
	error |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &bin_elems);
	error |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &bin_prefix);
	error |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &out_q);
	error |= clSetKernelArg(kernel, 5, sizeof(cl_mem), &out_C);

	if (error != CL_SUCCESS) {
		printf("One of the arguments could not be set: %d.\n", error);
		return -1;
	}

	const size_t dims[] = {(size_t)(bins_dim * bins_dim * bins_dim)};
	error = clEnqueueNDRangeKernel(q, kernel, 1, NULL, dims, NULL, 0, NULL,
			&time);
	if (error != CL_SUCCESS) {
		printf("Could not enqueue kernel execution: %d\n", error);
		return -1;
	}
	clFinish(q);

	time_diff = opencl_exec_time(time);
	printf("NDT mean/covariant mat: %lu ns\n", time_diff);
	if (time_ns) {
		*time_ns += time_diff;
		printf("*Per-cell mean/covariant: %lu ns\n", *time_ns);
	}

	printf("---------------------------------\n");

	return 0;
}

int
ndt_elem_qC(cl_context ctx, cl_command_queue q, cl_program prg,
		cl_mem data, unsigned int elems)
{
	cl_kernel kernel = 0;
	cl_int error;
	cl_mem out_q, out_C, cell;
	cl_event time;
	cl_ulong time_diff = 0l;
	cl_ulong time_total = 0l;
	cl_mem bin_elems;
	size_t bins;
	const int zero = 0;
	const cl_float fzero = 0.f;
	int ret = -1;
	unsigned int y;

	kernel = clCreateKernel(prg, "ndt_elem_q", &error);
	if (error != CL_SUCCESS) {
		printf("Could not create kernel\n");
		return -1;
	}

	cell = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS,
				elems * sizeof(float), NULL, &error); /* XXX */
	if (error != CL_SUCCESS) {
		printf("Could not create out buffer\n");
		return -1;
	}

	bins = prefix_sum_elems_ceil(ctx, bins_dim * bins_dim * bins_dim, NULL);
	bin_elems = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
			bins * sizeof(int), NULL, &error);
	if (error != CL_SUCCESS) {
		fprintf(stderr, "Could not create out buffer\n");
		goto error;
	}
	clEnqueueFillBuffer(q, bin_elems, &zero, sizeof(int), 0,
			bins * sizeof(int), 0, NULL, NULL);

	out_q = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
			elems * 3 * sizeof(float), NULL, &error); /* XXX */
	if (error != CL_SUCCESS) {
		printf("Could not create out buffer\n");
		goto error;
	}
	clEnqueueFillBuffer(q, out_q, &fzero, sizeof(float), 0,
			elems * 3  * sizeof(float), 0, NULL, NULL);

	out_C = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
			elems * 9 * sizeof(float), NULL, &error); /* XXX */
	if (error != CL_SUCCESS) {
		printf("Could not create out buffer\n");
		goto error;
	}
	clEnqueueFillBuffer(q, out_C, &fzero, sizeof(cl_float), 0,
				elems * 9  * sizeof(cl_float), 0, NULL, NULL);
	clFinish(q);

	error =  clSetKernelArg(kernel, 0, sizeof(cl_mem), &data);
	error |= clSetKernelArg(kernel, 1, sizeof(cl_uint), &elems);
	error |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &cell);
	error |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &bin_elems);
	error |= clSetKernelArg(kernel, 4, sizeof(cl_float), &bins_dim);
	error |= clSetKernelArg(kernel, 5, sizeof(cl_mem), &out_q);
	if (error != CL_SUCCESS) {
		printf("One of the arguments could not be set: %d.\n", error);
		return -1;
	}

	y = elems / 1024;
	if (elems % 1024)
		y++;

	const size_t dims[] = {1024, y};

	error = clEnqueueNDRangeKernel(q, kernel, 2, NULL, dims, NULL, 0, NULL,
			&time);
	if (error != CL_SUCCESS) {
		printf("Could not enqueue kernel execution: %d\n", error);
		return -1;
	}
	clFinish(q);

	time_diff = opencl_exec_time(time);
	time_total += time_diff;
	printf("NDT mean: %lu ns\n", time_diff);

	clReleaseKernel(kernel);
	kernel = 0;

	kernel = clCreateKernel(prg, "ndt_elem_C", &error);
	if (error != CL_SUCCESS) {
		printf("Could not create kernel\n");
		goto error;
	}
	error =  clSetKernelArg(kernel, 0, sizeof(cl_mem), &data);
	error |= clSetKernelArg(kernel, 1, sizeof(cl_uint), &elems);
	error |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &cell);
	error |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &bin_elems);
	error |= clSetKernelArg(kernel, 4, sizeof(cl_float), &bins_dim);
	error |= clSetKernelArg(kernel, 5, sizeof(cl_mem), &out_q);
	error |= clSetKernelArg(kernel, 6, sizeof(cl_mem), &out_C);
	if (error != CL_SUCCESS) {
		printf("One of the arguments could not be set: %d.\n", error);
		goto error;
	}

	error = clEnqueueNDRangeKernel(q, kernel, 2, NULL, dims, NULL, 0, NULL,
			&time);
	if (error != CL_SUCCESS) {
		printf("Could not enqueue kernel execution: %d\n", error);
		goto error;
	}
	clFinish(q);

	time_diff = opencl_exec_time(time);
	time_total += time_diff;

	printf("NDT covariant: %lu ns\n", time_diff);
	clReleaseKernel(kernel);
	kernel = 0;

	kernel = clCreateKernel(prg, "ndt_elem_qC_post", &error);
	if (error != CL_SUCCESS) {
		printf("Could not create kernel\n");
		return -1;
	}
	error =  clSetKernelArg(kernel, 0, sizeof(cl_mem), &data);
	error |= clSetKernelArg(kernel, 1, sizeof(cl_uint), &elems);
	error |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &bin_elems);
	error |= clSetKernelArg(kernel, 3, sizeof(cl_float), &bins_dim);
	error |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &out_q);
	error |= clSetKernelArg(kernel, 5, sizeof(cl_mem), &out_C);
	if (error != CL_SUCCESS) {
		printf("One of the arguments could not be set: %d.\n", error);
		goto error;
	}

	const size_t dims_cell[] = {(size_t)(bins_dim * bins_dim * bins_dim)};
	error = clEnqueueNDRangeKernel(q, kernel, 1, NULL, dims_cell, NULL, 0,
			NULL, &time);
	if (error != CL_SUCCESS) {
		printf("Could not enqueue kernel execution: %d\n", error);
		goto error;
	}
	clFinish(q);
	time_diff = opencl_exec_time(time);
	time_total += time_diff;

	printf("NDT post: %lu ns\n", time_diff);
	printf("* Per-elem mean/covariant: %lu ns\n", time_total);
	printf("---------------------------------\n");
	ret = 0;

error:
	clReleaseMemObject(cell);
	if (kernel)
		clReleaseKernel(kernel);
	clReleaseEvent(time);

	return ret;
}

int
ndt_elem_transform(cl_context ctx, cl_command_queue q, cl_program prg,
		float *in, unsigned int elems, cl_mem *out)
{
	cl_kernel kernel;
	cl_int error;
	cl_mem cl_in, trans;
	cl_event time;
	cl_ulong time_diff;
	float bias[12];
	int ret = -1;
	unsigned int y;

	kernel = clCreateKernel(prg, "ndt_vec_transform", &error);
	if (error != CL_SUCCESS) {
		printf("Could not create kernel\n");
		return -1;
	}

	cl_in = clCreateBuffer(ctx, CL_MEM_READ_ONLY,
			elems * 3 * sizeof(float), NULL, &error);
	if (error != CL_SUCCESS) {
		printf("Could not create in buffer\n");
		return -1;
	}
	trans = clCreateBuffer(ctx, CL_MEM_READ_ONLY,
			12 * sizeof(float), NULL, &error);
	if (error != CL_SUCCESS) {
		printf("Could not create biases buffer\n");
		return -1;
	}

	*out = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
			elems * 3 * sizeof(float), NULL, &error);
	if (error != CL_SUCCESS) {
		printf("Could not create out buffer\n");
		goto error;
	}

	error = clEnqueueWriteBuffer(q, cl_in, CL_FALSE, 0,
			elems * 3 * sizeof(cl_float), in, 0, NULL,
			NULL);
	error |= clEnqueueWriteBuffer(q, trans, CL_TRUE, 0,
			12 * sizeof(cl_float), bias, 0, NULL, NULL);
	if (error != CL_SUCCESS) {
		printf("Could not enqueue buffer write\n");
		goto error;
	}

	calc_translation(1.79387f, 0.720047f, 0.f, 0.f, bias);

	error =  clSetKernelArg(kernel, 0, sizeof(cl_mem), &cl_in);
	error |= clSetKernelArg(kernel, 1, sizeof(cl_uint), &elems);
	error |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &trans);
	error |= clSetKernelArg(kernel, 3, sizeof(cl_mem), out);
	if (error != CL_SUCCESS) {
		printf("One of the arguments could not be set: %d.\n", error);
		goto error;
	}

	y = elems / 1024;
	if (elems % 1024)
		y++;

	const size_t dims[] = {1024, y};
	error = clEnqueueNDRangeKernel(q, kernel, 2, NULL, dims, NULL, 0, NULL,
			&time);
	if (error != CL_SUCCESS) {
		printf("Could not enqueue kernel execution: %d\n", error);
		goto error;
	}
	clFinish(q);
	/* XXX: validate? */

	time_diff = opencl_exec_time(time);
	printf("* NDT data transform: %lu ns\n", time_diff);
	ret = 0;

error:
	clReleaseMemObject(cl_in);
	clReleaseMemObject(trans);
	clReleaseKernel(kernel);
	clReleaseEvent(time);

	return ret;
}

int test_3x3_inv(cl_context ctx, cl_command_queue q, cl_program prg)
{
/* XXX: Academic code... */
//	//float mat_in[9] = {1.0, 2.0, 3.0, 0.0, 1.0, 4.0, 5.0, 6.0, 0.0};
//	float mat_in[9] = {1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0};
//	float mat_mul[9];

//	kernel = clCreateKernel(prg, "invert_3x3", &error);
//	if (error != CL_SUCCESS) {
//		printf("Could not create kernel\n");
//		return -1;
//	}
//
//	in = clCreateBuffer(ctx, CL_MEM_READ_ONLY,
//				9 * sizeof(float), NULL, &error);
//	if (error != CL_SUCCESS) {
//		printf("Could not create in buffer\n");
//		return -1;
//	}
//	out = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,
//			9 * sizeof(float), NULL, &error); /* XXX */
//	if (error != CL_SUCCESS) {
//		printf("Could not create out buffer\n");
//		return -1;
//	}
//
//	error =  clSetKernelArg(kernel, 0, sizeof(cl_mem), &in);
//	error |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &out);
//	if (error != CL_SUCCESS) {
//		printf("One of the arguments could not be set: %d.\n", error);
//		return -1;
//	}
//
//	error = clEnqueueWriteBuffer(q, in, CL_FALSE, 0,
//			9 * sizeof(cl_float), mat_in, 0, NULL, NULL);
//
//	const size_t dims2[] = {1};
//	error = clEnqueueNDRangeKernel(q, kernel, 1, NULL, dims2, NULL, 0, NULL,
//			&time);
//	if (error != CL_SUCCESS) {
//		printf("Could not enqueue kernel execution: %d\n", error);
//		return -1;
//	}
//	clFinish(q);
//
//
//	clEnqueueReadBuffer(q, out, CL_TRUE, 0, sizeof(float) * 9, result, 0,
//			NULL, NULL);
//	printf("result:\n");
//	printf("%f, %f, %f\n", result[0], result[1], result[2]);
//	printf("%f, %f, %f\n", result[3], result[4], result[5]);
//	printf("%f, %f, %f\n", result[6], result[7], result[8]);
//
//	/* Verify */
//	mat_mul[0] = mat_in[0] * result[0] + mat_in[1] * result[3] + mat_in[2] * result[6];
//	mat_mul[1] = mat_in[0] * result[1] + mat_in[1] * result[4] + mat_in[2] * result[7];
//	mat_mul[2] = mat_in[0] * result[2] + mat_in[1] * result[5] + mat_in[2] * result[8];
//	mat_mul[3] = mat_in[3] * result[0] + mat_in[4] * result[3] + mat_in[5] * result[6];
//	mat_mul[4] = mat_in[3] * result[1] + mat_in[4] * result[4] + mat_in[5] * result[7];
//	mat_mul[5] = mat_in[3] * result[2] + mat_in[4] * result[5] + mat_in[5] * result[8];
//	mat_mul[6] = mat_in[6] * result[0] + mat_in[7] * result[3] + mat_in[8] * result[6];
//	mat_mul[7] = mat_in[6] * result[1] + mat_in[7] * result[4] + mat_in[8] * result[7];
//	mat_mul[8] = mat_in[6] * result[2] + mat_in[7] * result[5] + mat_in[8] * result[8];
//
//	printf("in * result:\n");
//	printf("%f, %f, %f\n", mat_mul[0], mat_mul[1], mat_mul[2]);
//	printf("%f, %f, %f\n", mat_mul[3], mat_mul[4], mat_mul[5]);
//	printf("%f, %f, %f\n", mat_mul[6], mat_mul[7], mat_mul[8]);

	return 0;
}


int main(int argc, char **argv)
{
	int c;
	int ret;
	char *file_1 = FILE_1;
	char *file_2 = FILE_2;
	int64_t data_entries;
	int64_t source_entries;
	uint32_t elems;
	float **data, **source;
	/*unsigned int sorted_elems;*/

	cl_context ctx;
	cl_command_queue q;
	cl_program prg;
	cl_mem src_unsorted, cl_data/*, src_sorted, bin_elems, bin_prefix*/;
	cl_int error;

	while ((c = getopt (argc, argv, "?i:"OPENCL_OPTS)) != -1)
	{
		switch (c) {
		case '?':
			usage();
			return 0;
		case 'i':
			file_1 = strdup(optarg);
			break;
		case 'b':
			file_2 = strdup(optarg);
			break;
		default:
			ret = opencl_parse_option(c, optarg);
			if (ret != 0) {
				usage();
				return -1;
			}
		}
	}

	source_entries = csv_file_read_float_n(file_1, 3, &source);
	printf("Read %"PRIi64" entries\n", source_entries);
	data_entries = csv_file_read_float_n(file_2, 3, &data);
	printf("Read %"PRIi64" entries\n", data_entries);
	elems = data_entries;

	ctx = opencl_create_context();
	if (!ctx) {
		usage();
		return -1;
	}

	q = opencl_create_cmdqueue(ctx);
	if (!q) {
		usage();
		return -1;
	}

	const char *programs = {
		"src/ndt/ndt.cl"
	};
	prg = opencl_compile_program(ctx, 1, &programs);

	ndt_elem_transform(ctx, q, prg, data[0], elems,
			&cl_data);

	src_unsorted = clCreateBuffer(ctx, CL_MEM_READ_ONLY,
			source_entries * 3 * sizeof(cl_float), NULL, &error);
	if (error != CL_SUCCESS) {
		printf("Could not create in buffer\n");
		return -1;
	}

	error = clEnqueueWriteBuffer(q, src_unsorted, CL_TRUE, 0,
			source_entries * 3 * sizeof(cl_float), source[0], 0,
			NULL, NULL);
	if (error != CL_SUCCESS) {
		printf("Could not enqueue buffer write\n");
		return -1;
	}
	clFinish(q);

	/*
	src_sorted = ndt_sort(ctx, q, prg, (unsigned int) source_entries,
			src_unsorted, &sorted_elems, &bin_elems, &bin_prefix,
			&time_sort);
	ndt_cell_qC(ctx, q, prg, src_sorted, sorted_elems, bin_elems,
			bin_prefix, &time_sort);
	 */
	ndt_elem_qC(ctx, q, prg, src_unsorted, (unsigned int) source_entries);

	/* test_inv_3x3() */

	/* Tear down */
	opencl_teardown(&ctx, &q, &prg);
	free(data);

	return 0;
}
