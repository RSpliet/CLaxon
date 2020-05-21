/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2019 Roy Spliet, University of Cambridge
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

void usage(char *prg)
{
	printf("%s\n", prg);
	printf("Options:\n");
	printf("\t-?\t\t This help\n");
	opencl_usage();
}

int main(int argc, char **argv)
{
	int c;
	int ret;
	int64_t data_entries;
	float *inData;
	float *inIndex;
	float *inPerm;
	float *inXVec;
	int *inJdsPtr;
	int *inShZcnt;

	cl_context ctx;
	cl_command_queue q;
	cl_program prg;
	cl_kernel kernel;
	cl_mem clInData, clInIndex, clInPerm, clInXVec, clInJdsPtr, clInShZcnt;
	cl_mem clOutVec;
	cl_int error;
	cl_event time;
	cl_ulong time_diff = 0l;
	cl_ulong time_avg = 0l;
	unsigned int i;
	const int xvec_sz = 11948;

	while ((c = getopt (argc, argv, "?"OPENCL_OPTS)) != -1)
	{
		switch (c) {
		case '?':
			usage(argv[0]);
			return 0;
		default:
			ret = opencl_parse_option(c, optarg);
			if (ret != 0) {
				usage(argv[0]);
				return -1;
			}
		}
	}

	data_entries = 150144;
	bin_file_read("data/spmv/data.bin", data_entries, (void **) &inData);
	bin_file_read("data/spmv/indices.bin", data_entries, (void **) &inIndex);
	bin_file_read("data/spmv/perm.bin", xvec_sz, (void **) &inPerm);
	bin_file_read("data/spmv/x_vector.bin", xvec_sz, (void **) &inXVec);
	bin_file_read("data/spmv/jds_ptr_int.bin", 50, (void **) &inJdsPtr);
	bin_file_read("data/spmv/sh_zcnt_int.bin", 374, (void **) &inShZcnt);

	printf("Read %"PRIi64" entries\n", data_entries);

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
		"src/spmv/kernel.cl"
	};
	prg = opencl_compile_program(ctx, 1, &programs);

	kernel = clCreateKernel(prg, "spmv_jds_naive", &error);
	if (error != CL_SUCCESS) {
		printf("Could not create kernel\n");
		return -1;
	}

	clInData = clCreateBuffer(ctx, CL_MEM_READ_ONLY,
			data_entries * sizeof(float), NULL, &error);
	if (error != CL_SUCCESS) {
		printf("Could not create in buffer\n");
		return -1;
	}

	clInIndex = clCreateBuffer(ctx, CL_MEM_READ_ONLY,
			data_entries * sizeof(int), NULL, &error);
	if (error != CL_SUCCESS) {
		printf("Could not create in buffer\n");
		return -1;
	}

	clInPerm = clCreateBuffer(ctx, CL_MEM_READ_ONLY,
			11948 * sizeof(int), NULL, &error);
	if (error != CL_SUCCESS) {
		printf("Could not create in buffer\n");
		return -1;
	}

	clInXVec = clCreateBuffer(ctx, CL_MEM_READ_ONLY,
			11948 * sizeof(float), NULL, &error);
	if (error != CL_SUCCESS) {
		printf("Could not create in buffer\n");
		return -1;
	}

	clInJdsPtr = clCreateBuffer(ctx, CL_MEM_READ_ONLY,
			50 * sizeof(int), NULL, &error);
	if (error != CL_SUCCESS) {
		printf("Could not create in buffer\n");
		return -1;
	}

	clInShZcnt = clCreateBuffer(ctx, CL_MEM_READ_ONLY,
			374 * sizeof(int), NULL, &error);
	if (error != CL_SUCCESS) {
		printf("Could not create in buffer\n");
		return -1;
	}

	clOutVec = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,
			data_entries * sizeof(float), NULL, &error);
	if (error != CL_SUCCESS) {
		printf("Could not create out buffer\n");
		return -1;
	}

	error = clEnqueueWriteBuffer(q, clInData, CL_FALSE, 0,
			data_entries * sizeof(float), inData, 0, NULL, NULL);
	error |= clEnqueueWriteBuffer(q, clInIndex, CL_FALSE, 0,
			data_entries * sizeof(cl_int), inIndex, 0, NULL, NULL);
	error |= clEnqueueWriteBuffer(q, clInPerm, CL_FALSE, 0,
			xvec_sz * sizeof(cl_int), inPerm, 0, NULL, NULL);
	error |= clEnqueueWriteBuffer(q, clInXVec, CL_FALSE, 0,
			xvec_sz * sizeof(float), inXVec, 0, NULL, NULL);
	error |= clEnqueueWriteBuffer(q, clInJdsPtr, CL_FALSE, 0,
			50 * sizeof(cl_int), inJdsPtr, 0, NULL, NULL);
	error |= clEnqueueWriteBuffer(q, clInShZcnt, CL_FALSE, 0,
			374 * sizeof(float), inShZcnt, 0, NULL, NULL);
	if (error != CL_SUCCESS) {
		printf("Could not enqueue buffer write\n");
		return -1;
	}

	error  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &clOutVec);
	error |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &clInData);
	error |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &clInIndex);
	error |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &clInPerm);
	error |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &clInXVec);
	error |= clSetKernelArg(kernel, 5, sizeof(cl_uint), &xvec_sz);
	error |= clSetKernelArg(kernel, 6, sizeof(cl_mem), &clInJdsPtr);
	error |= clSetKernelArg(kernel, 7, sizeof(cl_mem), &clInShZcnt);
	if (error != CL_SUCCESS) {
		printf("One of the arguments could not be set: %d.\n", error);
		return -1;
	}

	const size_t dims[] = {(xvec_sz % 256 ?
			(xvec_sz & ~255) + 256 : xvec_sz)};
	const size_t ldims[] = {256};

	for (i = 0; i < opencl_get_iterations(); i++) {
		error = clEnqueueNDRangeKernel(q, kernel, 1, NULL, dims, ldims,
				0, NULL, &time);
		if (error != CL_SUCCESS) {
			printf("Could not enqueue kernel execution: %d\n",
					error);
			return -1;
		}
		clFinish(q);

		time_diff = opencl_exec_time(time);
		time_avg += time_diff;
		printf("Time: %lu ns\n", time_diff);
	}

	if (opencl_compare_output()) {
		ret = opencl_compare_out_csv(q, clOutVec,
				"data/spmv/dst_vector.csv", xvec_sz, 0.05f,
				OPENCL_ERROR_FRAC);

		if (!ret)
			printf("Output valid\n");
		else
			fprintf(stderr, "Output comparison error: %i\n", ret);
	}

	printf("Time (avg of %u): %lu ns\n", opencl_get_iterations(),
			time_avg / opencl_get_iterations());

	/* Tear down */
	clReleaseEvent(time);
	clReleaseMemObject(clInData);
	clReleaseMemObject(clInIndex);
	clReleaseMemObject(clInJdsPtr);
	clReleaseMemObject(clInPerm);
	clReleaseMemObject(clInShZcnt);
	clReleaseMemObject(clInXVec);
	clReleaseMemObject(clOutVec);
	clReleaseKernel(kernel);

	opencl_teardown(&ctx, &q, &prg);
	free(inData);
	free(inIndex);
	free(inJdsPtr);
	free(inPerm);
	free(inXVec);
	free(inShZcnt);

	return ret;
}
