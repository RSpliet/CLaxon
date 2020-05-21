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
	float *in;
	unsigned int i;

	cl_context ctx;
	cl_command_queue q;
	cl_program prg;
	cl_kernel kernel;
	cl_mem clIn, clOut;
	cl_int error;
	cl_event time;
	cl_ulong time_diff = 0l;
	cl_ulong time_avg = 0l;
	const float c0 = 0.1666667;
	const float c1 = 0.0277778;

	const int d[3] = {128, 128, 32};

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

	data_entries = 128*128*32;
	bin_file_read("data/stencil/A0.bin", data_entries, (void **)&in);
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
		"src/stencil/kernel.cl"
	};
	prg = opencl_compile_program(ctx, 1, &programs);

	kernel = clCreateKernel(prg, "naive_kernel", &error);
	if (error != CL_SUCCESS) {
		printf("Could not create kernel\n");
		return -1;
	}

	clIn = clCreateBuffer(ctx, CL_MEM_READ_ONLY,
			data_entries * sizeof(float), NULL, &error);
	if (error != CL_SUCCESS) {
		printf("Could not create in buffer\n");
		return -1;
	}
	clOut = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,
			data_entries * sizeof(float), NULL, &error);
	if (error != CL_SUCCESS) {
		printf("Could not create out buffer\n");
		return -1;
	}

	error = clEnqueueWriteBuffer(q, clIn, CL_FALSE, 0,
			data_entries * sizeof(cl_int), in, 0, NULL, NULL);
	if (error != CL_SUCCESS) {
		printf("Could not enqueue buffer write\n");
		return -1;
	}

	error = clEnqueueWriteBuffer(q, clOut, CL_FALSE, 0,
			data_entries * sizeof(cl_int), in, 0, NULL, NULL);
	if (error != CL_SUCCESS) {
		printf("Could not enqueue buffer write\n");
		return -1;
	}

	error =  clSetKernelArg(kernel, 0, sizeof(float), &c0);
	error =  clSetKernelArg(kernel, 1, sizeof(float), &c1);
	error =  clSetKernelArg(kernel, 2, sizeof(cl_mem), &clIn);
	error |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &clOut);
	error |= clSetKernelArg(kernel, 4, sizeof(cl_int), &d[0]);
	error |= clSetKernelArg(kernel, 5, sizeof(cl_int), &d[1]);
	error |= clSetKernelArg(kernel, 6, sizeof(cl_int), &d[2]);
	if (error != CL_SUCCESS) {
		printf("One of the arguments could not be set: %d.\n", error);
		return -1;
	}

	const size_t dims[] = {128, 128, 32};
	for (i = 0; i < opencl_get_iterations(); i++) {
		error = clEnqueueNDRangeKernel(q, kernel, 3, NULL, dims, NULL,
				0, NULL, &time);
		if (error != CL_SUCCESS) {
			printf("Could not enqueue kernel execution: %d\n", error);
			return -1;
		}
		clFinish(q);

		time_diff = opencl_exec_time(time);
		time_avg += time_diff;
		printf("Time: %lu ns\n", time_diff);
	}

	if (opencl_compare_output()) {
		ret = opencl_compare_out_bin(q, clOut,
				"data/stencil/Anext.bin", data_entries, 0.001f,
				OPENCL_ERROR_ABS);

		if (!ret)
			printf("Output valid\n");
		else
			fprintf(stderr, "Output comparison error: %i\n", ret);
	}

	printf("Time (avg over %u): %lu ns\n", opencl_get_iterations(),
			time_avg / opencl_get_iterations());

	/* Tear down */
	clReleaseEvent(time);
	clReleaseMemObject(clIn);
	clReleaseMemObject(clOut);
	clReleaseKernel(kernel);

	opencl_teardown(&ctx, &q, &prg);
	free(in);

	return ret;
}
