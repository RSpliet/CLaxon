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
	printf("\t-i <file>\t Input file (default: data/cnn_convolution/in_large.txt)\n");
	printf("\t-k <file>\t Kernels input file (default: data/cnn_convolution/kernels_large.txt)\n");
	printf("\t-d <file>\t Download output buffer content to CSV file.\n");
	printf("\t-C <file>\t Comparison reference values (default: data/cnn_convolution/out.csv\n");
	opencl_usage();
}

int main(int argc, char **argv)
{
	int c;
	int ret;
	char *file = "data/cnn_convolution/in_large.txt";
	char *file_kernels = "data/cnn_convolution/kernels_large.txt";
	char *out_ref = "data/cnn_convolution/out.csv";
	int64_t data_entries;
	int64_t kernel_entries;
	float *data, *kernels;
	char *file_out = NULL;
	unsigned int i;
	int retval = 0;

	cl_context ctx;
	cl_command_queue q;
	cl_program prg;
	cl_kernel kernel;
	cl_mem in, in_kernels, out;
	cl_int error;
	cl_event time;
	cl_ulong time_diff = 0l;
	cl_ulong time_avg = 0l;

	const unsigned int three = 3;
	const unsigned int seven = 7;

	while ((c = getopt (argc, argv, "?i:d:k:C:"OPENCL_OPTS)) != -1)
	{
		switch (c) {
		case '?':
			usage(argv[0]);
			return 0;
		case 'i':
			file = strdup(optarg);
			break;
		case 'k':
			file_kernels = strdup(optarg);
			break;
		case 'd':
			file_out = strdup(optarg);
			break;
		case 'C':
			out_ref = strdup(optarg);
			break;
		default:
			ret = opencl_parse_option(c, optarg);
			if (ret != 0) {
				usage(argv[0]);
				return -1;
			}
		}
	}

	data_entries = csv_file_read_float(file, &data);
	printf("Read %"PRIi64" entries\n", data_entries);
	kernel_entries = csv_file_read_float(file_kernels, &kernels);
	printf("Read %"PRIi64" kernel entries\n", kernel_entries);

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
		"src/cnn_convolution/cnn_convolution.cl"
	};
	prg = opencl_compile_program(ctx, 1, &programs);

	kernel = clCreateKernel(prg, "cl_convolution", &error);
	if (error != CL_SUCCESS) {
		printf("Could not create kernel\n");
		return -1;
	}

	in = clCreateBuffer(ctx, CL_MEM_READ_ONLY,
			data_entries * sizeof(float), NULL, &error);
	if (error != CL_SUCCESS) {
		printf("Could not create in buffer\n");
		return -1;
	}
	in_kernels = clCreateBuffer(ctx, CL_MEM_READ_ONLY,
			kernel_entries * sizeof(float), NULL, &error);
	if (error != CL_SUCCESS) {
		printf("Could not create biases buffer\n");
		return -1;
	}
	out = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,
			(data_entries * 64 * sizeof(float)) / 3, NULL, &error);
	if (error != CL_SUCCESS) {
		printf("Could not create out buffer\n");
		return -1;
	}

	error = clEnqueueWriteBuffer(q, in, CL_FALSE, 0,
			data_entries * sizeof(cl_int), data, 0, NULL, NULL);
	error |= clEnqueueWriteBuffer(q, in_kernels, CL_TRUE, 0,
			kernel_entries * sizeof(cl_int), kernels, 0, NULL, NULL);
	if (error != CL_SUCCESS) {
		printf("Could not enqueue buffer write\n");
		return -1;
	}

	error =  clSetKernelArg(kernel, 0, sizeof(cl_mem), &in);
	error |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &in_kernels);
	error |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &out);
	error |= clSetKernelArg(kernel, 3, sizeof(int), &seven);
	error |= clSetKernelArg(kernel, 4, sizeof(int), &three);
	error |= clSetKernelArg(kernel, 5, seven * seven * three * sizeof(float),
			NULL);
	if (error != CL_SUCCESS) {
		printf("One of the arguments could not be set: %d.\n", error);
		return -1;
	}

	const size_t dims[] = {218, 218, 64};
	//const size_t ldims[] = {218, 1, 1};
	/* NVIDIA defaults to local workgroups of size {218,1,1}.
	 * More square-shaped configurations diminish performance. Bigger
	 * work groups (222,3,1), (111, 6, 1) reduce parallelism. Second channel
	 * in same work-group diminishes perf. Too many cores for amount
	 * of work?
	 */
	for (i = 0; i < opencl_get_iterations(); i++) {
		error = clEnqueueNDRangeKernel(q, kernel, 3, NULL, dims, NULL,
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

	/* Both don't make sense... really. */
	if (file_out) {
		opencl_download_float_csv(q, out, file_out, 218*218*64);
	} else if (opencl_compare_output()) {
		retval = opencl_compare_out_csv(q, out, out_ref, 218*218*64,
				0.001f, OPENCL_ERROR_ABS);

		if (!retval)
			printf("Output valid\n");
		else
			printf("Output invalid\n");
	}

	printf("Time (avg over %u): %lu ns\n", opencl_get_iterations(),
			time_avg / opencl_get_iterations());

	/* Tear down */
	clReleaseEvent(time);
	clReleaseMemObject(in);
	clReleaseMemObject(in_kernels);
	clReleaseMemObject(out);
	clReleaseKernel(kernel);

	opencl_teardown(&ctx, &q, &prg);
	free(data);
	free(kernels);

	return retval;
}
