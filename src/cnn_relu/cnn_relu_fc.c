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
	printf("\t-i <file>\t Input file (default: data/cnn_relu/cnn_relu.txt)\n");
	printf("\t-b <file>\t Bias input file (default: data/cnn_relu/cnn_relu_biases.txt)\n");
	printf("\t-C <file>\t Comparison reference values (default: data/cnn_relu/out.csv\n");
	printf("\t-d <file>\t Store output into <file>\n");
	opencl_usage();
}

int main(int argc, char **argv)
{
	int c;
	int ret;
	char *file = "data/cnn_relu/in_large.bin";
	char *file_bias = "data/cnn_relu/biases_large.bin";
	char *file_weights = "data/cnn_relu/weights_large.bin";
	char *out_ref = "data/cnn_relu/out_large.csv";
	float *data, *bias, *weight;
	unsigned int i;
	int retval = 0;

	cl_context ctx;
	cl_command_queue q;
	cl_program prg;
	cl_kernel kernel;
	cl_mem in, weights, biases, out;
	cl_int error;
	cl_event time;
	cl_ulong time_diff = 0l;
	cl_ulong time_avg = 0l;
	char *file_out = NULL;

	const int fourK = 4096;

	while ((c = getopt (argc, argv, "?i:b:C:d:"OPENCL_OPTS)) != -1)
	{
		switch (c) {
		case '?':
			usage(argv[0]);
			return 0;
		case 'i':
			file = strdup(optarg);
			break;
		case 'b':
			file_bias = strdup(optarg);
			break;
		case 'C':
			out_ref = strdup(optarg);
			break;
		case 'd':
			file_out = strdup(optarg);
			break;
		default:
			ret = opencl_parse_option(c, optarg);
			if (ret != 0) {
				usage(argv[0]);
				return -1;
			}
		}
	}

	bin_file_read(file, 4096, (void **) &data);
	bin_file_read(file_bias, 4096, (void **) &bias);
	bin_file_read(file_weights, 4096*4096,(void **) &weight);

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
		"src/cnn_relu/cnn_relu_fc.cl"
	};
	prg = opencl_compile_program(ctx, 1, &programs);

	kernel = clCreateKernel(prg, "cl_relu", &error);
	if (error != CL_SUCCESS) {
		printf("Could not create kernel\n");
		return -1;
	}

	in = clCreateBuffer(ctx, CL_MEM_READ_ONLY,
			4096 * sizeof(float), NULL, &error);
	if (error != CL_SUCCESS) {
		printf("Could not create in buffer\n");
		return -1;
	}
	biases = clCreateBuffer(ctx, CL_MEM_READ_ONLY,
			4096 * sizeof(float), NULL, &error);
	if (error != CL_SUCCESS) {
		printf("Could not create biases buffer\n");
		return -1;
	}
	weights = clCreateBuffer(ctx, CL_MEM_READ_ONLY,
			4096*4096 * sizeof(float), NULL, &error);
	if (error != CL_SUCCESS) {
		printf("Could not create biases buffer\n");
		return -1;
	}

	out = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,
			4096 * sizeof(float), NULL, &error);
	if (error != CL_SUCCESS) {
		printf("Could not create out buffer\n");
		return -1;
	}

	error = clEnqueueWriteBuffer(q, in, CL_FALSE, 0,
			4096 * sizeof(cl_int), data, 0, NULL, NULL);
	error |= clEnqueueWriteBuffer(q, biases, CL_TRUE, 0,
			4096 * sizeof(cl_int), bias, 0, NULL, NULL);
	error |= clEnqueueWriteBuffer(q, weights, CL_TRUE, 0,
			4096*4096 * sizeof(cl_int), weight, 0, NULL, NULL);
	if (error != CL_SUCCESS) {
		printf("Could not enqueue buffer write\n");
		return -1;
	}

	error =  clSetKernelArg(kernel, 0, sizeof(cl_mem), &in);
	error |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &biases);
	error |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &weights);
	error |= clSetKernelArg(kernel, 3, sizeof(int), &fourK);
	error |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &out);
	if (error != CL_SUCCESS) {
		printf("One of the arguments could not be set: %d.\n", error);
		return -1;
	}

	const size_t dims[] = {4096};
	for (i = 0; i < opencl_get_iterations(); i++) {
		error = clEnqueueNDRangeKernel(q, kernel, 1, NULL, dims, NULL,
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

	if (file_out) {
		opencl_download_float_csv(q, out, file_out, 4096);
	} else if (opencl_compare_output()) {
		retval = opencl_compare_out_csv(q, out, out_ref, 4096,
				0.0001f, OPENCL_ERROR_ABS);

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
	clReleaseMemObject(biases);
	clReleaseMemObject(weights);
	clReleaseMemObject(out);
	clReleaseKernel(kernel);

	opencl_teardown(&ctx, &q, &prg);
	free(data);
	free(bias);
	free(weight);

	return retval;
}
