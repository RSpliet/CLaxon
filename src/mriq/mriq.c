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
#include "macros.h"

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
	int64_t phi_entries, data_entries;
	float *inPhiR;
	float *inPhiI;
	float *inX;
	float *inY;
	float *inZ;
	struct kValues *inKValues;
	unsigned int QGrid;
	int QGridBase;
	const float zero = 0.f;

	cl_context ctx;
	cl_command_queue q;
	cl_program prg;
	cl_kernel computePhiMag, computeQ;
	cl_mem clInPhiR, clInPhiI, clInX, clInY, clInZ, clInKValues;
	cl_mem clOutPhiMag, clOutQr, clOutQi;
	cl_int error;
	cl_event time;
	cl_ulong time_diff = 0l;
	cl_ulong time_avg[2] = {0l,0l};
	unsigned int i;
	const cl_int numK = 2048;

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

	phi_entries = 2048;
	data_entries = 262144;
	bin_file_read("data/mriq/phiR.bin", phi_entries, (void **) &inPhiR);
	bin_file_read("data/mriq/phiI.bin", phi_entries, (void **) &inPhiI);
	csv_file_read_float("data/mriq/x.csv", &inX);
	csv_file_read_float("data/mriq/y.csv", &inY);
	csv_file_read_float("data/mriq/z.csv", &inZ);
	csv_file_read_float("data/mriq/kvalues.csv", (float **) &inKValues);

	printf("Read %"PRIi64" entries\n", phi_entries);

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
		"src/mriq/kernels.cl"
	};
	prg = opencl_compile_program(ctx, 1, &programs);

	computePhiMag = clCreateKernel(prg, "ComputePhiMag_GPU", &error);
	if (error != CL_SUCCESS) {
		printf("Could not create kernel ComputePhiMag_GPU\n");
		return -1;
	}

	computeQ = clCreateKernel(prg, "ComputeQ_GPU", &error);
	if (error != CL_SUCCESS) {
		printf("Could not create kernel ComputeQ_GPU\n");
		return -1;
	}

	clInPhiR = clCreateBuffer(ctx, CL_MEM_READ_ONLY,
			phi_entries * sizeof(float), NULL, &error);
	if (error != CL_SUCCESS) {
		printf("Could not create in buffer\n");
		return -1;
	}
	clInPhiI = clCreateBuffer(ctx, CL_MEM_READ_ONLY,
			phi_entries * sizeof(float), NULL, &error);
	if (error != CL_SUCCESS) {
		printf("Could not create in buffer\n");
		return -1;
	}
	clInX = clCreateBuffer(ctx, CL_MEM_READ_ONLY,
			data_entries * sizeof(float), NULL, &error);
	if (error != CL_SUCCESS) {
		printf("Could not create in buffer\n");
		return -1;
	}
	clInY = clCreateBuffer(ctx, CL_MEM_READ_ONLY,
			data_entries * sizeof(float), NULL, &error);
	if (error != CL_SUCCESS) {
		printf("Could not create in buffer\n");
		return -1;
	}

	clInZ = clCreateBuffer(ctx, CL_MEM_READ_ONLY,
			data_entries * sizeof(float), NULL, &error);
	if (error != CL_SUCCESS) {
		printf("Could not create in buffer\n");
		return -1;
	}

	clInKValues = clCreateBuffer(ctx, CL_MEM_READ_ONLY,
			KERNEL_Q_K_ELEMS_PER_GRID * sizeof(struct kValues),
			NULL, &error);
	if (error != CL_SUCCESS) {
		printf("Could not create in buffer\n");
		return -1;
	}

	clOutPhiMag = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,
			phi_entries * sizeof(float), NULL, &error);
	if (error != CL_SUCCESS) {
		printf("Could not create out buffer\n");
		return -1;
	}

	clOutQr = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
			data_entries * sizeof(float), NULL, &error);
	if (error != CL_SUCCESS) {
		printf("Could not create out buffer\n");
		return -1;
	}

	clOutQi = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
			data_entries * sizeof(float), NULL, &error);
	if (error != CL_SUCCESS) {
		printf("Could not create out buffer\n");
		return -1;
	}

	error = clEnqueueWriteBuffer(q, clInPhiR, CL_FALSE, 0,
			phi_entries * sizeof(float), inPhiR, 0, NULL, NULL);
	error |= clEnqueueWriteBuffer(q, clInPhiI, CL_FALSE, 0,
			phi_entries * sizeof(float), inPhiI, 0, NULL, NULL);
	error |= clEnqueueWriteBuffer(q, clInX, CL_FALSE, 0,
			data_entries * sizeof(float), inX, 0, NULL, NULL);
	error |= clEnqueueWriteBuffer(q, clInY, CL_FALSE, 0,
			data_entries * sizeof(float), inY, 0, NULL, NULL);
	error |= clEnqueueWriteBuffer(q, clInZ, CL_FALSE, 0,
			data_entries * sizeof(float), inZ, 0, NULL, NULL);
	if (error != CL_SUCCESS) {
		printf("Could not enqueue buffer write\n");
		return -1;
	}

	error  = clSetKernelArg(computePhiMag, 0, sizeof(cl_mem), &clInPhiR);
	error |= clSetKernelArg(computePhiMag, 1, sizeof(cl_mem), &clInPhiI);
	error |= clSetKernelArg(computePhiMag, 2, sizeof(cl_mem), &clOutPhiMag);
	error |= clSetKernelArg(computePhiMag, 3, sizeof(cl_int), &numK);

	if (error != CL_SUCCESS) {
		printf("One of the arguments could not be set: %d.\n", error);
		return -1;
	}

	error  = clSetKernelArg(computeQ, 0, sizeof(cl_int), &numK);
	error |= clSetKernelArg(computeQ, 2, sizeof(cl_mem), &clInX);
	error |= clSetKernelArg(computeQ, 3, sizeof(cl_mem), &clInY);
	error |= clSetKernelArg(computeQ, 4, sizeof(cl_mem), &clInZ);
	error |= clSetKernelArg(computeQ, 5, sizeof(cl_mem), &clOutQr);
	error |= clSetKernelArg(computeQ, 6, sizeof(cl_mem), &clOutQi);
	error |= clSetKernelArg(computeQ, 7, sizeof(cl_mem), &clInKValues);

	if (error != CL_SUCCESS) {
		printf("One of the arguments could not be set: %d.\n", error);
		return -1;
	}

	const size_t dims[] = {2048};
	const size_t Qdims[] = {data_entries};
	const size_t ldims[] = {KERNEL_PHI_MAG_THREADS_PER_BLOCK};
	for (i = 0; i < opencl_get_iterations(); i++) {
		error = clEnqueueNDRangeKernel(q, computePhiMag, 1, NULL, dims,
				ldims, 0, NULL, &time);
		if (error != CL_SUCCESS) {
			printf("Could not enqueue kernel execution: %d\n",
					error);
			return -1;
		}
		clFinish(q);

		time_diff = opencl_exec_time(time);
		time_avg[0] += time_diff;
		printf("computePhiMag Time: %lu ns\n", time_diff);

		time_diff = 0l;
		clEnqueueFillBuffer(q, clOutQi, &zero, sizeof(float), 0,
				data_entries * sizeof(float), 0, NULL, NULL);
		clEnqueueFillBuffer(q, clOutQr, &zero, sizeof(float), 0,
					data_entries * sizeof(float), 0, NULL,
					NULL);
		for (QGrid = 0; QGrid < (numK / KERNEL_Q_K_ELEMS_PER_GRID);
				QGrid++) {
			/* Put the tile of K values into constant mem. Seems
			 * wasteful to me to launch the kernel multiple times,
			 * but it's not my benchmark...¯ */
			QGridBase = QGrid * KERNEL_Q_K_ELEMS_PER_GRID;

			error  = clSetKernelArg(computeQ, 1, sizeof(cl_int),
					&QGridBase);
			if (error != CL_SUCCESS) {
				printf("One of the arguments could not be set: "
						"%d.\n", error);
				return -1;
			}

			error = clEnqueueWriteBuffer(q, clInKValues, CL_TRUE, 0,
					KERNEL_Q_K_ELEMS_PER_GRID *
							sizeof(struct kValues),
					&inKValues[QGridBase], 0, NULL, NULL);
			if (error != CL_SUCCESS) {
				printf("Could not enqueue buffer write\n");
				return -1;
			}
			clFinish(q);

			error = clEnqueueNDRangeKernel(q, computeQ, 1, NULL,
					Qdims, ldims, 0, NULL, &time);
			if (error != CL_SUCCESS) {
				printf("Could not enqueue kernel execution: "
						"%d\n", error);
				return -1;
			}
			clFinish(q);

			time_diff += opencl_exec_time(time);
		}
		time_avg[1] += time_diff;
		printf("computeQ Time: %lu ns\n", time_diff);
	}

	/* Validate outputs. */
	if (opencl_compare_output()) {
		ret = opencl_compare_out_csv(q,clOutPhiMag,
				"data/mriq/phimag_out.csv", phi_entries, 0.001f,
				OPENCL_ERROR_ABS);
		ret |= opencl_compare_out_bin(q, clOutQi,
				"data/mriq/qI_out.bin", data_entries, 0.02f,
				OPENCL_ERROR_ABS);
		ret |= opencl_compare_out_bin(q, clOutQr,
				"data/mriq/qR_out.bin", data_entries, 0.03f,
				OPENCL_ERROR_ABS);

		if (!ret)
			printf("Output valid\n");
		else
			printf("Output invalid\n");
	}

	printf("computePhiMag Time (avg of %u): %lu ns\n",
			opencl_get_iterations(),
			time_avg[0] / opencl_get_iterations());
	printf("computeQ Time (avg of %u): %lu ns\n",
			opencl_get_iterations(),
			time_avg[1] / opencl_get_iterations());

	/* Tear down */
	clReleaseEvent(time);
	clReleaseMemObject(clInX);
	clReleaseMemObject(clInY);
	clReleaseMemObject(clInZ);
	clReleaseMemObject(clOutPhiMag);
	clReleaseKernel(computePhiMag);
	clReleaseKernel(computeQ);

	opencl_teardown(&ctx, &q, &prg);
	free(inPhiR);
	free(inPhiI);
	free(inX);
	free(inY);
	free(inZ);
	free(inKValues);

	return ret;
}
