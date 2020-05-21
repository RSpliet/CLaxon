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
#include "main.h"

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
	float *diN, *diS, *djE, *djW, *dI, *dIReduce, *dSums2;
	const int Nr = 502;
	const int Nc = 458;
	const long Ne = 502 * 458;
	const float d_q0sqr = 0.0494804345f;
	const float d_lambda = .5f;
	unsigned int i;
	int mul;
	long no;
	size_t blocks_x;
	int blocks_work_size;
	size_t rdims[1];

	cl_context ctx;
	cl_command_queue q;
	cl_program prg;
	cl_kernel kSRAD, kSRAD2, kSRADReduce;
	cl_mem cldiN, cldiS, cldjE, cldjW, clddN, clddS, clddE, clddW, cldc,
		cldI, cldIReduce, cldSums2;
	cl_int error;
	cl_event time;
	cl_ulong time_diff = 0l;
	cl_ulong time_avg[3] = {0l,0l,0l};

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

	data_entries = 502*458;
	bin_file_read("data/srad/d_I.bin", data_entries, (void **) &dI);
	bin_file_read("data/srad/d_iN.bin", data_entries, (void **) &diN);
	bin_file_read("data/srad/d_iS.bin", data_entries, (void **) &diS);
	bin_file_read("data/srad/d_jE.bin", data_entries, (void **) &djE);
	bin_file_read("data/srad/d_jW.bin", data_entries, (void **) &djW);
	bin_file_read("data/srad/d_I_out.bin", data_entries,
			(void **) &dIReduce);
	bin_file_read("data/srad/d_sums2.bin", data_entries, (void **) &dSums2);

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
		"src/srad/kernel_gpu_opencl.cl"
	};
	prg = opencl_compile_program(ctx, 1, &programs);

	/** Create kernels */
	kSRAD = clCreateKernel(prg, "srad_kernel", &error);
	if (error != CL_SUCCESS) {
		printf("Could not create srad_kernel\n");
		return -1;
	}

	kSRAD2 = clCreateKernel(prg, "srad2_kernel", &error);
	if (error != CL_SUCCESS) {
		printf("Could not create srad2_kernel\n");
		return -1;
	}

	kSRADReduce = clCreateKernel(prg, "reduce_kernel", &error);
	if (error != CL_SUCCESS) {
		printf("Could not create reduce_kernel\n");
		return -1;
	}

	/** Create buffers */
	cldiN = clCreateBuffer(ctx, CL_MEM_READ_ONLY,
			data_entries * sizeof(cl_int), NULL, &error);
	if (error != CL_SUCCESS) {
		printf("Could not create in buffer\n");
		return -1;
	}

	cldiS = clCreateBuffer(ctx, CL_MEM_READ_ONLY,
			data_entries * sizeof(cl_int), NULL, &error);
	if (error != CL_SUCCESS) {
		printf("Could not create in buffer\n");
		return -1;
	}

	cldjE = clCreateBuffer(ctx, CL_MEM_READ_ONLY,
			data_entries * sizeof(cl_int), NULL, &error);
	if (error != CL_SUCCESS) {
		printf("Could not create in buffer\n");
		return -1;
	}

	cldjW = clCreateBuffer(ctx, CL_MEM_READ_ONLY,
			data_entries * sizeof(cl_int), NULL, &error);
	if (error != CL_SUCCESS) {
		printf("Could not create in buffer\n");
		return -1;
	}

	clddN = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
			data_entries * sizeof(float), NULL, &error);
	if (error != CL_SUCCESS) {
		printf("Could not create in buffer\n");
		return -1;
	}

	clddS = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
			data_entries * sizeof(float), NULL, &error);
	if (error != CL_SUCCESS) {
		printf("Could not create in buffer\n");
		return -1;
	}

	clddE = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
			data_entries * sizeof(float), NULL, &error);
	if (error != CL_SUCCESS) {
		printf("Could not create in buffer\n");
		return -1;
	}

	clddW = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
			data_entries * sizeof(float), NULL, &error);
	if (error != CL_SUCCESS) {
		printf("Could not create in buffer\n");
		return -1;
	}

	cldc = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
			data_entries * sizeof(float), NULL, &error);
	if (error != CL_SUCCESS) {
		printf("Could not create in buffer\n");
		return -1;
	}

	cldI = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
			data_entries * sizeof(float), NULL, &error);
	if (error != CL_SUCCESS) {
		printf("Could not create in buffer\n");
		return -1;
	}

	cldIReduce = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
			data_entries * sizeof(float), NULL, &error);
	if (error != CL_SUCCESS) {
		printf("Could not create in buffer\n");
		return -1;
	}
	cldSums2 = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
			data_entries * sizeof(float), NULL, &error);
	if (error != CL_SUCCESS) {
		printf("Could not create in buffer\n");
		return -1;
	}

	error = clEnqueueWriteBuffer(q, cldiN, CL_FALSE, 0,
			data_entries * sizeof(float), diN, 0, NULL, NULL);
	error |= clEnqueueWriteBuffer(q, cldiS, CL_FALSE, 0,
			data_entries * sizeof(float), diS, 0, NULL, NULL);
	error |= clEnqueueWriteBuffer(q, cldjE, CL_FALSE, 0,
			data_entries * sizeof(float), djE, 0, NULL, NULL);
	error |= clEnqueueWriteBuffer(q, cldjW, CL_FALSE, 0,
			data_entries * sizeof(float), djW, 0, NULL, NULL);
	error |= clEnqueueWriteBuffer(q, cldI, CL_FALSE, 0,
			data_entries * sizeof(float), dI, 0, NULL, NULL);
	if (error != CL_SUCCESS) {
		printf("Could not enqueue one-off buffer write.\n");
		return -1;
	}

	error  = clSetKernelArg(kSRAD, 0, sizeof(float), &d_lambda);
	error |= clSetKernelArg(kSRAD, 1, sizeof(cl_int), &Nr);
	error |= clSetKernelArg(kSRAD, 2, sizeof(cl_int), &Nc);
	error |= clSetKernelArg(kSRAD, 3, sizeof(cl_long), &Ne);
	error |= clSetKernelArg(kSRAD, 4, sizeof(cl_mem), &cldiN);
	error |= clSetKernelArg(kSRAD, 5, sizeof(cl_mem), &cldiS);
	error |= clSetKernelArg(kSRAD, 6, sizeof(cl_mem), &cldjE);
	error |= clSetKernelArg(kSRAD, 7, sizeof(cl_mem), &cldjW);
	error |= clSetKernelArg(kSRAD, 8, sizeof(cl_mem), &clddN);
	error |= clSetKernelArg(kSRAD, 9, sizeof(cl_mem), &clddS);
	error |= clSetKernelArg(kSRAD, 10, sizeof(cl_mem), &clddE);
	error |= clSetKernelArg(kSRAD, 11, sizeof(cl_mem), &clddW);
	error |= clSetKernelArg(kSRAD, 12, sizeof(float), &d_q0sqr);
	error |= clSetKernelArg(kSRAD, 13, sizeof(cl_mem), &cldc);
	error |= clSetKernelArg(kSRAD, 14, sizeof(cl_mem), &cldI);
	if (error != CL_SUCCESS) {
		printf("One of the arguments could not be set: %d.\n", error);
		return -1;
	}

	error  = clSetKernelArg(kSRAD2, 0, sizeof(float), &d_lambda);
	error |= clSetKernelArg(kSRAD2, 1, sizeof(cl_int), &Nr);
	error |= clSetKernelArg(kSRAD2, 2, sizeof(cl_int), &Nc);
	error |= clSetKernelArg(kSRAD2, 3, sizeof(cl_long), &Ne);
	error |= clSetKernelArg(kSRAD2, 4, sizeof(cl_mem), &cldiN);
	error |= clSetKernelArg(kSRAD2, 5, sizeof(cl_mem), &cldiS);
	error |= clSetKernelArg(kSRAD2, 6, sizeof(cl_mem), &cldjE);
	error |= clSetKernelArg(kSRAD2, 7, sizeof(cl_mem), &cldjW);
	error |= clSetKernelArg(kSRAD2, 8, sizeof(cl_mem), &clddN);
	error |= clSetKernelArg(kSRAD2, 9, sizeof(cl_mem), &clddS);
	error |= clSetKernelArg(kSRAD2, 10, sizeof(cl_mem), &clddE);
	error |= clSetKernelArg(kSRAD2, 11, sizeof(cl_mem), &clddW);
	error |= clSetKernelArg(kSRAD2, 12, sizeof(cl_mem), &cldc);
	error |= clSetKernelArg(kSRAD2, 13, sizeof(cl_mem), &cldI);
	if (error != CL_SUCCESS) {
		printf("One of the arguments could not be set: %d.\n", error);
		return -1;
	}

	error  = clSetKernelArg(kSRADReduce, 0, sizeof(cl_long), &Ne);
	error |= clSetKernelArg(kSRADReduce, 3, sizeof(cl_mem), &cldIReduce);
	error |= clSetKernelArg(kSRADReduce, 4, sizeof(cl_mem), &cldSums2);
	if (error != CL_SUCCESS) {
		printf("One of the arguments could not be set: %d.\n", error);
		return -1;
	}

	const size_t dims[] = {230144};
	const size_t ldims[1] = {NUMBER_THREADS};


	for (i = 0; i < opencl_get_iterations(); i++) {
		mul = 1;
		no = Ne;
		blocks_work_size = Ne/(int)ldims[0];
		if (Ne % (int)ldims[0] != 0){	// compensate for division
						// remainder above by adding one
						// grid
			blocks_work_size = blocks_work_size + 1;
		};
		rdims[0] = blocks_work_size * (int)ldims[0];
		time_diff = 0l;

		error  = clEnqueueWriteBuffer(q, cldIReduce, CL_FALSE, 0,
				data_entries * sizeof(float), dIReduce, 0, NULL,
				NULL);
		error  = clEnqueueWriteBuffer(q, cldIReduce, CL_FALSE, 0,
				data_entries * sizeof(float), dIReduce, 0, NULL,
				NULL);
		error |= clEnqueueWriteBuffer(q, cldI, CL_TRUE, 0,
				data_entries * sizeof(float), dI, 0, NULL,
				NULL);
		if (error != CL_SUCCESS) {
			printf("Could not enqueue buffer write\n");
			return -1;
		}

		while (blocks_work_size != 0) {
			// set arguments that were updated in this loop
			error  = clSetKernelArg(kSRADReduce, 1, sizeof(long),
					&no);
			error |= clSetKernelArg(kSRADReduce, 2, sizeof(int),
					&mul);
			error |= clSetKernelArg(kSRADReduce, 5, sizeof(int),
					&blocks_work_size);
			if (error != CL_SUCCESS) {
				printf("One of the arguments could not be set: "
						"%d.\n", error);
				return -1;
			}
			clFinish(q);

			// launch kernel
			error = clEnqueueNDRangeKernel(q, kSRADReduce, 1, NULL,
					rdims, ldims, 0, NULL, &time);
			if (error != CL_SUCCESS) {
				printf("Could not enqueue kSRADReduce "
						"execution: %d\n", error);
				return -1;
			}
			clFinish(q);

			time_diff += opencl_exec_time(time);

			// update execution parameters
			no = blocks_work_size;
			if (blocks_work_size == 1){
				blocks_work_size = 0;
			} else {
				mul = mul * NUMBER_THREADS;
				blocks_x = blocks_work_size/(int)ldims[0];
				if (blocks_work_size % (int)ldims[0] != 0) {
					blocks_x = blocks_x + 1;
				}
				blocks_work_size = blocks_x;
				rdims[0] = blocks_work_size * (int)ldims[0];
			}

		}

		time_avg[0] += time_diff;
		printf("Reduce Time: %lu ns\n", time_diff);

		error = clEnqueueNDRangeKernel(q, kSRAD, 1, NULL, dims, ldims,
				0, NULL, &time);
		if (error != CL_SUCCESS) {
			printf("Could not enqueue kSRAD execution: %d\n",
					error);
			return -1;
		}
		clFinish(q);

		time_diff = opencl_exec_time(time);
		time_avg[1] += time_diff;
		printf("kSRAD Time: %lu ns\n", time_diff);

		error = clEnqueueNDRangeKernel(q, kSRAD2, 1, NULL, dims, ldims,
				0, NULL, &time);
		if (error != CL_SUCCESS) {
			printf("Could not enqueue kSRAD2 execution: %d\n",
					error);
			return -1;
		}
		clFinish(q);

		time_diff = opencl_exec_time(time);
		time_avg[2] += time_diff;
		printf("kSRAD2 Time: %lu ns\n", time_diff);
	}

	if (opencl_compare_output()) {
		ret = opencl_compare_out_bin(q, cldc, "data/srad/d_c.bin",
				data_entries, 0.001f, OPENCL_ERROR_ABS);
		ret |= opencl_compare_out_bin(q, clddE, "data/srad/d_dE.bin",
				data_entries, 0.001f, OPENCL_ERROR_ABS);
		ret |= opencl_compare_out_bin(q, clddW, "data/srad/d_dW.bin",
				data_entries, 0.001f, OPENCL_ERROR_ABS);
		ret |= opencl_compare_out_bin(q, clddN, "data/srad/d_dN.bin",
				data_entries, 0.001f, OPENCL_ERROR_ABS);
		ret |= opencl_compare_out_bin(q, clddS, "data/srad/d_dS.bin",
				data_entries, 0.001f, OPENCL_ERROR_ABS);

		if (ret)
			fprintf(stderr, "SRAD output comparison error: %i\n",
					ret);
	}

	if (opencl_compare_output() && !ret) {
		ret = opencl_compare_out_bin(q, cldI, "data/srad/d_I_out.bin",
				data_entries, 0.001f, OPENCL_ERROR_ABS);

		if (ret)
			fprintf(stderr, "SRAD2 output comparison error: %i\n",
					ret);
	}

	if (opencl_compare_output() && !ret) {
		ret = opencl_compare_out_bin(q, cldIReduce,
				"data/srad/d_sums_res.bin", 1, 0.003f,
				OPENCL_ERROR_FRAC);
		ret |= opencl_compare_out_bin(q, cldSums2,
				"data/srad/d_sums2_res.bin", 1, 0.003f,
				OPENCL_ERROR_FRAC);

		if (!ret)
			printf("Output valid\n");
		else
			fprintf(stderr, "Reduce output comparison error: %i\n",
					ret);
	}

	printf("SRAD2 time (avg of %u): %lu ns\n", opencl_get_iterations(),
			time_avg[2] / opencl_get_iterations());
	printf("Reduce time (avg of %u): %lu ns\n", opencl_get_iterations(),
			time_avg[0] / opencl_get_iterations());
	printf("SRAD time (avg of %u): %lu ns\n", opencl_get_iterations(),
			time_avg[1] / opencl_get_iterations());

	/* Tear down */
	clReleaseEvent(time);
	clReleaseMemObject(cldI);
	clReleaseMemObject(cldc);
	clReleaseMemObject(clddE);
	clReleaseMemObject(clddN);
	clReleaseMemObject(clddS);
	clReleaseMemObject(clddW);
	clReleaseMemObject(cldjE);
	clReleaseMemObject(cldiN);
	clReleaseMemObject(cldiS);
	clReleaseMemObject(cldjW);

	clReleaseKernel(kSRAD);
	clReleaseKernel(kSRAD2);
	clReleaseKernel(kSRADReduce);

	opencl_teardown(&ctx, &q, &prg);
	free(dI);
	free(diN);
	free(diS);
	free(djE);
	free(djW);

	return ret;
}
