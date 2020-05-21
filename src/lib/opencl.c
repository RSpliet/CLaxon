/* SPDX-License-Identifier: MIT
 *
 * Copyright (c) 2016 Roy Spliet, University of Cambridge.
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
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <errno.h>
#include <math.h>
#include <stdbool.h>

/* We're targeting Clover amongst other APIs */
#include "lib/opencl.h"
#include "lib/csv.h"

struct {
	int platform;
	int device;
	bool compare_output;
	unsigned int iterations;

	cl_platform_id cl_platform;
	cl_device_id cl_device;
} state = {.platform = 0, .device = 0, .compare_output = false,
		.iterations = 10, .cl_platform = NULL, .cl_device = NULL};

const char *opt_generic = "-I .";
const char *opt_nv_sm_20 = "-I . -D NV_SM_20";

bool
opencl_compare_output()
{
	return state.compare_output;
}

unsigned int
opencl_get_iterations()
{
	return state.iterations;
}

size_t
opencl_kernel_size(const char *filename)
{
	struct stat fs;

	stat(filename, &fs);
	return fs.st_size;
}

char *
opencl_kernel_read(const char *filename)
{
	size_t size;
	char *file;
	FILE *fp;

	fp = fopen(filename, "r");
	if(!fp) {
		printf("Error: Could not open CL source file %s", filename);
		return NULL;
	}

	size = opencl_kernel_size(filename);
	file = calloc(1, size+1);
	if(!file) {
		printf("Error: Could not allocate space for %s", filename);
		return NULL;
	}

	fread(file, size, 1, fp);
	fclose(fp);

	return file;
}

cl_uint
opencl_nv_sm_major()
{
	char *exts;
	size_t size;
	cl_int error;
	cl_uint sm_major = 0;

	error = clGetDeviceInfo(state.cl_device, CL_DEVICE_EXTENSIONS, 0, NULL,
			&size);
	if (error != CL_SUCCESS) {
		printf("Error: could not read device extensions.\n");
		return 0;
	}

	exts = malloc(size);
	if (!exts) {
		printf("Error: could not allocate buffer for extensions.\n");
		return 0;
	}

	error = clGetDeviceInfo(state.cl_device, CL_DEVICE_EXTENSIONS, size,
			exts, NULL);
	if (!exts) {
		printf("Error: could not read device extensions\n");
		goto error;
	}

	if (strstr(exts, "cl_nv_device_attribute_query") != NULL) {
		error = clGetDeviceInfo(state.cl_device,
				CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV,
				sizeof(cl_uint), &sm_major, NULL);
		if (!exts)
			printf("Error: could not read compute capability.\n");
	}

error:
	free(exts);

	return sm_major;

}

cl_context
opencl_create_context()
{
	cl_uint c_platforms, c_devs;
	cl_int error = 0;
	cl_platform_id *l_platforms;
	cl_device_id *l_devs;
	cl_context ctx;

	/* Find our platform */
	clGetPlatformIDs(0, NULL, &c_platforms);
	if (c_platforms == 0) {
		fprintf(stderr, "Error: no OpenCL platforms found.\n");
		return NULL;
	}

	if (c_platforms < state.platform + 1) {
		fprintf(stderr, "Error: no OpenCL platform with index %u \n",
				state.platform);
		return NULL;
	}

	l_platforms = malloc(c_platforms * sizeof(cl_platform_id));
	clGetPlatformIDs(c_platforms, l_platforms, NULL);
	state.cl_platform = l_platforms[state.platform];
	free(l_platforms);

	/* Now the device */
	clGetDeviceIDs(state.cl_platform, CL_DEVICE_TYPE_ALL, 0, NULL, &c_devs);
	if(c_devs == 0) {
		fprintf(stderr, "Error: no OpenCL devices found.\n");
		return NULL;
	}

	if (c_devs < state.device + 1) {
		fprintf(stderr, "Error: no OpenCL device with index %u \n",
				state.device);
		return NULL;
	}

	l_devs = malloc(c_devs * sizeof(cl_device_id));
	clGetDeviceIDs(state.cl_platform, CL_DEVICE_TYPE_ALL, c_devs, l_devs,
			NULL);
	state.cl_device = l_devs[state.device];
	free(l_devs);

	/* Get the context */
	cl_context_properties ctx_props[] = {
			CL_CONTEXT_PLATFORM,
			(cl_context_properties)state.cl_platform,

			0
	};
	ctx = clCreateContext(ctx_props, 1, &state.cl_device, NULL, NULL,
			&error);
	if (error) {
		fprintf(stderr, "Error: Could not create OpenCL context on"
				"device (%u, %u).\n", state.platform,
				state.device);
		return NULL;
	}

	return ctx;
}

cl_command_queue
opencl_create_cmdqueue(cl_context ctx)
{
	cl_command_queue q;
	cl_int error;

	if (!state.cl_platform || !state.cl_device) {
		fprintf(stderr, "Error: Cannot create command queue for "
				"invalid context");
		return NULL;
	}

	/* XXX: properties? */
	q = clCreateCommandQueue (ctx, state.cl_device,
			CL_QUEUE_PROFILING_ENABLE, &error);
	if (error != CL_SUCCESS) {
		fprintf(stderr, "Error: Could not create OpenCL command queue "
				"on device (%u, %u).\n", state.platform,
				state.device);
		return NULL;
	}

	return q;
}

cl_program
opencl_compile_program(cl_context ctx, cl_uint source_cnt,
		const char **source_files)
{
	cl_program prg;
	cl_int error;
	int i;
	const char **sources;
	const char *options;
	char *status;
	int bStatus = 0;
	size_t ret_val_size;
	cl_uint sm_major;

	sources = malloc(source_cnt * sizeof (char *));
	if(!sources) {
		fprintf(stderr, "Error: Cannot allocate memory for source "
						"files");
		return NULL;
	}

	for (i = 0; i < source_cnt; i++) {
		sources[i] = opencl_kernel_read(source_files[i]);
	}

	prg = clCreateProgramWithSource(ctx, source_cnt, sources, NULL, &error);
	if (error) {
		fprintf(stderr, "Error: Cannot create program");
		return NULL;
	}

	sm_major = opencl_nv_sm_major();
	if (sm_major >= 2)
		options = opt_nv_sm_20;
	else
		options = opt_generic;

	error = clBuildProgram (prg, 1, &state.cl_device,
			options, NULL, NULL);
	if (error != CL_SUCCESS) {
		fprintf(stderr, "Error: failed to build CL program\n");

		error = clGetProgramBuildInfo(prg, state.cl_device,
				CL_PROGRAM_BUILD_STATUS,
				sizeof(cl_build_status), &bStatus, NULL);
		if(error != CL_SUCCESS) {
			printf("Build error: Could not read back build status:"
					" %i\n", error);
			return NULL;
		}

		if(bStatus != CL_BUILD_SUCCESS) {
			printf("Build error: %i\n\n",bStatus);
			printf("Compiler output:\n");
			clGetProgramBuildInfo(prg, state.cl_device,
					CL_PROGRAM_BUILD_LOG, 0, NULL,
					&ret_val_size);
			status = malloc(ret_val_size+1);
			clGetProgramBuildInfo(prg, state.cl_device,
					CL_PROGRAM_BUILD_LOG, ret_val_size,
					status, NULL);

			printf("%s",status);

			free(status);
			return NULL;
		}
		return NULL;
	}

	for (i = 0; i < source_cnt; i++)
		free((char *)sources[i]);
	free(sources);

	return prg;
}

cl_ulong
opencl_exec_time(cl_event time)
{
	cl_ulong time_start = 0l, time_end = 0l;

	clGetEventProfilingInfo (time, CL_PROFILING_COMMAND_START,
				sizeof(cl_ulong), &time_start, NULL);
	clGetEventProfilingInfo (time, CL_PROFILING_COMMAND_END,
				sizeof(cl_ulong), &time_end, NULL);

	return time_end - time_start;
}

size_t
opencl_max_workgroup_size()
{
	size_t max_items;

	clGetDeviceInfo(state.cl_device, CL_DEVICE_MAX_WORK_GROUP_SIZE,
			sizeof(size_t), &max_items, NULL);

	return max_items;
}

void
opencl_teardown(cl_context *ctx, cl_command_queue *q, cl_program *prg)
{
	if (prg && *prg) {
		clReleaseProgram(*prg);
		*prg = NULL;
	}

	if (q && *q) {
		clReleaseCommandQueue(*q);
		*q = NULL;
	}

	if (ctx && *ctx) {
		clReleaseContext(*ctx);
		*ctx = NULL;
	}
}

void
opencl_download_float_csv(cl_command_queue q, cl_mem out, char *file,
		size_t elems)
{
	float *result;

	result = malloc(elems*sizeof(float));
	if (!result) {
		printf("ERROR: could not allocate out-buffer, not downloading "
				"results.\n");
		return;
	}

	clEnqueueReadBuffer(q, out, CL_TRUE, 0, elems*sizeof(float), result, 0,
			NULL, NULL);
	csv_file_write(file, elems, result);

	free(result);
	result = NULL;
}

static int
opencl_compare_out_float(float *rvals, float *ovals, size_t elems, float delta,
		clErrorMarginType dType)
{
	float diff;
	size_t i;
	int retval;
	int errors;

	retval = 0;
	errors = 0;

	for (i = 0; i < elems; i++) {
		switch (dType) {
		case OPENCL_ERROR_FRAC:
			diff = fabs((ovals[i] / rvals[i]) - 1.f);
			break;
		default:
			diff = fabs(rvals[i] - ovals[i]);
		}

		if (diff > delta) {
			retval = -EINVAL;
			fprintf(stderr,"%06zx: MISMATCH %f != %f\n", i*4, ovals[i],
								rvals[i]);
			errors++;
		}

		if (errors >= 10) {
			fprintf(stderr,"Too many errors, quitting.\n");
			break;
		}
	}

	return retval;
}

int
opencl_compare_out_csv(cl_command_queue q, cl_mem out, char *file,
		size_t elems, float delta, clErrorMarginType dType)
{
	float *ovals;
	float *rvals;
	size_t relems;
	int retval;

	/* Allocate local buffer */
	ovals = malloc(elems * sizeof(float));
	if (!ovals)
		return -ENOMEM;

	/* Read CSV entries */
	relems = csv_file_read_float(file, &rvals);
	if (relems < elems) {
		retval = -EIO;
		goto out;
	}

	/* Download buffer */
	clEnqueueReadBuffer(q, out, CL_TRUE, 0, elems*sizeof(float), ovals, 0,
				NULL, NULL);

	/* Go compare */
	retval = opencl_compare_out_float(rvals, ovals, elems, delta, dType);

out:
	free(rvals);
	free(ovals);
	return retval;
}

int
opencl_compare_out_bin(cl_command_queue q, cl_mem out, char *file,
		size_t elems, float delta, clErrorMarginType dType)
{
	float *ovals;
	float *rvals;
	int retval;

	/* Allocate local buffer */
	ovals = malloc(elems * sizeof(float));
	if (!ovals)
		return -ENOMEM;

	/* Read binary float entries */
	if (bin_file_read(file, elems, (void **) &rvals)) {
		retval = -EIO;
		goto out;
	}

	/* Download buffer */
	clEnqueueReadBuffer(q, out, CL_TRUE, 0, elems*sizeof(float), ovals, 0,
				NULL, NULL);

	/* Go compare */
	retval = opencl_compare_out_float(rvals, ovals, elems, delta, dType);

out:
	free(rvals);
	free(ovals);
	return retval;
}

int
opencl_parse_option(int c, char *optarg)
{
	unsigned int optval;
	int ret;

	switch (c)
	{
	case 'P':
		ret = sscanf(optarg, "%u", &optval);
		if (ret != 1)
			return -EINVAL;

		state.platform = optval;
		return 0;
		break;
	case 'd':
		ret = sscanf(optarg, "%u", &optval);
		if (ret != 1)
			return -EINVAL;
		state.device = optval;
		return 0;
		break;
	case 'I':
		ret = sscanf(optarg, "%u", &optval);
		if (ret != 1)
			return -EINVAL;
		state.iterations = optval;
		return 0;
		break;
	case 'c':
		state.compare_output = true;
		return 0;
		break;
	default:
		break;
	}

	return -ENOSYS;
}

void
opencl_usage()
{
	printf("\t-P <platform id> OpenCL platform (default: 0)\n");
	printf("\t-d <device id>   OpenCL device (default: 0)\n");
	printf("\t-I <iterations>  Number of iterations (default: 10)\n");
	printf("\t-c               Compare output(s) (default: off)\n");
}
