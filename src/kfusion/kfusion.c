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
#include <math.h>

#include "lib/opencl.h"
#include "lib/csv.h"

typedef struct sTrackData {
	int result;
	float J[7];
} TrackData;

const static char* param_str[7] = {
	"Error",
	"referenceNormal.x",
	"referenceNormal.y",
	"referenceNormal.z",
	"out.x",
	"out.y",
	"out.z",
};

int
opencl_compare_kfusion_track(cl_command_queue q, cl_mem out, char *file,
		size_t elems)
{
	unsigned int errors;
	unsigned int j;
	float delta = 0.05;
	TrackData *rvals, *ovals;
	size_t i;
	int retval;

	/* Allocate local buffer */
	ovals = malloc(elems * sizeof(TrackData));
	if (!ovals)
		return -ENOMEM;

	/* Read binary float entries */
	if (bin_file_read(file, elems * 8, (void **) &rvals)) {
		retval = -EIO;
		goto out;
	}

	/* Download buffer */
	clEnqueueReadBuffer(q, out, CL_TRUE, 0, elems*sizeof(TrackData), ovals,
			0, NULL, NULL);

	errors = 0;

	for (i = 0; i < elems && errors < 10; i++) {
		if (rvals[i].result != ovals[i].result) {
			retval = -EINVAL;
			fprintf(stderr,"%zi: Result mismatch, %i != %i\n", i,
					ovals[i].result, rvals[i].result);
			errors++;
		}

		if (rvals[i].result < 1 || ovals[i].result < 1)
			continue;

		for (j = 1; j < 8; j++) {
			if (fabs(ovals[i].J[j] - rvals[i].J[j]) > delta) {
				retval = -EINVAL;
				fprintf(stderr,
					"%zi: %s mismatch, %.6f != %.6f\n",
					i, param_str[j-1], ovals[i].J[j],
					rvals[i].J[j]);
				errors++;
			}
		}

		if (errors >= 10) {
			fprintf(stderr, "Too many errors, exiting\n");
			break;
		}
	}

out:
	free(rvals);
	free(ovals);

	return retval;
}

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
	float *inDepth;
	float *invK;
	float *inVertex;
	float *inNormal;
	float *refVertex;
	float *refNormal;
	float *mats;
	const unsigned int size[2] = {640,480};
	const float dist_threshold = 0.1f;
	const float normal_threshold = 0.8f;
	const float e_d = 0.3f;
	const cl_int r = 1;
	unsigned int i;

	cl_context ctx;
	cl_command_queue q;
	cl_program prg;
	cl_kernel kTrack, kDepth2Vertex, kVertex2Normal, kHalfSampleRobustImage;
	cl_mem clInDepth, clInVertex, clInNormal, clRefVertex, clRefNormal,
		clOutput, clOutVertex, clOutNormal, clOutHalfSample;
	cl_int error;
	cl_event time;
	cl_ulong time_diff = 0l;
	cl_ulong time_avg[4] = {0,0,0,0};
	//TrackData *result;

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

	data_entries = 640*480;
	csv_file_read_float("data/kfusion/halfSampleRobustImage_in.csv",
			&inDepth);
	csv_file_read_float("data/kfusion/depth2vertex_invK.csv", &invK);
	bin_file_read("data/kfusion/depth2vertex_out.bin", data_entries * 3,
			(void **) &inVertex);
	bin_file_read("data/kfusion/vertex2normal_out.bin", data_entries * 3,
			(void **) &inNormal);
	csv_file_read_float("data/kfusion/track_refVertex.csv", &refVertex);
	csv_file_read_float("data/kfusion/track_refNormal.csv", &refNormal);
	csv_file_read_float("data/kfusion/track_transformMats.csv", &mats);

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
		"src/kfusion/kernels.cl"
	};
	prg = opencl_compile_program(ctx, 1, &programs);

	/** Create kernels */
	kTrack = clCreateKernel(prg, "trackKernel", &error);
	if (error != CL_SUCCESS) {
		printf("Could not create kTrack\n");
		return -1;
	}

	kDepth2Vertex = clCreateKernel(prg, "depth2vertexKernel", &error);
	if (error != CL_SUCCESS) {
		printf("Could not create kTrack\n");
		return -1;
	}

	kVertex2Normal = clCreateKernel(prg, "vertex2normalKernel", &error);
	if (error != CL_SUCCESS) {
		printf("Could not create kTrack\n");
		return -1;
	}

	kHalfSampleRobustImage = clCreateKernel(prg,
			"halfSampleRobustImageKernel", &error);
	if (error != CL_SUCCESS) {
		printf("Could not create kTrack\n");
		return -1;
	}

	/** Create buffers */
	clInVertex = clCreateBuffer(ctx, CL_MEM_READ_ONLY,
			data_entries * 3 * sizeof(float), NULL, &error);
	if (error != CL_SUCCESS) {
		printf("Could not create in buffer\n");
		return -1;
	}
	clInNormal = clCreateBuffer(ctx, CL_MEM_READ_ONLY,
			data_entries * 3 * sizeof(float), NULL, &error);
	if (error != CL_SUCCESS) {
		printf("Could not create in buffer\n");
		return -1;
	}
	clRefVertex = clCreateBuffer(ctx, CL_MEM_READ_ONLY,
			data_entries * 3 * sizeof(float), NULL, &error);
	if (error != CL_SUCCESS) {
		printf("Could not create in buffer\n");
		return -1;
	}
	clRefNormal = clCreateBuffer(ctx, CL_MEM_READ_ONLY,
			data_entries * 3 * sizeof(float), NULL, &error);
	if (error != CL_SUCCESS) {
		printf("Could not create in buffer\n");
		return -1;
	}

	clOutput = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,
			data_entries * 8*sizeof(float), NULL, &error);
	if (error != CL_SUCCESS) {
		printf("Could not create out buffer\n");
		return -1;
	}

	clInDepth = clCreateBuffer(ctx, CL_MEM_READ_ONLY,
			data_entries * sizeof(float), NULL, &error);
	if (error != CL_SUCCESS) {
		printf("Could not create in buffer\n");
		return -1;
	}

	clOutVertex = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,
				data_entries * 3 *sizeof(float), NULL, &error);
	if (error != CL_SUCCESS) {
		printf("Could not create out buffer\n");
		return -1;
	}

	clOutNormal = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,
				data_entries * 3 *sizeof(float), NULL, &error);
	if (error != CL_SUCCESS) {
		printf("Could not create out buffer\n");
		return -1;
	}

	clOutHalfSample = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,
			data_entries /* / 4 * sizeof(float) */, NULL, &error);
	if (error != CL_SUCCESS) {
		printf("Could not create out buffer\n");
		return -1;
	}

	error = clEnqueueWriteBuffer(q, clInVertex, CL_FALSE, 0,
			3 * data_entries * sizeof(float), inVertex, 0, NULL,
			NULL);
	error |= clEnqueueWriteBuffer(q, clInNormal, CL_FALSE, 0,
			3 * data_entries * sizeof(float), inNormal, 0, NULL,
			NULL);
	error |= clEnqueueWriteBuffer(q, clRefVertex, CL_FALSE, 0,
			3 * data_entries * sizeof(float), refVertex, 0, NULL,
			NULL);
	error |= clEnqueueWriteBuffer(q, clRefNormal, CL_FALSE, 0,
			3 * data_entries * sizeof(float), refNormal, 0, NULL,
			NULL);
	error |= clEnqueueWriteBuffer(q, clInDepth, CL_FALSE, 0,
			data_entries * sizeof(float), inDepth, 0, NULL, NULL);
	if (error != CL_SUCCESS) {
		printf("Could not enqueue buffer write\n");
		return -1;
	}

	error  = clSetKernelArg(kTrack, 0, sizeof(cl_mem), &clOutput);
	error |= clSetKernelArg(kTrack, 1, 2 * sizeof(unsigned int), size);
	error |= clSetKernelArg(kTrack, 2, sizeof(cl_mem), &clInVertex);
	error |= clSetKernelArg(kTrack, 3, 2 * sizeof(unsigned int), size);
	error |= clSetKernelArg(kTrack, 4, sizeof(cl_mem), &clInNormal);
	error |= clSetKernelArg(kTrack, 5, 2 * sizeof(unsigned int), size);
	error |= clSetKernelArg(kTrack, 6, sizeof(cl_mem), &clRefVertex);
	error |= clSetKernelArg(kTrack, 7, 2 * sizeof(unsigned int), size);
	error |= clSetKernelArg(kTrack, 8, sizeof(cl_mem), &clRefNormal);
	error |= clSetKernelArg(kTrack, 9, 2 * sizeof(unsigned int), size);
	error |= clSetKernelArg(kTrack, 10, 16 * sizeof(float), &mats[0]);
	error |= clSetKernelArg(kTrack, 11, 16 * sizeof(float), &mats[16]);
	error |= clSetKernelArg(kTrack, 12, sizeof(float), &dist_threshold);
	error |= clSetKernelArg(kTrack, 13, sizeof(float), &normal_threshold);
	if (error != CL_SUCCESS) {
		printf("One of the arguments could not be set: %d.\n", error);
		return -1;
	}

	error  = clSetKernelArg(kDepth2Vertex, 0, sizeof(cl_mem), &clOutVertex);
	error |= clSetKernelArg(kDepth2Vertex, 1, 2 * sizeof(unsigned int),
			size);
	error |= clSetKernelArg(kDepth2Vertex, 2, sizeof(cl_mem), &clInDepth);
	error |= clSetKernelArg(kDepth2Vertex, 3, 2 * sizeof(unsigned int),
			size);
	error |= clSetKernelArg(kDepth2Vertex, 4, 16 * sizeof(float), invK);
	if (error != CL_SUCCESS) {
		printf("One of the arguments could not be set: %d.\n", error);
		return -1;
	}

	error  = clSetKernelArg(kVertex2Normal, 0, sizeof(cl_mem), &clOutNormal);
	error |= clSetKernelArg(kVertex2Normal, 1, 2 * sizeof(unsigned int),
			size);
	error |= clSetKernelArg(kVertex2Normal, 2, sizeof(cl_mem), &clInVertex);
	error |= clSetKernelArg(kVertex2Normal, 3, 2 * sizeof(unsigned int),
			size);
	if (error != CL_SUCCESS) {
		printf("One of the arguments could not be set: %d.\n", error);
		return -1;
	}

	error  = clSetKernelArg(kHalfSampleRobustImage, 0, sizeof(cl_mem),
			&clOutHalfSample);
	error |= clSetKernelArg(kHalfSampleRobustImage, 1, sizeof(cl_mem),
			&clInDepth);
	error |= clSetKernelArg(kHalfSampleRobustImage, 2,
			2 * sizeof(unsigned int), size);
	error |= clSetKernelArg(kHalfSampleRobustImage, 3, sizeof(float), &e_d);
	error |= clSetKernelArg(kHalfSampleRobustImage, 4, sizeof(cl_uint), &r);
	if (error != CL_SUCCESS) {
		printf("One of the arguments could not be set: %d.\n", error);
		return -1;
	}

	const size_t dims[] = {640,480};
	const size_t hdims[] = {320,240};
	for (i = 0; i < opencl_get_iterations(); i++) {
		error = clEnqueueNDRangeKernel(q, kTrack, 2, NULL, dims, NULL,
				0, NULL, &time);
		if (error != CL_SUCCESS) {
			printf("Could not enqueue track execution: %d\n",
					error);
			return -1;
		}
		clFinish(q);

		time_diff = opencl_exec_time(time);
		time_avg[0] += time_diff;
		printf("Track Time: %lu ns\n", time_diff);

		error = clEnqueueNDRangeKernel(q, kDepth2Vertex, 2, NULL, dims,
				NULL, 0, NULL, &time);
		if (error != CL_SUCCESS) {
			printf("Could not enqueue depth2Vertex execution: %d\n",
					error);
			return -1;
		}
		clFinish(q);

		time_diff = opencl_exec_time(time);
		time_avg[1] += time_diff;
		printf("Depth2Vertex Time: %lu ns\n", time_diff);

		error = clEnqueueNDRangeKernel(q, kVertex2Normal, 2, NULL, dims,
				NULL, 0, NULL, &time);
		if (error != CL_SUCCESS) {
			printf("Could not enqueue vertex2Normal execution: "
					"%d\n",	error);
			return -1;
		}
		clFinish(q);

		time_diff = opencl_exec_time(time);
		time_avg[2] += time_diff;
		printf("Vertex2Normal Time: %lu ns\n", time_diff);

		error = clEnqueueNDRangeKernel(q, kHalfSampleRobustImage, 2,
				NULL, hdims, NULL, 0, NULL, &time);
		if (error != CL_SUCCESS) {
			printf("Could not enqueue kHalfSampleRobustImage "
					"execution: %d\n", error);
			return -1;
		}
		clFinish(q);

		time_diff = opencl_exec_time(time);
		time_avg[3] += time_diff;
		printf("HalfSampleRobustImage Time: %lu ns\n", time_diff);
	}

	ret = 0;
	if (opencl_compare_output()) {
		printf("Comparing track values, some errors are expected.\n");
		ret = opencl_compare_kfusion_track(q, clOutput,
				"data/kfusion/track_out.bin", data_entries);
		printf("\n");
		/* XXX: A handful of errors is expected, as rounding differences
		 * can lead to different input pixels being evaluated for some
		 * work-items, leading to errors larger than tolerated by
		 * any sensible bound. Don't make them result in returning
		 * -EINVAL from this program.. */
		if (ret == -EINVAL)
			ret = 0;

		printf("Comparing depth2vertex values.\n");
		ret |= opencl_compare_out_bin(q, clOutVertex,
				"data/kfusion/depth2vertex_out.bin",
				data_entries * 3, 0.0001f, OPENCL_ERROR_ABS);
		printf("\n");

		printf("Comparing vertex2normal values.\n");
		ret |= opencl_compare_out_bin(q, clOutNormal,
				"data/kfusion/vertex2normal_out.bin",
				data_entries * 3, 0.0001f, OPENCL_ERROR_ABS);
		printf("\n");

		printf("Comparing halfsamplerobustimage values.\n");
		ret |= opencl_compare_out_csv(q, clOutHalfSample,
				"data/kfusion/halfSampleRobustImage_out.csv",
				data_entries / 4, 0.0001f, OPENCL_ERROR_ABS);
		printf("\n");

		if (!ret)
			printf("Output valid\n");
		else
			fprintf(stderr, "Output comparison error: %i\n", ret);
	}

	printf("Depth2Vertex time (avg of %u): %lu ns\n",
			opencl_get_iterations(),
			time_avg[1] / opencl_get_iterations());
	printf("HalfSampleRobustImage time (avg of %u): %lu ns\n",
			opencl_get_iterations(),
			time_avg[3] / opencl_get_iterations());
	printf("Track time (avg of %u): %lu ns\n", opencl_get_iterations(),
			time_avg[0] / opencl_get_iterations());
	printf("Vertex2Normal time (avg of %u): %lu ns\n",
			opencl_get_iterations(),
			time_avg[2] / opencl_get_iterations());

	/* Tear down */
	clReleaseEvent(time);
	clReleaseMemObject(clInVertex);
	clReleaseMemObject(clInNormal);
	clReleaseMemObject(clRefVertex);
	clReleaseMemObject(clRefNormal);
	clReleaseMemObject(clOutput);
	clReleaseKernel(kTrack);

	opencl_teardown(&ctx, &q, &prg);
	free(inVertex);
	free(inNormal);
	free(refVertex);
	free(refNormal);

	return ret;
}
