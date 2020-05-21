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

#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <unistd.h>
#include <inttypes.h>

#include "lib/opencl.h"

#if __WORDSIZE == 64
# define clz(x) __builtin_clzl(x)
#else
# define clz(x) __builtin_clz(x)
#endif

/* Next power-of-two */
size_t next_pot(size_t in)
{
	int msb = 31 - clz(in - 1);

	return 1 << (msb + 1);
}

size_t prefix_sum_elems_ceil(cl_context ctx, size_t elems, size_t *wgs)
{
	size_t wg_size, work_groups, elems_ceil;

	wg_size = opencl_max_workgroup_size();
	work_groups = ((elems / 2) + wg_size - 1) / wg_size;

	if (work_groups > 1) {
		elems_ceil = 2 * work_groups * wg_size;
	} else {
		elems_ceil = next_pot(elems);
	}

	if (wgs)
		*wgs = work_groups;

	return elems_ceil;

}

int do_prefix_sum(cl_context ctx, cl_command_queue q, cl_kernel krnl,
		size_t work_group_size, size_t work_items, cl_mem in,
		cl_mem out, cl_mem incr, cl_ulong *time)
{
	cl_int error;
	cl_event e_time;
	cl_event *event_time = NULL;
	cl_ulong t;

	if (time)
		event_time = &e_time;

	error =  clSetKernelArg(krnl, 0, sizeof(cl_mem), &in);
	error |= clSetKernelArg(krnl, 1,
			work_group_size * 2 * sizeof(unsigned int), NULL);
	error |= clSetKernelArg(krnl, 2, sizeof(cl_mem), &out);
	error |= clSetKernelArg(krnl, 3, sizeof(cl_mem), &incr);
	if (error != CL_SUCCESS) {
		printf("One of the arguments could not be set: %d.\n", error);
		return -1;
	}

	const size_t dims_g[] = {work_items};
	const size_t dims_l[] = {work_group_size};
	error = clEnqueueNDRangeKernel(q, krnl, 1, NULL, dims_g, dims_l, 0,
			NULL, event_time);
	if (error != CL_SUCCESS) {
		printf("Could not enqueue prefix-sum kernel execution: %d\n",
				error);
		return -1;
	}

	if (time) {
		clFinish(q);
		t = opencl_exec_time(e_time);
		*time += t;
		printf("  Time do_prefix_sum: %lins\n", t);
		clReleaseEvent(e_time);
	}

	return 0;
}

int do_prefix_sum_post(cl_context ctx, cl_command_queue q, cl_kernel krnl,
		size_t work_group_size, size_t work_items, cl_mem data,
		cl_mem incr, cl_ulong *time)
{
	cl_int error;
	cl_event e_time;
	cl_event *event_time = NULL;
	cl_ulong t;

	if (time)
		event_time = &e_time;

	error =  clSetKernelArg(krnl, 0, sizeof(cl_mem), &data);
	error |= clSetKernelArg(krnl, 1, sizeof(cl_mem), &incr);
	if (error != CL_SUCCESS) {
		printf("One of the arguments could not be set: %d.\n", error);
		return -1;
	}

	const size_t dims_g[] = {work_items - (2 * work_group_size)};
	const size_t dims_l[] = {work_group_size};
	error = clEnqueueNDRangeKernel(q, krnl, 1, NULL, dims_g, dims_l, 0,
			NULL, event_time);
	if (error != CL_SUCCESS) {
		printf("Could not enqueue kernel execution: %d\n", error);
		return -1;
	}

	if (time) {
		clFinish(q);
		t = opencl_exec_time(e_time);
		*time += t;
		printf("  Time do_prefix_sum_post: %lins\n", t);
		clReleaseEvent(e_time);
	}

	return 0;
}

/*
 * Prefix sum (scan) that can do up to two "levels" of hierarchical scan,
 * thus limited to max_workgroup_size^2 elements (~1M on GT650)
 */
cl_mem prefix_sum(cl_context ctx, cl_command_queue q, cl_mem in, size_t elems,
		cl_ulong *time)
{
	size_t wg_size, work_items, work_groups, incrs;
	cl_mem out = (cl_mem)-1;
	cl_mem incr = 0;
	cl_program prg;
	cl_int error;
	cl_kernel k_prefix_sum, k_prefix_sum_post = 0;

	wg_size = opencl_max_workgroup_size();

	const char *programs = {
		"src/frnn/prefix_sum.cl"
	};
	prg = opencl_compile_program(ctx, 1, &programs);

	k_prefix_sum = clCreateKernel(prg, "prefix_sum", &error);
	if (error != CL_SUCCESS) {
		printf("Could not create prefix sum kernel\n");
		return (cl_mem)-1;
	}

	/* Calculate work dimension */
	work_items = prefix_sum_elems_ceil(ctx, elems, &work_groups);
	if (work_groups > 1) {
		incrs = next_pot(work_groups);
		if (incrs > wg_size) {
			printf("Data size prefix sum exceeds limits\n");
			goto error;
		}

		incr = clCreateBuffer(ctx,
				CL_MEM_READ_WRITE,
				incrs * sizeof(unsigned int), NULL, &error);
		if (error != CL_SUCCESS) {
			printf("Could not create prefix sum increment "
					"buffer\n");
			goto error;
		}

		k_prefix_sum_post = clCreateKernel(prg, "prefix_sum_post",
				&error);
		if (error != CL_SUCCESS) {
			printf("Could not create prefix sum post kernel\n");
			goto error;
		}
	} else {
		wg_size = work_items / 2;
	}

	/* Read-write for further processing */
	out = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
			work_items * sizeof(unsigned int), NULL, &error);
	if (error != CL_SUCCESS) {
		printf("Could not create prefix sum out buffer\n");
		out = (cl_mem)-1;
		goto error;
	}

	do_prefix_sum(ctx, q, k_prefix_sum, wg_size, work_items / 2, in, out,
			incr, time);

	if (work_groups > 1) {
		do_prefix_sum(ctx, q, k_prefix_sum, incrs / 2, incrs / 2, incr,
				incr, NULL, time);
		do_prefix_sum_post(ctx, q, k_prefix_sum_post, wg_size,
				work_items, out, incr, time);
	}

error:
	/* tear-down */
	clReleaseKernel(k_prefix_sum);
	if (k_prefix_sum_post)
		clReleaseKernel(k_prefix_sum_post);
	if (incr)
		clReleaseMemObject(incr);

	clReleaseProgram(prg);

	return out;
}
