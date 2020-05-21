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

#ifndef LIB_OPENCL_H
#define LIB_OPENCL_H

/* We're targeting Clover amongst other APIs */
#define CL_TARGET_OPENCL_VERSION 120
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/opencl.h>

#include <stdbool.h>

/** Command line options parsed by opencl_parse_option. Concatenate to your
 * own optargs to make use of library-provided options. */
#define OPENCL_OPTS "P:d:I:c"

typedef enum {
	OPENCL_ERROR_ABS,
	OPENCL_ERROR_FRAC,
} clErrorMarginType;

/**
 * Return true iff the user requested for the output buffer(s) to be validated
 * @return true iff output buffers should be compared.
 */
bool opencl_compare_output(void);

/**
 * Return the number of times each kernel should be run.
 *
 * Either indicated by the user using the -I flag, or defaults to 10.
 * @return The number of times each kernel should be run.
 */
unsigned int opencl_get_iterations(void);

/**
 * Create an OpenCL context for the platform and device specified by the P and
 * d optargs.
 *
 * If need be, one can manually call opencl_parse_option with the 'P' and 'd'
 * characters to hard-code values.
 * @return OpenCL context
 */
cl_context opencl_create_context(void);

/**
 * Create a command queue for a context.
 *
 * The command queue will have CL_QUEUE_PROFILING_ENABLE set, such that
 * profiling can be done either manually or using helper functions in this lib.
 * @param ctx Context created through opencl_create_context.
 * @return Command queue.
 */
cl_command_queue opencl_create_cmdqueue(cl_context ctx);

/**
 * Compile an OpenCL program from one or more source files.
 *
 * Takes a list of file paths and a context. This function will concatenate
 * their contents and run them through the OpenCL compiler. Upon errors, the
 * compiler output will be printed to stderr.
 * @param ctx OpenCL context
 * @param source_cnt Number of entries in the source file list
 * @param source_files List of source file paths/names to compile
 * @return The compiled program, or  NULL if compilation failed.
 */
cl_program opencl_compile_program(cl_context ctx, cl_uint source_cnt,
		const char **source_files);

/**
 * Destroy the context, command queue and program
 *
 * Short-hand in case only a single program was compiled.
 * @param ctx Context
 * @param q Command queue
 * @param prg Program
 */
void opencl_teardown(cl_context *ctx, cl_command_queue *q,  cl_program *prg);

/**
 * Obtain the execution time of a kernel run
 *
 * Provided an event handler pointer was passed to the kernel execution call.
 * @param time Time event taken from the OpenCL kernel execute invocation.
 * @return Time of execution in nanoseconds.
 */
cl_ulong opencl_exec_time(cl_event time);

/** Queries the device for the maximum number of work-items in a work-group.
 * @return the maximum number of work-items in a work-group.
 */
size_t opencl_max_workgroup_size();

/** Parse a key/value command line option pair.
 * @param c Character identifier for this option
 * @param optarg String provided as parameter to this option.
 * @return 0 on success.
 */
int opencl_parse_option(int c, char *optarg);

void opencl_download_float_csv(cl_command_queue q, cl_mem out, char *file,
		size_t elems);

/**
 * Compare output buffer against CSV file contents.
 *
 * @param q Command queue
 * @param out Buffer containing output values
 * @param file File containing reference values
 * @param elems Number of elements to compare
 * @param delta Tolerated error
 * @param dType Interpretation of error tolerance (absolute or as a fraction)
 * @return 0 upon success
 */
int opencl_compare_out_csv(cl_command_queue q, cl_mem out, char *file,
			size_t elems, float delta, clErrorMarginType dType);

/**
 * Compare output buffer against a binary file's contents.
 *
 * @param q Command queue
 * @param out Buffer containing output values
 * @param file File containing reference values
 * @param elems Number of elements to compare
 * @param delta Tolerated error
 * @param dType Interpretation of error tolerance (absolute or as a fraction)
 * @return 0 upon success
 */
int opencl_compare_out_bin(cl_command_queue q, cl_mem out, char *file,
			size_t elems, float delta, clErrorMarginType dType);

/** Print the library's parameter usage guidelines to stdout. */
void opencl_usage();

#endif /* LIB_OPENCL_H */
