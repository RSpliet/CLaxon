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
#include <errno.h>
#include <unistd.h>

#include "lib/opencl.h"

void usage()
{
	printf("cltest - test program for OpenCL routines\n");
	printf("Options:\n");
	printf("\t-?\t\t This help\n");
	opencl_usage();
}

int main(int argc, char **argv)
{
	int c;
	int ret;
	cl_context ctx;
	cl_command_queue q;

	while ((c = getopt (argc, argv, "?"OPENCL_OPTS)) != -1)
	{
		switch (c) {
		case '?':
			usage();
			return 0;
		default:
			ret = opencl_parse_option(c, optarg);
			if (ret != 0) {
				usage();
				return -1;
			}
		}
	}

	ctx = opencl_create_context();
	if (!ctx) {
		usage();
		return -1;
	}

	q = opencl_create_cmdqueue(ctx);
	if (!q) {
		usage();
		return -1;
	}


	opencl_teardown(&ctx, &q, NULL);

	return 0;
}
