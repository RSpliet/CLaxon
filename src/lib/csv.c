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

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <errno.h>
#include <inttypes.h>
#include <stdbool.h>

int64_t
csv_file_count(char *file)
{
	int64_t c = 0;
	float tmp;
	FILE *fp;

	fp = fopen(file, "r");
	if (!fp) {
		fprintf(stderr, "Could not open csv file %s\n", file);
		return -EINVAL;
	}

	while (fscanf(fp, "%f%*[, ]", &tmp) > 0) {
		c++;
	}

	fclose(fp);

	return c;
}

int64_t
csv_file_read(char *file, int **buf)
{
	int64_t count, i;
	int tmp;
	FILE *fp;

	count = csv_file_count(file);
	if (count < 0)
		return -1;

	*buf = malloc(count * sizeof(int));
	if (!*buf) {
		fprintf(stderr, "Could not allocate memory for data buffer\n");
		return -1;
	}

	fp = fopen(file, "r");
	if (!fp) {
		fprintf(stderr, "Could not open csv file %s\n", file);
		return -EINVAL;
	}

	for (i = 0; i < count; i++) {
		tmp = fscanf(fp, "%d%*[, ]", *buf+i);
		if (tmp <= 0)
			break;
	}

	fclose(fp);

	return i;
}

int64_t
csv_file_read_float(char *file, float **buf)
{
	int64_t count, i;
	int tmp;
	FILE *fp;

	count = csv_file_count(file);
	if (count < 0)
		return -1;

	*buf = malloc(count * sizeof(float));
	if (!*buf) {
		fprintf(stderr, "Could not allocate memory for data buffer\n");
		return -1;
	}

	fp = fopen(file, "r");
	if (!fp) {
		fprintf(stderr, "Could not open csv file %s\n", file);
		return -EINVAL;
	}

	for (i = 0; i < count; i++) {
		tmp = fscanf(fp, "%f%*[, ]", *buf+i);
		if (tmp <= 0)
			break;
	}

	fclose(fp);

	return i;
}

int64_t
csv_file_read_float_n(char *file, int n, float ***buf)
{
	int64_t count, i;
	int tmp;
	FILE *fp;

	count = csv_file_count(file);
	if (count < 0)
		return -1;

	if (count % n != 0) {
		fprintf(stderr, "Incomplete n-tuple found %"PRIu64"\n", count);
		return -1;
	}

	*buf = malloc(n * sizeof(float *));
	if (!*buf) {
		fprintf(stderr, "Could not allocate memory for data buffer\n");
		return -1;
	}

	/* Make one large contiguous buffer for easier param passing */
	(*buf)[0] = malloc(count * sizeof(float));
	if (!(*buf)[0]) {
		fprintf(stderr, "Could not allocate memory for data "
				"buffer\n");
		return -1;
	}

	for (i = 1; i < n; i++) {
		(*buf)[i] = (*buf)[0] + i * (count / n);
	}

	fp = fopen(file, "r");
	if (!fp) {
		fprintf(stderr, "Could not open csv file %s\n", file);
		return -EINVAL;
	}

	for (i = 0; i < count; i++) {
		tmp = fscanf(fp, "%f%*[, ]", (*buf)[i % n]+(i / n));
		if (tmp <= 0)
			break;
	}

	fclose(fp);

	return i/n;
}

bool
csv_file_write(char *file, size_t n, float *buf)
{
	FILE *fp;
	size_t i;

	if (n == 0) {
		fprintf(stderr, "Must have at least one item to print.");
		return false;
	}

	fp = fopen(file, "w");
	if (!fp) {
		fprintf(stderr, "Could not open file %s for writing.", file);
		return false;
	}

	fprintf(fp, "%.4f", buf[0]);

	for (i = 1; i < n; i++)
		fprintf(fp, ", %.4f", buf[i]);

	fclose(fp);

	return true;
}

bool
bin_file_read(char *file, size_t n, void **buf)
{
	FILE *fp;
	size_t rdbytes;

	*buf = malloc(n * sizeof(int));
	if (!*buf) {
		fprintf(stderr, "Could not allocate memory for data buffer\n");
		return -1;
	}

	fp = fopen(file, "r");
	if (!fp) {
		fprintf(stderr, "Could not open bin file %s\n", file);
		return -EINVAL;
	}

	rdbytes = fread(*buf, sizeof(int), n, fp);

	fclose(fp);

	return (rdbytes != n);
}
