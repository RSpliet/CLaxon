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

#ifndef LIB_CSV_H
#define LIB_CSV_H

#include <stdint.h>
#include <stdbool.h>

/**
 * Count the number of elements in a CSV file.
 *
 * @param file path and name of file to read words from
 * @return The number of read elements.
 */
int64_t csv_file_count(char *file);

/**
 * Read all integers from a CSV file, store in a newly-allocated buffer.
 *
 * This library function performs allocation, user must free manually.
 * @param file path and name of file to read words from
 * @param buf pointer to location where newly allocated buffer pointer must
 * 	be stored
 * @return The number of read elements.
 */
int64_t csv_file_read(char *file, int **buf);

/**
 * Read all floats from a CSV file, store in a newly-allocated buffer.
 *
 * This library function performs allocation, user must free manually.
 * @param file path and name of file to read words from
 * @param buf pointer to location where newly allocated buffer pointer must
 * 	be stored
 * @return The number of read elements.
 */
int64_t csv_file_read_float(char *file, float **buf);

/**
 * Read all n-tuples of floats from a file, store in "struct of arrays" buffer.
 *
 * This library function performs allocation, user must free manually.
 * @param file path and name of file to read words from
 * @param n Number of entries in a tuple.
 * @param buf pointer to location where newly allocated buffer pointer must
 * 	be stored
 * @return The number of read tuples.
 */
int64_t csv_file_read_float_n(char *file, int n, float ***buf);

/**
 * Read n numbers from a comma separated file into a buffer of floats.
 *
 * This library function performs allocation, user must free manually.
 * @param file path and name of file to read words from
 * @param n Number of words to read
 * @param buf pointer to location where newly allocated buffer pointer must
 * 	be stored
 * @return true iff all words were successfully read.
 */
bool csv_file_write(char *file, size_t n, float *buf);

/**
 * Read n 32-bit words from a binary file into a buffer.
 *
 * This library function performs allocation, user must free manually.
 * @param file path and name of file to read words from
 * @param n Number of words to read
 * @param buf pointer to location where newly allocated buffer pointer must
 * 	be stored
 * @return true iff all words were successfully read.
 */
bool bin_file_read(char *file, size_t n, void **buf);

#endif /* LIB_CSV_H */
