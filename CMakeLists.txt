# SPDX-License-Identifier: MIT
#
# Copyright (c) 2020 Roy Spliet, University of Cambridge
#
#Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice (including the next
# paragraph) shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Minimum is arbitrary, ~2.8.10 had ninja support merged
cmake_minimum_required(VERSION 2.8.10)

PROJECT(CLaxon)

find_package(OpenCL 1.2 REQUIRED)
if (OpenCL_FOUND)
  include_directories(${OpenCL_INCLUDE_DIRS})
  link_libraries (${OpenCL_LIBRARIES})
endif(OpenCL_FOUND)

set(CMAKE_C_FLAGS "-Wall")

# ADD_EXECUTABLE
include_directories(include)

add_library(CLaxon_libs OBJECT
        ${PROJECT_SOURCE_DIR}/src/lib/opencl.c
        ${PROJECT_SOURCE_DIR}/src/lib/csv.c
)

add_executable(cltest
	$<TARGET_OBJECTS:CLaxon_libs>
	src/cltest.c)

add_executable(cnn_maxpool
	$<TARGET_OBJECTS:CLaxon_libs>
	src/cnn_maxpool/cnn_maxpool.c)

add_executable(cnn_relu
	$<TARGET_OBJECTS:CLaxon_libs>
	src/cnn_relu/cnn_relu.c)

add_executable(cnn_relu_fc
	$<TARGET_OBJECTS:CLaxon_libs>
	src/cnn_relu/cnn_relu_fc.c)

add_executable(cnn_convolution
	$<TARGET_OBJECTS:CLaxon_libs>
	src/cnn_convolution/cnn_convolution.c)

add_executable(fft
	$<TARGET_OBJECTS:CLaxon_libs>
	src/fft/fft.c)

add_executable(kfusion
	$<TARGET_OBJECTS:CLaxon_libs>
	src/kfusion/kfusion.c)

add_executable(mriq
	$<TARGET_OBJECTS:CLaxon_libs>
	src/mriq/mriq.c)

add_executable(spmv
	$<TARGET_OBJECTS:CLaxon_libs>
	src/spmv/spmv.c)

add_executable(srad
	$<TARGET_OBJECTS:CLaxon_libs>
	src/srad/srad.c)

add_executable(stencil
	$<TARGET_OBJECTS:CLaxon_libs>
	src/stencil/stencil.c)

add_executable(frnn
	$<TARGET_OBJECTS:CLaxon_libs>
	src/frnn/frnn.c
	src/frnn/prefix_sum.c)
target_link_libraries(frnn m)

add_executable(ndt
	$<TARGET_OBJECTS:CLaxon_libs>
	src/ndt/ndt.c src/frnn/prefix_sum.c)

#add_executable(scratch $<TARGET_OBJECTS:CLaxon_libs> src/scratch/scratch.c)
