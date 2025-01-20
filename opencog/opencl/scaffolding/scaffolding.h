/**
 * scaffolding.h
 * OpenCL scaffolding.
 *
 * This provides minimalistic scaffolding to allow OpenCL experiments
 * to take place.
 *
 * Copyright (c) 2025 Linas Vepstas
 */

#ifndef __ATOMESE_OPENCL_SCAFFOLDING_H__
#define __ATOMESE_OPENCL_SCAFFOLDING_H__

#define CL_HPP_MINIMUM_OPENCL_VERSION 200
#define CL_HPP_TARGET_OPENCL_VERSION 300
#define CL_HPP_ENABLE_EXCEPTIONS
// #define CL_HPP_USE_IL_KHR  /* For SPV IL support */

#if defined __has_include
	#if __has_include(<CL/opencl.hpp>)
		#include <CL/opencl.hpp>
	#else
		#include <CL/cl.hpp>
	#endif
#else
	#include <CL/opencl.hpp>
#endif

/// Print rudimentary report of available OpenCL hardware.
void report_hardware(void);

/// Return the first device that has platsubstr and devsubstr as
/// substrings in the platform and device name.
cl::Device find_device(const char* platsubstr, const char* devsubstr);

/// Build kernel from source file, return context.
void build_kernel(cl::Context&, const char* srcfile, cl::Program&);

/// Load pre-build kernel from SPV file, return context.
void load_kernel(cl::Context&, const char* spvfile, cl::Program&);

#endif /* __ATOMESE_OPENCL_SCAFFOLDING_H__ */
