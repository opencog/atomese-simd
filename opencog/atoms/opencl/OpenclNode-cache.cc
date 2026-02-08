/*
 * opencog/atoms/opencl/OpenclNode-cache.cc
 *
 * Copyright (C) 2026 plankatronic-claude
 * All Rights Reserved
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License v3 as
 * published by the Free Software Foundation and including the exceptions
 * at http://opencog.org/wiki/Licenses
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program; if not, write to:
 * Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#include <fstream>
#include <sstream>
#include <iomanip>
#include <sys/stat.h>
#include <cstdlib>
#include <functional>

#include <opencog/util/Logger.h>

#include "OpenclNode.h"

using namespace opencog;

// ==============================================================
// Binary caching implementation.
// Caches compiled OpenCL programs to disk to avoid expensive JIT
// compilation on subsequent runs. Cache files are stored in
// ~/.cache/opencog/opencl/<device_hash>/<source_hash>.bin
//
// This follows the pattern used by hashcat, PyOpenCL, and game engines
// to dramatically reduce startup time (from seconds to milliseconds).

// XXX FIXME (Actually, delete me!?)
// This code was dreamed up by Claude, but it is not at all clear that
// it is a good idea ... in fact, it smells like a terrible idea, and
// a total misunderstanding of what Atomese is, and how it works. The
// core issue is that all sorts of Atomese will be flying in and out of
// of the system, doing god-knows-what; this Atomese will change from
// second to second, session-to-session. It's kind of fundamentally
// uncachable, because you don't know what it is, where it came from,
// whether it will ever be used again. So at best, this cache can hold
// maybe some basic startup stuffs ... but I dunno. Just even the idea
// of writing some garbage into the file system is ... a bad idea.
// These not what Atomese is or how its supposed to work.
// So XXX FIXME, review me, and maybe trash this code. The future is cloudy.

/// Compute a simple hash of a string using std::hash.
/// Returns a hex string representation.
std::string OpenclNode::compute_hash(const std::string& data) const
{
	std::hash<std::string> hasher;
	size_t hash = hasher(data);

	std::ostringstream oss;
	oss << std::hex << std::setfill('0') << std::setw(16) << hash;
	return oss.str();
}

/// Get the cache directory path.
/// Creates ~/.cache/opencog/opencl/<device_hash>/ if it doesn't exist.
std::string OpenclNode::get_cache_dir(void) const
{
	// Get home directory
	const char* home = std::getenv("HOME");
	if (nullptr == home) home = "/tmp";

	// Build device identifier for cache directory
	std::string device_id = _platform.getInfo<CL_PLATFORM_NAME>() + "_" +
	                        _device.getInfo<CL_DEVICE_NAME>() + "_" +
	                        _device.getInfo<CL_DRIVER_VERSION>();
	std::string device_hash = compute_hash(device_id);

	std::string cache_dir = std::string(home) + "/.cache/opencog/opencl/" + device_hash;

	// Create directories if they don't exist (mkdir -p equivalent)
	std::string path = std::string(home) + "/.cache";
	mkdir(path.c_str(), 0755);
	path += "/opencog";
	mkdir(path.c_str(), 0755);
	path += "/opencl";
	mkdir(path.c_str(), 0755);
	path += "/" + device_hash;
	mkdir(path.c_str(), 0755);

	return cache_dir;
}

/// Get the full cache file path for a given source.
std::string OpenclNode::get_cache_path(const std::string& src) const
{
	std::string source_hash = compute_hash(src);
	return get_cache_dir() + "/" + source_hash + ".bin";
}

/// Try to load a cached binary. Returns true if successful.
bool OpenclNode::load_cached_binary(const std::string& cache_path)
{
	std::ifstream binfile(cache_path, std::ios::binary);
	if (!binfile.is_open())
		return false;

	// Read the entire binary file
	std::vector<char> binary((std::istreambuf_iterator<char>(binfile)),
	                          std::istreambuf_iterator<char>());
	binfile.close();

	if (binary.empty())
		return false;

	try
	{
		// Create program from binary
		cl::Program::Binaries binaries;
		binaries.push_back(std::vector<unsigned char>(binary.begin(), binary.end()));

		std::vector<cl::Device> devices = {_device};
		std::vector<cl_int> binary_status;

		_program = cl::Program(_context, devices, binaries, &binary_status);

		// Check if binary was valid for this device
		if (binary_status[0] != CL_SUCCESS)
		{
			logger().info("OpenclNode: Cached binary invalid for device, will recompile\n");
			return false;
		}

		// Build the program (links the binary, much faster than JIT compile)
		_program.build("");

		logger().info("OpenclNode: Loaded cached binary from %s\n", cache_path.c_str());
		return true;
	}
	catch (const cl::Error& e)
	{
		logger().info("OpenclNode: Failed to load cached binary: %s\n", e.what());
		return false;
	}
}

/// Save the compiled program binary to cache.
void OpenclNode::save_binary_to_cache(const std::string& cache_path)
{
	try
	{
		// Get the binary sizes
		std::vector<size_t> binary_sizes = _program.getInfo<CL_PROGRAM_BINARY_SIZES>();
		if (binary_sizes.empty() || binary_sizes[0] == 0)
		{
			logger().info("OpenclNode: No binary available to cache\n");
			return;
		}

		// Get the binaries
		std::vector<std::vector<unsigned char>> binaries;
		binaries.resize(binary_sizes.size());
		for (size_t i = 0; i < binary_sizes.size(); i++)
			binaries[i].resize(binary_sizes[i]);

		// Use raw pointers for the API
		std::vector<unsigned char*> binary_ptrs;
		for (auto& b : binaries)
			binary_ptrs.push_back(b.data());

		clGetProgramInfo(_program(), CL_PROGRAM_BINARIES,
		                 binary_ptrs.size() * sizeof(unsigned char*),
		                 binary_ptrs.data(), nullptr);

		// Write to cache file
		std::ofstream binfile(cache_path, std::ios::binary);
		if (binfile.is_open())
		{
			binfile.write(reinterpret_cast<const char*>(binaries[0].data()),
			              binaries[0].size());
			binfile.close();
			logger().info("OpenclNode: Saved binary to cache: %s (%zu bytes)\n",
			              cache_path.c_str(), binaries[0].size());
		}
	}
	catch (const cl::Error& e)
	{
		logger().info("OpenclNode: Failed to save binary to cache: %s\n", e.what());
	}
}
