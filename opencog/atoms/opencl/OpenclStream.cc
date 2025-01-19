/*
 * opencog/atoms/opencl/OpenclStream.cc
 *
 * Copyright (C) 2025 Linas Vepstas
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

#include <opencog/util/exceptions.h>
#include <opencog/util/oc_assert.h>
#include <opencog/atomspace/AtomSpace.h>
#include <opencog/atoms/base/Link.h>
#include <opencog/atoms/base/Node.h>
#include <opencog/atoms/value/StringValue.h>
#include <opencog/atoms/value/ValueFactory.h>

#include <opencog/opencl/types/atom_types.h>
#include <opencog/sensory/types/atom_types.h>
#include "OpenclStream.h"

using namespace opencog;

OpenclStream::OpenclStream(const std::string& str)
	: OutputStream(OPENCL_STREAM)
{
	init(str);
}

OpenclStream::OpenclStream(const Handle& senso)
	: OutputStream(OPENCL_STREAM)
{
	if (SENSORY_NODE != senso->get_type())
		throw RuntimeException(TRACE_INFO,
			"Expecting SensoryNode, got %s\n", senso->to_string().c_str());

	init(senso->get_name());
}

OpenclStream::~OpenclStream()
{
	// Runs only if GC runs. This is a problem.
	halt();
}

void OpenclStream::halt(void) const
{
	_value.clear();
}

/// Attempt to open connection to OpenCL device
void OpenclStream::init(const std::string& url)
{
	do_describe();
	if (0 != url.compare(0, 9, "opencl://"))
		throw RuntimeException(TRACE_INFO,
			"Unsupported URL \"%s\"\n"
			"\tExpecting 'opencl://platform:device/file/path/kernel.cl'",
			url.c_str());

	// Make a copy, for debuggingg purposes.
	_uri = url;

	// Ignore the first 9 chars "opencl://"
	size_t pos = 9;
	size_t platend = url.find('/', pos);
	if (pos < platend)
		_platform = url.substr(pos, platend);

printf("duuude got %s %lu\n", _platform.c_str(), platend);
}

// ==============================================================

Handle _global_desc = Handle::UNDEFINED;

void OpenclStream::do_describe(void)
{
	if (_global_desc) return;

	HandleSeq cmds;

	// Describe exactly how to Open this stream.
	// It needs no special arguments.
	Handle open_cmd =
		make_description("Open connection to GPU",
		                 "OpenLink", "OpenclStream");
	cmds.emplace_back(open_cmd);

	// Write  XXX this is wrong.
	Handle write_cmd =
		make_description("Write kernel and data to GPU",
		                 "WriteLink", "ItemNode");
	cmds.emplace_back(write_cmd);

	_global_desc = createLink(cmds, CHOICE_LINK);
}

// This is totally bogus because it is unused.
// This should be class static member
ValuePtr OpenclStream::describe(AtomSpace* as, bool silent)
{
	if (_description) return as->add_atom(_description);
	_description = as->add_atom(_global_desc);
	return _description;
}

// ==============================================================

void OpenclStream::update() const
{
	_value.resize(1);
	_value[0] = createStringValue("foo");
}

// ==============================================================
// Send kernel and data
ValuePtr OpenclStream::write_out(AtomSpace* as, bool silent,
                                 const Handle& cref)
{
	return nullptr;
}

// ==============================================================

// Adds factory when library is loaded.
DEFINE_VALUE_FACTORY(OPENCL_STREAM, createOpenclStream, std::string)
DEFINE_VALUE_FACTORY(OPENCL_STREAM, createOpenclStream, Handle)

// ====================================================================

void opencog_opencl_init(void)
{
   // Force shared lib ctors to run
};
