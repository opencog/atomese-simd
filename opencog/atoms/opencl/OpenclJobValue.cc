/*
 * opencog/atoms/opencl/OpenclJobValue.cc
 *
 * Copyright (C) 2015 Linas Vepstas
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
#include <opencog/atomspace/AtomSpace.h>
#include <opencog/atoms/base/Link.h>
#include <opencog/atoms/core/NumberNode.h>
#include <opencog/atoms/core/TypeNode.h>
#include <opencog/atoms/value/StringValue.h>
#include <opencog/atoms/value/ValueFactory.h>
#include <opencog/opencl/types/atom_types.h>

#include "OpenclFloatValue.h"
#include "OpenclJobValue.h"
#include "OpenclNode.h"

using namespace opencog;

OpenclJobValue::OpenclJobValue(Handle defn) :
	LinkValue(OPENCL_JOB_VALUE),
	_kernel{}
{
	if (not defn->is_type(SECTION))
		throw RuntimeException(TRACE_INFO,
			"Expecting Section, got: %s", defn->to_string().c_str());
	_definition = defn;
}

OpenclJobValue::~OpenclJobValue()
{
	_kernel = {};
}

// ==============================================================

const std::string&
OpenclJobValue::get_kern_name (void) const
{
	Handle hk = _definition->getOutgoingAtom(0);
	ValuePtr vp = hk;
	if (hk->is_executable())
		vp = hk->execute();

	if (vp->is_node())
		return HandleCast(vp)->get_name();

	if (vp->is_type(STRING_VALUE))
		return StringValueCast(vp)->value()[0];

	throw RuntimeException(TRACE_INFO,
		"Expecting Value with kernel name, got %s\n",
		vp->to_string().c_str());
}

// ==============================================================

/// Find the vector length.
/// Look either for a length specification embedded in the list,
/// else obtain the shortest of all the vectors.
bool
OpenclJobValue::get_vec_len(const ValueSeq& vsq)
{
	bool have_length_spec = false;
	_dim = UINT_MAX;
	for (const ValuePtr& vp : vsq)
	{
		if (vp->is_type(TYPE_NODE)) continue;

		if (vp->is_type(NUMBER_NODE))
		{
			size_t sz = NumberNodeCast(vp)->size();
			if (sz < _dim) _dim = sz;
			continue;
		}

		if (vp->is_type(FLOAT_VALUE))
		{
			size_t sz = FloatValueCast(vp)->size();
			if (sz < _dim) _dim = sz;
			continue;
		}

		// Assume the length specification is wrapped like so:
		// (Connector (Number 42))
		// XXX FIXME check for insane structures here.
		if (vp->is_type(CONNECTOR))
		{
			have_length_spec = true;
			const Handle& h = HandleCast(vp)->getOutgoingAtom(0);
			double sz = NumberNodeCast(h)->get_value();
			if (sz < 0.0) continue;

			// round, just in case.
			_dim = (size_t) (sz+0.5);
			return true;
		}
	}

	// We got a length spec; maybe we had to guess the size.
	return have_length_spec;
}

/// Unwrap vector.
ValuePtr
OpenclJobValue::get_floats(const Handle& oclno, ValuePtr vp)
{
	// If we're already the right format, we're done. Do nothing.
	// Well, almost nothing. Make sure that the vector knows it's
	// context. It might not know, if the user created it and did
	// not explicitly do a *-write-* with it.
	if (vp->is_type(OPENCL_DATA_VALUE))
	{
		OpenclFloatValuePtr ofv = OpenclFloatValueCast(vp);
		ofv->set_context(oclno);
		return vp;
	}

	// Special-case location of the vector length specification.
	if (vp->is_type(CONNECTOR))
	{
		Handle hd = HandleCast(createNumberNode(_dim));
		AtomSpace* as = oclno->getAtomSpace();
		return as->add_link(CONNECTOR, hd);
	}

	const std::vector<double>* vals = nullptr;
	if (vp->is_type(FLOAT_VALUE))
		vals = &(FloatValueCast(vp)->value());
	else if (vp->is_type(NUMBER_NODE))
		vals = &(NumberNodeCast(vp)->value());
	else
		throw RuntimeException(TRACE_INFO,
			"Expecting vector of floats, got: %s", vp->to_string().c_str());

	OpenclFloatValuePtr ofv;
	if (vals->size() != _dim)
	{
		std::vector<double> cpy(*vals);
		cpy.resize(_dim);
		ofv = createOpenclFloatValue(cpy);
	}
	else
		ofv = createOpenclFloatValue(*vals);

	// We created a new createOpenclFloatValue and we know that
	// the kernel will use it as input. So upload the data now.
	ofv->set_context(oclno);
	ofv->send_buffer(oclno);
	return ofv;
}

/// Unpack kernel arguments
ValueSeq
OpenclJobValue::make_vectors(const Handle& oclno)
{
	// We could check that conseq is actually of type ConnectorSeq
	// and throw if not, but I don't see a need to enforce this yet.
	const Handle& conseq = _definition->getOutgoingAtom(1);
	const HandleSeq& oset = conseq->getOutgoingSet();

	// Execute any executable connectors...
	ValueSeq vsq;
	for (const Handle& oh : oset)
	{
		if (oh->is_executable())
			vsq.emplace_back(oh->execute());
		else
			vsq.push_back(oh);
	}

	// Find the shortest vector.
	bool have_size_spec = get_vec_len(vsq);
	ValueSeq flovec;
	for (const ValuePtr& v: vsq)
		flovec.emplace_back(get_floats(oclno, v));

	// If the user never specified an explicit location in which to pass
	// the vector size, assume it is the last location. Set it now.
	// Is this a good idea? I dunno. More thinking needed.
	if (not have_size_spec)
	{
		Handle hd = HandleCast(createNumberNode(_dim));
		AtomSpace* as = oclno->getAtomSpace();
		flovec.emplace_back(as->add_link(CONNECTOR, hd));
	}

	return flovec;
}

// ==============================================================

void
OpenclJobValue::check_signature(const Handle& kern, const Handle& iface,
                                const ValueSeq& flovecs)
{
	// iface is a ConnectorSeq oc Connectors
	if (flovecs.size() != iface->size())
		throw RuntimeException(TRACE_INFO,
			"Expected %zu arguments, got %zu for %s",
			iface->size(), flovecs.size(), kern->to_string().c_str());

	// Each Connector has the form
	//    (Connector (Type 'FloatValue) (Sex "input"))
	// or similar.
	// Each item in the flovecs array is going to either be
	//    (OpenclFloatValue ...)
	// or, for scalars,
	//    (Connector (Number 42))
	// We're going to blow off scalar checking, for now.
	const HandleSeq& cons = iface->getOutgoingSet();
	for (size_t i = 0; i < cons.size(); i++)
	{
		TypeNodePtr typ = TypeNodeCast(cons[i]->getOutgoingAtom(0));

		if (not flovecs[i]->is_type(typ->get_kind()) and
		    not flovecs[i]->is_type(CONNECTOR))
		{
			throw RuntimeException(TRACE_INFO,
				"Argument type mismatch at %zu: expected type %s for %s",
				i, typ->to_string().c_str(), kern->to_string().c_str());
		}
	}
}

// ==============================================================

void OpenclJobValue::build(const Handle& oclno)
{
	if (not oclno->is_type(OPENCL_NODE))
		throw RuntimeException(TRACE_INFO,
			"Expecting OpenclNode, got: %s", oclno->to_string().c_str());

	std::string kname = get_kern_name();

	// See if its a kernel that we know.
	// XXX FIXME this will fail for SPV files, because we don't
	// (yet) generate signatures for them.
	OpenclNodePtr ocn = OpenclNodeCast(oclno);
	AtomSpace* as = ocn->getAtomSpace();
	Handle kit = as->add_node(ITEM_NODE, std::string(kname));
	const HandleMap& ifmap = ocn->_kernel_interfaces;
	const auto& descr = ifmap.find(kit);
	if (descr == ifmap.end())
		throw RuntimeException(TRACE_INFO,
			"This OpenclNode does not know about the kernel \"%s\"\n",
			 kname.c_str());

	// Get our kernel from the program from the OpenclNode.
	const cl::Program& proggy = ocn->get_program();
	_kernel = cl::Kernel(proggy, kname.c_str());

	// Build the OpenclJobValue itself.
	ValueSeq flovecs = make_vectors (oclno);
	check_signature(descr->first, descr->second, flovecs);
	ValuePtr args = createLinkValue(flovecs);
	_value = ValueSeq{kit, args};

	// Bind the kernel to the kernel arguments
	size_t pos = 0;
	for (const ValuePtr& v: flovecs)
	{
		if (v->is_type(OPENCL_DATA_VALUE))
			_kernel.setArg(pos, OpenclFloatValueCast(v)->get_buffer());
		else
			_kernel.setArg(pos, _dim);
		pos++;
	}
}

void OpenclJobValue::run(const Handle& oclno)
{
	OpenclNodePtr onp = OpenclNodeCast(oclno);

	// Launch kernel
	cl::CommandQueue& queue = onp->get_queue();
	cl::Event& event_handler = onp->get_handler();

	queue.enqueueNDRangeKernel(_kernel,
		cl::NullRange,
		cl::NDRange(_dim),
		cl::NullRange,
		nullptr, &event_handler);
}

// ==============================================================

// Adds factory when the library is loaded.
DEFINE_VALUE_FACTORY(OPENCL_JOB_VALUE,
                     createOpenclJobValue, Handle)
