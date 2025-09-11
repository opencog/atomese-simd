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
#include <opencog/atoms/base/Link.h>
#include <opencog/atoms/core/NumberNode.h>
#include <opencog/atoms/value/ValueFactory.h>
#include <opencog/opencl/types/atom_types.h>

#include "OpenclFloatValue.h"
#include "OpenclJobValue.h"

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

ValuePtr
OpenclJobValue::get_kernel (ValuePtr kvec) const
{
	Handle hkl;
	if (kvec->is_type(SECTION))
		hkl = HandleCast(kvec)->getOutgoingAtom(0);
	else
	if (kvec->is_type(OPENCL_JOB_VALUE))
	{
		const ValueSeq& vsq = LinkValueCast(kvec)->value();
		hkl = HandleCast(vsq[0]);
	}

	if (nullptr == hkl or not hkl->is_type(OPENCL_KERNEL_LINK))
		throw RuntimeException(TRACE_INFO,
			"Expecting an OpenclKernelLink, got: %s", kvec->to_string().c_str());

	return hkl;
}

// ==============================================================

/// Find the vector length.
/// Look either for a length specification embedded in the list,
/// else obtain the shortest of all the vectors.
size_t
OpenclJobValue::get_vec_len(const ValueSeq& vsq, bool& have_size_spec) const
{
	have_size_spec = false;
	size_t dim = UINT_MAX;
	for (const ValuePtr& vp : vsq)
	{
		if (vp->is_type(TYPE_NODE)) continue;

		if (vp->is_type(NUMBER_NODE))
		{
			size_t sz = NumberNodeCast(vp)->size();
			if (sz < dim) dim = sz;
			continue;
		}

		if (vp->is_type(FLOAT_VALUE))
		{
			size_t sz = FloatValueCast(vp)->size();
			if (sz < dim) dim = sz;
			continue;
		}

		// Assume the length specification is wrapped like so:
		// (Connector (Number 42))
		// XXX FIXME check for insane structures here.
		if (vp->is_type(CONNECTOR))
		{
			have_size_spec = true;
			const Handle& h = HandleCast(vp)->getOutgoingAtom(0);
			double sz = NumberNodeCast(h)->get_value();
			if (sz < 0.0) continue;

			// round, just in case.
			return (size_t) (sz+0.5);
		}
	}

	return dim;
}

/// Unwrap vector.
ValuePtr
OpenclJobValue::get_floats(ValuePtr vp, size_t dim) const
{
	// If we're already the right format, we're done. Do nothing.
	if (vp->is_type(OPENCL_DATA_VALUE))
		return vp;

	// Special-case location of the vector length specification.
	if (vp->is_type(CONNECTOR))
	{
		Handle hd = HandleCast(createNumberNode(dim));
		return createLink(CONNECTOR, hd);
	}

	// XXX For now, we ignore the type. FIXME
	// XXX this API is a bad API. Needs rethinking.
	if (vp->is_type(TYPE_NODE))
	{
		std::vector<double> zero;
		zero.resize(dim);
		OpenclFloatValuePtr ofv = createOpenclFloatValue(zero);
		// ofv->set_context(_device, _context);
		return ofv;
	}

	const std::vector<double>* vals = nullptr;
	if (vp->is_type(NUMBER_NODE))
		vals = &(NumberNodeCast(vp)->value());

	if (vp->is_type(FLOAT_VALUE))
		vals = &(FloatValueCast(vp)->value());

	OpenclFloatValuePtr ofv;
	if (vals->size() != dim)
	{
		std::vector<double> cpy(*vals);
		cpy.resize(dim);
		ofv = createOpenclFloatValue(cpy);
	}
	else
		ofv = createOpenclFloatValue(*vals);

	// ofv->set_context(_device, _context);
	// ofv->send_buffer();
	return ofv;
}

ValueSeq
OpenclJobValue::make_vectors(ValuePtr kvec, size_t& dim) const
{
	// Unpack kernel arguments
	ValueSeq vsq;
	if (kvec->is_type(SECTION))
	{
		const Handle& conseq = HandleCast(kvec)->getOutgoingAtom(1);
		const HandleSeq& oset = conseq->getOutgoingSet();

		// Find the shortest vector.
		for (const Handle& oh : oset)
		{
			if (oh->is_executable())
				vsq.emplace_back(oh->execute());
			else
				vsq.push_back(oh);
		}
	}
	else
		throw RuntimeException(TRACE_INFO,
			"Unknown data type: got %s\n", kvec->to_string().c_str());

	// Find the shortest vector.
	bool have_size_spec = false;
	dim = get_vec_len(vsq, have_size_spec);
	ValueSeq flovec;
	for (const ValuePtr& v: vsq)
		flovec.emplace_back(get_floats(v, dim));

	// If the user never specified an explicit location in which to pass
	// the vector size, assume it is the last location. Set it now.
	// Is this a good idea? I dunno. More thinking needed.
	if (not have_size_spec)
	{
		Handle hd = HandleCast(createNumberNode(dim));
		flovec.emplace_back(createLink(CONNECTOR, hd));
	}

	return flovec;
}

// ==============================================================

void OpenclJobValue::build(const Handle& oclno)
{
#if 0
	ValuePtr hkl = get_kernel(kvec);
	OpenclKernelLinkPtr okp = OpenclKernelLinkCast(hkl);
	cl::Kernel kern = okp->get_kernel();

	size_t dim = 0;
	ValueSeq flovecs = make_vectors (kvec, dim);
	ValuePtr args = createLinkValue(flovecs);
	ValuePtr jobvec = createOpenclJobValue(ValueSeq{hkl, args});

	size_t pos = 0;
	for (const ValuePtr& v: flovecs)
	{
		if (v->is_type(OPENCL_FLOAT_VALUE))
			OpenclFloatValueCast(v)->set_arg(kern, pos);
		else
			kern.setArg(pos, dim);
		pos++;
	}
#endif
}

void OpenclJobValue::run(void)
{
#if 0
		// Launch kernel
		cl::Event event_handler;
		_queue.enqueueNDRangeKernel(kjob._kern,
			cl::NullRange,
			cl::NDRange(kjob._vecdim),
			cl::NullRange,
			nullptr, &event_handler);

		event_handler.wait();
		_qvp->add(kjob._kvec);
		return;
#endif
}

// ==============================================================

// Adds factory when the library is loaded.
DEFINE_VALUE_FACTORY(OPENCL_JOB_VALUE,
                     createOpenclJobValue, Handle)
