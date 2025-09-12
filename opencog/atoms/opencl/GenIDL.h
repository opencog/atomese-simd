/*
 * opencog/atoms/opencl/GenIDL.h
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

#ifndef _OPENCOG_GENIDL_H
#define _OPENCOG_GENIDL_H

#include <string>
#include <vector>
#include <opencog/atoms/base/Handle.h>

namespace opencog
{

/** \addtogroup grp_atomspace
 *  @{
 */

/**
 * GenIDL - Generate Interface Definition Language (IDL) for OpenCL kernels
 *
 * This class parses OpenCL kernel function declarations and generates
 * Atomese representation of their interfaces.
 */
class GenIDL
{
protected:
	// Pre-created connectors - these are the only ones we need
	Handle _fv_in_cnctr;
	Handle _fv_out_cnctr;
	Handle _fv_scalar_cnctr;

	// Helper methods for parsing
	std::vector<std::string> extract_kernels(const std::string& opencl_src) const;
	std::string extract_kernel_name(const std::string& kernel_decl) const;
	std::vector<std::string> parse_parameters(const std::string& kernel_decl) const;

	// Helper methods for Atomese generation
	Handle generate_kernel_section(const std::string& kernel_decl);

	// Utility methods
	std::string trim(const std::string& str) const;
	std::string determine_sex(const std::string& param_type) const;

	// Initialize common connectors
	void init_common_connectors();

public:
	GenIDL();
	~GenIDL();

	/**
	 * Generate IDL from OpenCL source code
	 *
	 * @param opencl_src The OpenCL source code containing kernel definitions
	 * @return HandleSeq containing Section for each kernel
	 */
	HandleSeq gen_idl(const std::string& opencl_src);
};

/** @}*/
} // namespace opencog

#endif // _OPENCOG_GENIDL_H