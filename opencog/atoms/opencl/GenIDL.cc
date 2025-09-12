/*
 * opencog/atoms/opencl/GenIDL.cc
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

#include <regex>
#include <sstream>
#include <algorithm>

#include <opencog/atoms/base/Node.h>
#include <opencog/atoms/base/Link.h>
#include <opencog/atoms/opencl/GenIDL.h>

using namespace opencog;

GenIDL::GenIDL()
{
	init_common_connectors();
}

GenIDL::~GenIDL()
{
}

void GenIDL::init_common_connectors()
{
	// Pre-create the only three connectors we need
	Handle fv_type = createNode(TYPE_NODE, "FloatValue");
	Handle in_sex = createNode(SEX_NODE, "input");
	Handle out_sex = createNode(SEX_NODE, "output");
	Handle scalar_sex = createNode(SEX_NODE, "scalar");

	_fv_in_cnctr = createLink(CONNECTOR, fv_type, in_sex);
	_fv_out_cnctr = createLink(CONNECTOR, fv_type, out_sex);
	_fv_scalar_cnctr = createLink(CONNECTOR, fv_type, scalar_sex);
}


std::string
GenIDL::trim(const std::string& str) const
{
	size_t first = str.find_first_not_of(" \t\n\r");
	if (first == std::string::npos)
		return "";
	size_t last = str.find_last_not_of(" \t\n\r");
	return str.substr(first, (last - first + 1));
}

std::string
GenIDL::determine_sex(const std::string& param_type) const
{
	// Check if this is a pointer type
	if (param_type.find("*") != std::string::npos)
	{
		// Pointer types are either input or output
		// Output if non-const pointer (no const qualifier)
		if (param_type.find("const") == std::string::npos)
			return "output";
		else
			return "input";  // const pointer is input
	}

	// Non-pointer types are scalar parameters
	return "scalar";
}

std::vector<std::string>
GenIDL::parse_parameters(const std::string& kernel_decl) const
{
	std::vector<std::string> param_types;

	// Find the parameter list between parentheses
	size_t paren_start = kernel_decl.find('(');
	size_t paren_end = kernel_decl.rfind(')');

	if (paren_start == std::string::npos || paren_end == std::string::npos)
		return param_types;

	std::string param_list = kernel_decl.substr(paren_start + 1, paren_end - paren_start - 1);

	// Split parameters by comma
	std::stringstream ss(param_list);
	std::string param;

	while (std::getline(ss, param, ','))
	{
		param = trim(param);
		if (param.empty()) continue;

		// Find the last word (parameter name) to exclude it
		size_t last_space = param.find_last_of(" \t");
		if (last_space != std::string::npos)
		{
			std::string param_type = param.substr(0, last_space);
			param_types.push_back(trim(param_type));
		}
	}

	return param_types;
}

std::string
GenIDL::extract_kernel_name(const std::string& kernel_decl) const
{
	// Find the function name between "kernel void" and "("
	size_t void_pos = kernel_decl.find("void");
	if (void_pos == std::string::npos)
		return "";

	size_t name_start = kernel_decl.find_first_not_of(" \t", void_pos + 4);
	size_t name_end = kernel_decl.find('(', name_start);

	if (name_start == std::string::npos || name_end == std::string::npos)
		return "";

	return trim(kernel_decl.substr(name_start, name_end - name_start));
}

std::vector<std::string>
GenIDL::extract_kernels(const std::string& opencl_src) const
{
	std::vector<std::string> kernels;

	// Regular expression to match kernel function declarations
	// Matches "kernel void functionName(...)"
	std::regex kernel_regex(R"(kernel\s+void\s+\w+\s*\([^)]*\))");

	auto kernels_begin = std::sregex_iterator(opencl_src.begin(), opencl_src.end(), kernel_regex);
	auto kernels_end = std::sregex_iterator();

	for (std::sregex_iterator i = kernels_begin; i != kernels_end; ++i)
	{
		std::smatch match = *i;
		kernels.push_back(match.str());
	}

	return kernels;
}

Handle
GenIDL::generate_kernel_section(const std::string& kernel_decl)
{
	// Extract kernel name
	std::string kernel_name = extract_kernel_name(kernel_decl);
	if (kernel_name.empty())
		return Handle::UNDEFINED;

	// Parse parameters
	auto param_types = parse_parameters(kernel_decl);

	// Build connector sequence
	HandleSeq connectors;
	for (const auto& param_type : param_types)
	{
		std::string sex = determine_sex(param_type);

		// Use the appropriate pre-created connector
		Handle connector;
		if (sex == "output")
			connector = _fv_out_cnctr;
		else if (sex == "scalar")
			connector = _fv_scalar_cnctr;
		else  // "input"
			connector = _fv_in_cnctr;

		connectors.push_back(connector);
	}

	return createLink(SECTION,
		createNode(ITEM_NODE, kernel_name),
		createLink(connectors, CONNECTOR_SEQ));
}

HandleSeq
GenIDL::gen_idl(const std::string& opencl_src)
{
	HandleSeq sections;

	// Extract all kernel declarations
	std::vector<std::string> kernels = extract_kernels(opencl_src);

	// Generate a Section for each kernel
	for (const std::string& kernel : kernels)
	{
		Handle section = generate_kernel_section(kernel);
		if (section != Handle::UNDEFINED)
			sections.emplace_back(section);
	}

	return sections;
}
