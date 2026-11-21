// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/power.hpp"
#include "utils.hpp"

namespace ov::frontend::pytorch::op {

using namespace ov::op;

OutputVector translate_float_power(const NodeContext& context) {
    // aten::float_power(Tensor/Scalar self, Tensor/Scalar exponent, *, Tensor? out=None) -> Tensor
    // Both operands are converted to float64 and pow is evaluated in double precision.
    // The result dtype is always float64. Complex inputs are not supported.
    num_inputs_check(context, 2, 3);
    auto lhs = context.mark_node(std::make_shared<v0::Convert>(context.get_input(0), element::f64));
    auto rhs = context.mark_node(std::make_shared<v0::Convert>(context.get_input(1), element::f64));
    return {context.mark_node(std::make_shared<v1::Power>(lhs, rhs))};
}

}  // namespace ov::frontend::pytorch::op
