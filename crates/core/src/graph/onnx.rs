use super::{Graph, GraphOp, NodeId};
use crate::kernel::{BinaryOp, ReduceOp, UnaryOp};

// Protobuf wire types
const VARINT: u8 = 0;
const LEN: u8 = 2;

fn encode_varint(buf: &mut Vec<u8>, mut v: u64) {
    loop {
        let byte = (v & 0x7F) as u8;
        v >>= 7;
        if v == 0 {
            buf.push(byte);
            return;
        }
        buf.push(byte | 0x80);
    }
}

fn encode_field_varint(buf: &mut Vec<u8>, field: u32, val: u64) {
    encode_varint(buf, ((field as u64) << 3) | VARINT as u64);
    encode_varint(buf, val);
}

fn encode_field_bytes(buf: &mut Vec<u8>, field: u32, data: &[u8]) {
    encode_varint(buf, ((field as u64) << 3) | LEN as u64);
    encode_varint(buf, data.len() as u64);
    buf.extend_from_slice(data);
}

fn encode_field_string(buf: &mut Vec<u8>, field: u32, s: &str) {
    encode_field_bytes(buf, field, s.as_bytes());
}

/// ONNX TensorProto.DataType.DOUBLE = 11
const ONNX_DOUBLE: u64 = 11;

/// Map our UnaryOp to ONNX operator type strings.
fn unary_op_name(op: UnaryOp) -> &'static str {
    match op {
        UnaryOp::Neg => "Neg",
        UnaryOp::Abs => "Abs",
        UnaryOp::Sqrt => "Sqrt",
        UnaryOp::Rsqrt => "Rsqrt",
        UnaryOp::Exp => "Exp",
        UnaryOp::Log => "Log",
        UnaryOp::Sin => "Sin",
        UnaryOp::Cos => "Cos",
        UnaryOp::Tanh => "Tanh",
        UnaryOp::Relu => "Relu",
        UnaryOp::Sigmoid => "Sigmoid",
        UnaryOp::Floor => "Floor",
        UnaryOp::Ceil => "Ceil",
        UnaryOp::Scale(_) => "Mul",
        UnaryOp::Offset(_) => "Add",
    }
}

fn binary_op_name(op: BinaryOp) -> &'static str {
    match op {
        BinaryOp::Add => "Add",
        BinaryOp::Sub => "Sub",
        BinaryOp::Mul => "Mul",
        BinaryOp::Div => "Div",
        BinaryOp::Min => "Min",
        BinaryOp::Max => "Max",
        BinaryOp::Pow => "Pow",
    }
}

fn reduce_op_name(op: ReduceOp) -> &'static str {
    match op {
        ReduceOp::Sum => "ReduceSum",
        ReduceOp::Min => "ReduceMin",
        ReduceOp::Max => "ReduceMax",
        ReduceOp::Mean => "ReduceMean",
        ReduceOp::Product => "ReduceProd",
    }
}

/// Encode a single ONNX NodeProto.
fn encode_node(buf: &mut Vec<u8>, op_type: &str, inputs: &[&str], outputs: &[&str], name: &str) {
    let mut node = Vec::new();
    for inp in inputs {
        encode_field_string(&mut node, 1, inp); // input
    }
    for out in outputs {
        encode_field_string(&mut node, 2, out); // output
    }
    encode_field_string(&mut node, 3, name); // name
    encode_field_string(&mut node, 4, op_type); // op_type
    encode_field_bytes(buf, 1, &node); // GraphProto.node (field 1)
}

/// Encode a TensorProto for an initializer (constant data).
fn encode_initializer(buf: &mut Vec<u8>, name: &str, data: &[f64]) {
    let mut tensor = Vec::new();
    // dims (field 1, repeated int64)
    encode_field_varint(&mut tensor, 1, data.len() as u64);
    // data_type (field 2)
    encode_field_varint(&mut tensor, 2, ONNX_DOUBLE);
    // name (field 8)
    encode_field_string(&mut tensor, 8, name);
    // double_data (field 10, packed doubles)
    let raw: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    encode_field_bytes(&mut tensor, 10, &raw);
    encode_field_bytes(buf, 5, &tensor); // GraphProto.initializer (field 5)
}

/// Encode a ValueInfoProto (input/output type annotation).
fn encode_value_info(buf: &mut Vec<u8>, field: u32, name: &str) {
    let mut vi = Vec::new();
    encode_field_string(&mut vi, 1, name); // name
    // TypeProto (field 2) with tensor_type
    let mut tp = Vec::new();
    // TensorTypeProto: elem_type = DOUBLE (field 1)
    let mut ttp = Vec::new();
    encode_field_varint(&mut ttp, 1, ONNX_DOUBLE);
    encode_field_bytes(&mut tp, 1, &ttp); // TypeProto.tensor_type (field 1)
    encode_field_bytes(&mut vi, 2, &tp);
    encode_field_bytes(buf, field, &vi);
}

/// Encode the full ONNX ModelProto.
pub(super) fn encode(graph: &Graph, model_name: &str) -> Vec<u8> {
    let mut graph_proto = Vec::new();

    // Collect inputs, outputs, and initializers
    let mut initializer_names: Vec<String> = Vec::new();

    // Process all nodes in topological order from all leaf outputs
    for (i, node) in graph.nodes.iter().enumerate() {
        let out_name = node_name(graph, i);

        match &node.op {
            GraphOp::Constant(data) => {
                let init_name = out_name.clone();
                encode_initializer(&mut graph_proto, &init_name, data);
                initializer_names.push(init_name);
            }
            GraphOp::Input(name) => {
                encode_value_info(&mut graph_proto, 11, name); // GraphProto.input (field 11)
            }
            GraphOp::Unary { input, op } => {
                let inp = node_name(graph, input.0 as usize);
                match op {
                    UnaryOp::Scale(scalar) => {
                        // Scale = multiply by constant
                        let s: f64 = (*scalar).into();
                        let scale_name = format!("{out_name}_scale");
                        encode_initializer(&mut graph_proto, &scale_name, &[s]);
                        initializer_names.push(scale_name.clone());
                        encode_node(
                            &mut graph_proto,
                            "Mul",
                            &[&inp, &scale_name],
                            &[&out_name],
                            &out_name,
                        );
                    }
                    UnaryOp::Offset(scalar) => {
                        let o: f64 = (*scalar).into();
                        let off_name = format!("{out_name}_offset");
                        encode_initializer(&mut graph_proto, &off_name, &[o]);
                        initializer_names.push(off_name.clone());
                        encode_node(
                            &mut graph_proto,
                            "Add",
                            &[&inp, &off_name],
                            &[&out_name],
                            &out_name,
                        );
                    }
                    _ => {
                        encode_node(
                            &mut graph_proto,
                            unary_op_name(*op),
                            &[&inp],
                            &[&out_name],
                            &out_name,
                        );
                    }
                }
            }
            GraphOp::Binary { lhs, rhs, op } => {
                let l = node_name(graph, lhs.0 as usize);
                let r = node_name(graph, rhs.0 as usize);
                encode_node(
                    &mut graph_proto,
                    binary_op_name(*op),
                    &[&l, &r],
                    &[&out_name],
                    &out_name,
                );
            }
            GraphOp::Reduce { input, op } => {
                let inp = node_name(graph, input.0 as usize);
                encode_node(
                    &mut graph_proto,
                    reduce_op_name(*op),
                    &[&inp],
                    &[&out_name],
                    &out_name,
                );
            }
            GraphOp::Scan { input, .. } => {
                let inp = node_name(graph, input.0 as usize);
                encode_node(&mut graph_proto, "CumSum", &[&inp], &[&out_name], &out_name);
            }
            GraphOp::Gemm { lhs, rhs, .. } => {
                let l = node_name(graph, lhs.0 as usize);
                let r = node_name(graph, rhs.0 as usize);
                encode_node(
                    &mut graph_proto,
                    "MatMul",
                    &[&l, &r],
                    &[&out_name],
                    &out_name,
                );
            }
            GraphOp::Gather { data, indices } => {
                let d = node_name(graph, data.0 as usize);
                let idx = node_name(graph, indices.0 as usize);
                encode_node(
                    &mut graph_proto,
                    "Gather",
                    &[&d, &idx],
                    &[&out_name],
                    &out_name,
                );
            }
            GraphOp::Reshape { input, .. } => {
                let inp = node_name(graph, input.0 as usize);
                encode_node(
                    &mut graph_proto,
                    "Reshape",
                    &[&inp],
                    &[&out_name],
                    &out_name,
                );
            }
            GraphOp::Concat(nodes) => {
                let inp_names: Vec<String> = nodes
                    .iter()
                    .map(|n| node_name(graph, n.0 as usize))
                    .collect();
                let inp_refs: Vec<&str> = inp_names.iter().map(|s| s.as_str()).collect();
                encode_node(
                    &mut graph_proto,
                    "Concat",
                    &inp_refs,
                    &[&out_name],
                    &out_name,
                );
            }
            GraphOp::Slice { input, .. } => {
                let inp = node_name(graph, input.0 as usize);
                encode_node(&mut graph_proto, "Slice", &[&inp], &[&out_name], &out_name);
            }
            GraphOp::Custom { input, name } => {
                let inp = node_name(graph, input.0 as usize);
                encode_node(&mut graph_proto, name, &[&inp], &[&out_name], &out_name);
            }
        }
    }

    // Graph output = last node
    if !graph.nodes.is_empty() {
        let last_name = node_name(graph, graph.nodes.len() - 1);
        encode_value_info(&mut graph_proto, 12, &last_name); // GraphProto.output (field 12)
    }

    // GraphProto.name (field 2)
    encode_field_string(&mut graph_proto, 2, model_name);

    // Wrap in ModelProto
    let mut model = Vec::new();
    encode_field_varint(&mut model, 1, 8); // ir_version = 8
    // opset_import (field 8) — OperatorSetIdProto
    let mut opset = Vec::new();
    encode_field_string(&mut opset, 1, ""); // domain = "" (default ONNX)
    encode_field_varint(&mut opset, 2, 18); // version = 18
    encode_field_bytes(&mut model, 8, &opset);
    encode_field_string(&mut model, 2, "any-compute"); // producer_name
    encode_field_string(&mut model, 3, "0.1.0"); // producer_version
    encode_field_string(&mut model, 5, model_name); // doc_string
    encode_field_bytes(&mut model, 7, &graph_proto); // ModelProto.graph (field 7)

    model
}

fn node_name(graph: &Graph, idx: usize) -> String {
    NodeId(idx as u32).label(graph)
}
