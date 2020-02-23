#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict, OrderedDict
import io
import os
import sys
import datetime

import numpy as np  # type: ignore

from onnx import defs, FunctionProto, helper, OperatorStatus
from onnx.defs import OpSchema, ONNX_DOMAIN, ONNX_ML_DOMAIN
from onnx.backend.test.case import collect_snippets
from onnx.backend.sample.ops import collect_sample_implementations
from typing import Any, Text, Sequence, Dict, List, Type, Set, Tuple


#controls on ONNF code gen
#specify attr default value 
special_attr_defaults = dict([
       # ("AveragePool.kernel_shape", ('ints', '{}')),
       # ("MaxPool.kernel_shape", ('ints', '{}')),
       # ("Cast.to", ('int', '0')),
       # ("Concat.axis", ('int', '0')),
       # ("Conv.group", ('int', '1')),
       # ("Unsqueeze.axes", ('ints', '{}')),
       # ("RNN.activation_alpha", ('floats', '{}')),
       # ("RNN.activation_beta", ('floats', '{}')),
        ])

#specify the function name in src/builder/frontend_dialect_transformer.cpp
#the reason for Conv and MaPool is to handled optional arguments
special_op_handler = dict([
        ("Conv", "ImportNodeConv"),
        ("MaxPool", "ImportNodeMaxPool"),
        ("BatchNormalization", "ImportNodeBatchNormalization"),
        ("Gemm", "ImportNodeGemm"),
        ("Pad", "ImportNodePad"),
        #("Transpose", "ImportNodeTranspose")
        ])

#add an Op in this list if ShapeInterference is defined for this Op
OpsWithShapeInference = ['Exp', 'Tanh', 'Sinh', 'Cosh', 'Sigmoid', 'Relu',
                   'Add', 'Mul', 'Div', 'Sub', 'And', 'Or', 'Xor',
                   'Sum', 'Max', 'Min', 'MatMul', 'Gemm', 'LeakyRelu',
                   'Elu', 'Selu', 'HardSigmoid', 'Reshape', 'Reciprocal',
                   'Identity', 'Cos', 'Log', 'Transpose', 'Softmax',
                   'ReduceMax', 'ReduceMin', 'ReduceProd', 'ReduceSum',
                   'Softplus', 'Softsign', 'Sqrt', 'Unsqueeze', 'Sign']

OpsWithCanonicalizer = ['Add', 'Identity', 'ReduceL1', 'ReduceL2', 'ReduceLogSum',
               'ReduceLogSumExp', 'ReduceSumSquare']

#add an Op in this list if the Op needs result type deduction which is required
#when writing declarative rewriting rules. Deduced type is always
#an UnrankedTensorType whose element type is the same as the first operand's
#element type.
#currenlty, there are only two build methods generated:
# - one with operands and attributes having a separate parameter, and
# - one with operands and attributes having aggregated parameters.
custom_builder_ops_list = ['Abs', 'Mul', 'Exp', 'ReduceSum', 'ReduceSumSquare']

manual_code_in_op_def = dict([
      ('DummyExample', '  let extraClassDeclaration = [{ \n'+
                    '    static StringRef getPermAttrName() { return "perm"; }\n'+
                    '    }];\n')
      ])


SNIPPETS = collect_snippets()
SAMPLE_IMPLEMENTATIONS = collect_sample_implementations()
ONNX_ML = not bool(os.getenv('ONNX_ML') == '0')

ONNX_ML = False
print("ONNX_ML", ONNX_ML)


if ONNX_ML:
    ext = '-ml.md'
else:
    ext = '.md'


def should_render_domain(domain):  # type: (Text) -> bool
    if domain == ONNX_ML_DOMAIN and not ONNX_ML:
        return False
    elif ONNX_ML and domain != ONNX_ML_DOMAIN:
        return False
    return True

def display_attr_type(v):  # type: (OpSchema.AttrType) -> Text
    assert isinstance(v, OpSchema.AttrType)
    s = Text(v)
    s = s[s.rfind('.') + 1:].lower()
    if s[-1] == 's':
        s = 'list of ' + s
    return s

def get_unique_output_name(schema, name):
    for input in schema.inputs :
        if input.name == name :
            return 'out_'+name
    return name

def convert_type(tstr) :
    tfrom = np.array(['bool', 'int8', 'int16', 'int32', 'int64',
            'unkown', 'float16', 'float', 'double'])
    tto =np.array(['I1', 'I8', 'I16', 'I32', 'I64',
         'BF16', 'F16', 'F32', 'F64'])
    index = -1
    for i in range(len(tfrom)) :
        if tfrom[i] in tstr :
            index = i
            break
    if index == -1 :
        print("error", tstr)
        return ''
    else :
        return tto[i]

def get_allowed_elem_types(schema, input):
    allowed_types_str = None
    return allowed_types_str
    # TODO: enable type constraints.
    # if input.typeStr :
    #     tstr = input.typeStr
    # else :
    #     return allwedTypeStr
    # if schema.type_constraints:
    #     for type_constraint in schema.type_constraints:
    #         if type_constraint.type_param_str != tstr :
    #             continue
    #         allowedTypes = type_constraint.allowed_type_strs
    #         allowedTypeStr=''
    #         if (len(allowedTypes) > 0):
    #             t = convert_type(allowedTypes[0])
    #             if t == '' :
    #                 return ''
    #             allowedTypeStr += t
    #         for allowedType in allowedTypes[1:]:
    #             t = convert_type(allowedType)
    #             if t == '' :
    #                 return ''
    #             if  not t in allowedTypeStr :
    #                 allowedTypeStr += ', '+t
    #
    #         return allowedTypeStr
    #
    # return allowedTypeStr

def inc_indent(indent = None):
    return "" if indent is None else indent + ' ' * 2

def dec_indent(indent):
    return indent[:-2]

def join_args(args):
    return ", ".join(args)

def gen_schema(schema) :
    indent = inc_indent()
    s = 'def ONNX{0}Op:ONNX_Op<"{0}",\n'.format(schema.name)

    # Generate decl for op traits.
    traits = ["NoSideEffect"]
    if schema.name in OpsWithShapeInference:
        traits.append("DeclareOpInterfaceMethods<ShapeInferenceOpInterface>")
    s += inc_indent(indent) + '[{}]> {{\n'.format(join_args(traits))

    # Generate decl for canonicalizer.
    indent = inc_indent(indent)
    if schema.name in OpsWithCanonicalizer:
        s += indent + 'let hasCanonicalizer = 1;\n'

    # Generate decl for summary.
    s += indent + 'let summary = "ONNX {} operation";\n'.format(schema.name)

    # Generate description.
    s += indent + 'let description = [{\n'
    if schema.doc:
        lines = schema.doc.lstrip().splitlines()
        for line in lines:
            escaped_line = line.replace('"', '\\"')\
                               .replace('}]', '\\}\\]')
            s += indent + '"{}"\n'.format(escaped_line)
    s += indent+'}];\n'

    # Generate ins (consisting of operands and attributes).
    ins = get_operand_ins(schema)
    ins.update(get_attr_ins(schema))
    ins_strs = ["{1}:${0}".format(*i) for i in ins.items()]
    s += indent + 'let arguments = (ins {});'.format((',\n' + inc_indent(indent)).join(ins_strs))

    # outs
    s+= '\n'+indent+'let results = (outs '
    if schema.outputs:
        for output in schema.outputs:
            if output != schema.outputs[0] :
                s+= ',\n           '
            #need to interpret output.typeStr
            etypes=get_allowed_elem_types(schema, output)
            if etypes is None:
                s+= 'AnyTypeOf<[AnyMemRef, AnyTensor]>'
            else:
                s+= 'TensorOf<['+etypes+']>'
            s += ':$'+get_unique_output_name(schema, output.name)
    s+= ');\n'

    #TODO: any better way to do this.
    def get_attr_type_for_builder(attr_type):
        if 'I64Attr' in attr_type :
            mytype = 'IntegerAttr'
        elif 'F32Attr' in attr_type :
            mytype = 'FloatAttr'
        elif 'I64ArrayAttr' in attr_type or 'F32ArrayAttr' in attr_type:
            mytype = 'ArrayAttr'
        elif 'StrAttr' in attr_type :
            mytype = 'StringAttr'
        elif 'strings' in attr_type :
            mytype = 'ArrayAttr'
        else :
            mytype ='Attribute'
        return mytype

    def get_op_type_for_builder(op_type):
        if op_type.startswith('Variadic'):
            mytype = 'ValueRange'
        else:
            mytype = 'Value'
        return mytype

    # add custom builders
    # use element type of the first operand to construct an UnrankedTensorType for the output.
    if schema.name in custom_builder_ops_list:
        if len(ins) == 0:
            print("warning: not generate custom build methods for " + schema.name + " since it does not have operands.")
        else:
            if get_op_type_for_builder(list(ins.items())[0][1]) == 'ValueRange':
                first_operand = list(ins.items())[0][0]+'[0]'
            else:
                first_operand = list(ins.items())[0][0]

            s += indent+'let builders = [\n'

            # custom builders with operands and attributes having a seperate parameter.
            # E.g. OpBuilder<"Builder *builder, OperationState &state, Value X, Value, Y, Attribute A", [{}]>
            s += indent*2+'OpBuilder<"Builder *builder, OperationState &state'
            for arg_name, arg_type in get_operand_ins(schema).items():
                s += ', '+get_op_type_for_builder(arg_type)+' '+arg_name
            for attr_name, attr_type in get_attr_ins(schema).items():
                s += ', '+get_attr_type_for_builder(attr_type)+' '+attr_name
            s += '", [{\n'
            s += indent*3+'auto elementType = '+first_operand+'.getType().cast<TensorType>().getElementType();\n'
            s += indent*3+'build(builder, state, UnrankedTensorType::get(elementType)'
            for arg_name, _ in ins.items():
                s += ', '+arg_name
            s += ');\n'
            s += indent*2+'}]>,\n'

            # custom builders with all operands and attributes having aggregate parameters.
            # E.g. OpBuilder<"Builder *builder, OperationState &state, ValueRange operands, ArrayRef<NamedAttribute> attributes", [{}]>'
            s += indent*2+'OpBuilder<"Builder *builder, OperationState &state, ValueRange operands, ArrayRef<NamedAttribute> attributes", [{\n'
            s += indent*3+'auto elementType = operands[0].getType().cast<TensorType>().getElementType();\n'
            s += indent*3+'std::vector<mlir::Type> outputTypes;\n'
            s += indent*3+'outputTypes.emplace_back(UnrankedTensorType::get(elementType));\n'
            s += indent*3+'build(builder, state, outputTypes, operands, attributes);\n'
            s += indent*2+'}]>'

            s += '\n'+indent+'];\n'

    # add special code
    if schema.name in manual_code_in_op_def :
        s += manual_code_in_op_def[schema.name]

    s += '}\n\n'

    return s

"""
special cases:
* Split: attr split default value: sizeof(output1) namely 1
* Conv: attr dilations default value is {num_dim of first input - 2, 1}
* Conv: attr kernel_shape type is ints
* Transpose: attr perm default value is {} empty int list
"""

def gen_code(schema, fefile):
    indent = inc_indent()
    s = ''
    fefile.write(indent + 'if (opName == "'+schema.name+'")\n')
    op_type_str = 'mlir::ONNX{}Op'.format(schema.name)

    expected_num_operands = len(schema.inputs)
    expected_num_results = len(schema.outputs)
    for input in schema.inputs:
        if OpSchema.FormalParameterOption.Variadic == input.option:
            expected_num_operands = -1
    for output in schema.outputs:
        if OpSchema.FormalParameterOption.Variadic == output.option:
            expected_num_results = -1

    handler_func = special_op_handler.get(schema.name, "buildOperation<{}>".format(op_type_str))
    inner_indent = indent + ' ' * 2

    args = ["node"]
    # Special handlers currently require expected num operands/results to be specified.
    # TODO: remove special handlers.
    if expected_num_operands == -1 or expected_num_results == -1 or "buildOperation" not in handler_func:
        args.append("/* expected_num_operands = */ {}".format(expected_num_operands))
        args.append('/* expected_num_results = */ {}'.format(expected_num_results))
    fefile.write(inner_indent + "return {}({});\n".format(handler_func, ", ".join(args)))

def get_operand_ins(schema):
    if not schema.inputs:
        return OrderedDict()

    def any_type_of(types):
        assert isinstance(types, list)
        if len(types) == 1:
            return types[0]
        else:
            return "AnyTypeOf<[{}]>".format(", ".join(types))

    name_to_types = OrderedDict()
    for input in schema.inputs:
        elem_types = get_allowed_elem_types(schema, input)

        if elem_types is None:
            types = ["AnyMemRef", "AnyTensor"]
        else:
            types = ["TensorOf<[{}]>", "MemRefOf<[{}]>"]
            types = list(map(lambda x: x.format(elem_types), types))

        if OpSchema.FormalParameterOption.Optional == input.option:
            #TODO : handle optional
            print("warning: optional input for"+schema.name+' '+input.name)
        elif OpSchema.FormalParameterOption.Variadic == input.option:
            if input.isHomogeneous:
                types = ["Variadic<{}>".format(any_type_of(types))]
            else:
                #TODO handle(variadic, heterogeneous) "
                print("warning: (variadic, heterogeneous) for"+schema.name+' '+input.name)
        name_to_types[input.name] = any_type_of(types)
    return name_to_types

def get_attr_ins(schema):
    def get_attr_type_basic(attr_type):
        if attr_type == 'int':
            mlir_attr_type = 'I64Attr'
        elif attr_type == 'float':
            mlir_attr_type = 'F32Attr'
        elif attr_type == 'ints':
            mlir_attr_type = 'I64ArrayAttr'
        elif attr_type == 'floats':
            mlir_attr_type = 'F32ArrayAttr'
        elif attr_type == "string":
            mlir_attr_type = 'StrAttr'
        elif attr_type == "strings":
            mlir_attr_type = 'StrArrayAttr'
        else:
            mlir_attr_type = 'AnyAttr'
        #TODO: tensor and sparse tensor
        return mlir_attr_type

    def get_attr_type_optional(attr_type):
        return 'OptionalAttr<{}>'.format(get_attr_type_basic(attr_type))

    def get_attr_type_with_default(attr_type, attr_default):
        return 'DefaultValuedAttr<{}, "{}">'.format(
            get_attr_type_basic(attr_type), attr_default)

    if not schema.attributes:
        return OrderedDict()

    name_to_type = OrderedDict()
    for _, attr in sorted(schema.attributes.items()):
        qualified_attr_name = "{}.{}".format(schema.name, attr.name)
        if qualified_attr_name in special_attr_defaults:
            name_to_type[attr.name] = get_attr_type_with_default(
                *special_attr_defaults[qualified_attr_name])

        # option holds either required or default value
        elif attr.required:
            s = Text(attr.type)
            attr_type_str = s[s.rfind('.') + 1:].lower()
            name_to_type[attr.name] = get_attr_type_basic(attr_type_str)
        elif attr.default_value.name:
            s = Text(attr.type)
            attr_type_str = s[s.rfind('.') + 1:].lower()
            default_value = helper.get_attribute_value(attr.default_value)

            def format_value(value):  # type: (Any) -> Text
                if isinstance(value, float):
                    formatted = str(np.round(value, 5))
                    # use default formatting, unless too long.
                    if (len(formatted) > 10):
                        formatted = str("({:e})".format(value))
                    return formatted
                elif isinstance(
                        value,
                        (bytes, bytearray)) and sys.version_info[0] == 3:
                    return str(value.decode('utf-8'))
                return str(value)

            if isinstance(default_value, list):
                default_value = [format_value(val) for val in default_value]
                attr_option_str = '{}'.format(default_value)
                attr_option_str = attr_option_str.replace('[', '{', 1)
                attr_option_str = attr_option_str.replace(']', '}', 1)
                if attr_type_str == 'strings':
                    attr_option_str = attr_option_str.replace("'", '\\"')
                else:
                    attr_option_str = attr_option_str.replace("'", '')
            else:
                default_value = format_value(default_value)
                attr_option_str = default_value
            name_to_type[attr.name] = get_attr_type_with_default(
                attr_type_str, attr_option_str)
        else:
            s = Text(attr.type)
            attr_type_str = s[s.rfind('.') + 1:].lower()
            name_to_type[attr.name] = get_attr_type_optional(attr_type_str)
    return name_to_type

def build_operator_schemas():
    # domain -> support level -> name -> [schema]
    index = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))  # type: Dict[Text, Dict[int, Dict[Text, List[OpSchema]]]]
    for schema in defs.get_all_schemas_with_history():
        index[schema.domain][int(schema.support_level)][schema.name].append(schema)

    # Preprocess the Operator Schemas
    # [(domain, [(support_level, [(schema name, current schema, all versions schemas)])])]
    operator_schemas = list()  # type: List[Tuple[Text, List[Tuple[int, List[Tuple[Text, OpSchema, List[OpSchema]]]]]]]
    exsting_ops = set()  # type: Set[Text]
    for domain, _supportmap in sorted(index.items()):
        if not should_render_domain(domain):
            continue

        processed_supportmap = list()
        for _support, _namemap in sorted(_supportmap.items()):
            processed_namemap = list()
            for n, unsorted_versions in sorted(_namemap.items()):
                versions = sorted(unsorted_versions, key=lambda s: s.since_version)
                schema = versions[-1]
                #print("check point 2", schema)
                if schema.name in exsting_ops:
                    continue
                exsting_ops.add(schema.name)
                processed_namemap.append((n, schema, versions))
            processed_supportmap.append((_support, processed_namemap))
        operator_schemas.append((domain, processed_supportmap))
    return operator_schemas

def main(args):  # type: (Type[Args]) -> None
    with io.open(args.output, 'w', newline='', encoding="utf-8") as fout:
        curr_utc_time = datetime.datetime.now(datetime.timezone.utc).strftime("%m/%d/%Y, %H:%M:%S")
        autogen_warning = ('//********************************************************\n'
                          '//   This file is generated on UTC-{}.\n'
                          '//   Do not modify this file directly.\n'
                          '//   This file is automatically generated via script.\n'
                          '//   Details can be found in doc/readonnxdefs.md .\n'
                          '//********************************************************\n\n')
        autogen_warning = autogen_warning.format(curr_utc_time)

        tdfile= io.open(args.tdfile, 'w', newline='') 
        tdfile.write(autogen_warning)

        fefile=io.open('op_build_table.inc', 'w', newline='')
        fefile.write(autogen_warning)

        for domain, supportmap in build_operator_schemas():
            for _, namemap in supportmap:
                for op_type, schema, versions in namemap:
                    gen_code(schema, fefile)
                    r = gen_schema(schema)
                    tdfile.write(r)
        fefile.close()


if __name__ == '__main__':
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    class Args(object):
        output = os.path.join(curr_dir, 'Operators' + ext)
        changelog = os.path.join(curr_dir, 'Changelog' + ext)
        tdfile = os.path.join(curr_dir, 'onnxop.inc')
    main(Args)
