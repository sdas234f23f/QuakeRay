"""
Cross-checks that an HLSL shader produces the same shader interface and the same memory
layout as the GLSL shader it replaces, by compiling both to SPIR-V and comparing the
disassembly.

Both compilers are given the same job: a SPIR-V module for Vulkan 1.2. glslc (through
glslang) and dxc spell a few things differently, so those are normalized instead of being
compared as text:

  * matrices: an HLSL matrix is declared as the transpose of the GLSL one, GLSL matCxR
    becomes floatRxC, and dxc emits RowMajor where glslang emits ColMajor. A transposed
    declaration occupies the same bytes (same offsets, MatrixStride and ArrayStride, which
    are compared and not normalized), so the GLSL side is described as its transpose and the
    majorness is not compared. Describing the transpose is also what makes the checker pin
    the transposed declaration the port is required to use.
  * dxc wraps a structured buffer in a Block struct that holds the runtime array, and puts
    NonWritable on that member instead of on the variable. A read-only storage buffer is
    read-only in either spelling.
  * a GLSL `readonly uniform` block is NonWritable, an HLSL ConstantBuffer cannot express
    that flag. It has no effect on the descriptor, so it is ignored for uniform blocks.
  * glslang wraps a uniform block in a one-member struct, dxc puts the members of the block
    directly into the block. The wrapper is stepped over on both sides.
  * dxc reports an image of unknown depth as depth 2 where glslang says 0 (not depth). A real
    depth image is 1 on both sides, so an unknown depth is compared as not-depth.
  * dxc declares SPV_EXT_descriptor_indexing for the descriptor-indexing instructions it emits,
    while glslang leaves the extension out at vulkan1.2, where descriptor indexing is core and
    the capabilities it gates are declared by both compilers. An extension the target
    environment has promoted to core is therefore not compared; the capabilities are.
  * a scalar followed by a vec2: GLSL uses std430 alignment, dxc uses HLSL packing. This is
    a real difference and it is reported, not normalized. Fix it in HLSL with [[vk::offset]].

Compared properties: entry point and execution model, workgroup size, specialization
constants, descriptor set/binding assignment, resource kind and storage class (the kind of
descriptor the host binds: a uniform buffer, a storage buffer, or a sampler/image/TLAS), the
byte layout of every block (recursively, including nested structs and array strides),
input/output locations, the builtins the entry point uses, and the capabilities and extensions
the module declares. The capability comparison exists because the host loads the *HLSL* blob:
a capability there that the GLSL blob does not need (RayQueryKHR was the first) is a capability
the device is not required to enable, and the validation layer rejects the module with
VUID-VkShaderModuleCreateInfo-pCode-08740/08742. dxc has flags that change the capabilities it
generates, which is why the comparison cannot be left to a source review.

A header has no stage of its own and is therefore never compiled on its own, so a header is
pinned by a probe: a shader that instantiates what the header declares and touches every
member of it. The probe's HLSL lives in Probes/, out of the shader build, while its GLSL
half lives in GLSL/ like the GLSL half of any other pair.

Usage:
    python CheckShaderProperties.py [--rebuild] [shader ...]

With no arguments, every <name>.<stage>.hlsl next to this script and every
Probes/<name>.<stage>.hlsl is checked against GLSL/<name>.<stage>. Exits with a non-zero
status if a property does not match.
"""

import difflib
import os
import re
import subprocess
import sys


GLSL_FOLDER_PATH = "GLSL/"
HLSL_FOLDER_PATH = "HLSL/"
PROBES_FOLDER_PATH = "Probes/"
GLSL_EXTENSIONS = [".comp", ".vert", ".frag", ".rgen", ".rahit", ".rchit", ".rmiss"]
HLSL_SUFFIX = ".hlsl"
TEMP_FOLDER_PATH = "Build/"
ALLOW_LIST_FILE_NAME = "ShaderPropertiesAllowList.txt"

HLSL_PROFILES = {
    ".comp":    "cs_6_2",
    ".vert":    "vs_6_2",
    ".frag":    "ps_6_2",
    ".rgen":    "lib_6_3",
    ".rahit":   "lib_6_3",
    ".rchit":   "lib_6_3",
    ".rmiss":   "lib_6_3",
}

# The same extension allow-list GenerateShaders.py passes to dxc, so that the HLSL half is built
# with the flags the host build uses. Without it dxc declares SPV_KHR_ray_query/RayQueryKHR for
# every module that has an OpTypeAccelerationStructureKHR, even one that only calls OpTraceRayKHR,
# which is what the capability comparison below is there to catch.
SPIRV_EXTENSIONS = [
    "SPV_KHR_ray_tracing",
    "SPV_EXT_descriptor_indexing",
    "SPV_KHR_compute_shader_derivatives",
]

# Extensions the target environment (vulkan1.2) has promoted to core. Both compilers are free to
# declare or omit them, and the capabilities they gate are compared separately.
SPIRV_CORE_EXTENSIONS = [
    "SPV_EXT_descriptor_indexing",
]

TAB = "  "


def findTool(name, arguments):
    try:
        r = subprocess.run([name] + arguments, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    except FileNotFoundError:
        return None, name + " is not in PATH"
    if r.returncode != 0:
        return None, r.stdout
    return r.stdout, None


def addVulkanSdkToPath():
    sdkBin = os.path.join(os.environ.get("VULKAN_SDK", ""), "Bin")
    if os.path.isdir(sdkBin):
        os.environ["PATH"] = sdkBin + os.pathsep + os.environ["PATH"]


def getHLSLProfile(filename):
    for ext in GLSL_EXTENSIONS:
        if filename.endswith(ext + HLSL_SUFFIX):
            return ext
    return None


def readAllowList():
    allowList = []

    if not os.path.isfile(ALLOW_LIST_FILE_NAME):
        return allowList

    with open(ALLOW_LIST_FILE_NAME, "r") as f:
        for line in f:
            line = line.strip()
            if len(line) == 0 or line.startswith("#"):
                continue
            parts = line.split("=", 1)
            if len(parts) != 2:
                print("> Malformed line in " + ALLOW_LIST_FILE_NAME + ": " + line)
                continue
            allowList.append((parts[0].strip(), parts[1].strip()))

    return allowList


class Instruction:
    def __init__(self, resultId, opcode, operands):
        self.resultId = resultId
        self.opcode = opcode
        self.operands = operands


class Module:
    def __init__(self, text, transposeMatrices=False):
        self.transposeMatrices = transposeMatrices
        self.instructions = []
        self.resultInstruction = {}
        # decorations[id][name] = [args]
        self.decorations = {}
        self.memberDecorations = {}
        self.names = {}
        self.entryPoints = []
        self.capabilities = []
        self.extensions = []
        self.executionModes = {}

        for line in text.splitlines():
            line = line.strip()
            if not line.startswith("%") and not line.startswith("Op"):
                continue

            tokens = line.split()
            resultId = None

            if tokens[0].startswith("%"):
                if len(tokens) < 3 or tokens[1] != "=":
                    continue
                resultId = tokens[0]
                tokens = tokens[2:]

            instruction = Instruction(resultId, tokens[0], tokens[1:])
            self.instructions.append(instruction)

            if resultId is not None:
                self.resultInstruction[resultId] = instruction

            self.parseSpecial(instruction)

        self.buildUsedIds()

    def parseSpecial(self, instruction):
        op = instruction.opcode
        o = instruction.operands

        if op == "OpDecorate":
            self.decorations.setdefault(o[0], {})[o[1]] = o[2:]
        elif op == "OpMemberDecorate":
            self.memberDecorations.setdefault((o[0], int(o[1])), {})[o[2]] = o[3:]
        elif op == "OpName":
            self.names[o[0]] = o[1].strip('"')
        elif op == "OpEntryPoint":
            self.entryPoints.append((o[0], o[1], o[2].strip('"'), o[3:]))
        elif op == "OpCapability":
            self.capabilities.append(o[0])
        elif op == "OpExtension":
            self.extensions.append(o[0].strip('"'))
        elif op == "OpExecutionMode":
            self.executionModes.setdefault(o[0], []).append(o[1:])

    def buildUsedIds(self):
        # A resource is used if it is referenced from a function body. Both compilers drop
        # unused resources, but they do not have to agree on which ones those are, and the
        # difference is worth reporting.
        self.usedIds = set()
        isInFunction = False

        for instruction in self.instructions:
            if instruction.opcode == "OpFunction":
                isInFunction = True
            elif instruction.opcode == "OpFunctionEnd":
                isInFunction = False
            elif isInFunction:
                for operand in instruction.operands:
                    if operand.startswith("%"):
                        self.usedIds.add(operand)

    # -- type helpers ------------------------------------------------------------------

    def typeOf(self, id):
        instruction = self.resultInstruction.get(id)
        if instruction is None or not instruction.opcode.startswith("OpType"):
            return None, []
        return instruction.opcode, instruction.operands

    def constantValue(self, id):
        instruction = self.resultInstruction.get(id)
        if instruction is None or not instruction.opcode.startswith("OpConstant"):
            return id
        return instruction.operands[1]

    def describeType(self, id, depth=0):
        op, o = self.typeOf(id)

        if op is None:
            return id
        if op == "OpTypeFloat" or op == "OpTypeInt":
            return ("float" if op == "OpTypeFloat" else ("int" if o[1] == "1" else "uint")) + o[0]
        if op == "OpTypeBool":
            return "bool"
        if op == "OpTypeVector":
            return "v" + o[1] + "(" + self.describeType(o[0], depth) + ")"
        if op == "OpTypeMatrix":
            if self.transposeMatrices:
                # GLSL matCxR is C vectors of R components, HLSL floatRxC is R vectors of C
                # components. Describing the GLSL matrix transposed makes it directly
                # comparable with the HLSL spelling dxc emits for the ported declaration.
                vectorOp, vectorOperands = self.typeOf(o[0])
                if vectorOp == "OpTypeVector":
                    return "m" + vectorOperands[1] + "(v" + o[1] + "(" + \
                        self.describeType(vectorOperands[0], depth) + "))"
            return "m" + o[1] + "(" + self.describeType(o[0], depth) + ")"
        if op == "OpTypeArray":
            return "array[" + self.constantValue(o[1]) + "](" + self.describeType(o[0], depth) + ")"
        if op == "OpTypeRuntimeArray":
            return "array[](" + self.describeType(o[0], depth) + ")"
        if op == "OpTypeStruct":
            return "struct"
        if op == "OpTypePointer":
            return "ptr(" + self.describeType(o[1], depth) + ")"
        if op == "OpTypeImage":
            # dxc reports "depth unknown" (2) for a color image where glslang reports
            # "not depth" (0). A real depth image is reported as 1 by both compilers, so an
            # unknown depth is treated as not-depth instead of as a difference.
            dimension, depth, arrayed, ms, sampled, imageFormat = o[1:7]
            return "image:" + ":".join([dimension, "0" if depth == "2" else depth,
                                        arrayed, ms, sampled, imageFormat])
        if op == "OpTypeSampler":
            return "sampler"
        if op == "OpTypeSampledImage":
            return "sampledimage(" + self.describeType(o[0], depth) + ")"
        if op == "OpTypeAccelerationStructureKHR":
            return "accelerationstructure"
        if op == "OpTypeVoid":
            return "void"

        return op

    def arrayStride(self, id):
        return self.decorations.get(id, {}).get("ArrayStride", [None])[0]

    def isNonWritable(self, variableId, blockTypeId):
        if "NonWritable" in self.decorations.get(variableId, {}):
            return True
        return "NonWritable" in self.memberDecorations.get((blockTypeId, 0), {})

    # -- property extraction -----------------------------------------------------------

    def getDescribedBlocks(self):
        """Returns {property-path: [layout lines]} for blocks, keyed by where they are bound."""
        blocks = {}

        for instruction in self.instructions:
            if instruction.opcode != "OpVariable":
                continue

            storageClass = instruction.operands[1]
            if storageClass not in ["Uniform", "StorageBuffer", "PushConstant"]:
                continue
            if instruction.resultId not in self.usedIds:
                continue

            blockTypeId = instruction.operands[0]
            blockTypeId = self.pointee(blockTypeId)
            if self.typeOf(blockTypeId)[0] != "OpTypeStruct":
                continue

            path = self.bindingPath(instruction, storageClass)
            blocks["block " + path] = self.describeBlock(self.unwrapBlock(blockTypeId))

        return blocks

    def unwrapBlock(self, typeId):
        """Steps over the wrappers that spell one memory layout in two different shapes.

        glslang emits a uniform block as a struct holding a single struct member at offset
        0, dxc puts the members of the block directly into the block. Stepping over the
        wrapper on both sides makes the two spellings comparable.

        A single-instance storage block (``buffer B { T x; }``) has no HLSL spelling: a struct
        member of a block cannot be addressed in HLSL, so the port is ``StructuredBuffer<T>``
        (read) or ``RWStructuredBuffer<T>`` (written), and dxc describes that as an array of
        the struct. Stepping over that array compares the layout of the element, which is
        what the host binds for both spellings; the offsets of every member and the stride
        between elements stay under check.
        """
        while True:
            op, members = self.typeOf(typeId)
            if op != "OpTypeStruct" or len(members) != 1:
                return typeId
            if self.memberDecorations.get((typeId, 0), {}).get("Offset", ["?"])[0] != "0":
                return typeId

            memberOp, memberOperands = self.typeOf(members[0])
            if memberOp == "OpTypeStruct":
                typeId = members[0]
                continue
            if memberOp in ["OpTypeRuntimeArray", "OpTypeArray"]:
                elementId = memberOperands[0]
                if self.typeOf(elementId)[0] == "OpTypeStruct":
                    typeId = elementId
                    continue
            return typeId

    def pointee(self, id):
        op, o = self.typeOf(id)
        while op in ["OpTypePointer", "OpTypeArray", "OpTypeRuntimeArray"]:
            # OpTypePointer is <storage class> <pointee type>, the arrays are <element type> <length>.
            id = o[1] if op == "OpTypePointer" else o[0]
            op, o = self.typeOf(id)
        return id

    def describeBlock(self, typeId, depth=0, visited=None):
        visited = visited or set()
        if typeId in visited or depth > 4:
            return [TAB * depth + "... (recursion)"]

        visited = visited | {typeId}
        op, members = self.typeOf(typeId)
        lines = []

        for index, memberId in enumerate(members):
            decor = self.memberDecorations.get((typeId, index), {})
            offset = decor.get("Offset", ["?"])[0]

            description = TAB * depth + "[%d] offset %s %s" % (index, offset, self.describeType(memberId))

            # ArrayStride is decorated on the array type itself, not on its element type: a plain
            # array member therefore carries its stride here, where a lookup on the element would
            # find nothing and silently compare no stride at all.
            stride = self.arrayStride(memberId)
            if stride is not None:
                description += " stride " + stride
            if "MatrixStride" in decor:
                description += " matrixstride " + decor["MatrixStride"][0]

            lines.append(description)

            nested = self.nestedStruct(memberId)
            if nested is not None:
                lines += self.describeBlock(self.unwrapBlock(nested), depth + 1, visited)

        return lines

    def elementType(self, id):
        op, o = self.typeOf(id)
        if op in ["OpTypeArray", "OpTypeRuntimeArray"]:
            return o[0]
        return id

    def nestedStruct(self, id):
        op, o = self.typeOf(id)
        if op in ["OpTypeArray", "OpTypeRuntimeArray"]:
            return self.nestedStruct(o[0])
        if op == "OpTypeStruct":
            return id
        return None

    def bindingPath(self, instruction, storageClass):
        decor = self.decorations.get(instruction.resultId, {})
        if storageClass == "PushConstant":
            return "pushconstant"
        return "set " + decor.get("DescriptorSet", ["?"])[0] + " binding " + decor.get("Binding", ["?"])[0]

    def getDescribedDescriptors(self):
        """Returns {property-path: description} for the descriptors, keyed by their binding.

        Two properties are reported per descriptor: the pointee kind under
        "descriptor <set/binding>", and the storage class of the variable under
        "descriptor <set/binding>: storage class".
        """
        descriptors = {}

        for instruction in self.instructions:
            if instruction.opcode != "OpVariable":
                continue

            storageClass = instruction.operands[1]
            if storageClass not in ["UniformConstant", "Uniform", "StorageBuffer"]:
                continue
            if instruction.resultId not in self.usedIds:
                continue

            path = self.bindingPath(instruction, storageClass)
            pointeeId = self.pointee(instruction.operands[0])
            description = self.describeType(pointeeId)
            decor = self.decorations.get(instruction.resultId, {})

            # GLSL `readonly uniform` gets NonWritable, an HLSL ConstantBuffer cannot
            # express that flag, and for a uniform block it does not affect the descriptor.
            if storageClass != "Uniform" and self.isNonWritable(instruction.resultId, pointeeId):
                description += " nonwritable"

            descriptors[path] = description

            # The storage class is the kind of descriptor the host binds: a Uniform block is
            # a VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, a StorageBuffer is a ..._STORAGE_BUFFER,
            # and a UniformConstant is a sampler, an image or a TLAS. It needs a property of
            # its own, because the pointee kind above cannot separate an HLSL
            # StructuredBuffer from the GLSL uniform block it replaces: dxc spells the
            # former as a struct holding a runtime array and glslang wraps the latter in a
            # struct too, so both sides describe as "struct". The storage class separates
            # them, and reporting it separately names the resource and lets a deliberate
            # difference be allow-listed on its own.
            descriptors[path + ": storage class"] = storageClass

        return descriptors

    def getEntryPointProperty(self):
        return ", ".join([model + " " + name for model, funcId, name, iface in self.entryPoints])

    def getWorkgroupSizeProperty(self):
        result = []
        for funcId, modes in self.executionModes.items():
            for mode in modes:
                if mode[0] == "LocalSize":
                    result.append(" ".join(mode[1:]))
        return ", ".join(sorted(result))

    def getSpecializationConstants(self):
        constants = {}

        for id, decor in self.decorations.items():
            if "SpecId" not in decor:
                continue
            instruction = self.resultInstruction.get(id)
            typeName = self.describeType(instruction.operands[0]) if instruction else "?"
            constants[decor["SpecId"][0]] = typeName + " = " + (instruction.operands[1] if instruction else "?")

        return constants

    def getInterfaceProperties(self):
        properties = {}

        for model, funcId, name, interface in self.entryPoints:
            for id in interface:
                decor = self.decorations.get(id, {})
                instruction = self.resultInstruction.get(id)
                pointeeId = self.pointee(instruction.operands[0]) if instruction else None

                # Builtins are the only interface of a compute or an RT shader, and their
                # names are language independent in SPIR-V (SV_DispatchThreadID becomes
                # GlobalInvocationId for both compilers).
                if "BuiltIn" in decor:
                    properties["builtin " + decor["BuiltIn"][0]] = self.builtinType(pointeeId)
                    continue

                # A block whose members carry BuiltIn decorations is compared through those
                # members: glslang puts the gl_PerVertex block's Position on a *member* of a
                # block variable, dxc puts it on a plain output variable, and the property list
                # has to see the same builtin on both sides. Only the members the module
                # accesses are expanded -- the block type always carries PointSize,
                # ClipDistance and CullDistance, which a shader writing only gl_Position
                # neither reads nor writes.
                if self.typeOf(pointeeId)[0] == "OpTypeStruct":
                    expanded = self.expandBlockBuiltins(id, pointeeId)
                    if expanded:
                        properties.update(expanded)
                        continue

                if model not in ["Vertex", "Fragment"] or "Location" not in decor:
                    continue

                storageClass = instruction.operands[1] if instruction else "?"
                if storageClass not in ["Input", "Output"]:
                    continue

                key = storageClass.lower() + " location " + decor["Location"][0]
                properties[key] = self.describeType(pointeeId)

        return properties

    def builtinType(self, pointeeId):
        """The builtin's type with the integer sign normalized away.

        glslang spells the vertex/instance index builtins as signed and dxc as unsigned
        (SV_VertexID/SV_InstanceID are uint in HLSL while gl_VertexIndex/gl_InstanceIndex are
        int in GLSL); SPIR-V fixes the width but not the sign, the host cannot see either, and
        the DXIL path rejects the signed spelling outright. The comparison therefore keeps the
        width and drops the sign, so a genuinely different width still shows up.
        """
        description = self.describeType(pointeeId)
        if description.startswith("uint"):
            return "int" + description[len("uint"):]
        if "(uint" in description:
            return description.replace("(uint", "(int")
        return description

    def expandBlockBuiltins(self, blockVariableId, blockTypeId):
        """{property name: type} for the builtin members of a block the module accesses.

        The per-vertex block of a GLSL vertex shader declares Position, PointSize,
        ClipDistance and CullDistance; only the members the shader touches are compared, so a
        port that writes just the position is not asked to declare the rest.
        """
        accessed = set()
        for instruction in self.instructions:
            if instruction.opcode != "OpAccessChain":
                continue
            if len(instruction.operands) < 3 or instruction.operands[1] != blockVariableId:
                continue
            accessed.add(self.constantValue(instruction.operands[2]))

        expanded = {}
        op, members = self.typeOf(blockTypeId)
        if op != "OpTypeStruct":
            return expanded
        for index, memberId in enumerate(members):
            decor = self.memberDecorations.get((blockTypeId, index), {})
            if "BuiltIn" not in decor or str(index) not in accessed:
                continue
            expanded["builtin " + decor["BuiltIn"][0]] = self.describeType(memberId)
        return expanded

    def getPropertySet(self):
        properties = {
            "entry point": self.getEntryPointProperty(),
            "workgroup size": self.getWorkgroupSizeProperty(),
        }

        for specId, value in sorted(self.getSpecializationConstants().items(), key=lambda i: int(i[0])):
            properties["constant_id " + specId] = value

        for path, description in self.getDescribedDescriptors().items():
            properties["descriptor " + path] = description

        for path, lines in self.getDescribedBlocks().items():
            properties[path] = "\n" + "\n".join(lines)

        for key, value in self.getInterfaceProperties().items():
            properties[key] = value

        # The capabilities and extensions the module declares. They are compared like any other
        # property, so the allow-list and the [MISMATCH] report cover them too. A capability is
        # named in the property path ("capability RayQueryKHR"), which is what a validator error
        # or an allow-list line refers to.
        for capability in self.capabilities:
            properties["capability " + capability] = "declared"
        for extension in self.extensions:
            if extension in SPIRV_CORE_EXTENSIONS:
                continue
            properties["extension " + extension] = "declared"

        return properties


def compileShader(sourcePath, outputPath, isHLSL):
    if isHLSL:
        profile = getHLSLProfile(sourcePath)

        if profile is None:
            return False, sourcePath + " does not name a known shader stage"

        command = ["dxc", "-spirv", "-T", HLSL_PROFILES[profile], "-fspv-target-env=vulkan1.2"] + \
            ["-fspv-extension=" + ext for ext in SPIRV_EXTENSIONS] + \
            ["-I", ".", "-I", "LPM", "-I", "CAS", "-I", "../Generated/", sourcePath, "-Fo", outputPath]
    else:
        # -O on the golden side, so that both compilers are compared at their own full
        # optimization: dxc optimizes by default and drops the resources of dead code, while
        # plain glslc keeps them, which used to surface as a descriptor difference where only
        # liveness differed (the RFL/Q2 raygen configurations were the first to show it).
        #
        # The two header subfolders are on the include path as they are in the host build
        # (GenerateShaders.py adds every subfolder): CmPrepareFinal.comp includes the AMD
        # header as a bare "ffx_a.h", which lives in LPM/ and CAS/. LPM first, the copy its
        # ported twin names.
        command = ["glslc", "-O", "--target-env=vulkan1.2", "-I", ".", "-I", "LPM", "-I", "CAS", "-I", "../Generated/",
            sourcePath, "-o", outputPath]

    try:
        r = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    except FileNotFoundError:
        return False, command[0] + " is not in PATH"

    if r.returncode != 0:
        return False, r.stdout

    return True, None


def disassemble(spvPath, txtPath):
    try:
        r = subprocess.run(["spirv-dis", "--no-color", spvPath, "-o", txtPath],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    except FileNotFoundError:
        return False, "spirv-dis is not in PATH"

    if r.returncode != 0:
        return False, r.stdout

    return True, None


def compareProperties(name, glslProperties, hlslProperties, allowList):
    """Returns (messages, mismatchCount). Ignored differences are reported as such."""
    messages = []
    mismatches = 0
    keys = []

    for key in glslProperties:
        if key not in keys:
            keys.append(key)
    for key in hlslProperties:
        if key not in keys:
            keys.append(key)

    for key in keys:
        glslValue = glslProperties.get(key, None)
        hlslValue = hlslProperties.get(key, None)

        if glslValue == hlslValue:
            continue

        path = name + " " + key
        allowed = None
        for fragment, reason in allowList:
            if fragment in path:
                allowed = reason
                break

        if allowed is not None:
            messages.append(TAB + "[allowed] " + key + " -- " + allowed)
            continue

        mismatches += 1
        messages.append(TAB + "[MISMATCH] " + key)

        glslText = "" if glslValue is None else str(glslValue)
        hlslText = "" if hlslValue is None else str(hlslValue)

        if "\n" in glslText or "\n" in hlslText:
            # A block mismatch is easier to read as a diff than as two full dumps.
            diff = difflib.unified_diff(glslText.split("\n"), hlslText.split("\n"),
                "GLSL", "HLSL", n=1, lineterm="")
            for line in diff:
                messages.append(TAB + TAB + line)
        else:
            messages.append(TAB + TAB + "GLSL: " + glslText)
            messages.append(TAB + TAB + "HLSL: " + hlslText)

    return messages, mismatches


def checkPair(name, allowList):
    # A probe carries its folder in the name, its GLSL original does not.
    baseName = os.path.basename(name)
    glslPath = GLSL_FOLDER_PATH + baseName
    hlslPath = name + HLSL_SUFFIX

    if not os.path.isfile(glslPath):
        return 0, [TAB + "skipped: " + glslPath + " does not exist"], True

    glslSpvPath = TEMP_FOLDER_PATH + baseName + ".glsl.spv"
    hlslSpvPath = TEMP_FOLDER_PATH + baseName + ".hlsl.spv"

    success, compilerOutput = compileShader(glslPath, glslSpvPath, isHLSL=False)
    if not success:
        return 1, [TAB + "glslc failed", compilerOutput], False

    success, compilerOutput = compileShader(hlslPath, hlslSpvPath, isHLSL=True)
    if not success:
        return 1, [TAB + "dxc failed", compilerOutput], False

    glslTxtPath = TEMP_FOLDER_PATH + baseName + ".glsl.spv.txt"
    hlslTxtPath = TEMP_FOLDER_PATH + baseName + ".hlsl.spv.txt"

    for spvPath, txtPath in [(glslSpvPath, glslTxtPath), (hlslSpvPath, hlslTxtPath)]:
        if not disassemble(spvPath, txtPath)[0]:
            return 1, [TAB + "spirv-dis failed"], False

    with open(glslTxtPath, "r", encoding="utf-8") as f:
        glslModule = Module(f.read(), transposeMatrices=True)
    with open(hlslTxtPath, "r", encoding="utf-8") as f:
        hlslModule = Module(f.read())

    messages, mismatches = compareProperties(name, glslModule.getPropertySet(),
        hlslModule.getPropertySet(), allowList)

    return mismatches, messages, False


def resolveProbeName(name):
    if os.path.isfile(name + HLSL_SUFFIX):
        return name
    if os.path.isfile(HLSL_FOLDER_PATH + name + HLSL_SUFFIX):
        return HLSL_FOLDER_PATH + name
    if os.path.isfile(PROBES_FOLDER_PATH + name + HLSL_SUFFIX):
        return PROBES_FOLDER_PATH + name
    return name


def getPairs(arguments):
    if len(arguments) > 0:
        # Arguments may name the HLSL file or the shader itself, e.g. EfWaves.comp.hlsl or
        # EfWaves.comp. A probe can be named without its folder.
        names = [a[:-len(HLSL_SUFFIX)] if a.endswith(HLSL_SUFFIX) else a for a in arguments]
        return [resolveProbeName(name) for name in names]

    pairs = []
    for folder in ["./", HLSL_FOLDER_PATH, PROBES_FOLDER_PATH]:
        if not os.path.isdir(folder):
            continue
        for entry in sorted(os.listdir(folder)):
            if entry.endswith(HLSL_SUFFIX) and getHLSLProfile(entry) is not None:
                pairs.append(folder + entry[:-len(HLSL_SUFFIX)])

    return pairs


def main():
    addVulkanSdkToPath()

    if not os.path.isdir(TEMP_FOLDER_PATH):
        os.makedirs(TEMP_FOLDER_PATH)

    arguments = [a for a in sys.argv[1:] if not a.startswith("-")]
    pairs = getPairs(arguments)

    if len(pairs) == 0:
        print("> No HLSL shaders to check. Nothing to do.")
        return 0

    allowList = readAllowList()
    mismatches = 0
    skipped = 0

    for pair in pairs:
        print("=== " + pair)
        pairMismatches, messages, pairSkipped = checkPair(pair, allowList)
        mismatches += pairMismatches

        for message in messages:
            print(message)

        if pairSkipped:
            skipped += 1
        elif pairMismatches == 0:
            print(TAB + "all properties match")

    if mismatches > 0:
        print("")
        print("> " + str(mismatches) + " propertie(s) do not match. Fix them in HLSL, or add a")
        print("> justified line to " + ALLOW_LIST_FILE_NAME + ".")
        return 1

    if skipped > 0:
        print("")
        print("> " + str(skipped) + " pair(s) were skipped: nothing was checked for them. A pair")
        print("> is named after its HLSL file, e.g. ShaderCommon.probe.comp or EfWaves.comp.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
