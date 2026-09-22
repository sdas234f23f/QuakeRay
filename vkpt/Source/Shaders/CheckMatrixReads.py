#!/usr/bin/env python3
#
# Reads a matrix through a single index and compares it with the golden.
#
# The port keeps GLSL `matCxR` as HLSL `floatRxC`, so both languages show the shader the same
# logical matrix: HLSL `m(r, c)` equals GLSL `m(r, c)`. What differs is what one index means. In
# GLSL `m[i]` is column i, in HLSL it is row i, so the golden `m[i]` has to be read as
# `getColumn(m, i)` or as `transpose(m)[i]` on the HLSL side. A transcribed `m[i]` reads a row
# instead, and the two halves of a probe then compare different numbers while
# CheckShaderProperties.py, which compares interfaces and not values, stays green.
#
# Most such reads are invisible by accident, and this script reports only the ones that are not:
#
#   * `m[i].c` is the same element on both sides when `c` is the component of index i, `m[0].x`,
#     `m[1].y` and `m[2].z` are diagonal reads. Anything else, `m[0].y` included, is a finding.
#   * `m[a][b]` is the same element when a == b, and a finding otherwise. The golden `m[a][b]` is
#     `m[b][a]` in HLSL, so the HLSL spelling of it is `getColumn(m, a)[b]`.
#   * a local that was declared as `transpose(...)` of a matrix holds the columns in its rows, so a
#     row read of that local is the column read the golden wrote and is not reported.
#   * the left side of an assignment is not a read: `m[i] = v` is how a golden member that is filled
#     column by column is built, see the constructor rule in ShaderCommonHLSL.hlsli.
#
# Usage:
#
#   python CheckMatrixReads.py [file ...]
#
# Without arguments every header of the base and every probe half is checked. A finding is a
# candidate defect, not a proven one: it means that the value the shader reads is not the value the
# golden wrote, so either the read is wrong or the difference is intended and belongs in a comment.
#
# Known limit: locals are resolved by the nearest declaration of the same name that precedes the
# read, and by whether that declaration is a matrix or an array of matrices. A name that is a matrix
# in one function and an array in another is therefore reported only where the declaration says so.

import os
import re
import sys
import glob

SHADERS_FOLDER_PATH = "./"
PROBES_FOLDER_PATH = "Probes/"

MATRIX_TYPE = r"float[2-4]x[2-4]"
COMPONENTS = "xyzw"

# A declaration of any type, used to resolve a name to its nearest preceding declaration.
ANY_DECLARATION = re.compile(
    r"(?:^|[;{(,]|\))\s*(?:const\s+|static\s+|uniform\s+)*([A-Za-z_][\w:<>]*)\s+"
    r"([A-Za-z_]\w*)\s*(\[[^\]]*\])?\s*(?:=|;|,)")
# A declaration whose type is a matrix, used for the struct member registry.
MATRIX_DECLARATION = re.compile(r"\b(" + MATRIX_TYPE + r")\s+([A-Za-z_]\w*)\s*(\[[^\]]*\])?\s*(?:=|;|,)")
# A local that holds the transpose of a matrix, whose rows are the columns of that matrix.
TRANSPOSE_LOCAL = re.compile(r"\b(?:const\s+)?" + MATRIX_TYPE + r"\s+([A-Za-z_]\w*)\s*=\s*transpose\s*\(")
# A name followed by at least one index.
INDEXED_NAME = re.compile(r"(?P<dot>\.?)(?P<name>[A-Za-z_]\w*)(?P<subs>(?:\s*\[[^\]]*\])+)")
STRUCT_BODY = re.compile(r"\bstruct\s+\w+\s*\{(.*?)\n\s*\};", flags=re.S)
ASSIGNMENT = re.compile(r"\s*=(?!=)")
SWIZZLE = re.compile(r"\s*\.\s*([xyzw]+)\b")


def stripComments(text):
    text = re.sub(r"/\*.*?\*/", lambda m: "\n" * m.group(0).count("\n"), text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def readFile(path):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def collectMatrixMembers():
    """Every struct member of the base that is a matrix, and whether it is an array of matrices."""
    members = {}
    for pattern in ("**/*.hlsli", "**/*.h"):
        for path in glob.glob(pattern, recursive=True):
            if os.sep + "Build" + os.sep in path or os.sep + "Generated" + os.sep in path:
                continue
            text = stripComments(readFile(path))
            for body in STRUCT_BODY.finditer(text):
                for match in MATRIX_DECLARATION.finditer(body.group(1)):
                    members[match.group(2)] = bool(match.group(3))
    return members


def findUnsafeReads(path, matrixMembers):
    text = stripComments(readFile(path))
    declarations = [(match.start(), match.group(2),
                     "matrix" if re.fullmatch(MATRIX_TYPE, match.group(1)) else "other",
                     bool(match.group(3)))
                    for match in ANY_DECLARATION.finditer(text)]
    transposedLocals = set(TRANSPOSE_LOCAL.findall(text))

    findings = []
    for match in INDEXED_NAME.finditer(text):
        name = match.group("name")
        isMember = bool(match.group("dot"))
        subscripts = re.findall(r"\[[^\]]*\]", match.group("subs"))

        if isMember:
            if name not in matrixMembers:
                continue
            isArray = matrixMembers[name]
        else:
            preceding = [d for d in declarations if d[1] == name and d[0] < match.start()]
            if not preceding:
                continue
            _, _, kind, isArray = preceding[-1]
            if kind != "matrix":
                continue

        if name in transposedLocals:
            continue

        # The first index of an array of matrices selects the matrix, not a row of one.
        considered = subscripts[1:] if isArray else subscripts
        if not considered:
            continue

        rest = text[match.end():]
        if ASSIGNMENT.match(rest):
            continue

        indexes = [group.strip()[1:-1].strip() for group in considered]
        swizzle = SWIZZLE.match(rest)

        if len(indexes) == 1 and indexes[0].isdigit() and swizzle and len(swizzle.group(1)) == 1:
            if swizzle.group(1) == COMPONENTS[int(indexes[0])]:
                continue
        if len(indexes) >= 2 and indexes[0].isdigit() and indexes[1].isdigit() and indexes[0] == indexes[1]:
            continue

        line = text[:match.start()].count("\n") + 1
        findings.append((line, match.group(0).strip()))

    return findings


def getTargets(arguments):
    if len(arguments) > 0:
        return list(arguments)

    targets = sorted(glob.glob(SHADERS_FOLDER_PATH + "*.hlsli"))
    targets += sorted(glob.glob(PROBES_FOLDER_PATH + "*.probe.comp.hlsl"))
    return [t for t in targets
            if os.sep + "Build" + os.sep not in t and os.sep + "Generated" + os.sep not in t]


def main():
    arguments = [a for a in sys.argv[1:] if not a.startswith("-")]
    targets = getTargets(arguments)

    if len(targets) == 0:
        print("> No HLSL file to check. Nothing to do.")
        return 0

    matrixMembers = collectMatrixMembers()
    unsafe = 0

    for path in targets:
        if not os.path.isfile(path):
            print("=== " + path)
            print("  skipped: it does not exist")
            continue

        findings = findUnsafeReads(path, matrixMembers)
        if len(findings) == 0:
            continue

        unsafe += len(findings)
        print("=== " + path)
        for line, spelling in findings:
            print("  line " + str(line) + ": " + spelling)

    if unsafe > 0:
        print("")
        print("> " + str(unsafe) + " matrix read(s) do not read what the golden read. Read the")
        print("> column with getColumn(m, i) of ShaderCommonHLSLFunc.hlsli, and the element of a")
        print("> double index with getColumn(m, i)[j].")
        return 1

    print("> Every matrix read of the checked files reads the element the golden read.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
