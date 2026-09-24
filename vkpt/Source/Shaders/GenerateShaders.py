# Copyright (c) 2021 Sultim Tsyrendashiev
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import sys
import os
import subprocess
import pathlib


CACHE_FOLDER_PATH           = "Build/"
OUTPUT_FOLDER_PATH          = "../../Build/"
CACHE_FILE_NAME             = "GenerateShadersCache.txt"
EXTENSIONS                  = [ ".comp", ".vert", "frag", ".rgen", ".rahit", ".rchit", ".rmiss" ]
DEPENDENCY_EXTENSIONS       = [ ".h", ".inl", ".glsl", ".hlsl", ".hlsli" ]
DEPENDENCY_FOLDERS          = { "", "../Generated/" }
# HLSL_FOLDER_PATH holds stage *sources*, not dependencies: keeping it off this list is what lets
# the build loop see a changed source (its mtime is compared against the cache), because the
# dependency scan would otherwise seed the cache with the source's own mtime first and the source
# would look up to date forever. GLSL/ holds the goldens, which nothing includes.
HLSL_FOLDER_PATH            = "HLSL/"
SOURCE_FOLDERS              = [ "", HLSL_FOLDER_PATH ]
DEPENDENCY_FOLDERS_IGNORE   = [ CACHE_FOLDER_PATH, ".vscode/", "GLSL/", HLSL_FOLDER_PATH ]
DEPENDENCY_IGNORE           = [ "BlueNoiseFileNames.h", "ShaderCommonC.h", "ShaderCommonCFramebuf.h" ]

# HLSL sources are named <name><stage>.hlsl (e.g. CmBloomUpsample.comp.hlsl), so the produced
# blob keeps the name the host already looks up (CmBloomUpsample.comp.spv). Plain .hlsl and
# .hlsli files are headers and are never compiled on their own.
# The ported stage sources live in HLSL/ beside their GLSL originals in GLSL/; the root holds the
# shared headers (.hlsli and the goldens). Both folders below are scanned for stage sources, the
# root first.
# Ported shaders move their GLSL original to GLSL/ so that CheckShaderProperties.py can keep
# comparing them against the HLSL replacement.
HLSL_SUFFIX                 = ".hlsl"
HLSL_PROFILES               = {
    ".comp":    "cs_6_2",
    ".vert":    "vs_6_2",
    "frag":     "ps_6_2",
    ".rgen":    "lib_6_3",
    ".rahit":   "lib_6_3",
    ".rchit":   "lib_6_3",
    ".rmiss":   "lib_6_3",
}

# dxc adds one of the two acceleration-structure providers to every module at module level, even
# one that does not trace, and picks SPV_KHR_ray_query by default. Its own capability trim pass
# was expected to remove the pair again, but the SPIR-V grammar lists RayQueryKHR among the
# capabilities of OpTypeAccelerationStructureKHR, so every RT module (which has that type) reads
# as a user of ray query and keeps the capability along with the extension, although no
# OpRayQuery* instruction exists. The validation layer then rejects the blob on a device that does
# not enable the ray query feature (VUID-VkShaderModuleCreateInfo-pCode-08740).
# An explicit extension list replaces dxc's default choice; without SPV_KHR_ray_query dxc declares
# SPV_KHR_ray_tracing instead, the extension these shaders really use (OpTraceRayKHR and
# OpTypeAccelerationStructureKHR come from it). The list names every extension the sources need on
# vulkan1.2: dxc fails the build with an explicit error if one of them starts needing an extension
# that is missing here, so this is the only place the permitted set is maintained.
# SPV_KHR_compute_shader_derivatives is on it for a compute shader that samples with an implicit
# lod: dxc keeps the sample implicit and adds the derivative group execution mode, which requires
# the capability the extension provides. No production compute shader samples that way today, but
# the two probes CheckShaderProperties.py compiles do, and both tools pass the same list.
SPIRV_EXTENSIONS            = [
    "SPV_KHR_ray_tracing",
    "SPV_EXT_descriptor_indexing",
    "SPV_KHR_compute_shader_derivatives",
]


CACHE_FILE_DEPENDENCY_MAP_SEPARATOR_LINE = "DEPENDENCY\n"


MARKED_FILES = []
def wereDependentModified(dependencyMap, modifiedDependent, cache, baseFile, firstTime=True):
    global MARKED_FILES
    if firstTime:
        MARKED_FILES = []

    for dpd in dependencyMap[baseFile]:
        if dpd in modifiedDependent or dpd not in cache:
            return True
        elif dpd not in MARKED_FILES:
            MARKED_FILES.append(dpd)
            if wereDependentModified(dependencyMap, modifiedDependent, cache, dpd, firstTime=False):
                return True

    return False


def printInPowerShell(msg, color):
    p = subprocess.run([
        "PowerShell",
        "Write-Host",
        "\"" + msg + "\"",
        "-ForegroundColor", color
    ])


def getDependentFoldersProcArg():
    # The root ("") is part of DEPENDENCY_FOLDERS and reaches the compiler as `-I .`: a stage source
    # in HLSL/ includes the shared headers by their bare names, and its own folder is not the root.
    return [a for p in DEPENDENCY_FOLDERS for a in ("-I", p if p != "" else ".")]


def getHLSLStage(filename):
    if not filename.endswith(HLSL_SUFFIX):
        return None

    for ext in EXTENSIONS:
        if filename.endswith(ext + HLSL_SUFFIX):
            return ext

    return None


def isShaderSource(filename):
    return any([filename.endswith(ext) for ext in EXTENSIONS]) or getHLSLStage(filename) is not None


def getOutputFilename(filename):
    base = os.path.basename(filename)

    if base.endswith(HLSL_SUFFIX):
        base = base[:-len(HLSL_SUFFIX)]

    return OUTPUT_FOLDER_PATH + base + ".spv"


# Returns the command line that compiles the shader to the SPIR-V blob the host loads.
# GLSL goes through glslc, HLSL goes through dxc, which emits SPIR-V for Vulkan and can also
# emit DXIL for D3D12 from the very same source.
def getCompileCommand(filename, outputFilename):
    stage = getHLSLStage(filename)

    if stage is None:
        return [
            "glslc", "--target-env=vulkan1.2"
            ] + getDependentFoldersProcArg() + [
            filename,
            "-o", outputFilename]

    return [
        "dxc",
        "-spirv",
        "-T", HLSL_PROFILES[stage],
        "-fspv-target-env=vulkan1.2"
        ] + ["-fspv-extension=" + ext for ext in SPIRV_EXTENSIONS] + getDependentFoldersProcArg() + [
        filename,
        "-Fo", outputFilename]


def abspath(filename):
    return os.path.abspath(filename).replace('\\','/')


def getAllSubfolders(folder):
    folder = folder if folder != "" else "."
    return {os.path.relpath(root).replace('\\','/') + '/' for root, _, _ in os.walk(folder) if abspath(root) != abspath(folder)}


def fillDependencyFolders():
    global DEPENDENCY_FOLDERS

    fs = DEPENDENCY_FOLDERS
    for f in DEPENDENCY_FOLDERS:
        fs = fs.union(getAllSubfolders(f))

    for i in DEPENDENCY_FOLDERS_IGNORE:
        fs.discard(i)

    DEPENDENCY_FOLDERS = fs


def main():
    if "--help" in sys.argv or "-help" in sys.argv or "-h" in sys.argv or "--h" in sys.argv:
        print("-rebuild  : clear cache and rebuild all shaders")
        print("-gencomm  : invoke GenerateShaderCommon.py script")
        print("-psout    : use PowerShell for printing colored output")
        print("-r        : same as \"-rebuild\"")
        print("-g        : same as \"-gencomm\"")
        print("-ps       : same as \"-psout\"")
        print("")
        print("GLSL shaders are compiled with glslc, HLSL shaders (<name>.<stage>.hlsl)")
        print("are compiled with dxc into the very same <name>.<stage>.spv blobs.")
        return

    forceRebuild = False
    powerShellOutput = False
    if "-rebuild" in sys.argv or "--rebuild" in sys.argv or "-r" in sys.argv or "--r" in sys.argv:
        forceRebuild = True
    if "-gencomm" in sys.argv or "--gencomm" in sys.argv or "-g" in sys.argv or "--g" in sys.argv:
        subprocess.run(["python", "../Generated/GenerateShaderCommon.py", "--path", "../Generated/"])
    if "-psout" in sys.argv or "--psout" in sys.argv or "-ps" in sys.argv or "--ps" in sys.argv:
        powerShellOutput = True
    #elif len(sys.argv) > 1:
    #    print("> Couldn't parse arguments")
    #    return

    fillDependencyFolders()

    if not os.path.exists(CACHE_FOLDER_PATH):
        try:
            os.mkdir(CACHE_FOLDER_PATH)
        except OSError:
            print("> Coudn't create cache folder")
            return

    if not os.path.exists(OUTPUT_FOLDER_PATH):
        try:
            os.makedirs(OUTPUT_FOLDER_PATH)
        except OSError:
            print("> Coudn't create output folder")
            return

    if not os.path.exists(CACHE_FOLDER_PATH + CACHE_FILE_NAME):
        try:
            with open(CACHE_FOLDER_PATH + CACHE_FILE_NAME, "w"): pass
        except OSError:
            print("> Coudn't create cache file")
            return
    with open(CACHE_FOLDER_PATH + CACHE_FILE_NAME, "r+") as cacheFile:
        cache = {}
        dependencyMap = {}

        if not forceRebuild:
            try:
                parsingDpdncy = False
                for line in cacheFile:
                    if line == CACHE_FILE_DEPENDENCY_MAP_SEPARATOR_LINE:
                        parsingDpdncy = True
                    else:
                        words = line.split()
                        if not parsingDpdncy and len(words) >= 2:
                            # filename + st_mtime
                            cache[words[0]] = int(words[1])

                        if parsingDpdncy:
                            if len(words) >= 2:
                                # filename + (list of files it dependent on)
                                checkedDpds = set()
                                dpds = set(words[1:])
                                for dpd in dpds:
                                    if os.path.exists(dpd):
                                        checkedDpds.add(dpd)
                                dependencyMap[words[0]] = checkedDpds
                            else:
                                dependencyMap[words[0]] = set()
            except:
                cache = {}
                dependencyMap = {}
    
    modifiedDependent = set()

    msgWasAnyShaderRebuilt = False
    msgErrorCount = 0

    for folder in DEPENDENCY_FOLDERS:
        if not forceRebuild:
            print("> Checking dependency files in " + ("current folder" if folder == "" else folder))

        fileList = os.listdir() if folder == "" else os.listdir(folder)
        for otherFolderFilename in fileList:
            if otherFolderFilename in DEPENDENCY_IGNORE:
                continue

            filename = abspath(folder + otherFolderFilename)
            isDependentOn = any([filename.endswith(ext) for ext in DEPENDENCY_EXTENSIONS])

            if not isDependentOn:
                continue

            if ' ' in folder + filename:
                print("> File \"" + folder + filename + "\" has spaces in its path. Skipping.")
                continue

            lastModifTime = int(pathlib.Path(filename).stat().st_mtime)

            isOutdated = filename in cache and lastModifTime != cache[filename]

            if filename not in cache or isOutdated:
                modifiedDependent.add(filename)

            cache[filename] = lastModifTime

            if filename not in dependencyMap or isOutdated:
                dependencyMap[filename] = set()

                with open(filename, "r") as dpd:
                    for line in dpd:
                        if "#include" in line and '"' in line:
                            dpdFile = line.split("\"")[1]
                            for dpdFolder in DEPENDENCY_FOLDERS:
                                dpd = abspath(dpdFolder + dpdFile)
                                if os.path.exists(dpd) and dpd != filename:
                                    dependencyMap[filename].add(dpd)

    #if wereDependentModified and not forceRebuild:
    #    print("> Dependency files were modified. Rebuilding all...")
    # print()

    for filenameRelative in [folder + entry for folder in SOURCE_FOLDERS for entry in os.listdir(folder or ".")]:
        filename = abspath(filenameRelative)

        if not isShaderSource(filename):
            continue

        if ' ' in filename:
            print("> File \"" + filename + "\" has spaces in its name. Skipping.")
            continue

        lastModifTime = int(pathlib.Path(filename).stat().st_mtime)
        isOutdated = filename in cache and lastModifTime != cache[filename]

        outputFilename = getOutputFilename(filename)

        if filename not in dependencyMap or isOutdated:
            dependencyMap[filename] = set()

            with open(filename, "r") as dpd:
                for line in dpd:
                    if "#include" in line and '"' in line:
                        dpdFile = line.split("\"")[1]
                        for dpdFolder in DEPENDENCY_FOLDERS:
                            dpd = abspath(dpdFolder + dpdFile)
                            if os.path.exists(dpd):
                                dependencyMap[filename].add(dpd)

        if forceRebuild or filename not in cache or isOutdated or not os.path.exists(outputFilename) or wereDependentModified(dependencyMap, modifiedDependent, cache, filename):
            print("> Building " + os.path.basename(filename))

            command = getCompileCommand(filename, outputFilename)

            try:
                r = subprocess.run(command,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                compilerOutput = r.stdout
                isFailed = r.returncode != 0
            except FileNotFoundError:
                compilerOutput = "> " + command[0] + " is not in PATH"
                isFailed = True

            if isFailed:
                if powerShellOutput:
                    printInPowerShell(compilerOutput, "Red")
                else:
                    print(compilerOutput)

                msgErrorCount += 1

                # Do not leave the previously built blob behind: a failed build must not be
                # masked by deploying the stale one.
                if os.path.exists(outputFilename):
                    os.remove(outputFilename)

                if filename in cache:
                    del cache[filename]
            else:
                # dxc prints warnings that do not fail the build, e.g. about attributes that
                # only apply to one of the two backends.
                if len(compilerOutput) > 0:
                    if powerShellOutput:
                        printInPowerShell(compilerOutput, "Yellow")
                    else:
                        print(compilerOutput)

                cache[filename] = lastModifTime

            msgWasAnyShaderRebuilt = True

    with open(CACHE_FOLDER_PATH + CACHE_FILE_NAME, "w") as cacheFile:
        for name, tm in cache.items():
            cacheFile.write(name + " " + str(tm) + "\n")
        cacheFile.write(CACHE_FILE_DEPENDENCY_MAP_SEPARATOR_LINE)
        for name, arr in dependencyMap.items():
            arrStr = " ".join(arr)
            cacheFile.write(name + " " + arrStr + "\n")

    #if wereDependentModified:
    #    print()

    msg = ""
    color = ""

    if msgErrorCount > 0:
        msg = "> " + str(msgErrorCount) + (" shader build failed." if msgErrorCount == 1 else " shader builds failed.")
        color = "DarkRed"
    elif not msgWasAnyShaderRebuilt:
        msg = "> Everything is up-to-date."
        color = "Green"
    else:
        msg = "> Done."
        color = "Green"

    if powerShellOutput:
        printInPowerShell(msg, color)
    else:
        print(msg)

    if msgErrorCount > 0:
        sys.exit(1)


# main
if __name__ == "__main__":
    main()