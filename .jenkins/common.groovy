// This file is for internal AMD use.
// If you are interested in running your own Jenkins, please raise a github issue for assistance.

def runCompileCommand(platform, project, jobName, settings)
{
    project.paths.construct_build_prefix()

    String buildTypeArg = settings.debug ? '-DCMAKE_BUILD_TYPE=Debug' : '-DCMAKE_BUILD_TYPE=Release'
    String buildTypeDir = settings.debug ? 'debug' : 'release'
    String cmake = platform.jenkinsLabel.contains('centos') ? 'cmake3' : 'cmake'
    //Set CI node's gfx arch as target if PR, otherwise use default targets of the library
    String amdgpuTargets = env.BRANCH_NAME.startsWith('PR-') ? '-DAMDGPU_TARGETS=\$gfx_arch' : ''
    String asanFlag = settings.addressSanitizer ? '-DBUILD_ADDRESS_SANITIZER=ON' : ''
    boolean sameOrg = settings.sameOrg ?: false

    def getRocPRIM = auxiliary.getLibrary('rocPRIM', platform.jenkinsLabel, null, sameOrg)

    def command = """#!/usr/bin/env bash
                set -x
                ${getRocPRIM}
                cd ${project.paths.project_build_prefix}
                mkdir -p build/${buildTypeDir} && cd build/${buildTypeDir}
                ${auxiliary.gfxTargetParser()}
                ${cmake} -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc ${buildTypeArg} ${amdgpuTargets} ${asanFlag} -DBUILD_TEST=ON -DBUILD_BENCHMARK=ON ../..
                make -j\$(nproc)
                """

    platform.runCommand(this, command)
}

def runTestCommand (platform, project, settings)
{
    String sudo = auxiliary.sudo(platform.jenkinsLabel)

    def testCommand = "ctest --output-on-failure"
    def hmmTestCommand = ''
    def excludeRegex = 'reduce_by_key.hip'
    def LD_PATH = ''
    
    if (platform.jenkinsLabel.contains('gfx11'))
    {
        excludeRegex = /(reduce_by_key.hip|partition.hip|sort.hip|sort_by_key.hip|stable_sort_by_key.hip|stable_sort.hip|async_copy.hip|async_reduce.hip|async_scan.hip|async_sort.hip|async_transform.hip)/
    }
    testCommandExclude = "--exclude-regex \"${excludeRegex}\""
    
    if (platform.jenkinsLabel.contains('gfx90a'))
    {
        hmmTestCommand = ""
                        // temporarily disable hmm testing
                        //  """
                        //     export HSA_XNACK=1
                        //     export ROCTHRUST_USE_HMM=1
                        //     ${testCommand} ${testCommandExclude}
                        //  """
    }

    if (settings.addressSanitizer)
    {
        LD_PATH = """
                    export ASAN_LIB_PATH=\$(/opt/rocm/llvm/bin/clang -print-file-name=libclang_rt.asan-x86_64.so)
                    export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:\$(dirname "\${ASAN_LIB_PATH}")
                  """
    }

    def command = """
                    #!/usr/bin/env bash
                    set -x
                    cd ${project.paths.project_build_prefix}
                    cd ${project.testDirectory}
                    ${LD_PATH}
                    ${testCommand} ${testCommandExclude}
                    ${hmmTestCommand}
                  """

    platform.runCommand(this, command)
}

def runPackageCommand(platform, project)
{
    def packageHelper = platform.makePackage(platform.jenkinsLabel,"${project.paths.project_build_prefix}/build/release")

    platform.runCommand(this, packageHelper[0])
        platform.archiveArtifacts(this, packageHelper[1])
}

return this

