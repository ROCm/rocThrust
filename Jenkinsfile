#!/usr/bin/env groovy
// This shared library is available at https://github.com/ROCmSoftwarePlatform/rocJENKINS/
@Library('rocJenkins') _

// This file is for internal AMD use.
// If you are interested in running your own Jenkins, please raise a github issue for assistance.

import com.amd.project.*
import com.amd.docker.*
import java.nio.file.Path;

rocThrustCI:
{

    def rocthrust = new rocProject('rocThrust')
    // customize for project
    rocthrust.paths.build_command = './install -c'

    // Define test architectures, optional rocm version argument is available
    def nodes = new dockerNodes(['gfx803 && ubuntu && hip-clang', 'gfx900 && ubuntu && hip-clang', 'gfx906 && centos7 && hip-clang'], rocthrust)

    boolean formatCheck = false

    def compileCommand =
    {
        platform, project->

        project.paths.construct_build_prefix()
        
        def command = """#!/usr/bin/env bash
                    set -x
                    cd ${project.paths.project_build_prefix}
                    LD_LIBRARY_PATH=/opt/rocm/lib CXX=/opt/rocm/bin/hipcc ${project.paths.build_command} --hip-clang
                """
        
        platform.runCommand(this, command)
    }

    def testCommand =
    {
        platform, project->

        def command = """#!/usr/bin/env bash
                    set -x
                    cd ${project.paths.project_build_prefix}/build/release
                    make -j4
                    sudo ctest --output-on-failure
                """
        
        platform.runCommand(this, command)
    }

    def packageCommand = null

    buildProject(rocthrust, formatCheck, nodes.dockerArray, compileCommand, testCommand, packageCommand)

}
