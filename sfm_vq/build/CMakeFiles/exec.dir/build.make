# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.9

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/Cellar/cmake/3.9.6/bin/cmake

# The command to remove a file.
RM = /usr/local/Cellar/cmake/3.9.6/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/thejuspathmakumar/cv-works/sfm_vq

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/thejuspathmakumar/cv-works/sfm_vq/build

# Include any dependencies generated for this target.
include CMakeFiles/exec.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/exec.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/exec.dir/flags.make

CMakeFiles/exec.dir/src.cxx.o: CMakeFiles/exec.dir/flags.make
CMakeFiles/exec.dir/src.cxx.o: ../src.cxx
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/thejuspathmakumar/cv-works/sfm_vq/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/exec.dir/src.cxx.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/exec.dir/src.cxx.o -c /Users/thejuspathmakumar/cv-works/sfm_vq/src.cxx

CMakeFiles/exec.dir/src.cxx.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/exec.dir/src.cxx.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/thejuspathmakumar/cv-works/sfm_vq/src.cxx > CMakeFiles/exec.dir/src.cxx.i

CMakeFiles/exec.dir/src.cxx.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/exec.dir/src.cxx.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/thejuspathmakumar/cv-works/sfm_vq/src.cxx -o CMakeFiles/exec.dir/src.cxx.s

CMakeFiles/exec.dir/src.cxx.o.requires:

.PHONY : CMakeFiles/exec.dir/src.cxx.o.requires

CMakeFiles/exec.dir/src.cxx.o.provides: CMakeFiles/exec.dir/src.cxx.o.requires
	$(MAKE) -f CMakeFiles/exec.dir/build.make CMakeFiles/exec.dir/src.cxx.o.provides.build
.PHONY : CMakeFiles/exec.dir/src.cxx.o.provides

CMakeFiles/exec.dir/src.cxx.o.provides.build: CMakeFiles/exec.dir/src.cxx.o


# Object files for target exec
exec_OBJECTS = \
"CMakeFiles/exec.dir/src.cxx.o"

# External object files for target exec
exec_EXTERNAL_OBJECTS =

exec: CMakeFiles/exec.dir/src.cxx.o
exec: CMakeFiles/exec.dir/build.make
exec: /usr/local/lib/libopencv_stitching.3.3.1.dylib
exec: /usr/local/lib/libopencv_superres.3.3.1.dylib
exec: /usr/local/lib/libopencv_videostab.3.3.1.dylib
exec: /usr/local/lib/libopencv_aruco.3.3.1.dylib
exec: /usr/local/lib/libopencv_bgsegm.3.3.1.dylib
exec: /usr/local/lib/libopencv_bioinspired.3.3.1.dylib
exec: /usr/local/lib/libopencv_ccalib.3.3.1.dylib
exec: /usr/local/lib/libopencv_dpm.3.3.1.dylib
exec: /usr/local/lib/libopencv_face.3.3.1.dylib
exec: /usr/local/lib/libopencv_fuzzy.3.3.1.dylib
exec: /usr/local/lib/libopencv_img_hash.3.3.1.dylib
exec: /usr/local/lib/libopencv_line_descriptor.3.3.1.dylib
exec: /usr/local/lib/libopencv_optflow.3.3.1.dylib
exec: /usr/local/lib/libopencv_reg.3.3.1.dylib
exec: /usr/local/lib/libopencv_rgbd.3.3.1.dylib
exec: /usr/local/lib/libopencv_saliency.3.3.1.dylib
exec: /usr/local/lib/libopencv_stereo.3.3.1.dylib
exec: /usr/local/lib/libopencv_structured_light.3.3.1.dylib
exec: /usr/local/lib/libopencv_surface_matching.3.3.1.dylib
exec: /usr/local/lib/libopencv_tracking.3.3.1.dylib
exec: /usr/local/lib/libopencv_xfeatures2d.3.3.1.dylib
exec: /usr/local/lib/libopencv_ximgproc.3.3.1.dylib
exec: /usr/local/lib/libopencv_xobjdetect.3.3.1.dylib
exec: /usr/local/lib/libopencv_xphoto.3.3.1.dylib
exec: /usr/local/lib/libopencv_shape.3.3.1.dylib
exec: /usr/local/lib/libopencv_photo.3.3.1.dylib
exec: /usr/local/lib/libopencv_calib3d.3.3.1.dylib
exec: /usr/local/lib/libopencv_phase_unwrapping.3.3.1.dylib
exec: /usr/local/lib/libopencv_video.3.3.1.dylib
exec: /usr/local/lib/libopencv_datasets.3.3.1.dylib
exec: /usr/local/lib/libopencv_plot.3.3.1.dylib
exec: /usr/local/lib/libopencv_text.3.3.1.dylib
exec: /usr/local/lib/libopencv_dnn.3.3.1.dylib
exec: /usr/local/lib/libopencv_features2d.3.3.1.dylib
exec: /usr/local/lib/libopencv_flann.3.3.1.dylib
exec: /usr/local/lib/libopencv_highgui.3.3.1.dylib
exec: /usr/local/lib/libopencv_ml.3.3.1.dylib
exec: /usr/local/lib/libopencv_videoio.3.3.1.dylib
exec: /usr/local/lib/libopencv_imgcodecs.3.3.1.dylib
exec: /usr/local/lib/libopencv_objdetect.3.3.1.dylib
exec: /usr/local/lib/libopencv_imgproc.3.3.1.dylib
exec: /usr/local/lib/libopencv_core.3.3.1.dylib
exec: CMakeFiles/exec.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/thejuspathmakumar/cv-works/sfm_vq/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable exec"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/exec.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/exec.dir/build: exec

.PHONY : CMakeFiles/exec.dir/build

CMakeFiles/exec.dir/requires: CMakeFiles/exec.dir/src.cxx.o.requires

.PHONY : CMakeFiles/exec.dir/requires

CMakeFiles/exec.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/exec.dir/cmake_clean.cmake
.PHONY : CMakeFiles/exec.dir/clean

CMakeFiles/exec.dir/depend:
	cd /Users/thejuspathmakumar/cv-works/sfm_vq/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/thejuspathmakumar/cv-works/sfm_vq /Users/thejuspathmakumar/cv-works/sfm_vq /Users/thejuspathmakumar/cv-works/sfm_vq/build /Users/thejuspathmakumar/cv-works/sfm_vq/build /Users/thejuspathmakumar/cv-works/sfm_vq/build/CMakeFiles/exec.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/exec.dir/depend

