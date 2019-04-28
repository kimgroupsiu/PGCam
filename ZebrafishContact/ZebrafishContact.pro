#-------------------------------------------------
#
# Project created by QtCreator 2019-02-07T10:56:58
#
#-------------------------------------------------

CONFIG += c++11
QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = ZebrafishContact
TEMPLATE = app

DEFINES += QT_DEPRECATED_WARNINGS

SOURCES += main.cpp\
        mainwindow.cpp \
    inputdevice.cpp \
    inputdevicesetting.cpp \
    PointGreyCamera.cpp \
    proctime.cpp \
    bolbimgproc.cpp \
    camera_spk.cpp \
    DSC_GPU/dsp_gpu.cpp \
    PointGreyCamSpinnaker.cpp

HEADERS  += mainwindow.h \
    inputdevice.h \
    inputdevicesetting.h \
    PointGreyCamera.h \
    proctime.h \
    bolbimgproc.h \
    camera_spk.h \
    cuda_kernels.h \
    DSC_GPU/cuda_kernels.h \
    DSC_GPU/dsp_gpu.h \
    PointGreyCamSpinnaker.h

FORMS    += mainwindow.ui

DISTFILES += DSC_GPU/vectorAdd_kernel.cu \
    DSC_GPU/convolutionSeparable.cu \
    DSC_GPU/BlobImageProc_Kernel.cu

# Camera include / Library
LIBS += -L/usr/lib/aarch64-linux-gnu/ -lQt5Core -lQt5Gui -lQt5Widgets
INCLUDEPATH += /home/nvidia/Downloads/flycapture.2.12.3.2_arm64/include/
LIBS += -L/home/nvidia/Downloads/flycapture.2.12.3.2_arm64/lib/ -lflycapture

# open cv library
INCLUDEPATH += /usr/include/
LIBS += -L/usr/lib/ -lopencv_core -lopencv_highgui -lopencv_imgproc
#LIBS += -L/usr/lib/ -lstdc++ -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_video -lopencv_videoio -lgstreamer-1.0


INCLUDEPATH += /usr/include/spinnaker
LIBS += -lSpinnaker
LIBS += -lGenApi_gcc540_v3_0 -lGCBase_gcc540_v3_0

##############################

#INCLUDEPATH += /usr/local/cuda-9.0/include/
#LIBS += -L/usr/local/cuda-9.0/lib64/ -lcuda -lcudart

SYSTEM_TYPE = 64

DESTDIR = $$system(pwd)
CUDA_OBJECTS_DIR = ./
OBJECTS_DIR = $$DESTDIR/Obj

QMAKE_CXXFLAGS_RELEASE =-O3 -DCMAKE_C_COMPILER=/usr/bin/gcc-5  -DCMAKE_CXX_COMPILER=/usr/bin/g++-5

CUDA_SOURCES += $$DISTFILES

# Path to cuda toolkit install
CUDA_DIR      = /usr/local/cuda-9.0/
# Path to header and libs files
INCLUDEPATH += /usr/local/cuda-9.0/include
QMAKE_LIBDIR += /usr/local/cuda-9.0/lib64/    # Note I'm using a 64 bits Operating system
# libs used in your code
LIBS += -L/usr/local/cuda-9.0/lib64/ -lcudart -lcuda
# GPU architecture
CUDA_ARCH     = sm_62 # Yeah! I've a new device. Adjust with your compute capability
# Here are some NVCC flags I've always used by default.
NVCC_OPTIONS = --use_fast_math  -DCMAKE_C_COMPILER=/usr/bin/gcc-5  -DCMAKE_CXX_COMPILER=/usr/bin/g++-5
# NVCCFLAGS     = --compiler-options -fno-strict-aliasing -use_fast_math --ptxas-options=-v  -DCMAKE_C_COMPILER=/usr/bin/gcc-5  -DCMAKE_CXX_COMPILER=/usr/bin/g++-5

# Prepare the extra compiler configuration (taken from the nvidia forum - i'm not an expert in this part)
CUDA_INC = $$join(INCLUDEPATH,' -I','-I',' ')
#-arch=$$CUDA_ARCH
#cuda.commands = $$CUDA_DIR/bin/nvcc -m64 -O3  -c $$NVCCFLAGS  -DCMAKE_C_COMPILER=/usr/bin/gcc-5  -DCMAKE_CXX_COMPILER=/usr/bin/g++-5 \
#               $$CUDA_INC $$LIBS  ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT} \
#               2>&1 | sed -r \"s/\\(([0-9]+)\\)/:\\1/g\" 1>&2
# nvcc error printout format ever so slightly different from gcc
# http://forums.nvidia.com/index.php?showtopic=171651

# SPECIFY THE R PATH FOR NVCC (this caused me a lot of trouble before)
QMAKE_LFLAGS += -DCMAKE_C_COMPILER=/usr/bin/gcc-5  -DCMAKE_CXX_COMPILER=/usr/bin/g++-5 -Wl,-rpath,/usr/local/cuda-9.1/lib64/ # <-- added this

# NVCCFLAGS =  -DCMAKE_C_COMPILER=/usr/bin/gcc-5  -DCMAKE_CXX_COMPILER=/usr/bin/g++-5 -Xlinker -rpath,/usr/local/cuda-9.1/lib64/ # <-- and this
#-arch=$$CUDA_ARCH
CONFIG(debug, debug|release) {
    # Debug mode
    cuda_d.input = CUDA_SOURCES
    cuda_d.output = $$CUDA_OBJECTS_DIR/${QMAKE_FILE_BASE}_cuda.o
    cuda_d.commands = $$CUDA_DIR/bin/nvcc -D_DEBUG $$NVCC_OPTIONS $$CUDA_INC $$NVCC_LIBS --machine $$SYSTEM_TYPE  -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
    cuda_d.dependency_type = TYPE_C
    QMAKE_EXTRA_COMPILERS += cuda_d
}
else {
    # Release mode
    cuda.input = CUDA_SOURCES
    cuda.output = $$CUDA_OBJECTS_DIR/${QMAKE_FILE_BASE}_cuda.o
    cuda.commands = $$CUDA_DIR/bin/nvcc $$NVCC_OPTIONS $$CUDA_INC $$NVCC_LIBS --machine $$SYSTEM_TYPE -arch=$$CUDA_ARCH -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
    cuda.dependency_type = TYPE_C
    QMAKE_EXTRA_COMPILERS += cuda
}
