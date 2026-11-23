# Makefile for compiling standalone sakhiya-cli linked against ggml
CC      ?= gcc
CXX     ?= g++
GGML_DIR ?= /tmp/ggml

BUILD_DIR = build

CFLAGS   = -O3 -march=native -fPIC -pthread -fopenmp -D_GNU_SOURCE \
           -I$(BUILD_DIR) -I$(GGML_DIR)/include -I$(GGML_DIR)/src -I$(GGML_DIR)/src/ggml-cpu \
           -DGGML_USE_CPU -DGGML_USE_OPENMP -DGGML_VERSION='"1.0.0"' -DGGML_COMMIT='"d9fe7bf"'
CXXFLAGS = -std=c++17 $(CFLAGS) -funroll-loops -Icpp
LDFLAGS  = -pthread -fopenmp -lm -ldl

GGML_C_SRCS = \
	$(GGML_DIR)/src/ggml.c \
	$(GGML_DIR)/src/ggml-alloc.c \
	$(GGML_DIR)/src/ggml-quants.c \
	$(GGML_DIR)/src/ggml-cpu/ggml-cpu.c \
	$(GGML_DIR)/src/ggml-cpu/quants.c \
	$(GGML_DIR)/src/ggml-cpu/arch/x86/quants.c

GGML_CXX_SRCS = \
	$(GGML_DIR)/src/ggml.cpp \
	$(GGML_DIR)/src/gguf.cpp \
	$(GGML_DIR)/src/ggml-backend.cpp \
	$(GGML_DIR)/src/ggml-backend-reg.cpp \
	$(GGML_DIR)/src/ggml-backend-dl.cpp \
	$(GGML_DIR)/src/ggml-backend-meta.cpp \
	$(GGML_DIR)/src/ggml-opt.cpp \
	$(GGML_DIR)/src/ggml-threading.cpp \
	$(GGML_DIR)/src/ggml-cpu/ggml-cpu.cpp \
	$(GGML_DIR)/src/ggml-cpu/ops.cpp \
	$(GGML_DIR)/src/ggml-cpu/binary-ops.cpp \
	$(GGML_DIR)/src/ggml-cpu/unary-ops.cpp \
	$(GGML_DIR)/src/ggml-cpu/vec.cpp \
	$(GGML_DIR)/src/ggml-cpu/repack.cpp \
	$(GGML_DIR)/src/ggml-cpu/traits.cpp \
	$(GGML_DIR)/src/ggml-cpu/hbm.cpp \
	$(GGML_DIR)/src/ggml-cpu/amx/amx.cpp \
	$(GGML_DIR)/src/ggml-cpu/amx/mmq.cpp \
	$(GGML_DIR)/src/ggml-cpu/tiled/tiled.cpp \
	$(GGML_DIR)/src/ggml-cpu/tiled/tiled-kernel.cpp \
	$(GGML_DIR)/src/ggml-cpu/arch/x86/repack.cpp

GGML_OBJS = $(patsubst $(GGML_DIR)/src/%.c,$(BUILD_DIR)/ggml/%.c.o,$(GGML_C_SRCS)) \
            $(patsubst $(GGML_DIR)/src/%.cpp,$(BUILD_DIR)/ggml/%.cpp.o,$(GGML_CXX_SRCS))

all: $(BUILD_DIR)/sakhiya-cli

$(BUILD_DIR)/ggml-version.h:
	@mkdir -p $(BUILD_DIR)
	@printf '#pragma once\n#define GGML_VERSION "1.0.0"\n#define GGML_COMMIT "d9fe7bf"\n' > $@

$(BUILD_DIR)/ggml/%.c.o: $(GGML_DIR)/src/%.c $(BUILD_DIR)/ggml-version.h
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/ggml/%.cpp.o: $(GGML_DIR)/src/%.cpp $(BUILD_DIR)/ggml-version.h
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/sakhiya-cli: cpp/sakhiya_cli.cpp cpp/sakhiya_ggml.hpp cpp/sakhiya_g2p.hpp $(GGML_OBJS)
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) cpp/sakhiya_cli.cpp $(GGML_OBJS) $(LDFLAGS) -o $@
	cp $@ sakhiya-cli

clean:
	rm -rf $(BUILD_DIR) sakhiya-cli
