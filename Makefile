PROGRAM_NAME := test

BUILD_PATH := build
BIN_PATH := bin
COMPILER := nvcc
STANDART := --std=c++11
NVCC_FLAGS := --cudart static --relocatable-device-code=false
LIBS := -lcudart -lpthread -lgtest
CUDA_INCLUDES := -I"/usr/local/cuda/include"
WARNINGS_ERRORS := -pedantic -Wall -Wextra -Wno-deprecated -Wno-unused-parameter  -Wno-enum-compare -Weffc++

SRC_FILES := $(shell find . -name '*.cpp' | sort -k 1nr | cut -f2-)

OBJS := $(SRC_FILES:%.cpp=$(BUILD_PATH)/%.o)
DEP := $(OBJS:.o=.d)

# Test build for gtests
.PHONY: test
test: dirs $(BIN_PATH)/$(PROGRAM_NAME)
	@$(RM) $(PROGRAM_NAME)
	@ln -s $(BIN_PATH)/$(PROGRAM_NAME) $(PROGRAM_NAME)

.PHONY: dirs
dirs:
	@echo "Creating directories"
	@mkdir -p $(dir $(OBJS))
	@mkdir -p $(BIN_PATH)
	@echo "Directories created"

.PHONY: clean
clean:
	@echo "Deleting $(PROGRAM_NAME) symlink"
	@$(RM) $(PROGRAM_NAME)
	@echo "Deleting directories"
	@$(RM) -r build
	@$(RM) -r bin

$(BIN_PATH)/$(PROGRAM_NAME): $(OBJS)
	@echo "Linking"
	$(COMPILER) $(STANDART) $(NVCC_FLAGS) $(LIBS) -link -o $(BIN_PATH)/$(PROGRAM_NAME) $(OBJS)
	chmod +x $(BIN_PATH)/$(PROGRAM_NAME)

$(BUILD_PATH)/%.o: %.cpp
	@echo "Building"
	$(COMPILER) $(STANDART) --compile -o "$@" "$<"

