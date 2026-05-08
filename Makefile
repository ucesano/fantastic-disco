CC=gcc
NVCC=nvcc

LIBS=
INCLUDES=-I../../
LIB_FLAGS=-lm


BIN_FOLDER := bin
OBJ_FOLDER := obj
SRC_FOLDER := src
BATCH_OUT_FOLDER := outputs


MAIN_NAME=main
MAIN_BIN=spmv
MAIN_SRC=$(MAIN_NAME).cu

OBJECTS = $(OBJ_FOLDER)/mmio.o $(OBJ_FOLDER)/matrix.o $(OBJ_FOLDER)/spmv.o

all: $(BIN_FOLDER)/$(MAIN_BIN)

$(OBJ_FOLDER)/mmio.o: $(SRC_FOLDER)/mmio.c
	@mkdir -p $(BIN_FOLDER) $(OBJ_FOLDER) $(BATCH_OUT_FOLDER)
	$(CC) -c $(SRC_FOLDER)/mmio.c -o $@ $(LIB_FLAGS)

$(OBJ_FOLDER)/matrix.o: $(SRC_FOLDER)/matrix.c
	@mkdir -p $(BIN_FOLDER) $(OBJ_FOLDER) $(BATCH_OUT_FOLDER)
	$(CC) -c $(SRC_FOLDER)/matrix.c -o $@ $(LIB_FLAGS)

$(OBJ_FOLDER)/spmv.o: $(SRC_FOLDER)/spmv.c
	@mkdir -p $(BIN_FOLDER) $(OBJ_FOLDER) $(BATCH_OUT_FOLDER)
	$(CC) -c $(SRC_FOLDER)/spmv.c -o $@ $(LIB_FLAGS)

$(BIN_FOLDER)/$(MAIN_BIN): $(MAIN_SRC) $(OBJECTS)
	mkdir -p $(BIN_FOLDER)
	$(NVCC) $^ -o $@ $(LIBS) $(INCLUDES) $(LIB_FLAGS)

clean:
	rm -rf $(BIN_FOLDER) $(OBJ_FOLDER)
