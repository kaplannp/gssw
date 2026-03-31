CC:=gcc-11
CFLAGS+=-Wall -O3 -g
OBJ_DIR:=obj
BIN_DIR:=bin
SRC_DIR:=src
LIB_DIR:=lib

OBJ=gssw.o
EXE=gssw_example
EXEADJ=gssw_example_adj
EXETEST=gssw_test

.PHONY:all clean cleanlocal test

all:$(LIB_DIR)/libgssw.a

$(OBJ_DIR)/$(OBJ):$(SRC_DIR)/gssw.h $(SRC_DIR)/gssw.c
	@mkdir -p $(@D)
	$(CC) $(LDFLAGS) $(CPPFLAGS) $(CFLAGS) -c -o $@ $(SRC_DIR)/gssw.c

$(LIB_DIR)/libgssw.a:$(OBJ_DIR)/$(OBJ)
	@mkdir -p $(@D)
	ar rvs $@ $<
	
test:$(BIN_DIR)/$(EXETEST)
	$(BIN_DIR)/$(EXETEST)

cleanlocal:
	$(RM) -r lib/
	$(RM) -r bin/
	$(RM) -r obj/

clean:cleanlocal




