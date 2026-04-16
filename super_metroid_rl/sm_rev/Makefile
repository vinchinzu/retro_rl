TARGET_EXEC := sm_rev

SRCS := $(wildcard src/*.c src/snes/*.c) third_party/gl_core/gl_core_3_1.c
OBJS := $(SRCS:%.c=%.o)

PYTHON := /usr/bin/env python3
CFLAGS := $(if $(CFLAGS),$(CFLAGS),-O2 -fno-strict-aliasing -Werror)
CFLAGS += $(shell sdl2-config --cflags) -DSYSTEM_VOLUME_MIXER_AVAILABLE=0 -I.

ifeq (${OS},Windows_NT)
    WINDRES := windres
    SDLFLAGS := -Wl,-Bstatic $(shell sdl2-config --static-libs)
else
    SDLFLAGS := $(shell sdl2-config --libs) -lm
endif

.PHONY: all clean clean_obj run test test-fast

all: $(TARGET_EXEC)

$(TARGET_EXEC): $(OBJS)
	$(CC) $^ -o $@ $(LDFLAGS) $(SDLFLAGS)

%.o: %.c
	$(CC) -c $(CFLAGS) $< -o $@

run: all
	./$(TARGET_EXEC)

clean: clean_obj
clean_obj:
	@$(RM) $(OBJS) $(TARGET_EXEC)

test: all
	$(PYTHON) tests/run_tests.py -v

test-fast: all
	$(PYTHON) tests/run_tests.py --fast -v
