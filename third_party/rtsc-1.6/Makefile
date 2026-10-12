# DEBUG = y

TRIMESHDIR = ../trimesh2
MAKERULESDIR = $(TRIMESHDIR)
INCLUDES = -I$(TRIMESHDIR)/include
LIBDIR = -L$(TRIMESHDIR)/lib.$(UNAME)
NOGUI_LIBS = $(LIBS) -lglfw -lGL -lGLEW

include $(MAKERULESDIR)/Makerules
ifneq (,$(filter Darwin Darwin64,$(UNAME)))
  # Homebrew dependencies are not on Clang's default include/library paths.
  INCLUDES += $(shell pkg-config --cflags glfw3 glew)
  NOGUI_LIBS = $(shell pkg-config --libs glfw3 glew) -framework OpenGL
endif
ifeq ($(UNAME),Win32)
  COPTS += -mwindows
endif

# 原始程序的源文件和目标文件
# sources_rtsc = rtsc.cc apparentridge.cc
# ofiles_rtsc = $(addprefix $(OBJDIR)/,$(sources_rtsc:.cc=.o))

# # contour_renderer的源文件和目标文件
# sources_contour = contour_renderer.cc apparentridge.cc
# ofiles_contour = $(addprefix $(OBJDIR)/,$(sources_contour:.cc=.o))

# nogui_contour_renderer的源文件和目标文件
sources_nogui_contour = nogui_contour_renderer.cc apparentridge.cc
ofiles_nogui_contour = $(addprefix $(OBJDIR)/,$(sources_nogui_contour:.cc=.o))

# 所有可执行程序
PROGS = rtsc$(EXE) contour_renderer$(EXE) nogui_contour_renderer$(EXE)

default: $(PROGS)

LIBS += -ltrimesh

# 原始程序的链接规则
rtsc$(EXE) : $(OBJDIR)/rtsc.o $(OBJDIR)/apparentridge.o
	$(LINK) $(GLLIBS)

# contour_renderer的链接规则
contour_renderer$(EXE) : $(OBJDIR)/contour_renderer.o $(OBJDIR)/apparentridge.o
	$(LINK) $(GLLIBS) 

# nogui_contour_renderer的链接规则
nogui_contour_renderer$(EXE) : $(OBJDIR)/nogui_contour_renderer.o $(OBJDIR)/apparentridge.o
	$(LINK) $(NOGUI_LIBS)

clean :
	-rm -f $(ofiles_rtsc) $(ofiles_contour) $(ofiles_nogui_contour) $(OBJDIR)/Makedepend $(OBJDIR)/*.d
	-rm -rf $(OBJDIR)/ii_files
	-rmdir $(OBJDIR)

spotless : clean
	-rm -f $(PROGS)
	-rmdir $(DESTDIR)

VERSION = 1.6

tar:
	cd .. && tar zcvf rtsc-$(VERSION).tar.gz `find rtsc-$(VERSION) -name 'OBJ*' -prune -o -name CVS -prune -o -type f -print | sort`

zip:
	cd .. && find rtsc-$(VERSION) -name 'OBJ*' -prune -o -name CVS -prune -o -print | sort | zip -9 rtsc-$(VERSION).zip -@
