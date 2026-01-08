// Stub header for SDL_opengles2.h
// This file provides minimal OpenGL ES 2.0 type definitions
// to allow compilation without SDL2/OpenGL dependencies

#ifndef SDL_OPENGLES2_H
#define SDL_OPENGLES2_H

// Minimal OpenGL ES 2.0 type definitions
typedef unsigned int GLuint;
typedef int GLint;
typedef float GLfloat;
typedef double GLdouble;
typedef unsigned char GLboolean;
typedef int GLsizei;
typedef void GLvoid;
typedef char GLchar;
typedef unsigned short GLushort;
typedef short GLshort;
typedef unsigned char GLubyte;
typedef signed char GLbyte;
typedef int GLenum;
typedef int GLbitfield;

// OpenGL ES 2.0 constants (minimal set)
#define GL_FALSE 0
#define GL_TRUE 1
#define GL_TRIANGLES 0x0004
#define GL_ARRAY_BUFFER 0x8892
#define GL_ELEMENT_ARRAY_BUFFER 0x8893
#define GL_STATIC_DRAW 0x88E4
#define GL_DYNAMIC_DRAW 0x88E8
#define GL_VERTEX_SHADER 0x8B31
#define GL_FRAGMENT_SHADER 0x8B30
#define GL_COMPILE_STATUS 0x8B81
#define GL_LINK_STATUS 0x8B82
#define GL_INFO_LOG_LENGTH 0x8B84

// Stub function declarations (will not be called if OpenGL is not used)
#ifdef __cplusplus
extern "C" {
#endif

// These are just stubs - they won't be called if OpenGL is not actually used
static inline void glGenBuffers(GLsizei n, GLuint* buffers) { (void)n; (void)buffers; }
static inline void glBindBuffer(GLenum target, GLuint buffer) { (void)target; (void)buffer; }
static inline void glBufferData(GLenum target, GLsizeiptr size, const void* data, GLenum usage) { (void)target; (void)size; (void)data; (void)usage; }
static inline void glGenVertexArrays(GLsizei n, GLuint* arrays) { (void)n; (void)arrays; }
static inline void glBindVertexArray(GLuint array) { (void)array; }
static inline void glEnableVertexAttribArray(GLuint index) { (void)index; }
static inline void glVertexAttribPointer(GLuint index, GLint size, GLenum type, GLboolean normalized, GLsizei stride, const void* pointer) { (void)index; (void)size; (void)type; (void)normalized; (void)stride; (void)pointer; }
static inline void glUseProgram(GLuint program) { (void)program; }
static inline void glDrawElements(GLenum mode, GLsizei count, GLenum type, const void* indices) { (void)mode; (void)count; (void)type; (void)indices; }
static inline void glDeleteBuffers(GLsizei n, const GLuint* buffers) { (void)n; (void)buffers; }
static inline void glDeleteVertexArrays(GLsizei n, const GLuint* arrays) { (void)n; (void)arrays; }
static inline void glDeleteProgram(GLuint program) { (void)program; }
static inline GLuint glCreateShader(GLenum type) { (void)type; return 0; }
static inline void glShaderSource(GLuint shader, GLsizei count, const GLchar* const* string, const GLint* length) { (void)shader; (void)count; (void)string; (void)length; }
static inline void glCompileShader(GLuint shader) { (void)shader; }
static inline void glGetShaderiv(GLuint shader, GLenum pname, GLint* params) { (void)shader; (void)pname; (void)params; }
static inline void glGetShaderInfoLog(GLuint shader, GLsizei bufSize, GLsizei* length, GLchar* infoLog) { (void)shader; (void)bufSize; (void)length; (void)infoLog; }
static inline GLuint glCreateProgram(void) { return 0; }
static inline void glAttachShader(GLuint program, GLuint shader) { (void)program; (void)shader; }
static inline void glLinkProgram(GLuint program) { (void)program; }
static inline void glGetProgramiv(GLuint program, GLenum pname, GLint* params) { (void)program; (void)pname; (void)params; }
static inline void glGetProgramInfoLog(GLuint program, GLsizei bufSize, GLsizei* length, GLchar* infoLog) { (void)program; (void)bufSize; (void)length; (void)infoLog; }
static inline void glDeleteShader(GLuint shader) { (void)shader; }
static inline GLint glGetUniformLocation(GLuint program, const GLchar* name) { (void)program; (void)name; return -1; }
static inline void glUniformMatrix4fv(GLint location, GLsizei count, GLboolean transpose, const GLfloat* value) { (void)location; (void)count; (void)transpose; (void)value; }
static inline void glGenTextures(GLsizei n, GLuint* textures) { (void)n; (void)textures; }
static inline void glBindTexture(GLenum target, GLuint texture) { (void)target; (void)texture; }
static inline void glTexImage2D(GLenum target, GLint level, GLint internalformat, GLsizei width, GLsizei height, GLint border, GLenum format, GLenum type, const void* data) { (void)target; (void)level; (void)internalformat; (void)width; (void)height; (void)border; (void)format; (void)type; (void)data; }
static inline void glTexParameteri(GLenum target, GLenum pname, GLint param) { (void)target; (void)pname; (void)param; }
static inline void glActiveTexture(GLenum texture) { (void)texture; }
static inline void glDeleteTextures(GLsizei n, const GLuint* textures) { (void)n; (void)textures; }
static inline void glDisableVertexAttribArray(GLuint index) { (void)index; }
static inline void glClear(GLbitfield mask) { (void)mask; }
static inline void glClearColor(GLfloat red, GLfloat green, GLfloat blue, GLfloat alpha) { (void)red; (void)green; (void)blue; (void)alpha; }

#ifdef __cplusplus
}
#endif

// Additional types that might be needed
typedef ptrdiff_t GLsizeiptr;
typedef ptrdiff_t GLintptr;

#endif // SDL_OPENGLES2_H

