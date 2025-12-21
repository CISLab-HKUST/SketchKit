// Pre-include header to ensure Eigen definitions are set before any includes
// This is especially important for MSVC to avoid template parsing issues

#ifndef TEDDY_SKETCH2MODEL_PREINCLUDE_H
#define TEDDY_SKETCH2MODEL_PREINCLUDE_H

// Disable IterativeLinearSolvers to avoid MSVC template parsing issues
// This must be defined before any Eigen includes
#ifndef EIGEN_DONT_INCLUDE_ITERATIVELINEARSOLVERS
#define EIGEN_DONT_INCLUDE_ITERATIVELINEARSOLVERS
#endif

// Other important definitions
#ifndef EIGEN_SPARSESOLVERBASE_H
#define EIGEN_SPARSESOLVERBASE_H
#endif

// Disable SDL/OpenGL dependencies (not needed for our use case)
// Define a stub for SDL_opengles2.h to avoid compilation errors
#ifndef NO_SDL
#define NO_SDL
#endif

// Stub SDL_opengles2.h if it's not available
#ifndef SDL_OPENGLES2_H
#define SDL_OPENGLES2_H
// Minimal stub definitions to avoid compilation errors
typedef unsigned int GLuint;
typedef int GLint;
typedef float GLfloat;
#endif

#endif // TEDDY_SKETCH2MODEL_PREINCLUDE_H
