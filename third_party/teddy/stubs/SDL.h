// Stub header for SDL.h
// This file provides minimal SDL type definitions
// to allow compilation without SDL2 dependencies

#ifndef SDL_H
#define SDL_H

// Minimal SDL type definitions
typedef struct SDL_Window SDL_Window;
typedef struct SDL_Renderer SDL_Renderer;
typedef struct SDL_Surface SDL_Surface;
typedef struct SDL_Texture SDL_Texture;
typedef struct SDL_Event SDL_Event;
typedef void* SDL_GLContext;

// SDL integer types
typedef int Sint32;
typedef long long Sint64;
typedef unsigned int Uint32;
typedef unsigned long long Uint64;

// SDL specific types
typedef Sint64 SDL_FingerID;

// SDL Color structure
typedef struct {
    unsigned char r;
    unsigned char g;
    unsigned char b;
    unsigned char a;
} SDL_Color;

// SDL constants (minimal set)
#define SDL_INIT_VIDEO 0x00000020
#define SDL_WINDOW_SHOWN 0x00000004
#define SDL_WINDOW_RESIZABLE 0x00000020
#define SDL_WINDOW_OPENGL 0x00000002

// SDL event types
#define SDL_QUIT 0x100
#define SDL_WINDOWEVENT 0x200
#define SDL_KEYDOWN 0x300
#define SDL_KEYUP 0x301
#define SDL_MOUSEBUTTONDOWN 0x401
#define SDL_MOUSEBUTTONUP 0x402
#define SDL_MOUSEMOTION 0x400
#define SDL_MOUSEWHEEL 0x403

// SDL key codes
#define SDLK_ESCAPE 27
#define SDLK_RETURN 13
#define SDLK_SPACE 32

// SDL mouse button codes
#define SDL_BUTTON_LEFT 1
#define SDL_BUTTON_MIDDLE 2
#define SDL_BUTTON_RIGHT 3

// SDL window event types
#define SDL_WINDOWEVENT_RESIZED 5
#define SDL_WINDOWEVENT_CLOSE 16

// Stub function declarations (will not be called if SDL is not used)
#ifdef __cplusplus
extern "C" {
#endif

// These are just stubs - they won't be called if SDL is not actually used
static inline int SDL_Init(unsigned int flags) { (void)flags; return 0; }
static inline void SDL_Quit(void) {}
static inline SDL_Window* SDL_CreateWindow(const char* title, int x, int y, int w, int h, unsigned int flags) { (void)title; (void)x; (void)y; (void)w; (void)h; (void)flags; return nullptr; }
static inline void SDL_DestroyWindow(SDL_Window* window) { (void)window; }
static inline SDL_Renderer* SDL_CreateRenderer(SDL_Window* window, int index, unsigned int flags) { (void)window; (void)index; (void)flags; return nullptr; }
static inline void SDL_DestroyRenderer(SDL_Renderer* renderer) { (void)renderer; }
static inline int SDL_PollEvent(SDL_Event* event) { (void)event; return 0; }
static inline void SDL_Delay(unsigned int ms) { (void)ms; }
static inline unsigned int SDL_GetTicks(void) { return 0; }
static inline void SDL_GetWindowSize(SDL_Window* window, int* w, int* h) { (void)window; if(w) *w = 0; if(h) *h = 0; }
static inline void SDL_GetMouseState(int* x, int* y) { if(x) *x = 0; if(y) *y = 0; }
static inline unsigned int SDL_GetMouseState_buttons(void) { return 0; }
static inline const unsigned char* SDL_GetKeyboardState(int* numkeys) { if(numkeys) *numkeys = 0; return nullptr; }

#ifdef __cplusplus
}
#endif

#endif // SDL_H

