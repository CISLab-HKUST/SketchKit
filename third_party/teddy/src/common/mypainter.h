// Copyright 2020-2021 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Stub header for MyPainter class
// This file provides only forward declaration to avoid SDL/OpenGL dependencies
// The actual drawing functionality is not needed for the core sketch2model algorithm

#ifndef MYPAINTER_H
#define MYPAINTER_H

// Forward declarations
class MyWindow;

// Include image.h for Cu type (used in function signatures)
#include <image/image.h>

// Full class declaration (minimal, only what's needed for compilation)
class MyPainter {
 public:
  MyPainter();
  MyPainter(MyWindow *myWindow);
  MyPainter(Imguc &I);
  virtual ~MyPainter();

  // Drawing methods (declarations only, implementations not needed)
  void setColor(const Cu &c);
};

#endif  // MYPAINTER_H

