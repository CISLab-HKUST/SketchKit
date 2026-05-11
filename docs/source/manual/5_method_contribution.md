# Method Contribution Guide


This guide provides a comprehensive walkthrough for developers who want to contribute a new method of sketch processing to the SketchKit toolkit.

---

## Phase 1: Collaboration Workflow (GitHub Flow)

We utilize the **GitHub Flow** to manage contributions. Please ensure your local environment is configured with Git and your GitHub credentials.

### 1. Fork the Repository

If you don't have write access to the main SketchKit repository, fork it to your personal account.

1. Click the **Fork** button on the GitHub repository page.
2. Clone your fork to your local machine:
```bash
git clone https://github.com/{your_username}/SketchKit.git
cd SketchKit
```



### 2. Create a Branch

Create a branch with a descriptive name related to the task or method you are adding.

```bash
# Example: git checkout -b feature/sketch2image-gan
git checkout -b feature/{task_name}-{method_name}
```

### 3. Make Changes

Implement your logic within the appropriate module. Ensure each commit is focused.

```bash
# Check your progress
git status

# Stage only specific files/directories
git add sketchkit/{task_name}/methods/your_new_method.py
git add sketchkit/{task_name}/__init__.py
git add sketchkit/{task_name}/processor.py 

# Commit with a clear message 
git commit -m "feat: add new_method to {task_name} module."
```

### 4. Push Changes

Upload your branch to your remote fork.

```bash
git push origin feature/{task_name}-{method_name}
```

### 5. Create a Pull Request (PR)

On GitHub, open a Pull Request.

1. Go to your branch and click **New pull request**.
2. Summarize changes, link related issues, and add visuals if helpful.
3. Use **Draft PR** for early feedback.

### 6. Address Review Comments

If reviewers suggest changes, apply them locally and push again. The PR updates automatically.

```bash
git add path/to/revised_file.py
git commit -m "fix: input handling for high-res sketches."
git push origin feature/{task_name}-{method_name}
```

### 7. Merge & Cleanup

Once merged, sync your local environment and delete the feature branch.

```bash
git checkout main
git pull origin main
git branch -d feature/{task_name}-{method_name}
```

---

## Phase 2: Technical Integration Standards

SketchKit relies on a **Unified Interface** pattern. Whether you are adding `sketch2model` or `sketch2image`, follow these architectural rules:

### 1. Module Structure

Each major task should be its own directory (e.g., `sketchkit/sketch2image/`). Within that directory, maintain:

* **`methods/`**: A folder containing the specific implementation files.
* **`__init__.py`**: The entry point defining the unified functional API (e.g., `sketch_to_image()`).
* **`task_processor.py`**: (e.g., `renderer.py` or `modeler.py`) A class-based wrapper for stateful operations.

### 2. Unified Interface Requirements

Your integration must support multiple input formats to stay user-friendly and robust:

* **Sketch Objects**: The internal SketchKit data structure.
* **File Paths**: Strings pointing to `.png`, `.jpg`, etc.
* **NumPy Arrays**: Raw image data.

### 3. Handling Heavy Assets (Models/Weights)

* **DO NOT** commit large pre-trained models (e.g., `.pth`, `.onnx`) to the repository.
* Implement a utility to check for weights locally and download them from a provided URL if missing.

### 4. Third-Party Dependencies

* If your method requires specific external libraries (e.g., `diffvg`), verify that they do not conflict with the core SketchKit environment.
* Use the `third_party/` directory for modified external source code.

### 5. Security & API Protection

* Paid APIs (e.g., OpenAI, Doubao, Midjourney) must fetch keys via `os.getenv()`.
* Provide clear error messages if an expected API key is missing from the environment.

---

## Phase 3: Implementation Example Template

When adding a new method to a module, follow this code pattern for the **Task Class**:

```python
class TaskProcessor:
    def __init__(self, method="default_algo"):
        self.method = method

    def run(self, input_data, output_path=None, **kwargs):
        # 1. Standardize Input
        processed_input = self._prepare_input(input_data)
        
        # 2. Route to specific method
        if self.method == "new_method":
            return self._run_new_method(processed_input, output_path, **kwargs)
        
    def _run_new_method(self, input_data, output_path, **kwargs):
        # Implementation logic goes here
        pass
```

## Phase 4: Documentation

A manual document for the new method is required. Follow the steps below to write the document. You can refer to the documents of {ref}`Other documents <methods>` for a reference.

a) Create a markdown file `MyMethod.md` in the `docs/source/manual/methods` directory.

b) The document includes the following parts:

- **Introduction** to the method: a brief description of the input and output of the method.
- **Source**: `tests/unit/test_mymethod.py`
- **Available Methods**: a list of supported implementations. Introduce each method in detail about the paper title, its source (URL), publication venue and year, etc.
- **Code**: code snippet of quick usage of the method, including initializing a method instance, loading an input, and executing the method for the output.
- **Arguments**: introduction of necessary arguments in the code.

---

## Final Checklist

* [ ] Method handles all three input types (Sketch object, Path, NumPy).
* [ ] No large model weights committed to Git.
* [ ] API keys are secured via environment variables.
* [ ] `__init__.py` updated to export the new method.
* [ ] Documentation/Docstrings updated with usage examples.

