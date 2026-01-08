# SketchKit

## Documentation

The docs can found on [https://cislab.hkust-gz.edu.cn/projects/sketchkit/docs/](https://cislab.hkust-gz.edu.cn/projects/sketchkit/docs/). The api docs will be auto generated after each commit.

If you want to contribute to manual docs, please refer to documents in `docs/source/manual` folder, just like:

- [docs/source/manual/0_representation.md](docs/source/manual/0_representation.md)

### Docstring

A [Python Docstring](https://peps.python.org/pep-0257/) (short for documentation string) is a special kind of string in Python used to describe what your code does. It’s written as a string literal — usually triple quotes """ ... """ or ''' ... ''' — placed immediately after a function, class, or module definition.

We can auto-generate documentation with these docstrings using some tools.

We recommand [Google Style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) or [reST Style](https://sphinx-rtd-tutorial.readthedocs.io/en/latest/docstrings.html) docsrtings.

A Google Style docstring is like this:

```python
def add_numbers(a: int, b: int) -> int:
    """Add two numbers.

    Args:
        a (int): First number.
        b (int): Second number.

    Returns:
        int: Sum of the two numbers.
    """
    return a + b
```

**Tip:** you can just let chatGPT do this for you! 😆

## Development

We adapt the [GitHub flow](https://docs.github.com/en/get-started/using-github/github-flow) for collaboration.

### 1. Fork the Repository

If you don’t have write access to the original repository, fork it to your account.

- Go to the repository page and click **Fork**.
- This creates your own copy where you can make changes.
- [Learn more about forking](https://docs.github.com/en/get-started/quickstart/fork-a-repo)

### 2. Create a Branch

- Use a short, descriptive name (e.g., `fix-typo`, `add-feature-x`).
- Keeps changes isolated from the default branch.
- [How to create a branch](https://docs.github.com/en/repositories/creating-and-managing-branches)

```bash
git checkout -b {new_branch_name}
```

### 3. Make Changes

- Edit, add, rename, move, or delete files.
- Commit changes with descriptive messages.
- Keep each commit focused on a single change for easier review/revert.

### 4. Push Changes

- Push commits to your branch on GitHub.
- Work is backed up remotely and visible to collaborators.

### 5. Create a Pull Request (PR)

- Go to your branch and click **New pull request**.
- Summarize changes, link related issues, and add visuals if helpful.
- Use **draft PR** for early feedback.

### 6. Address Review Comments

- Reviewers may suggest changes or leave questions.
- Commit and push updates—PR updates automatically.

### 7. Merge the Pull Request

- After approval and passing checks, merge your branch.
- Branch protection rules may require specific reviews or checks.

### 8. Delete Your Branch

- Removes clutter and signals work completion.
- You can restore or revert changes later if needed.

```bash
git branch -d your_local_branch
```

### Tips

- Create separate branches for unrelated changes.
- Link issues with keywords so they close automatically upon merge.
- Use status checks to catch errors before merging.

