# JEPA-WAM project page

This directory is a dependency-free, single-column project page for GitHub Pages.

Before the public launch, update the project links in `index.html`:

- Replace the disabled Paper element with an anchor to the final ArXiv URL.
- Remove the `Coming soon` label from Code when the repository is public.

The deployment workflow publishes this directory from the `main` branch. The site can also be
served locally from the repository root with:

```bash
python -m http.server 8000 --directory docs
```
