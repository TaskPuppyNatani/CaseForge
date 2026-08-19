# CaseForge maintainer packaging

The normal user workflow is graphical: install the generated `.deb`, then
launch **CaseForge** from the application menu.

Maintainers can build the Linux frozen application and Debian package with the
repository-local environment:

```bash
./.venv/bin/python packaging/build_deb.py
```

The script runs the tracked `CaseForge.spec`, stages the one-directory bundle
under `/opt/caseforge`, installs the desktop entry and icon, and writes the
package to `dist/caseforge_<version>_amd64.deb`.
