# xscope

`xscope` is a local, zero-config ML experiment visualization dashboard. It is designed to be used alongside the **`mtrick`** logging library and runs as a standalone tool to monitor and visualize your metrics.

https://pypi.org/project/mtrick/ 

![Dashboard Interface](https://raw.githubusercontent.com/AvikArefin/xscope/main/interface.png)

## Installation

You can install `xscope` as a global command-line tool using `uv` or `pipx`:

```bash
# Using uv (recommended)
uv tool install xscope

# Or run without installing using uvx
uvx xscope

# Or using pipx
pipx install xscope
```

## Usage

To launch the dashboard, run:

```bash
xscope
```

## For development

You can use 
```bash
uv run src/cli.py -—reload
```
