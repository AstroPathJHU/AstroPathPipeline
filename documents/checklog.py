import argparse, textwrap
from texoutparse import LatexLogParser

def checklatex(filename):
  p = LatexLogParser()
  with open(filename) as f:
    p.process(f)

  message = []
  if p.errors:
    message.append("Errors:\n\n" + "\n\n".join(textwrap.indent(str(e), "  ") for e in p.errors))
  if p.warnings:
    message.append("Warnings:\n\n" + "\n\n".join(textwrap.indent(str(e), "  ") for e in p.warnings))
  if p.badboxes:
    message.append("Bad boxes:\n\n" + "\n\n".join(textwrap.indent(str(e), "  ") for e in p.badboxes))
  if hasattr(p, "missing_refs"):
    if p.missing_refs:
      message.append("Bad refs:\n\n" + "\n\n".join(textwrap.indent(str(e), "  ") for e in p.missing_refs))

  if message:
    raise RuntimeError(f"Latex gave some {'errors' if p.errors else 'warnings'}:\n\n\n" + "\n\n\n".join(message))

if __name__ == "__main__":
  p = argparse.ArgumentParser()
  p.add_argument("filename")
  args = p.parse_args()
  checklatex(args.filename)
