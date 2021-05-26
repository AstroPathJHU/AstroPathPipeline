import re, uncertainties as unc

def siunitxformat(distance, *, pixelsormicrons="pixels", fmt="", power=1, pscale=None):
  from . import pixels, microns  #has to be in the function to keep up with setup
  quantity = {"pixels": pixels, "microns": microns}[pixelsormicrons](distance=distance, power=power, pscale=pscale)[()]

  formattedquantity = format(quantity, fmt)
  if unc.std_dev(quantity):
    match = re.match(r"(?P<p>\()?(?P<n>-?[0-9.]+)\+/-(?P<s>[0-9.]+)(?(p)\))(?P<e>(?(p)[Ee][-+0-9]*))", formattedquantity)
    if match is None: raise ValueError(f"{formattedquantity} doesn't match expected regex")
    n = match.group("n")
    s = match.group("s").replace(".", "").lstrip("0")
    e = match.group("e")
    formattedquantity = f"{n}({s}){e}"

  if power:
    formattedunit = {"pixels": "pixels", "microns": r"\micro\meter"}[pixelsormicrons]
    if power != 1: formattedunit += f"^{{{power}}}"
    return rf"\SI{{{formattedquantity}}}{{{formattedunit}}}"
  else:
    return rf"\num{{{formattedquantity}}}"
