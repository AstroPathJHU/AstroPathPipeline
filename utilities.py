import matplotlib.pyplot as plt, numpy as np, uncertainties as unc

def covariance_matrix(*args, **kwargs):
  result = np.array(unc.covariance_matrix(*args, **kwargs))
  return (result + result.T) / 2

def removepdfpagegroup(filename):
  with open(filename, "rb") as f:
    content = f.read()
  content = content.replace(b"/Group << /CS /DeviceRGB /S /Transparency /Type /Group >>\n", b"")
  with open(filename, "wb") as f:
    f.write(content)

def savefig(fname, *args, **kwargs):
  plt.savefig(fname, *args, **kwargs)
  removepdfpagegroup(fname)
