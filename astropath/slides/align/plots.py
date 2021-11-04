#!/usr/bin/env python

import logging, matplotlib.cm, matplotlib.collections, matplotlib.colors, matplotlib.patches, matplotlib.pyplot as plt, more_itertools, mpl_toolkits.axes_grid1, networkx as nx, numpy as np, uncertainties.unumpy as unp
from ...utilities import units
from ...utilities.misc import floattoint, pullhist

logger = logging.getLogger("alignmentplots")

def rectanglelayout(alignsample, *, xrange=None, yrange=None, primaryarea=True, figurekwargs={}, showplot=None, saveas=None):
  fig = plt.figure(**figurekwargs)
  ax = fig.add_subplot(1, 1, 1)
  onepixel = alignsample.onepixel
  shape = alignsample.fields[0].shape/onepixel
  if primaryarea:
    if xrange is None:
      xrange = (
        np.min([r.mx1 for r in alignsample.fields]) / onepixel,
        np.max([r.mx2 for r in alignsample.fields]) / onepixel,
      )
    if yrange is None:
      yrange = (
        np.max([r.my2 for r in alignsample.fields]) / onepixel,
        np.min([r.my1 for r in alignsample.fields]) / onepixel,
      )
  else:
    xmin, ymin = np.min([r.xvec for r in alignsample.fields], axis=0) / onepixel
    xmax, ymax = np.max([r.xvec+r.shape for r in alignsample.fields], axis=0) / onepixel
    if xrange is None: xrange = xmin, xmax
    if yrange is None: yrange = ymax, ymin
  ax.set_xlim(*xrange)
  ax.set_ylim(*yrange)
  xparity = {x: i for i, x in enumerate(sorted({r.x for r in alignsample.fields}))}
  yparity = {y: i for i, y in enumerate(sorted({r.y for r in alignsample.fields}))}
  colors = ["red", "blue", "green", "yellow", "purple", "orange", "magenta"]
  for r in alignsample.fields:
    parity = xparity[r.x] + 3*yparity[r.y]
    color = colors[parity % len(colors)]
    if primaryarea:
      xvec = np.array([r.mx1, r.my1])
      shape = np.array([r.mx2 - r.mx1, r.my2 - r.my1])
    else:
      xvec = r.xvec
      shape = r.shape
    box = matplotlib.patches.Rectangle(
      xvec / onepixel,
      *shape / onepixel,
      facecolor=color,
      edgecolor="black",
      alpha=0.5
    )
    ax.add_patch(box)
    plt.text(*xvec+shape/2, str(r.n), horizontalalignment="center", verticalalignment="center")
  for o in alignsample.overlaps:
    r1, r2 = o.rectangles
    center = (r1.xvec+r1.shape/2 + r2.xvec+r2.shape/2) / 2
    boxsize = r1.shape/10
    box = matplotlib.patches.Rectangle(
      (center-boxsize/2) / onepixel,
      *boxsize / onepixel,
      facecolor="tab:red" if o.result.exit else "tab:green",
      edgecolor=None,
      alpha=0.5
    )
    ax.add_patch(box)
  if showplot is None: showplot = saveas is None
  if showplot:
    plt.show()
  if saveas is not None:
    plt.savefig(saveas)
  if not showplot:
    plt.close()

def plotpairwisealignments(alignsample, *, stitched=False, tags=[1, 2, 3, 4, 6, 7, 8, 9], plotstyling=lambda fig, ax: None, errorbars=True, saveas=None, showplot=None, figurekwargs={}, pull=False, pixelsormicrons=None, pullkwargs={}, pullbinning=None):
  logger.debug(alignsample.samp)
  fig = plt.figure(**figurekwargs)
  ax = fig.add_subplot(1, 1, 1)

  vectors = np.array([
    o.result.dxvec - (o.stitchresult if stitched else 0)
    for o in alignsample.overlaps
    if not o.result.exit
    and o.tag in tags
  ])
  if not errorbars: vectors = units.nominal_values(vectors)
  if pull:
    if pullbinning is None: pullbinning = np.linspace(-5, 5, 51)
    pullhist(
      vectors[:,0],
      label="$x$ pulls",
      stdinlabel=True,
      alpha=0.5,
      binning=pullbinning,
      **pullkwargs,
    )
    pullhist(
      vectors[:,1],
      label="$y$ pulls",
      stdinlabel=True,
      alpha=0.5,
      binning=pullbinning,
      **pullkwargs,
    )
  else:
    if pullkwargs != {} or pullbinning is not None:
      raise ValueError("Can't provide pull kwargs for a scatter plot")
    if pixelsormicrons is None:
      raise ValueError("Have to provide pixelsormicrons for a scatterplot")
    divideby = {"pixels": alignsample.onepixel, "microns": alignsample.onemicron}[pixelsormicrons]
    kwargs = dict(
      x=(units.nominal_values(vectors[:,0]) / divideby).astype(float),
      y=(units.nominal_values(vectors[:,1]) / divideby).astype(float),
    )
    if errorbars:
      plt.errorbar(
        **kwargs,
        xerr=(units.std_devs(vectors[:,0]) / divideby).astype(float),
        yerr=(units.std_devs(vectors[:,1]) / divideby).astype(float),
        fmt='o',
      )
    else:
      plt.scatter(
        **kwargs,
        s=4,
      )

  plotstyling(fig=fig, ax=ax)

  if showplot is None:
    showplot = saveas is None

  if showplot:
    plt.show()
  if saveas is not None:
    plt.savefig(saveas)
  if not showplot:
    plt.close()

  logger.debug("done")
  return vectors

def closedlooppulls(alignsample, *, tagsequence, binning=np.linspace(-5, 5, 51), quantileforstats=1, verbose=True, stitchresult=None, saveas=None, figurekwargs={}, plotstyling=lambda fig, ax: None):
  logger.debug(alignsample.samp)
  dct = {
    1: (-1, -1),
    2: ( 0, -1),
    3: ( 1, -1),
    4: (-1,  0),
    6: ( 1,  0),
    7: (-1,  1),
    8: ( 0,  1),
    9: ( 1,  1),
  }
  totaloffset = sum(np.array(dct[tag]) for tag in tagsequence)
  if np.any(totaloffset):
    raise ValueError(f"please check your tag sequence - it ends with an offset of {tuple(totaloffset)}")

  overlaps = alignsample.overlaps
  g = alignsample.overlapgraph()
  xresiduals, yresiduals = [], []
  overlapdict = nx.get_edge_attributes(g, "overlap")

  for o in overlaps:
    if o.result.exit: continue
    if o.tag != tagsequence[-1]: continue
    for path in nx.algorithms.simple_paths.shortest_simple_paths(g, o.p2, o.p1):
      if len(path) < len(tagsequence): continue
      if len(path) > len(tagsequence): break
      path = path[:]
      path.append(path[0]) #make a full circle

      overlaps = [overlapdict[nodepair] for nodepair in more_itertools.pairwise(path)]
      assert overlaps[-1] is o
      tags = [o.tag for o in overlaps]
      if tags != tagsequence: continue

      if any(o.result.exit for o in overlaps): continue
      if stitchresult is not None:
        dxvecs = [stitchresult.dx(o) for o in overlaps]
      else:
        dxvecs = [o.result.dxvec for o in overlaps]

      dxs, dys = zip(*dxvecs)

      if verbose is True or verbose is not False and verbose(path, dxs, dys):
        alignsample.printlogger.info(" --> ".join(f"{node:4d}" for node in path))
        for nodepair, dx, dy in zip(more_itertools.pairwise(path), dxs, dys):
          alignsample.printlogger.info(f"  {nodepair[0]:4d} --> {nodepair[1]:4d}: {dx:10} {dy:10}")
        alignsample.printlogger.info(f"          total: {sum(dxs):10} {sum(dys):10}")

      totaldx = sum(dxs)
      totaldy = sum(dys)
      xresiduals.append(totaldx)
      yresiduals.append(totaldy)

  if verbose:
    alignsample.printlogger.info()
    alignsample.printlogger.info()
    alignsample.printlogger.info()
  fig = plt.figure(**figurekwargs)
  ax = fig.add_subplot(1, 1, 1)
  alignsample.printlogger.info("x pulls:")
  pullhist(xresiduals, binning=binning, verbose=True, alpha=0.5, label="$x$ pulls", stdinlabel=True)
  alignsample.printlogger.info()
  alignsample.printlogger.info()
  alignsample.printlogger.info()
  alignsample.printlogger.info("y pulls:")
  pullhist(yresiduals, binning=binning, verbose=True, alpha=0.5, label="$y$ pulls", stdinlabel=True)
  plotstyling(fig=fig, ax=ax)

  if saveas is not None:
    plt.savefig(saveas)
    plt.close()

  logger.debug("done")
  return xresiduals, yresiduals

def shiftplot2D(alignsample, *, saveasx=None, saveasy=None, figurekwargs={}, plotstyling=lambda fig, ax, cbar, xory: None, island=None, showplot=None):
  logger.debug(alignsample.samp)
  fields = alignsample.fields
  onepixel = alignsample.onepixel
  if island is not None:
    fields = [field for field in fields if field.n in alignsample.islands()[island]]
  deltax = min(abs(a.x-b.x) for a, b in more_itertools.pairwise(fields) if a.x != b.x)
  deltay = min(abs(a.y-b.y) for a, b in more_itertools.pairwise(fields) if a.y != b.y)
  deltaxvec = deltax, deltay
  x0vec = np.min([[f.x, f.y] for f in fields], axis=0)
  f = fields[0]
  shape = tuple(reversed(floattoint(np.max([(f.xvec - x0vec) / deltaxvec for f in fields], axis=0).astype(float)) + 1))
  xyarray = np.full(shape=(2,)+shape, fill_value=-999.*onepixel)

  extent = np.array([x0vec[0], shape[1] * deltaxvec[0] + x0vec[0], shape[0] * deltaxvec[1] + x0vec[1], x0vec[1]])

  for f in fields:
    idx = (slice(None),) + tuple(reversed(floattoint(((f.xvec - x0vec) / deltaxvec).astype(float))))
    xyarray[idx] = units.nominal_values(f.pxvec - alignsample.T@(f.xvec-alignsample.position))

  xyarraypixels = (xyarray / onepixel).astype(float)

  vmin = min(np.min(xyarraypixels[xyarraypixels != -999]), -np.max(xyarraypixels[xyarraypixels != -999]))
  vmax = -vmin
  norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
  cmap = matplotlib.cm.get_cmap()
  xycolor = cmap(norm(xyarraypixels))
  xycolor[xyarraypixels == -999] = 0

  if showplot is None: showplot = saveasx is saveasy is None

  for colorplot, xory, saveas in zip(xycolor, "xy", (saveasx, saveasy)):
    fig = plt.figure(**figurekwargs)
    ax = plt.gca()
    ax.imshow(colorplot, extent=(extent / onepixel).astype(float))
    divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    plotstyling(fig=fig, ax=ax, cbar=cbar, xory=xory)
    if showplot:
      plt.show()
    if saveas is not None:
      plt.savefig(saveas)
    if not showplot:
      plt.close()

  logger.debug("done")
  return xyarray, extent


def shiftplotprofile(alignsample, *, deltaxory, vsxory, saveas=None, figurekwargs={}, plotstyling=lambda fig, ax, deltaxory, vsxory: None, drawfourier=False, guessparameters=None, plotsine=True, sinetext=True, **kwargs):
  onepixel = alignsample.onepixel
  fig = plt.figure(**figurekwargs)
  ax = fig.add_subplot(1, 1, 1)

  xyarray, extent = shiftplot2D(alignsample, showplot=False, **kwargs)

  if deltaxory == "x":
    array2D = xyarray[0]
    yidx = 0
  elif deltaxory == "y":
    array2D = xyarray[1]
    yidx = 1
  else:
    assert False, deltaxory

  if vsxory == "x":
    array2D = array2D.T
    edges = extent[0:2]
    xidx = 0
  elif vsxory == "y":
    edges = extent[2:4]
    xidx = 1
  else:
    assert False, vsxory

  mean = np.mean(array2D[array2D / onepixel != -999])
  RMS = np.std(array2D[array2D / onepixel != -999])

  x = []
  y = []
  yerr = []
  binedges = units.np.linspace(*edges, num=len(array2D)+1)
  for rowcolumn, (binlow, binhigh) in more_itertools.zip_equal(array2D, more_itertools.pairwise(binedges)):
    ys = [_ for _ in rowcolumn if _ != -999*onepixel]
    if not ys: continue
    x.append((binlow+binhigh)/2)
    y.append(np.mean(ys))
    yerr.append(np.std(ys))

  x = np.array(x)
  y = np.array(y)
  yerr = np.array(yerr)

  errorzero = abs(yerr/(sum(y**2)/len(y))**.5) < 1e-3
  errornonzero = ~errorzero

  xwitherror = x[errornonzero]
  ywitherror = y[errornonzero]
  yerrwitherror = yerr[errornonzero]
  xnoerror = x[errorzero]
  ynoerror = y[errorzero]

  if not drawfourier:
    plt.errorbar(
      x=(xwitherror / onepixel).astype(float),
      y=(ywitherror / onepixel).astype(float),
      yerr=(yerrwitherror / onepixel).astype(float),
      fmt='o',
      color='b',
    )
    if xnoerror.size:
      plt.scatter(
        x=(xnoerror / onepixel).astype(float),
        y=(ynoerror / onepixel).astype(float),
        facecolors='none',
        edgecolors='b',
      )

  #fit to sine wave
  #for the initial parameter estimation, do an fft
  #can only do fft on evenly spaced data
  # ==> if there are islands, we have to pick the biggest one
  alldeltaxs = {b-a for a, b in more_itertools.pairwise(x)}
  deltax = min(alldeltaxs)
  rtol = 1e-7
  for _ in alldeltaxs:
    units.np.testing.assert_allclose(_*(1+2*rtol) // deltax, _ / deltax, rtol=rtol)

  chunkstarts = [
    xx for xx in x if not np.any(units.np.isclose(xx-abs(deltax), x, rtol=rtol))
  ]
  chunkends = [
    xx for xx in x if not np.any(units.np.isclose(xx+abs(deltax), x, rtol=rtol))
  ]
  biggestchunkstart, biggestchunkend = max(
    more_itertools.zip_equal(chunkstarts, chunkends),
    key=lambda startend: startend[1]-startend[0],
  )
  biggestchunkxs = np.array([xx for xx in x if biggestchunkstart <= xx <= biggestchunkend])
  biggestchunkys = np.array([yy for xx, yy in zip(x, y) if xx in biggestchunkxs])

  k = np.fft.fftfreq(len(biggestchunkys), deltax)
  f = units.np.fft.fft(biggestchunkys)

  if drawfourier:
    plt.scatter(k * onepixel, abs(f) / onepixel)
    plotstyling(fig=fig, ax=ax)

    if saveas is None:
      plt.show()
    else:
      plt.savefig(saveas)
      plt.close()

    return k, f

  def cosfunction(xx, amplitude, kk, phase):
    @np.vectorize
    def cos(thing):
      try:
        return np.cos(thing)
      except TypeError:
        return unp.cos(thing)
    return amplitude * cos(units.asdimensionless(kk*(xx - biggestchunkxs[0]) + phase)) + mean

  bestk, bestf = max(zip(k[1:], f[1:]), key=lambda kf: abs(kf[1]))  #[1:]: exclude k=0 term
  initialguess = [
    abs(bestf) / (len(biggestchunkxs) / 2),
    bestk * 2 * np.pi,
    units.np.angle(bestf),
  ]
  if guessparameters is not None:
    for i, parameter in enumerate(guessparameters):
      if parameter is not None:
        initialguess[i] = parameter

  try:
    p, cov = units.scipy.optimize.curve_fit(
      cosfunction, xwitherror, ywitherror, p0=initialguess, sigma=yerrwitherror, absolute_sigma=True,
    )
    amplitude, kk, phase = units.correlated_distances(distances=p, covariance=cov)
  except (RuntimeError, np.linalg.LinAlgError):
    alignsample.printlogger.info("fit failed")
    amplitude = kk = phase = 0
    p = amplitude, kk, phase
    cov = np.diag([1, 1, 1])
    amplitude, kk, phase = units.correlated_distances(distances=p, covariance=cov)

  if amplitude < 0:
    amplitude *= -1
    phase += np.pi
  if kk < 0:
    kk *= -1
    phase *= -1
  p = amplitude, kk, phase

  alignsample.printlogger.info("Average:")
  alignsample.printlogger.info(f"  {mean}")
  try:
    o = alignsample.overlaps[0]
    expected = ((alignsample.T - np.identity(2)) @ (o.x1vec - o.x2vec))[yidx]
  except AttributeError:
    pass
  else:
    alignsample.printlogger.info(f"  (expected from T matrix: {expected})")
  alignsample.printlogger.info("Sine wave:")
  alignsample.printlogger.info(f"  amplitude: {amplitude}")
  if abs(amplitude.n) > 5*amplitude.s and np.count_nonzero(abs(amplitude.n) > yerr) > len(yerr)/4:
    wavelength = 2*np.pi / kk
    alignsample.printlogger.info(f"  wavelength: {wavelength}")
    alignsample.printlogger.info(f"              = field size * {wavelength / o.rectangles[0].shape[xidx]}")
  else:
    alignsample.printlogger.info("  (not significant)")
    plotsine = False

  alignsample.printlogger.info("Remaining noise:")
  alignsample.printlogger.info(f"  RMS     = {RMS}")

  oldylim = ax.get_ylim()
  plotstyling(fig=fig, ax=ax)
  adjustylim = oldylim == ax.get_ylim()

  xplot = units.np.linspace(min(x), max(x), 1000)
  if plotsine:
    #plt.plot(xplot, cosfunction(xplot, *initialguess), color='g')
    plt.plot(xplot / onepixel, cosfunction(xplot, *units.nominal_values(p)) / onepixel, color='b')
    if sinetext:
      xcenter = np.average(ax.get_xlim())
      bottom, top = ax.get_ylim()
      if adjustylim:
        top += (top-bottom) * .3
        ax.set_ylim(bottom, top)
      amplitudetext = units.drawing.siunitxformat(amplitude, power=1)
      wavelengthtext = units.drawing.siunitxformat(wavelength, power=1)
      RMStext = units.drawing.siunitxformat(RMS, power=1, fmt=".2f")
      plt.text(xcenter, top, f"amplitude: {amplitudetext}", horizontalalignment="center", verticalalignment="top")
      plt.text(xcenter, 0.92*top+0.08*bottom, f"wavelength: {wavelengthtext}", horizontalalignment="center", verticalalignment="top")
      plt.text(xcenter, 0.84*top+0.16*bottom, f"RMS of noise: {RMStext}", horizontalalignment="center", verticalalignment="top")
  else:
    if sinetext:
      xcenter = np.average(ax.get_xlim())
      bottom, top = ax.get_ylim()
      if adjustylim:
        top += (top-bottom) * .1
        ax.set_ylim(bottom, top)
      RMStext = units.drawing.siunitxformat(RMS, power=1, fmt=".2f")
      plt.text(xcenter, top, f"RMS of noise: {RMStext}", horizontalalignment="center", verticalalignment="top")

  if vsxory == "y":
    ax.invert_xaxis()

  if saveas is None:
    plt.show()
  else:
    plt.savefig(saveas)
    plt.close()

  logger.debug("done")
  return x, y, yerr, p
