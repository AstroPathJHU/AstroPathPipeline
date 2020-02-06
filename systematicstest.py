from .alignmentset import AlignmentSet
from .overlap import Overlap

class OverlapWithSystematicShift(Overlap):
  def prepimage(self, systematicshift=0):
    result = super().prepimage()

    try:
      assert not self.systematicshift%2
    except AttributeError:
      raise AttributeError("Have to set the systematic shift before calling prepimage")

    if self.systematicshift:
      self.cutimages[0][:self.cutimages[0].shape[0]//2,...] = self.cutimages[0][self.systematicshift//2:self.cutimages[0].shape[0]//2+self.systematicshift//2,...]
      self.cutimages[0][self.cutimages[0].shape[0]//2:,...] = self.cutimages[0][self.cutimages[0].shape[0]//2-self.systematicshift//2:-self.systematicshift//2,...]


class AlignmentSetWithSystematicShift(AlignmentSet):
  overlaptype = OverlapWithSystematicShift
  def __init__(self, *args, systematicshift, **kwargs):
    super().__init__(*args, **kwargs)
    self.systematicshift = systematicshift
    for o in self.overlaps: o.systematicshift = systematicshift
