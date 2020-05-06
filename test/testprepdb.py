import pathlib, unittest

thisfolder = pathlib.Path(__file__).parent

class TestPrepDb(unittest.TestCase):
  def testPrepDb(self):
    sample = Sample(thisfolder/"data", "M21_1")
    sample.writemetadata()
